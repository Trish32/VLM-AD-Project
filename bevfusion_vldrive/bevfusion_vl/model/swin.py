"""
SwinTransformer (mmdet 2.x layout) — pure-PyTorch port for the MIT BEVFusion
The pipeline: image → PatchEmbed (4×4 patchify via stride-4 conv) → 4 stages of windowed attention, downsampling between them 
→ a 3-level feature pyramid (192, 384, 768) at strides (8, 16, 32).
camera backbone. Matches checkpoint keys: patch_embed.projection/norm,
stages.N.blocks.N.{norm1, attn.w_msa.(qkv,proj,relative_position_bias_table,
relative_position_index), norm2, ffn.layers.0.0/ffn.layers.1}, stages.N.downsample
.(norm,reduction), and output norms norm0..norm3 (here norm1/2/3 for out_indices [1,2,3]).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Split the image into 4x4 patches and linearly embed each -> token sequence.
    A stride-4 conv does both at once. img (B,3,256,704) -> tokens (B, 64*176, 96).
    256 x 704 image(180,224 tokens)  →  (256/4) x (704/4) = 64 x 176 = 11,264 patch-tokens
    """

    def __init__(self, in_channels=3, embed_dims=96, patch_size=4):
        super().__init__()
        # The operation "take a 4x4x3(48-value patch) block of pixels and linearly map it to a 96-vector" is exactly what a conv with kernel_size=4, stride=4 computes:
        # kernel_size=4 → each output looks at a 4×4 patch
        # stride=4 → patches don't overlap (the kernel jumps a full patch each step)
        # out_channels=96 → each patch becomes a 96-dim embedding
        self.projection = nn.Conv2d(in_channels, embed_dims, kernel_size=patch_size,
                                    stride=patch_size)        # 4x4 stride-4 = patchify+embed
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, x):
        x = self.projection(x)               # (B, embed_dims, H, W) = (B,96,64,176)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)     # (B, H*W, C) token sequence
        x = self.norm(x)
        return x, (H, W)                     # tokens + spatial shape (needed to re-grid later)


class PatchMerging(nn.Module):
    """Downsample between stages: merge each 2x2 block of tokens into one, halving H,W and
    doubling channels (the Swin analogue of a stride-2 conv).

    *** THE BUG (bug_log #13) LIVED HERE. *** The 2x2 neighbors must be concatenated in the
    EXACT channel order the checkpoint's `reduction` weight expects. mmcv/mmdet (which trained
    this checkpoint) uses nn.Unfold, giving order [c@00, c@01, c@10, c@11]. My first port used
    the microsoft cat order [00,10,01,11] -> the 4C reduction input was permuted -> stages 1-3
    silently corrupted -> soft monocular depth -> camera BEV radial smear. Using nn.Unfold here
    makes the merge bit-exact vs official. Lesson: 'loads 0/0 + runs' does NOT prove fidelity."""

    def __init__(self, in_channels):
        super().__init__()
        # Unfold pulls each 2x2 window into a 4C column in mmcv's channel order
        self.sampler = nn.Unfold(kernel_size=2, dilation=1, padding=0, stride=2)
        self.norm = nn.LayerNorm(4 * in_channels)
        self.reduction = nn.Linear(4 * in_channels, 2 * in_channels, bias=False)  # 4C -> 2C

    def forward(self, x, hw_shape):
        B, L, C = x.shape
        H, W = hw_shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)   # (B, C, H, W) back to image grid
        if H % 2 or W % 2:                            # pad to even dims if needed
            x = F.pad(x, (0, W % 2, 0, H % 2))
        x = self.sampler(x)                          # (B, 4C, L') gather 2x2 blocks (unfold order)
        x = x.transpose(1, 2)                        # (B, L', 4C)
        x = self.norm(x)
        x = self.reduction(x)                        # (B, L', 2C) linear merge
        return x, ((H + 1) // 2, (W + 1) // 2)       # tokens + halved spatial shape


def window_partition(x, ws):
    """Cut the feature map into non-overlapping ws x ws windows, stacked on the batch axis.
    x: (B, H, W, C) -> (B * num_windows, ws, ws, C). Attention then runs WITHIN each window."""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(windows, ws, H, W):
    """Inverse of window_partition: (B*num_windows, ws, ws, C) -> (B, H, W, C)."""
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowMSA(nn.Module):
    """Multi-head self-attention restricted to a single window, with a learned
    relative-position bias (Swin's replacement for absolute position embeddings)."""

    def __init__(self, embed_dims, num_heads, window_size):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = (window_size, window_size)   # (Wh, Ww), e.g. (7,7)
        head_dim = embed_dims // num_heads
        self.scale = head_dim ** -0.5                   # 1/sqrt(d) attention scaling
        Wh, Ww = self.window_size
        # one learnable bias per (relative offset, head); table size covers all offsets
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), num_heads))
        # precompute, for every (query, key) pair in a window, its flattened relative-offset index
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))   # (2,Wh,Ww)
        coords_flatten = torch.flatten(coords, 1)                                   # (2, Wh*Ww)
        rel = coords_flatten[:, :, None] - coords_flatten[:, None, :]               # (2, N, N) offsets
        rel = rel.permute(1, 2, 0).contiguous()                                     # (N, N, 2)
        rel[:, :, 0] += Wh - 1                          # shift to non-negative
        rel[:, :, 1] += Ww - 1
        rel[:, :, 0] *= 2 * Ww - 1                      # ravel the 2D offset to 1D table index
        self.register_buffer('relative_position_index', rel.sum(-1))   # (N, N)
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=True)    # produce q,k,v together
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # x: (B_, N, C) where B_ = batch*num_windows, N = ws*ws tokens per window
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]               # each (B_, heads, N, head_dim)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)                 # (B_, heads, N, N) attention logits
        Wh, Ww = self.window_size
        # look up the relative-position bias for each (q,k) pair, per head
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            Wh * Ww, Wh * Ww, -1).permute(2, 0, 1).contiguous()   # (heads, N, N)
        attn = attn + bias.unsqueeze(0)
        if mask is not None:                           # shifted windows: block cross-region attention
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)   # (B_, N, C) attended features
        return self.proj(x)


class ShiftWindowMSA(nn.Module):
    """Window attention with optional cyclic SHIFT. Alternating blocks shift the window
    grid by ws//2 so windows in the next block straddle the previous block's boundaries —
    that's how information flows ACROSS windows despite attention being window-local."""

    def __init__(self, embed_dims, num_heads, window_size, shift_size):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size            # 0 (regular) or ws//2 (shifted block)
        self.w_msa = WindowMSA(embed_dims, num_heads, window_size)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        x = query.view(B, H, W, C)
        ws = self.window_size
        pad_r = (ws - W % ws) % ws              # pad so H,W are multiples of ws
        pad_b = (ws - H % ws) % ws
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = x.shape[1], x.shape[2]         # padded dims

        if self.shift_size > 0:
            # roll the map so shifted windows align to the grid; mask blocks wrapped-around regions
            x = torch.roll(x, (-self.shift_size, -self.shift_size), dims=(1, 2))
            mask = self._attn_mask(Hp, Wp, x.device)
        else:
            mask = None

        windows = window_partition(x, ws).view(-1, ws * ws, C)   # (B*nW, ws*ws, C)
        attn = self.w_msa(windows, mask=mask)                    # attention within each window
        attn = attn.view(-1, ws, ws, C)
        x = window_reverse(attn, ws, Hp, Wp)                     # stitch windows back to (B,Hp,Wp,C)

        if self.shift_size > 0:
            x = torch.roll(x, (self.shift_size, self.shift_size), dims=(1, 2))   # undo the roll
        if pad_r or pad_b:
            x = x[:, :H, :W, :].contiguous()    # remove padding
        return x.view(B, H * W, C)

    def _attn_mask(self, Hp, Wp, device):
        """Build the attention mask for shifted windows: after rolling, a window can contain
        tokens from non-adjacent image regions; this masks attention between different regions
        (different region id -> -100 logit -> ~0 after softmax)."""
        ws, ss = self.window_size, self.shift_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        cnt = 0
        # label the 3x3 regions created by the roll with distinct ids
        for h in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
            for w in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mw = window_partition(img_mask, ws).view(-1, ws * ws)    # region id per token per window
        mask = mw.unsqueeze(1) - mw.unsqueeze(2)                 # 0 if same region, else nonzero
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)


class FFN(nn.Module):
    """mmcv FFN (the MLP after attention): layers.0=Sequential(Linear,GELU,Dropout),
    layers.1=Linear, with a residual add. Layout matches the checkpoint keys exactly."""

    def __init__(self, embed_dims, feedforward_channels):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dims, feedforward_channels), nn.GELU(), nn.Dropout(0.)),
            nn.Linear(feedforward_channels, embed_dims),
        ])
        self.dropout = nn.Dropout(0.)

    def forward(self, x, identity):
        out = self.layers[0](x)              # expand -> GELU
        out = self.layers[1](out)            # project back
        return identity + self.dropout(out)  # residual


class SwinBlock(nn.Module):
    """One Swin block: pre-norm (shifted-)window attention + residual, then pre-norm FFN +
    residual. `shift` alternates per block so windows reshuffle every other block."""

    def __init__(self, embed_dims, num_heads, feedforward_channels, window_size, shift):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dims)
        self.attn = ShiftWindowMSA(embed_dims, num_heads, window_size,
                                   window_size // 2 if shift else 0)   # shift = ws//2 on odd blocks
        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = FFN(embed_dims, feedforward_channels)

    def forward(self, x, hw_shape):
        identity = x
        x = self.norm1(x)
        x = self.attn(x, hw_shape)
        x = x + identity                     # attention residual
        identity = x
        x = self.norm2(x)
        x = self.ffn(x, identity=identity)   # FFN residual (added inside FFN)
        return x


class SwinBlockSequence(nn.Module):
    """One Swin STAGE: `depth` blocks (alternating shift) optionally followed by a
    PatchMerging downsample. Returns BOTH the pre-downsample features (for the FPN output
    at this stage's resolution) and the downsampled features (input to the next stage)."""

    def __init__(self, embed_dims, num_heads, feedforward_channels, depth,
                 window_size, downsample):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(embed_dims, num_heads, feedforward_channels, window_size,
                      shift=(i % 2 == 1))    # even block regular, odd block shifted
            for i in range(depth)])
        self.downsample = downsample

    def forward(self, x, hw_shape):
        for blk in self.blocks:
            x = blk(x, hw_shape)             # blocks keep resolution
        if self.downsample is not None:
            x_down, down_hw = self.downsample(x, hw_shape)   # PatchMerging -> half res, 2x channels
            return x, hw_shape, x_down, down_hw              # (stage out, its hw, next-stage in, its hw)
        return x, hw_shape, x, hw_shape       # last stage: no downsample


class SwinTransformer(nn.Module):
    """Swin-T camera backbone (mmdet-2.x layout). Produces a feature pyramid for the FPN.
    depths (2,2,6,2), heads (3,6,12,24), embed 96, window 7. out_indices (1,2,3) ->
    channels (192, 384, 768) at strides (8, 16, 32). 187 checkpoint tensors load 0/0."""

    def __init__(self, embed_dims=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4, out_indices=(1, 2, 3)):
        super().__init__()
        self.out_indices = out_indices            # which stages feed the FPN
        self.num_layers = len(depths)
        self.patch_embed = PatchEmbed(3, embed_dims, 4)   # img -> 96-dim tokens at stride 4
        self.drop_after_pos = nn.Dropout(0.)

        self.stages = nn.ModuleList()
        in_dims = embed_dims
        for i in range(self.num_layers):
            # stages 0-2 end with PatchMerging (downsample); last stage has none
            downsample = PatchMerging(in_dims) if i < self.num_layers - 1 else None
            self.stages.append(SwinBlockSequence(
                in_dims, num_heads[i], mlp_ratio * in_dims, depths[i],
                window_size, downsample))
            if i < self.num_layers - 1:
                in_dims *= 2                       # channels double each stage: 96->192->384->768

        self.num_features = [embed_dims * 2 ** i for i in range(self.num_layers)]   # [96,192,384,768]
        # one output LayerNorm per emitted stage
        for i in out_indices:
            self.add_module(f'norm{i}', nn.LayerNorm(self.num_features[i]))

    def forward(self, x):
        # x: (B*N_cam, 3, 256, 704)
        x, hw_shape = self.patch_embed(x)         # tokens (B,64*176,96), hw (64,176)
        x = self.drop_after_pos(x)
        outs = []
        for i, stage in enumerate(self.stages):
            # `out` = features at this stage's resolution; `x` = downsampled, fed to next stage
            out, out_hw, x, hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm = getattr(self, f'norm{i}')
                out = norm(out)
                B = out.shape[0]
                # tokens -> image grid (B, C, H, W) for the conv-based FPN
                out = out.view(B, *out_hw, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs                               # [(B,192,32,88),(B,384,16,44),(B,768,8,22)]
