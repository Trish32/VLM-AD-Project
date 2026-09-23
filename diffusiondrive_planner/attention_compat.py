"""Device-agnostic replacement for upstream's flash-attention modules.

`MultiheadFlashAttention` hard-requires the `flash_attn` package, and its inner
`FlashAttention.forward` asserts `q.is_cuda` and fp16/bf16 dtype. That makes the whole
`diff_operation_order` loop unrunnable on a Mac, and unrunnable on CPU anywhere.

Flash attention is a *kernel*, not a different algorithm: it computes exactly
`softmax(QK^T / sqrt(d)) V`, just with better memory traffic. `F.scaled_dot_product_attention`
computes the same thing and dispatches to a fused kernel on CUDA, memory-efficient
attention elsewhere. So this is a faithful substitution, not an approximation.

**Parameter names are load-bearing.** Upstream's state_dict keys are
`attn.in_proj_weight`, `attn.in_proj_bias`, `attn.out_proj.{weight,bias}`. These classes
reproduce them exactly so the official checkpoint loads 0 missing / 0 unexpected — that
is asserted in `tests/test_attention_compat.py`, not assumed.

Numerics: upstream forces fp16 on CUDA (`dtype=torch.float16` in the FlashMHA ctor).
We keep the input dtype, so a CPU/MPS run is fp32 and will differ from a CUDA fp16 run
in the usual last-few-digits way. That is a precision difference, not a behavioural one.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


#: torch >= 2.0 has a fused SDPA kernel; the DiffusionDrive nusc stack pins torch 1.12
#: (mmdet 2.x era), so a manual path is required for the module to be importable and
#: testable in that env at all. Both compute exactly softmax(QK^T/sqrt(d))V.
_HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


def _attend(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
    """scaled_dot_product_attention, with a manual fallback for torch < 2.0."""
    if _HAS_SDPA:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )

    scale = q.shape[-1] ** -0.5
    scores = (q @ k.transpose(-2, -1)) * scale
    if is_causal:
        sq, sk = q.shape[-2], k.shape[-2]
        causal = torch.ones(sq, sk, dtype=torch.bool, device=q.device).tril(diagonal=sk - sq)
        scores = scores.masked_fill(~causal, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    attn = scores.softmax(dim=-1)
    if dropout_p:
        attn = F.dropout(attn, p=dropout_p)
    return attn @ v


def _in_projection_packed(q, k, v, w, b=None):
    """Port of attention.py:26. Splits the packed QKV weight into three linears."""
    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


class FlashMHACompat(nn.Module):
    """Drop-in for `FlashMHA`, built on `scaled_dot_product_attention`.

    Same parameters, same packed-QKV layout, same default softmax scale
    (`1/sqrt(head_dim)`), so weights transfer verbatim.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        batch_first: bool = True,
        attention_dropout: float = 0.0,
        causal: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        assert batch_first, "upstream asserts batch_first; keeping the same contract"
        if embed_dim % num_heads:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.dropout_p = attention_dropout

        # Names must match upstream exactly — see module docstring.
        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None
        if not bias:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, key_padding_mask=None):
        """(B, S, E) x3 -> (out (B, S, E), None).

        Returns a 2-tuple because upstream's caller indexes `[0]`.
        """
        q, k, v = _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)

        b, sq, _ = q.shape
        sk = k.shape[1]
        # (B, S, E) -> (B, H, S, D) as SDPA expects
        q = q.view(b, sq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, sk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, sk, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            # True = keep, matching upstream's unpad_input semantics.
            attn_mask = key_padding_mask[:, None, None, :].to(torch.bool)

        context = _attend(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=self.causal,
        )
        context = context.transpose(1, 2).reshape(b, sq, self.embed_dim)
        return self.out_proj(context), None


class MultiheadAttentionCompat(nn.Module):
    """Drop-in for `MultiheadFlashAttention`.

    Reproduces the wrapper's positional-encoding and residual semantics, which are the
    parts most easily got wrong:

    * `query_pos` is added to the query and `key_pos` to the key, but **never to value**
      (the mmcv convention — the same one that mattered in the Sparse4D port);
    * when `key_pos` is absent but `query_pos` is present, `query_pos` is reused as
      `key_pos` **only if the shapes match**, otherwise it is dropped with a warning;
    * the residual is `identity + dropout(proj_drop(out))`, where `identity` defaults to
      the ORIGINAL query — i.e. before `query_pos` was added.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        dropout_layer: dict | None = None,
        batch_first: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if "dropout" in kwargs:  # upstream's deprecated alias
            attn_drop = kwargs.pop("dropout")

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = True  # upstream hardcodes this too
        self.attn = FlashMHACompat(
            embed_dim=embed_dims, num_heads=num_heads, attention_dropout=attn_drop
        )
        self.proj_drop = nn.Dropout(proj_drop)
        drop_prob = (dropout_layer or {}).get("drop_prob", 0.0)
        self.dropout_layer = nn.Dropout(drop_prob) if dropout_layer else nn.Identity()

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        assert attn_mask is None, "attn mask not supported now."
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query  # BEFORE query_pos is added
        if key_pos is None and query_pos is not None:
            if query_pos.shape == key.shape:
                key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        out = self.attn(q=query, k=key, v=value, key_padding_mask=key_padding_mask)[0]
        return identity + self.dropout_layer(self.proj_drop(out))
