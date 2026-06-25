"""QCNet AttentionLayer, reimplemented without torch_geometric ``MessagePassing``.

The module structure (submodule names, LayerNorm placement, gating, FFN) is identical to
the official ``layers/attention_layer.py`` so the official checkpoint loads unchanged. The
``propagate/message/update`` message passing is replaced by explicit gather +
segment-softmax + scatter-add over the edge index, which runs on MPS.
"""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from utils import scatter_sum
from utils import segment_softmax
from utils import weight_init


class AttentionLayer(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 bipartite: bool,
                 has_pos_emb: bool,
                 **kwargs) -> None:
        super(AttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.has_pos_emb = has_pos_emb
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_k = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
        self.to_v = nn.Linear(hidden_dim, head_dim * num_heads)
        if has_pos_emb:
            self.to_k_r = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
            self.to_v_r = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_s = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_g = nn.Linear(head_dim * num_heads + hidden_dim, head_dim * num_heads)
        self.to_out = nn.Linear(head_dim * num_heads, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.ff_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        if bipartite:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = nn.LayerNorm(hidden_dim)
        else:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = self.attn_prenorm_x_src
        if has_pos_emb:
            self.attn_prenorm_r = nn.LayerNorm(hidden_dim)
        self.attn_postnorm = nn.LayerNorm(hidden_dim)
        self.ff_prenorm = nn.LayerNorm(hidden_dim)
        self.ff_postnorm = nn.LayerNorm(hidden_dim)
        self.apply(weight_init)

    def forward(self,
                x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                r: Optional[torch.Tensor],
                edge_index: torch.Tensor) -> torch.Tensor:
        # x: node features [N, D] (or (src[Ns,D], dst[Nd,D]) when bipartite).
        # r: per-edge relative-position embedding [E, D]. edge_index: [2, E] = [src, dst].
        # Pre-norm transformer block: residual( attn ) then residual( feed-forward ).
        if isinstance(x, torch.Tensor):
            x_src = x_dst = self.attn_prenorm_x_src(x)
        else:
            x_src, x_dst = x
            x_src = self.attn_prenorm_x_src(x_src)
            x_dst = self.attn_prenorm_x_dst(x_dst)
            x = x[1]  # residual stream is the destination nodes
        if self.has_pos_emb and r is not None:
            r = self.attn_prenorm_r(r)
        x = x + self.attn_postnorm(self._attn_block(x_src, x_dst, r, edge_index))
        x = x + self.ff_postnorm(self._ff_block(self.ff_prenorm(x)))
        return x

    def _attn_block(self,
                    x_src: torch.Tensor,
                    x_dst: torch.Tensor,
                    r: Optional[torch.Tensor],
                    edge_index: torch.Tensor) -> torch.Tensor:
        # Multi-head attention as explicit message passing (replaces PyG MessagePassing).
        H, D = self.num_heads, self.head_dim
        q = self.to_q(x_dst).view(-1, H, D)      # [Nd, H, D] queries from destination nodes
        k = self.to_k(x_src).view(-1, H, D)      # [Ns, H, D] keys from source nodes
        v = self.to_v(x_src).view(-1, H, D)      # [Ns, H, D] values from source nodes
        src, dst = edge_index[0], edge_index[1]  # flow source_to_target -> j=src, i=dst
        q_i = q[dst]                             # [E, H, D] query of each edge's target
        k_j = k[src]                             # [E, H, D] key of each edge's source
        v_j = v[src]                             # [E, H, D] value of each edge's source
        if self.has_pos_emb and r is not None:   # inject relative geometry into key & value
            k_j = k_j + self.to_k_r(r).view(-1, H, D)
            v_j = v_j + self.to_v_r(r).view(-1, H, D)
        sim = (q_i * k_j).sum(dim=-1) * self.scale            # [E, H] scaled dot-product score
        attn = segment_softmax(sim, dst, num_nodes=x_dst.size(0))  # softmax over edges per target node
        attn = self.attn_drop(attn)
        msg = v_j * attn.unsqueeze(-1)                        # [E, H, D] weighted values
        agg = scatter_sum(msg, dst, dim_size=x_dst.size(0))   # [Nd, H, D] sum messages into targets
        inputs = agg.reshape(-1, H * D)                       # [Nd, H*D]
        # gated residual update: blend aggregated message with a self-projection of the dst node
        g = torch.sigmoid(self.to_g(torch.cat([inputs, x_dst], dim=-1)))
        agg = inputs + g * (self.to_s(x_dst) - inputs)
        return self.to_out(agg)                               # [Nd, D]

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff_mlp(x)
