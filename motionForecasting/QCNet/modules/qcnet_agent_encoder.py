"""QCNet agent encoder. Faithful port; PyG/torch_cluster ops replaced by
``utils.pyg_compat``, HeteroData accessed as nested dict (single-scene / bs=1).

Note: even for a single scene, ``batch_s``/``batch_pl`` group nodes by timestep so that
``radius``/``radius_graph`` only connect agents/polygons within the same historical step."""
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn

from layers.attention_layer import AttentionLayer
from layers.fourier_embedding import FourierEmbedding
from utils import angle_between_2d_vectors
from utils import dense_to_sparse
from utils import radius
from utils import radius_graph
from utils import subgraph
from utils import weight_init
from utils import wrap_angle


class QCNetAgentEncoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(QCNetAgentEncoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        if dataset == 'argoverse_v2':
            input_dim_x_a = 4
            input_dim_r_t = 4
            input_dim_r_pl2a = 3
            input_dim_r_a2a = 3
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))

        if dataset == 'argoverse_v2':
            self.type_a_emb = nn.Embedding(10, hidden_dim)
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))
        self.x_a_emb = FourierEmbedding(input_dim=input_dim_x_a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_t_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2a_emb = FourierEmbedding(input_dim=input_dim_r_pl2a, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=input_dim_r_a2a, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.t_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.apply(weight_init)

    def forward(self,
                data: Dict,
                map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Shape symbols: A = num agents, T = num_historical_steps (50), D = hidden_dim (128),
        # Pl = num map polygons, H/hd = attention heads / head_dim.
        # Two node orderings appear below and the code transposes between them each layer:
        #   * "agent-major" (a*T + t): used by the temporal (t) attention.
        #   * "time-major"  (t*A + a): used by the spatial (pl2a, a2a) attention, so that a
        #     per-timestep batch index can confine neighbor search to the same step.
        mask = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()  # [A, T]
        pos_a = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].contiguous()  # [A, T, 2]
        # per-step displacement (zero at t=0); the agent's "motion vector"
        motion_vector_a = torch.cat([pos_a.new_zeros(data['agent']['num_nodes'], 1, self.input_dim),
                                     pos_a[:, 1:] - pos_a[:, :-1]], dim=1)  # [A, T, 2]
        head_a = data['agent']['heading'][:, :self.num_historical_steps].contiguous()  # [A, T]
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)  # [A, T, 2] unit heading
        pos_pl = data['map_polygon']['position'][:, :self.input_dim].contiguous()
        orient_pl = data['map_polygon']['orientation'].contiguous()
        if self.dataset == 'argoverse_v2':
            vel = data['agent']['velocity'][:, :self.num_historical_steps, :self.input_dim].contiguous()
            length = width = height = None
            categorical_embs = [
                self.type_a_emb(data['agent']['type'].long()).repeat_interleave(repeats=self.num_historical_steps,
                                                                                dim=0),
            ]
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        if self.dataset == 'argoverse_v2':
            # 4 rotation-invariant per-step features: |motion|, angle(motion vs heading),
            # |velocity|, angle(velocity vs heading)  -> [A, T, 4]
            x_a = torch.stack(
                [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]),
                 torch.norm(vel[:, :, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel[:, :, :2])], dim=-1)
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        # Fourier-embed the 4 features (+ agent-type embedding) -> token per (agent, step)
        x_a = self.x_a_emb(continuous_inputs=x_a.view(-1, x_a.size(-1)), categorical_embs=categorical_embs)  # [A*T, D]
        x_a = x_a.view(-1, self.num_historical_steps, self.hidden_dim)  # [A, T, D]

        # ---- temporal edges: connect steps of the SAME agent (agent-major nodes a*T+t) ----
        pos_t = pos_a.reshape(-1, self.input_dim)          # [A*T, 2]
        head_t = head_a.reshape(-1)                        # [A*T]
        head_vector_t = head_vector_a.reshape(-1, 2)       # [A*T, 2]
        mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)     # [A, T, T] valid (t_i, t_j) pairs
        edge_index_t = dense_to_sparse(mask_t)[0]          # [2, E_t], nodes flattened as a*T+t
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]                 # causal: attend to past only
        edge_index_t = edge_index_t[:, edge_index_t[1] - edge_index_t[0] <= self.time_span]  # within time_span steps
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
             rel_head_t,
             edge_index_t[0] - edge_index_t[1]], dim=-1)
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)

        # ---- spatial nodes: switch to TIME-major (t*A+a / t*Pl+pl) so a per-step batch index
        #      keeps neighbor search within one timestep ----
        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)   # [T*A, 2]
        head_s = head_a.transpose(0, 1).reshape(-1)                 # [T*A]
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)  # [T*A, 2]
        mask_s = mask.transpose(0, 1).reshape(-1)                   # [T*A]
        pos_pl = pos_pl.repeat(self.num_historical_steps, 1)        # [T*Pl, 2] (map is static, tiled over time)
        orient_pl = orient_pl.repeat(self.num_historical_steps)     # [T*Pl]
        batch_s = torch.arange(self.num_historical_steps,           # [T*A] step id per agent-node
                               device=pos_a.device).repeat_interleave(data['agent']['num_nodes'])
        batch_pl = torch.arange(self.num_historical_steps,          # [T*Pl] step id per polygon-node
                                device=pos_pl.device).repeat_interleave(data['map_polygon']['num_nodes'])
        # polygon -> agent edges within pl2a_radius, same timestep: [2, E] = [pl_node, agent_node]
        edge_index_pl2a = radius(x=pos_s[:, :2], y=pos_pl[:, :2], r=self.pl2a_radius, batch_x=batch_s, batch_y=batch_pl,
                                 max_num_neighbors=300)
        edge_index_pl2a = edge_index_pl2a[:, mask_s[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]])
        r_pl2a = torch.stack(
            [torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2]),
             rel_orient_pl2a], dim=-1)
        r_pl2a = self.r_pl2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)  # [E_pl2a, D]
        # agent <-> agent edges within a2a_radius, same timestep; then keep only valid endpoints
        edge_index_a2a = radius_graph(x=pos_s[:, :2], r=self.a2a_radius, batch=batch_s, loop=False,
                                      max_num_neighbors=300)
        edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
             rel_head_a2a], dim=-1)
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

        # Each layer = temporal attn (agent-major) -> map->agent attn -> agent<->agent attn
        # (both time-major). The reshape/transpose pairs convert between the two node orderings.
        for i in range(self.num_layers):
            x_a = x_a.reshape(-1, self.hidden_dim)                       # [A*T, D] agent-major
            x_a = self.t_attn_layers[i](x_a, r_t, edge_index_t)          # temporal self-attention
            x_a = x_a.reshape(-1, self.num_historical_steps,
                              self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)  # -> [T*A, D] time-major
            x_a = self.pl2a_attn_layers[i]((map_enc['x_pl'].transpose(0, 1).reshape(-1, self.hidden_dim), x_a), r_pl2a,
                                           edge_index_pl2a)              # map polygons -> agents
            x_a = self.a2a_attn_layers[i](x_a, r_a2a, edge_index_a2a)    # agent <-> agent (same step)
            x_a = x_a.reshape(self.num_historical_steps, -1, self.hidden_dim).transpose(0, 1)  # -> [A, T, D]

        return {'x_a': x_a}  # [A, T, D] per-agent per-step scene encoding
