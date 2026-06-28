"""Pure-PyTorch, MPS-friendly replacements for the PyTorch-Geometric / torch_cluster /
torch_scatter operations used by QCNet.

Semantics are matched to the upstream libraries so that the ported model is numerically
faithful to the official implementation:

* ``segment_softmax``  -> ``torch_geometric.utils.softmax`` (segment softmax over dst index)
* ``scatter_sum``      -> ``torch_scatter.scatter(..., reduce='sum')`` via ``index_add_``
* ``radius``           -> ``torch_cluster.radius``      (per-``y`` neighbor search, [y, x])
* ``radius_graph``     -> ``torch_cluster.radius_graph`` (per-target neighbor search, [src, dst])
* ``dense_to_sparse``  -> ``torch_geometric.utils.dense_to_sparse`` (2D and batched 3D)
* ``bipartite_dense_to_sparse`` / ``coalesce`` / ``merge_edges`` -> QCNet utils/graph.py
* ``subgraph``         -> ``torch_geometric.utils.subgraph`` (filter, no relabel)
"""
from typing import List, Optional, Tuple

import torch


# --------------------------------------------------------------------------------------
# scatter / segment-softmax (replaces torch_scatter + torch_geometric.utils.softmax)
# --------------------------------------------------------------------------------------
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # src: [E, *] per-edge values; index: [E] dst node of each edge -> out: [dim_size, *]
    out = src.new_zeros((dim_size,) + tuple(src.shape[1:]))
    idx = index
    if src.dim() > 1:
        idx = index.view((-1,) + (1,) * (src.dim() - 1)).expand_as(src)  # broadcast index to src's shape
    return out.scatter_add(0, idx, src)


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # per-group max; empty groups stay 0 but are never gathered (every present index has >=1 elem)
    out = src.new_zeros((dim_size,) + tuple(src.shape[1:]))
    idx = index
    if src.dim() > 1:
        idx = index.view((-1,) + (1,) * (src.dim() - 1)).expand_as(src)
    return out.scatter_reduce(0, idx, src, reduce='amax', include_self=False)


def segment_softmax(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Numerically-stable softmax of ``src`` grouped by ``index`` (the dst node of each edge),
    matching ``torch_geometric.utils.softmax``. src: [E, H] scores; index: [E] dst node;
    returns [E, H] attention weights that sum to 1 within each dst node's incoming edges."""
    src_max = scatter_max(src.detach(), index, num_nodes)  # [num_nodes, H] per-dst max (for stability)
    src_max = src_max.index_select(0, index)               # [E, H] broadcast back to edges
    out = (src - src_max).exp()                            # [E, H] exp of shifted scores
    out_sum = scatter_sum(out, index, num_nodes).index_select(0, index) + 1e-16  # [E, H] per-dst denom
    return out / out_sum


# --------------------------------------------------------------------------------------
# radius graphs (replaces torch_cluster.radius / radius_graph)
# --------------------------------------------------------------------------------------
def _capped_neighbors(dist: torch.Tensor, r: float, max_num_neighbors: int
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given ``dist[q, c]`` (distance from each query ``q`` to each candidate ``c``), return
    ``(q_idx, c_idx)`` for all pairs within radius ``r``, keeping at most ``max_num_neighbors``
    nearest candidates per query."""
    Q, C = dist.shape                       # Q queries (centers), C candidates
    k = min(max_num_neighbors, C)
    if k == 0:
        empty = dist.new_zeros(0, dtype=torch.long)
        return empty, empty
    masked = dist.masked_fill(dist > r, float('inf'))   # drop out-of-radius candidates
    vals, cols = masked.topk(k, dim=1, largest=False)   # [Q, k] nearest-k per query (cap)
    valid = torch.isfinite(vals)                        # discard inf padding (fewer than k in radius)
    q_idx = torch.arange(Q, device=dist.device).unsqueeze(1).expand(-1, k)[valid]  # [E]
    c_idx = cols[valid]                                                            # [E]
    return q_idx, c_idx


def _batch_groups(batch: Optional[torch.Tensor], n: int, device) -> List[torch.Tensor]:
    if batch is None:
        return [torch.arange(n, device=device)]
    return [(batch == b).nonzero(as_tuple=False).flatten() for b in torch.unique(batch)]


def radius(x: torch.Tensor, y: torch.Tensor, r: float,
           batch_x: Optional[torch.Tensor] = None, batch_y: Optional[torch.Tensor] = None,
           max_num_neighbors: int = 32) -> torch.Tensor:
    """For each point in ``y`` find points in ``x`` within distance ``r``.
    Returns edge_index ``[2, E]`` with row0 = index into ``y``, row1 = index into ``x``."""
    device = x.device
    y_groups = _batch_groups(batch_y, y.size(0), device)
    if batch_x is None:
        x_groups = [torch.arange(x.size(0), device=device)] * len(y_groups)
    else:
        ub = torch.unique(batch_y) if batch_y is not None else None
        x_groups = [(batch_x == b).nonzero(as_tuple=False).flatten() for b in ub]
    rows, cols = [], []
    for yg, xg in zip(y_groups, x_groups):  # process one batch group (e.g. one timestep) at a time
        if yg.numel() == 0 or xg.numel() == 0:
            continue
        dist = torch.cdist(y[yg], x[xg])     # [|yg|, |xg|] pairwise distances within the group
        q, c = _capped_neighbors(dist, r, max_num_neighbors)  # local indices
        rows.append(yg[q])  # map back to global y index
        cols.append(xg[c])  # map back to global x index
    if len(rows) == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=device)
    return torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)


def radius_graph(x: torch.Tensor, r: float, batch: Optional[torch.Tensor] = None,
                 loop: bool = False, max_num_neighbors: int = 32) -> torch.Tensor:
    """Build a radius graph over ``x``. Returns edge_index ``[2, E] = [source, target]``
    (flow ``source_to_target``), capped at ``max_num_neighbors`` neighbors per target."""
    device = x.device
    groups = _batch_groups(batch, x.size(0), device)
    srcs, dsts = [], []
    for g in groups:
        if g.numel() == 0:
            continue
        dist = torch.cdist(x[g], x[g])     # [|g|, |g|] within-group pairwise distances
        if not loop:
            dist.fill_diagonal_(float('inf'))  # exclude self (avoid 0*inf=nan from eye trick)
        # query = target node (cap per target); candidate = source neighbor
        tgt, src = _capped_neighbors(dist, r, max_num_neighbors)
        srcs.append(g[src])  # global source index
        dsts.append(g[tgt])  # global target index
    if len(srcs) == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=device)
    return torch.stack([torch.cat(srcs), torch.cat(dsts)], dim=0)


# --------------------------------------------------------------------------------------
# dense <-> sparse + edge utilities (replaces torch_geometric.utils + QCNet utils/graph.py)
# --------------------------------------------------------------------------------------
def dense_to_sparse(adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match ``torch_geometric.utils.dense_to_sparse`` for 2D and batched 3D adjacency."""
    if adj.dim() == 2:
        edge_index = adj.nonzero().t().contiguous()      # [2, E]
        return edge_index, adj[edge_index[0], edge_index[1]]
    elif adj.dim() == 3:
        # batched [B, N, M]: flatten to a single node index per batch (b*N+row, b*M+col),
        # exactly matching PyG so it lines up with the agent-major node ordering upstream.
        idx = adj.nonzero()  # [E, 3] = (batch, row, col)
        edge_attr = adj[idx[:, 0], idx[:, 1], idx[:, 2]]
        row = idx[:, 1] + adj.size(-2) * idx[:, 0]
        col = idx[:, 2] + adj.size(-1) * idx[:, 0]
        return torch.stack([row, col], dim=0), edge_attr
    else:
        raise ValueError('dense_to_sparse expects a 2D or 3D adjacency')


def bipartite_dense_to_sparse(adj: torch.Tensor) -> torch.Tensor:
    """Ported verbatim from QCNet utils/graph.py."""
    index = adj.nonzero(as_tuple=True)
    if len(index) == 3:
        batch_src = index[0] * adj.size(1)
        batch_dst = index[0] * adj.size(2)
        index = (batch_src + index[1], batch_dst + index[2])
    return torch.stack(index, dim=0)


def subgraph(subset: torch.Tensor, edge_index: torch.Tensor,
             edge_attr: Optional[torch.Tensor] = None
             ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Filter edges to those with both endpoints in ``subset`` (boolean node mask).
    No node relabeling, matching ``torch_geometric.utils.subgraph`` defaults."""
    mask = subset[edge_index[0]] & subset[edge_index[1]]
    edge_index = edge_index[:, mask]
    if edge_attr is not None:
        edge_attr = edge_attr[mask]
    return edge_index, edge_attr


def coalesce(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None,
             reduce: str = 'add') -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Merge duplicate edges, reducing their attributes. Matches the subset of
    ``torch_geometric.utils.coalesce`` behaviour used by QCNet (reduce='max'/'add')."""
    if edge_index.numel() == 0:
        return edge_index, edge_attr
    num_nodes = int(edge_index.max()) + 1
    key = edge_index[0].long() * num_nodes + edge_index[1].long()  # unique (src,dst) -> scalar key
    perm = key.argsort()                          # sort edges so duplicates are adjacent
    key, edge_index = key[perm], edge_index[:, perm]
    is_first = torch.ones_like(key, dtype=torch.bool)
    is_first[1:] = key[1:] != key[:-1]            # mark first edge of each unique group
    out_edge_index = edge_index[:, is_first]      # [2, E_unique]
    if edge_attr is None:
        return out_edge_index, None
    edge_attr = edge_attr[perm]
    group = is_first.cumsum(0) - 1                # group id per (sorted) edge
    num_groups = int(group[-1]) + 1
    reduce_map = {'add': 'sum', 'sum': 'sum', 'mean': 'mean', 'max': 'amax', 'min': 'amin'}
    out_attr = edge_attr.new_zeros((num_groups,) + tuple(edge_attr.shape[1:])).float()
    gidx = group if edge_attr.dim() == 1 else group.view((-1,) + (1,) * (edge_attr.dim() - 1)).expand_as(edge_attr)
    out_attr = out_attr.scatter_reduce(0, gidx, edge_attr.float(), reduce=reduce_map[reduce],
                                       include_self=False)
    return out_edge_index, out_attr.to(edge_attr.dtype)


def merge_edges(edge_indices: List[torch.Tensor], edge_attrs: Optional[List[torch.Tensor]] = None,
                reduce: str = 'add') -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Ported from QCNet utils/graph.py."""
    edge_index = torch.cat(edge_indices, dim=1)
    edge_attr = torch.cat(edge_attrs, dim=0) if edge_attrs is not None else None
    return coalesce(edge_index=edge_index, edge_attr=edge_attr, reduce=reduce)
