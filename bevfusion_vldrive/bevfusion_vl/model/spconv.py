"""
Pure-PyTorch sparse 3D convolution, replacing CUDA spconv for the BEVFusion
SparseEncoder. Implements submanifold (SubMConv3d) and regular strided
(SparseConv3d) convolutions via rulebook gather / matmul / scatter-add.

Conventions (match spconv / the checkpoint):
- A sparse tensor = (features (N,C), indices (N,4) int [b, d0, d1, d2],
  spatial_shape (S0,S1,S2), batch_size).
- Weight layout from the checkpoint is (k0, k1, k2, Cin, Cout); kernel dim i
  aligns with coord spatial dim i and spatial_shape[i].
- Cross-correlation: out[c] = sum_o W[o] @ in[c + (o - center)]   (submanifold)
  For strided conv, output position q receives input c = q*s - p + o.
Convs are bias-free (BatchNorm follows separately).
"""
from __future__ import annotations

import itertools

import torch
import torch.nn as nn


class SparseTensor:
    """A 3D sparse tensor = ONLY the N occupied voxels, never the full dense grid.
    The det grid is 1440x1440x41 (~85M cells) but a LiDAR sweep fills only ~12k of
    them, so we store the occupied set and treat everything else as zero.

    features      (N, C)      one feature vector per occupied voxel
    indices       (N, 4)      int64 coords [b, d0, d1, d2] of those voxels
    spatial_shape (S0,S1,S2)  full dense grid dims (defines what is "in bounds")
    batch_size    int
    """
    __slots__ = ("features", "indices", "spatial_shape", "batch_size", "_hash")

    def __init__(self, features, indices, spatial_shape, batch_size):
        self.features = features            # (N, C)
        self.indices = indices              # (N, 4) int64 [b, d0, d1, d2]
        self.spatial_shape = list(spatial_shape)   # [S0, S1, S2]
        self.batch_size = batch_size
        self._hash = None                   # (N,) int64 coord hashes

    def hashes(self):
        # Cache the per-voxel flattened-coordinate hash; reused across all 27 kernel taps.
        if self._hash is None:
            self._hash = _coord_hash(self.indices, self.spatial_shape, self.batch_size)
        return self._hash                   # (N,) int64


def _coord_hash(indices, spatial_shape, batch_size):
    """Flatten a 4D coord (b,d0,d1,d2) into ONE int64 key (a ravel_multi_index).
    Two voxels collide iff they are the same voxel -> equality test == voxel identity.
    indices: (M, 4)  ->  (M,) int64."""
    S0, S1, S2 = spatial_shape
    b, d0, d1, d2 = indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]  # each (M,)
    return ((b.long() * S0 + d0.long()) * S1 + d1.long()) * S2 + d2.long()  # (M,) unique int64 per (b,x,y,z)


def _build_lookup(keys):
    """Sort occupied-voxel hashes once so neighbors can be found by binary search later.
    keys: (N,) int64  ->  sorted_keys (N,), sorted_idx (N,)  where
    sorted_idx[i] = original row of the i-th smallest key."""
    order = torch.argsort(keys)             # (N,) permutation that sorts keys
    return keys[order], order               # sorted_keys (N,), sorted_idx (N,)


def _query(sorted_keys, sorted_idx, query_keys):
    """For each query key, return the matching occupied-voxel row, or -1 if that cell is empty.
    This is the MPS-friendly replacement for a hash map: one vectorized binary search.
    sorted_keys/sorted_idx: (N,);  query_keys: (Q,)  ->  out: (Q,) int64 row indices."""
    pos = torch.searchsorted(sorted_keys, query_keys)   # (Q,) (indices)where each query_key WOULD insert into sorted_key
    pos = pos.clamp(max=sorted_keys.numel() - 1)        # guard right-edge out-of-bounds
    matched = sorted_keys[pos] == query_keys            # (Q,) bool: is the key actually present?
    out = torch.full_like(query_keys, -1)               # (Q,) default -1 = cell not occupied
    out[matched] = sorted_idx[pos[matched]]             # hits -> map back to original row index
    return out                                          # (Q,) int64


class SubMConv3d(nn.Module):
    """Submanifold sparse conv: output sites == input sites (sparsity pattern is FROZEN).
    Crucial because a normal conv would dilate the occupied set every layer (1 voxel ->
    27), making the tensor dense within a few layers. Submanifold keeps it sparse.

    out[c] = sum_o  W[o] @ in[c + (o - center)]    over occupied neighbors only.
    Cost = 27 matmuls over N voxels, NOT anything over the 85M-cell grid."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        k = kernel_size
        self.k = (k, k, k) if isinstance(k, int) else tuple(k)
        # weight layout (k0,k1,k2,Cin,Cout) is dictated by the checkpoint
        self.weight = nn.Parameter(torch.zeros(*self.k, in_channels, out_channels))

    def forward(self, x: SparseTensor) -> SparseTensor:
        coords = x.indices                  # (N, 4) [b, d0, d1, d2]
        N = coords.shape[0]
        Cout = self.weight.shape[-1]
        out = x.features.new_zeros((N, Cout))           # (N, Cout) accumulator
        sk, si = _build_lookup(x.hashes())              # sorted hashes (N,), rows (N,)
        center = [(ks - 1) // 2 for ks in self.k]       # kernel center, e.g. (1,1,1) for 3x3x3
        spat = coords[:, 1:]                            # (N, 3) spatial coords (drop batch col)
        # Loop over the (up to) 27 kernel taps -- the ONLY python loop; everything inside is vectorized over N.
        for o in itertools.product(*[range(ks) for ks in self.k]):
            off = torch.tensor([o[i] - center[i] for i in range(3)],
                               device=coords.device, dtype=coords.dtype)   # (3,) this tap's offset
            nb = spat + off                  # (N, 3) for each output voxel, the input neighbor at this tap
            nb_full = torch.cat([coords[:, :1], nb], dim=1)   # (N, 4) re-attach batch col
            valid = ((nb >= 0).all(1) &      # (N,) bool: neighbor inside the grid?
                     (nb[:, 0] < x.spatial_shape[0]) &
                     (nb[:, 1] < x.spatial_shape[1]) &
                     (nb[:, 2] < x.spatial_shape[2]))
            qk = _coord_hash(nb_full, x.spatial_shape, x.batch_size)       # (N,) neighbor hashes
            in_idx = torch.full((N,), -1, device=coords.device, dtype=torch.long)   # (N,) -1 = empty/OOB
            if valid.any():
                in_idx[valid] = _query(sk, si, qk[valid])   # look up which neighbors are occupied
            hit = in_idx >= 0                # (N,) bool: output voxels with an occupied neighbor at this tap
            if hit.any():
                W = self.weight[o[0], o[1], o[2]]           # (Cin, Cout) this tap's weight slice
                # gather occupied neighbor feats (n_hit,Cin) -> matmul -> scatter-add into outputs
                out[hit] += x.features[in_idx[hit]] @ W     # (n_hit, Cout)
        return SparseTensor(out, coords, x.spatial_shape, x.batch_size)  # same coords, new feats


class SparseConv3d(nn.Module):
    """Regular strided sparse conv (the stride-2 downsampling between encoder stages).
    Unlike submanifold conv, output coords are NEW and FEWER, so we cannot reuse the
    input index set -- we derive each output voxel from the inputs that map onto it.

    Relation (inverse of submanifold): input voxel c with tap o feeds output voxel
        q = (c + padding - o) / stride        (only when that division is exact).
    Two passes: (1) enumerate every (input, tap) -> output coord; (2) dedup the produced
    outputs and scatter-add each tap's gathered+matmul'd contributions."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.k = tuple(kernel_size) if not isinstance(kernel_size, int) else (kernel_size,) * 3
        self.stride = tuple(stride) if not isinstance(stride, int) else (stride,) * 3
        self.padding = tuple(padding) if not isinstance(padding, int) else (padding,) * 3
        self.weight = nn.Parameter(torch.zeros(*self.k, in_channels, out_channels))  # (k0,k1,k2,Cin,Cout)

    def _out_shape(self, in_shape):
        # standard conv output-size formula, applied per spatial dim
        return [(in_shape[i] + 2 * self.padding[i] - self.k[i]) // self.stride[i] + 1
                for i in range(3)]

    def forward(self, x: SparseTensor) -> SparseTensor:
        coords = x.indices                  # (N, 4) [b, d0, d1, d2]
        device = coords.device
        out_shape = self._out_shape(x.spatial_shape)    # [O0, O1, O2] downsampled dims
        spat = coords[:, 1:]                # (N, 3) input spatial coords
        b = coords[:, :1]                  # (N, 1) input batch column

        # First pass: enumerate all (input, offset) -> output coords, collect unique outputs.
        pair_in, pair_off, pair_outcoord = [], [], []   # lists of (P_o,), (P_o,), (P_o,4)
        for oi, o in enumerate(itertools.product(*[range(ks) for ks in self.k])):
            # output voxel q such that input voxel c = q*s - p + o  => q = (c + padding - o)/s
            num = spat + torch.tensor([self.padding[i] - o[i] for i in range(3)],
                                      device=device, dtype=coords.dtype)   # (N, 3) numerator (c + p - o)
            s = torch.tensor(self.stride, device=device, dtype=coords.dtype)   # (3,)
            divisible = ((num % s) == 0).all(1)  # (N,) a tap only lands on a valid strided output if (c+p−o) % stride == 0
            q = torch.div(num, s, rounding_mode='floor')    # (N, 3) candidate output coord
            in_range = ((q >= 0).all(1) &                   # (N,) is q inside the downsampled grid?
                        (q[:, 0] < out_shape[0]) & (q[:, 1] < out_shape[1]) & (q[:, 2] < out_shape[2]))
            mask = divisible & in_range                     # (N,) inputs that contribute via THIS tap
            if mask.any():
                pair_in.append(torch.nonzero(mask, as_tuple=False).squeeze(1))   # (P_o,) input row indices
                pair_off.append(torch.full((int(mask.sum()),), oi, device=device, dtype=torch.long))  # (P_o,) tap id
                pair_outcoord.append(torch.cat([b[mask], q[mask]], dim=1))       # (P_o, 4) output coords

        if not pair_in:                     # nothing maps anywhere -> empty output tensor
            empty = x.features.new_zeros((0, self.weight.shape[-1]))
            ec = coords.new_zeros((0, 4))
            return SparseTensor(empty, ec, out_shape, x.batch_size)

        all_in = torch.cat(pair_in)         # (P,) input row per pair   (P = sum of P_o over taps)
        all_off = torch.cat(pair_off)       # (P,) tap id per pair
        all_out = torch.cat(pair_outcoord)  # (P, 4) output coord per pair

        # unique output coords: many pairs land on the same output voxel
        out_hash = _coord_hash(all_out, out_shape, x.batch_size)     # (P,) hash of each output coord
        uniq, inv = torch.unique(out_hash, return_inverse=True)      # uniq (M,); inv (P,) pair->unique row
        M = uniq.numel()                    # number of distinct output voxels
        # recover one representative 4-coord per unique output voxel
        rep = torch.zeros((M, 4), device=device, dtype=coords.dtype)     # (M, 4)
        rep[inv] = all_out                  # duplicates write identical coords, so this is well-defined

        Cout = self.weight.shape[-1]
        out_feat = x.features.new_zeros((M, Cout))                       # (M, Cout) accumulator
        Wflat = self.weight.reshape(-1, self.weight.shape[-2], self.weight.shape[-1])  # (K,Cin,Cout)
        for oi in range(Wflat.shape[0]):    # loop taps again, accumulating each tap's contribution
            sel = all_off == oi             # (P,) pairs belonging to this tap
            if not sel.any():
                continue
            in_idx = all_in[sel]            # (P_o,) input rows for this tap
            out_idx = inv[sel]              # (P_o,) destination unique-output rows
            contrib = x.features[in_idx] @ Wflat[oi]      # (P_o, Cout) gather -> matmul
            out_feat.index_add_(0, out_idx, contrib)  # multiple input voxels summing into one output voxel get accumulated correctly and in parallel
        return SparseTensor(out_feat, rep, out_shape, x.batch_size)


def to_dense(x: SparseTensor):
    """Exit the sparse world: scatter the N occupied voxels back into a full dense grid.
    SparseTensor -> dense (B, C, S0, S1, S2), mostly zeros. The downstream BEV code then
    collapses the z-axis (S2) into channels, turning the 3D volume into a 2D BEV map."""
    B = x.batch_size
    C = x.features.shape[1]
    S0, S1, S2 = x.spatial_shape
    dense = x.features.new_zeros((B, S0, S1, S2, C))    # (B, S0, S1, S2, C)
    idx = x.indices                                    # (N, 4)
    # place each occupied voxel's feature at its [b, d0, d1, d2] cell
    dense[idx[:, 0].long(), idx[:, 1].long(), idx[:, 2].long(), idx[:, 3].long()] = x.features
    return dense.permute(0, 4, 1, 2, 3).contiguous()   # (B, C, S0, S1, S2)
