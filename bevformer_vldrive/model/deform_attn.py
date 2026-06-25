"""
Pure PyTorch multi-scale deformable attention core.
Replaces the mmcv CUDA extension (ms_deform_attn_forward / backward).
Works on MPS, CUDA, and CPU via F.grid_sample bilinear interpolation.
"""

import torch
import torch.nn.functional as F


def ms_deform_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Args:
        value:                (N, S, num_heads, head_dim)
        value_spatial_shapes: (num_levels, 2) long tensor  [(H0,W0), ...]
        sampling_locations:   (N, Q, num_heads, num_levels, num_points, 2)  in [0,1]
        attention_weights:    (N, Q, num_heads, num_levels, num_points)     softmax weights
    Returns:
        output:               (N, Q, num_heads * head_dim)
    """
    N, _, num_heads, head_dim = value.shape
    _, Q, _, num_levels, num_points, _ = sampling_locations.shape  # e.g. (6, 900, 8, 1, 8, 2)

    value_list = value.split(
        [int(H) * int(W) for H, W in value_spatial_shapes], dim=1
    )

    # grid_sample expects coordinates in [-1, 1]
    sampling_grids = 2.0 * sampling_locations - 1.0  # (N, Q, num_heads, num_levels, num_points, 2)

    sampled_list = []
    for lid in range(num_levels):
        H, W = int(value_spatial_shapes[lid, 0]), int(value_spatial_shapes[lid, 1])

        # Reshape image features: (N, H*W, num_heads, head_dim) -> (N*num_heads, head_dim, H, W)
        # the C5 ResNet feature map is 15 high × 25 wide (after 32× stride on 480×800 input)
        #   (6, 15*25, 8_heads, 32_headdim) → (6*8, 32, 15, 25)
        v = value_list[lid].flatten(2).transpose(1, 2).reshape(N * num_heads, head_dim, H, W)

        # Sampling grid: (N, Q, num_heads, num_points, 2) -> (N*num_heads, Q, num_points, 2)
        sg = sampling_grids[:, :, :, lid].transpose(1, 2).flatten(0, 1)

        # bilinear grid_sample -> (N*num_heads, head_dim, Q, num_points) (6*8, 32, k, 8_points)
        sampled = F.grid_sample(v, sg,  # input features and sampling grid
                                mode='bilinear', 
                                padding_mode='zeros',   # ← out-of-FOV points [-1, 1] return 0, not reflected/clamped
                                align_corners=False)
        sampled_list.append(sampled)

    # (N*num_heads, head_dim, Q, num_levels*num_points)
    sampling_values = torch.stack(sampled_list, dim=-2).flatten(-2)

    # attention_weights -> (N*num_heads, 1, Q, num_levels*num_points)
    attn = attention_weights.transpose(1, 2).reshape(
        N * num_heads, 1, Q, num_levels * num_points
    )

    # (N*num_heads, head_dim, Q) -> (N, Q, embed_dim)
    output = (sampling_values * attn).sum(-1)
    output = output.view(N, num_heads * head_dim, Q).transpose(1, 2).contiguous()  # (N, Q, num_heads*head_dim)
    return output
