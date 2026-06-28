# Geometry helpers (ported verbatim from official QCNet utils/geometry.py).
import math

import torch


def angle_between_2d_vectors(
        ctr_vector: torch.Tensor,
        nbr_vector: torch.Tensor) -> torch.Tensor:
    # signed angle (-pi, pi] of nbr relative to ctr, via atan2(cross, dot). This is the core
    # of QCNet's rotation invariance: relative geometry is expressed as angles in each node's frame.
    return torch.atan2(ctr_vector[..., 0] * nbr_vector[..., 1] - ctr_vector[..., 1] * nbr_vector[..., 0],
                       (ctr_vector[..., :2] * nbr_vector[..., :2]).sum(dim=-1))


def angle_between_3d_vectors(
        ctr_vector: torch.Tensor,
        nbr_vector: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.cross(ctr_vector, nbr_vector, dim=-1).norm(p=2, dim=-1),
                       (ctr_vector * nbr_vector).sum(dim=-1))


def side_to_directed_lineseg(
        query_point: torch.Tensor,
        start_point: torch.Tensor,
        end_point: torch.Tensor) -> str:
    cond = ((end_point[0] - start_point[0]) * (query_point[1] - start_point[1]) -
            (end_point[1] - start_point[1]) * (query_point[0] - start_point[0]))
    if cond > 0:
        return 'LEFT'
    elif cond < 0:
        return 'RIGHT'
    else:
        return 'CENTER'


def wrap_angle(
        angle: torch.Tensor,
        min_val: float = -math.pi,
        max_val: float = math.pi) -> torch.Tensor:
    # wrap an angle (e.g. a heading difference) into (-pi, pi] so errors don't blow up at ±pi
    return min_val + (angle + max_val) % (max_val - min_val)
