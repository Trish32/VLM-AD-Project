# Ported from official QCNet transforms/target_builder.py (HeteroData -> dict, callable).
from typing import Dict

import torch

from utils import wrap_angle


class TargetBuilder(object):
    """
    query-centric: everything is expressed relative to each agent's pose at the last observed step
    rotates/translates the 60 future positions into that local frame → data['agent']['target']. 
    Metrics are computed in this same local frame, so no global transform is needed for eval.
    """

    def __init__(self,
                 num_historical_steps: int,
                 num_future_steps: int) -> None:
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps

    def __call__(self, data: Dict) -> Dict:
        # A = num agents, Tf = num_future_steps. Anchor = each agent's pose at the last
        # observed step; rot_mat rotates world deltas into that agent's local frame.
        origin = data['agent']['position'][:, self.num_historical_steps - 1]  # [A, 3] anchor position
        theta = data['agent']['heading'][:, self.num_historical_steps - 1]    # [A] anchor heading
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(data['agent']['num_nodes'], 2, 2)           # [A, 2, 2] world->local rotation
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        data['agent']['target'] = origin.new_zeros(data['agent']['num_nodes'], self.num_future_steps, 4)  # [A, Tf, 4]=(x,y,z,heading)
        # future positions relative to anchor, rotated into local frame -> [A, Tf, 2]
        data['agent']['target'][..., :2] = torch.bmm(data['agent']['position'][:, self.num_historical_steps:, :2] -
                                                     origin[:, :2].unsqueeze(1), rot_mat)
        if data['agent']['position'].size(2) == 3:
            data['agent']['target'][..., 2] = (data['agent']['position'][:, self.num_historical_steps:, 2] -
                                               origin[:, 2].unsqueeze(-1))         # relative height
        data['agent']['target'][..., 3] = wrap_angle(data['agent']['heading'][:, self.num_historical_steps:] -
                                                     theta.unsqueeze(-1))          # heading relative to anchor
        return data
