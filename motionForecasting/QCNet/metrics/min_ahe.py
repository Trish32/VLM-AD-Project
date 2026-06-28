# Ported from official QCNet metrics/min_ahe.py (torchmetrics -> plain accumulator).
from typing import Optional

import torch

from metrics.base import Metric
from metrics.utils import topk
from metrics.utils import valid_filter
from utils import wrap_angle


class minAHE(Metric):
    """minimum Average Heading Error: average wrapped heading error of the best-of-K mode over
    valid steps. (Heading is derived from finite differences of positions when output_head=False.)"""

    def __init__(self, max_guesses: int = 6, device: torch.device = torch.device('cpu')) -> None:
        super(minAHE, self).__init__(device=device)
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True,
               min_criterion: str = 'FDE') -> None:
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, _ = topk(self.max_guesses, pred, prob)
        if min_criterion == 'FDE':
            inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
            inds_best = torch.norm(
                pred_topk[torch.arange(pred.size(0)), :, inds_last, :-1] -
                target[torch.arange(pred.size(0)), inds_last, :-1].unsqueeze(-2), p=2, dim=-1).argmin(dim=-1)
        elif min_criterion == 'ADE':
            inds_best = (torch.norm(pred_topk[..., :-1] - target[..., :-1].unsqueeze(1), p=2, dim=-1) *
                         valid_mask.unsqueeze(1)).sum(dim=-1).argmin(dim=-1)
        else:
            raise ValueError('{} is not a valid criterion'.format(min_criterion))
        self.sum += ((wrap_angle(pred_topk[torch.arange(pred.size(0)), inds_best, :, -1] - target[..., -1]).abs() *
                      valid_mask).sum(dim=-1) / valid_mask.sum(dim=-1)).sum()
        self.count += pred.size(0)
