# Ported from official QCNet metrics/min_ade.py (torchmetrics -> plain accumulator).
from typing import Optional

import torch

from metrics.base import Metric
from metrics.utils import topk
from metrics.utils import valid_filter


class minADE(Metric):
    """minimum Average Displacement Error: pick the best of K modes (by final-step distance,
    min_criterion='FDE') and average its per-step L2 error over valid steps, then over agents."""

    def __init__(self, max_guesses: int = 6, device: torch.device = torch.device('cpu')) -> None:
        super(minADE, self).__init__(device=device)
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
                pred_topk[torch.arange(pred.size(0)), :, inds_last] -
                target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2), p=2, dim=-1).argmin(dim=-1)
            self.sum += ((torch.norm(pred_topk[torch.arange(pred.size(0)), inds_best] - target, p=2, dim=-1) *
                          valid_mask).sum(dim=-1) / valid_mask.sum(dim=-1)).sum()
        elif min_criterion == 'ADE':
            self.sum += ((torch.norm(pred_topk - target.unsqueeze(1), p=2, dim=-1) *
                          valid_mask.unsqueeze(1)).sum(dim=-1).min(dim=-1)[0] / valid_mask.sum(dim=-1)).sum()
        else:
            raise ValueError('{} is not a valid criterion'.format(min_criterion))
        self.count += pred.size(0)
