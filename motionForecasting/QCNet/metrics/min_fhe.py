# Ported from official QCNet metrics/min_fhe.py (torchmetrics -> plain accumulator).
from typing import Optional

import torch

from metrics.base import Metric
from metrics.utils import topk
from metrics.utils import valid_filter
from utils import wrap_angle


class minFHE(Metric):
    """minimum Final Heading Error: wrapped heading error at the final step for the mode whose
    endpoint is closest to the ground truth, averaged over agents."""

    def __init__(self, max_guesses: int = 6, device: torch.device = torch.device('cpu')) -> None:
        super(minFHE, self).__init__(device=device)
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True) -> None:
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, _ = topk(self.max_guesses, pred, prob)
        inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
        inds_best = torch.norm(
            pred_topk[torch.arange(pred.size(0)), :, inds_last, :-1] -
            target[torch.arange(pred.size(0)), inds_last, :-1].unsqueeze(-2), p=2, dim=-1).argmin(dim=-1)
        self.sum += wrap_angle(pred_topk[torch.arange(pred.size(0)), inds_best, inds_last, -1] -
                               target[torch.arange(pred.size(0)), inds_last, -1]).abs().sum()
        self.count += pred.size(0)
