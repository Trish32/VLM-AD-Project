# Ported from official QCNet metrics/min_fde.py (torchmetrics -> plain accumulator).
from typing import Optional

import torch

from metrics.base import Metric
from metrics.utils import topk
from metrics.utils import valid_filter


class minFDE(Metric):
    """minimum Final Displacement Error over K modes: the smallest endpoint L2 distance to
    the ground truth, averaged over agents. pred [N,K,T,2], target [N,T,2]."""

    def __init__(self, max_guesses: int = 6, device: torch.device = torch.device('cpu')) -> None:
        super(minFDE, self).__init__(device=device)
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True) -> None:
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, _ = topk(self.max_guesses, pred, prob)  # [N, K, T, 2]
        inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)  # [N] last valid step
        self.sum += torch.norm(pred_topk[torch.arange(pred.size(0)), :, inds_last] -          # endpoint per mode [N,K,2]
                               target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),    # gt endpoint [N,1,2]
                               p=2, dim=-1).min(dim=-1)[0].sum()                               # min over K, sum over N
        self.count += pred.size(0)
