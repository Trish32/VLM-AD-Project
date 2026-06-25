# Ported from official QCNet metrics/mr.py (torchmetrics -> plain accumulator).
from typing import Optional

import torch

from metrics.base import Metric
from metrics.utils import topk
from metrics.utils import valid_filter


class MR(Metric):
    """Miss Rate: fraction of agents whose best-of-K endpoint error exceeds miss_threshold
    (2 m). Lower is better; noisiest of the metrics on small subsets."""

    def __init__(self, max_guesses: int = 6, device: torch.device = torch.device('cpu')) -> None:
        super(MR, self).__init__(device=device)
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True,
               miss_criterion: str = 'FDE',
               miss_threshold: float = 2.0) -> None:
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, _ = topk(self.max_guesses, pred, prob)
        if miss_criterion == 'FDE':
            inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
            self.sum += (torch.norm(pred_topk[torch.arange(pred.size(0)), :, inds_last] -
                                    target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                                    p=2, dim=-1).min(dim=-1)[0] > miss_threshold).sum()
        elif miss_criterion == 'MAXDE':
            self.sum += (((torch.norm(pred_topk - target.unsqueeze(1),
                                      p=2, dim=-1) * valid_mask.unsqueeze(1)).max(dim=-1)[0]).min(dim=-1)[0] >
                         miss_threshold).sum()
        else:
            raise ValueError('{} is not a valid criterion'.format(miss_criterion))
        self.count += pred.size(0)
