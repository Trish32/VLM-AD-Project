# Ported from official QCNet losses/mixture_nll_loss.py.
# The joint/ptr branch (only used for multi-agent joint prediction) relied on
# torch_scatter.segment_csr; it is not exercised by AV2 marginal eval (joint=False) and
# raises if requested rather than pulling in torch_scatter.
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.gaussian_nll_loss import GaussianNLLLoss
from losses.laplace_nll_loss import LaplaceNLLLoss
from losses.von_mises_nll_loss import VonMisesNLLLoss


class MixtureNLLLoss(nn.Module):

    def __init__(self,
                 component_distribution: Union[str, List[str]],
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(MixtureNLLLoss, self).__init__()
        self.reduction = reduction

        loss_dict = {
            'gaussian': GaussianNLLLoss,
            'laplace': LaplaceNLLLoss,
            'von_mises': VonMisesNLLLoss,
        }
        if isinstance(component_distribution, str):
            self.nll_loss = loss_dict[component_distribution](eps=eps, reduction='none')
        else:
            self.nll_loss = nn.ModuleList([loss_dict[dist](eps=eps, reduction='none')
                                           for dist in component_distribution])

    def forward(self,
                pred: torch.Tensor,          # [A, K, T, 2*D] per-mode (loc, scale) over future steps
                target: torch.Tensor,        # [A, T, D] ground-truth future
                prob: torch.Tensor,          # [A, K] mode logits
                mask: torch.Tensor,          # [A, T] valid future steps
                ptr: Optional[torch.Tensor] = None,
                joint: bool = False) -> torch.Tensor:
        # Classification loss = NLL of a mixture model: -log sum_k pi_k * p_k(target).
        # Per-mode, per-step NLL first:
        if isinstance(self.nll_loss, nn.ModuleList):
            nll = torch.cat(
                [self.nll_loss[i](pred=pred[..., [i, target.size(-1) + i]],
                                  target=target[..., [i]].unsqueeze(1))
                 for i in range(target.size(-1))],
                dim=-1)
        else:
            nll = self.nll_loss(pred=pred, target=target.unsqueeze(1))
        nll = (nll * mask.view(-1, 1, target.size(-2), 1)).sum(dim=(-2, -1))  # [A, K] masked sum over steps/dims
        if joint:
            if ptr is None:
                nll = nll.sum(dim=0, keepdim=True)
            else:
                raise NotImplementedError('joint prediction with ptr requires torch_scatter; not used for AV2 marginal')
        else:
            pass
        log_pi = F.log_softmax(prob, dim=-1)               # [A, K] log mixture weights
        loss = -torch.logsumexp(log_pi - nll, dim=-1)      # [A] mixture NLL (logsumexp_k log_pi_k + log p_k)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
