# Ported verbatim from official QCNet losses/laplace_nll_loss.py.
import torch
import torch.nn as nn


class LaplaceNLLLoss(nn.Module):
    """Negative log-likelihood of a Laplace distribution. ``pred`` packs [loc, scale] in its
    last dim; returns -log p(target) = log(2*scale) + |target - loc| / scale (elementwise)."""

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)   # each [..., 1]: predicted mean and Laplace scale b
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=self.eps)       # keep scale strictly positive (no grad through clamp)
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale  # [..., 1] per-element NLL
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
