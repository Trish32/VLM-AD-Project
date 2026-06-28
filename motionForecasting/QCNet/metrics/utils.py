# Ported from official QCNet metrics/utils.py. The ptr/joint code paths relied on
# torch_scatter (segment_csr/gather_csr); they are only used for batched joint metrics and
# raise here rather than introducing torch_scatter. The AV2 marginal eval path (ptr=None,
# joint=False) is unaffected.
from typing import Optional, Tuple

import torch


def topk(
        max_guesses: int,
        pred: torch.Tensor,            # [N, K, T, 2] candidate trajectories
        prob: Optional[torch.Tensor] = None,  # [N, K] mode probabilities
        ptr: Optional[torch.Tensor] = None,
        joint: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep the ``max_guesses`` highest-probability modes per agent (here K=6=max_guesses,
    so it just renormalizes the probabilities and returns all modes)."""
    max_guesses = min(max_guesses, pred.size(1))
    if max_guesses == pred.size(1):
        if prob is not None:
            prob = prob / prob.sum(dim=-1, keepdim=True)
        else:
            prob = pred.new_ones((pred.size(0), max_guesses)) / max_guesses
        return pred, prob
    else:
        if prob is not None:
            if joint:
                if ptr is None:
                    inds_topk = torch.topk((prob / prob.sum(dim=-1, keepdim=True)).mean(dim=0, keepdim=True),
                                           k=max_guesses, dim=-1, largest=True, sorted=True)[1]
                    inds_topk = inds_topk.repeat(pred.size(0), 1)
                else:
                    raise NotImplementedError('joint topk with ptr requires torch_scatter; not used for AV2 marginal')
            else:
                inds_topk = torch.topk(prob, k=max_guesses, dim=-1, largest=True, sorted=True)[1]
            pred_topk = pred[torch.arange(pred.size(0)).unsqueeze(-1).expand(-1, max_guesses), inds_topk]
            prob_topk = prob[torch.arange(pred.size(0)).unsqueeze(-1).expand(-1, max_guesses), inds_topk]
            prob_topk = prob_topk / prob_topk.sum(dim=-1, keepdim=True)
        else:
            pred_topk = pred[:, :max_guesses]
            prob_topk = pred.new_ones((pred.size(0), max_guesses)) / max_guesses
        return pred_topk, prob_topk


def valid_filter(
        pred: torch.Tensor,
        target: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
        keep_invalid_final_step: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
                                                       torch.Tensor, torch.Tensor]:
    """Drop agents with no valid future step (keep_invalid_final_step=True keeps an agent if
    ANY step is valid). Returns the filtered pred/target/prob/valid_mask + a trivial ptr."""
    if valid_mask is None:
        valid_mask = target.new_ones(target.size()[:-1], dtype=torch.bool)
    if keep_invalid_final_step:
        filter_mask = valid_mask.any(dim=-1)
    else:
        filter_mask = valid_mask[:, -1]
    pred = pred[filter_mask]
    target = target[filter_mask]
    if prob is not None:
        prob = prob[filter_mask]
    valid_mask = valid_mask[filter_mask]
    if ptr is not None:
        raise NotImplementedError('valid_filter with ptr requires torch_scatter; not used for AV2 marginal')
    else:
        ptr = target.new_tensor([0, target.size(0)])
    return pred, target, prob, valid_mask, ptr
