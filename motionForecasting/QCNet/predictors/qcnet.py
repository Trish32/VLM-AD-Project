"""QCNet predictor as a plain ``nn.Module`` (no PyTorch Lightning).

Submodule names (``encoder``/``decoder`` and all their children) are identical to the
official Lightning module, so the released ``QCNet_AV2.ckpt`` state_dict loads unchanged
(the only difference is the Lightning wrapper, whose ``Brier``/``minADE``/... metric buffers
and ``reg_loss``/``cls_loss`` carry no parameters). The loss assembly and the AV2 marginal
evaluation logic are ported from the official ``training_step``/``validation_step``.
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import MixtureNLLLoss
from losses import NLLLoss
from metrics import Brier
from metrics import MR
from metrics import minADE
from metrics import minAHE
from metrics import minFDE
from metrics import minFHE
from modules import QCNetDecoder
from modules import QCNetEncoder

HYPERPARAM_KEYS = [
    'dataset', 'input_dim', 'hidden_dim', 'output_dim', 'output_head', 'num_historical_steps',
    'num_future_steps', 'num_modes', 'num_recurrent_steps', 'num_freq_bands', 'num_map_layers',
    'num_agent_layers', 'num_dec_layers', 'num_heads', 'head_dim', 'dropout', 'pl2pl_radius',
    'time_span', 'pl2a_radius', 'a2a_radius', 'num_t2m_steps', 'pl2m_radius', 'a2m_radius',
]


class QCNet(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 num_recurrent_steps: int,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_dec_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 pl2pl_radius: float,
                 time_span,
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_t2m_steps,
                 pl2m_radius: float,
                 a2m_radius: float,
                 **kwargs) -> None:
        super(QCNet, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps

        self.encoder = QCNetEncoder(
            dataset=dataset, input_dim=input_dim, hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps, pl2pl_radius=pl2pl_radius, time_span=time_span,
            pl2a_radius=pl2a_radius, a2a_radius=a2a_radius, num_freq_bands=num_freq_bands,
            num_map_layers=num_map_layers, num_agent_layers=num_agent_layers, num_heads=num_heads,
            head_dim=head_dim, dropout=dropout)
        self.decoder = QCNetDecoder(
            dataset=dataset, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            output_head=output_head, num_historical_steps=num_historical_steps, num_future_steps=num_future_steps,
            num_modes=num_modes, num_recurrent_steps=num_recurrent_steps, num_t2m_steps=num_t2m_steps,
            pl2m_radius=pl2m_radius, a2m_radius=a2m_radius, num_freq_bands=num_freq_bands,
            num_layers=num_dec_layers, num_heads=num_heads, head_dim=head_dim, dropout=dropout)

        self.reg_loss = NLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                reduction='none')
        self.cls_loss = MixtureNLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                       reduction='none')

    # ------------------------------------------------------------------ checkpoint
    @classmethod
    def from_checkpoint(cls, ckpt_path: str, map_location='cpu') -> 'QCNet':
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        hp = ckpt['hyper_parameters']
        hp = {k: hp[k] for k in HYPERPARAM_KEYS}
        model = cls(**hp)
        model.hparams = hp
        missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
        # Drop metric/loss buffers that the Lightning module saved but this module does not own.
        missing = [k for k in missing if k.startswith(('encoder.', 'decoder.'))]
        unexpected = [k for k in unexpected if k.startswith(('encoder.', 'decoder.'))]
        if missing or unexpected:
            raise RuntimeError(f'checkpoint load mismatch: missing={missing}, unexpected={unexpected}')
        return model

    # ------------------------------------------------------------------ forward
    def forward(self, data: Dict) -> Dict[str, torch.Tensor]:
        scene_enc = self.encoder(data)
        pred = self.decoder(data, scene_enc)
        return pred

    # ------------------------------------------------------------------ training
    def training_step(self, data: Dict) -> Dict[str, torch.Tensor]:
        """Compute the QCNet training loss for one scene (propose + refine regression NLL
        + winner-take-all classification NLL), ported from the official training_step
        (AV2, output_head=False)."""
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]
        pred = self(data)
        traj_propose, traj_refine = self._assemble_traj(pred, self.output_dim, self.output_head)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        od, oh = self.output_dim, self.output_head
        l2_norm = (torch.norm(traj_propose[..., :od] - gt[..., :od].unsqueeze(1), p=2, dim=-1) *
                   reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        reg_loss_propose = self.reg_loss(traj_propose_best, gt[..., :od + oh]).sum(dim=-1) * reg_mask
        reg_loss_propose = (reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)).mean()
        reg_loss_refine = self.reg_loss(traj_refine_best, gt[..., :od + oh]).sum(dim=-1) * reg_mask
        reg_loss_refine = (reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)).mean()
        cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(), target=gt[:, -1:, :od + oh],
                                 prob=pi, mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        loss = reg_loss_propose + reg_loss_refine + cls_loss
        return {'loss': loss, 'reg_loss_propose': reg_loss_propose, 'reg_loss_refine': reg_loss_refine,
                'cls_loss': cls_loss}

    def configure_optimizers(self, lr: float = 5e-4, weight_decay: float = 1e-4, T_max: int = 64):
        """AdamW with the official decay / no-decay parameter split + CosineAnnealingLR."""
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        assert len(decay & no_decay) == 0
        assert len(param_dict.keys() - (decay | no_decay)) == 0
        optim_groups = [
            {'params': [param_dict[pn] for pn in sorted(decay)], 'weight_decay': weight_decay},
            {'params': [param_dict[pn] for pn in sorted(no_decay)], 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0.0)
        return optimizer, scheduler

    @staticmethod
    def _assemble_traj(pred, output_dim, output_head):
        # pack the decoder's loc/scale heads into [A, K, Tf, 2*output_dim] tensors
        # (location dims followed by scale dims) for both the propose and refine stages.
        if output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :output_dim], pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :output_dim], pred['conc_refine_head']], dim=-1)
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :output_dim], pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :output_dim], pred['conc_propose_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :output_dim],
                                     pred['scale_refine_pos'][..., :output_dim]], dim=-1)
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :output_dim],
                                      pred['scale_propose_pos'][..., :output_dim]], dim=-1)
        return traj_propose, traj_refine

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate_step(self, data: Dict, metrics: Dict) -> None:
        """Run one scene and update the marginal-prediction metrics (focal/scored agents),
        mirroring the official ``validation_step`` (AV2, output_head=False path)."""
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]  # [A, Tf] which future steps count
        pred = self(data)
        _, traj_refine = self._assemble_traj(pred, self.output_dim, self.output_head)  # [A, K, Tf, 2] (+scale)
        pi = pred['pi']                                                                # [A, K] mode logits
        # ground truth in the agent's local frame: [A, Tf, 3] = (x, y, heading)
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)

        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] == 3  # only the scored FOCAL agent(s) count for AV2 metrics
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        valid_mask_eval = reg_mask[eval_mask]                                       # [Ne, Tf]
        traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head]  # [Ne, K, Tf, 2]
        if not self.output_head:
            # output_head=False -> derive heading from consecutive positions (finite differences)
            traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
                                                     traj_eval[..., :2]], dim=-2)         # prepend anchor (origin)
            motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
            head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
            traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)         # [Ne, K, Tf, 3]
        pi_eval = F.softmax(pi[eval_mask], dim=-1)  # [Ne, K] normalized mode probabilities
        gt_eval = gt[eval_mask]                      # [Ne, Tf, 3]

        if eval_mask.sum() == 0:
            return
        metrics['Brier'].update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim],
                                prob=pi_eval, valid_mask=valid_mask_eval)
        metrics['minADE'].update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim],
                                 prob=pi_eval, valid_mask=valid_mask_eval)
        metrics['minAHE'].update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        metrics['minFDE'].update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim],
                                 prob=pi_eval, valid_mask=valid_mask_eval)
        metrics['minFHE'].update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        metrics['MR'].update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim],
                             prob=pi_eval, valid_mask=valid_mask_eval)

    def make_metrics(self, device: torch.device) -> Dict:
        return {
            'Brier': Brier(max_guesses=6, device=device),
            'minADE': minADE(max_guesses=6, device=device),
            'minAHE': minAHE(max_guesses=6, device=device),
            'minFDE': minFDE(max_guesses=6, device=device),
            'minFHE': minFHE(max_guesses=6, device=device),
            'MR': MR(max_guesses=6, device=device),
        }
