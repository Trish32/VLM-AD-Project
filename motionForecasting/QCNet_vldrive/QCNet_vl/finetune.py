"""Fine-tune the pure-PyTorch QCNet on Argoverse 2 (MPS).

Starts from the official QCNet_AV2 checkpoint and continues training with the official loss
(propose + refine regression NLL + winner-take-all classification NLL), AdamW with the
official decay/no-decay split, and a cosine schedule. The model runs one scene per forward
(bs=1), so a target batch size is emulated with gradient accumulation (--accum_steps).

Example (fine-tune demo on the extracted val scenes; point --split train once train.tar is
extracted under the AV2 root):
    python finetune.py --root "/Users/trish/Downloads/Argoverse 2" --split val \
        --ckpt_path ckpt/QCNet_AV2.ckpt --max_scenarios 100 --max_epochs 3 \
        --accum_steps 16 --lr 5e-5
"""
import argparse
import os
import random
import time

import torch

from datasets import ArgoverseV2Dataset
from predictors import QCNet
from transforms import TargetBuilder
from utils.data_utils import to_device


def run_eval(model, dataset, device, limit):
    model.eval()
    metrics = model.make_metrics(device)
    n = min(limit, len(dataset))
    for i in range(n):
        model.evaluate_step(to_device(dataset[i], device), metrics)
    model.train()
    return {k: metrics[k].compute().item() for k in ['minADE', 'minFDE', 'MR']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, default='ckpt/QCNet_AV2.ckpt')
    parser.add_argument('--split', type=str, default='val',
                        help="split to fine-tune on ('train' once downloaded; 'val' for a demo)")
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--max_scenarios', type=int, default=None)
    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--accum_steps', type=int, default=16, help='scenes per optimizer step (batch emulation)')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--save_dir', type=str, default='ckpt')
    parser.add_argument('--eval_limit', type=int, default=40)
    parser.add_argument('--seed', type=int, default=2023)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    model = QCNet.from_checkpoint(args.ckpt_path, map_location='cpu').to(device)
    model.train()
    optimizer, scheduler = model.configure_optimizers(lr=args.lr, weight_decay=args.weight_decay,
                                                      T_max=args.max_epochs)

    dataset = ArgoverseV2Dataset(
        root=args.root, split=args.split,
        transform=TargetBuilder(model.num_historical_steps, model.num_future_steps),
        dim=3, num_historical_steps=model.num_historical_steps, num_future_steps=model.num_future_steps,
        max_scenarios=args.max_scenarios)
    order = list(range(len(dataset)))
    print(f'fine-tune on {len(order)} {args.split} scenes | lr={args.lr} accum={args.accum_steps} '
          f'epochs={args.max_epochs}')
    print('before:', run_eval(model, dataset, device, args.eval_limit))

    for epoch in range(args.max_epochs):
        random.shuffle(order)
        optimizer.zero_grad()
        running = {'loss': 0.0, 'reg_loss_propose': 0.0, 'reg_loss_refine': 0.0, 'cls_loss': 0.0}
        t0 = time.time()
        for step, idx in enumerate(order):
            data = to_device(dataset[idx], device)
            out = model.training_step(data)
            (out['loss'] / args.accum_steps).backward()
            for k in running:
                running[k] += out[k].item()
            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
        scheduler.step()
        m = {k: running[k] / len(order) for k in running}
        print(f'epoch {epoch}: loss={m["loss"]:.4f} (propose {m["reg_loss_propose"]:.4f} '
              f'refine {m["reg_loss_refine"]:.4f} cls {m["cls_loss"]:.4f}) | lr={scheduler.get_last_lr()[0]:.2e} '
              f'| {time.time()-t0:.0f}s')

    print('after :', run_eval(model, dataset, device, args.eval_limit))
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, 'QCNet_AV2_finetuned.ckpt')
    torch.save({'state_dict': model.state_dict(), 'hyper_parameters': model.hparams}, out_path)
    print(f'saved fine-tuned checkpoint -> {out_path}')


if __name__ == '__main__':
    main()
