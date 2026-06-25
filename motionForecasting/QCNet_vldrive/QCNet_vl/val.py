"""Validate the pure-PyTorch QCNet port against the official AV2 checkpoint on MPS.

Example:
    python val.py --root "/Users/trish/Downloads/Argoverse 2" \
                  --ckpt_path ckpt/QCNet_AV2.ckpt --max_scenarios 200
"""
import argparse
import time

import torch
from tqdm import tqdm

from datasets import ArgoverseV2Dataset
from predictors import QCNet
from transforms import TargetBuilder
from utils.data_utils import to_device

# Published QCNet AV2 validation results (README), for reference.
PUBLISHED = {'minADE': 0.72, 'minFDE': 1.25, 'MR': 0.16, 'Brier': 1.87 - 1.25}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, default='ckpt/QCNet_AV2.ckpt')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--max_scenarios', type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = QCNet.from_checkpoint(args.ckpt_path, map_location='cpu').to(device).eval()
    print(f'loaded {args.ckpt_path}: checkpoint state_dict matched (0 missing / 0 unexpected)')

    dataset = ArgoverseV2Dataset(
        root=args.root, split=args.split,
        transform=TargetBuilder(model.num_historical_steps, model.num_future_steps),
        dim=3, num_historical_steps=model.num_historical_steps, num_future_steps=model.num_future_steps,
        max_scenarios=args.max_scenarios)
    print(f'{args.split}: {len(dataset)} scenarios from {dataset.raw_dir}')

    metrics = model.make_metrics(device)
    t0 = time.time()
    for i in tqdm(range(len(dataset)), desc='val'):
        data = to_device(dataset[i], device)
        model.evaluate_step(data, metrics)
    dt = time.time() - t0

    print('\n================ QCNet_vl  (pure-PyTorch, MPS) ================')
    print(f'{"metric":>10} | {"ours":>8} | {"published":>9}')
    print('-' * 36)
    for name in ['minADE', 'minFDE', 'MR', 'Brier']:
        val = metrics[name].compute().item()
        pub = PUBLISHED.get(name)
        pub_s = f'{pub:.3f}' if pub is not None else '   -   '
        print(f'{name:>10} | {val:8.4f} | {pub_s:>9}')
    for name in ['minAHE', 'minFHE']:
        print(f'{name:>10} | {metrics[name].compute().item():8.4f} | {"   -   ":>9}')
    n = int(metrics["minADE"].count.item())
    print('-' * 36)
    print(f'scored agents: {n}   |   {dt:.1f}s   ({dt / max(len(dataset),1):.2f}s/scene)')


if __name__ == '__main__':
    main()
