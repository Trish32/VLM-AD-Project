"""Run the pure-PyTorch QCNet port on a single AV2 scenario and dump the focal agent's
multimodal predicted trajectories in the world frame.

Example:
    python infer.py --root "/Users/trish/Downloads/Argoverse 2" --index 0
"""
import argparse

import torch
import torch.nn.functional as F

from datasets import ArgoverseV2Dataset
from predictors import QCNet
from transforms import TargetBuilder
from utils.data_utils import to_device


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, default='ckpt/QCNet_AV2.ckpt')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--out', type=str, default=None, help='optional .pt path to save predictions')
    args = parser.parse_args()

    device = torch.device(args.device)
    model = QCNet.from_checkpoint(args.ckpt_path, map_location='cpu').to(device).eval()
    dataset = ArgoverseV2Dataset(
        root=args.root, split=args.split,
        transform=TargetBuilder(model.num_historical_steps, model.num_future_steps),
        dim=3, num_historical_steps=model.num_historical_steps, num_future_steps=model.num_future_steps)

    data = to_device(dataset[args.index], device)
    pred = model(data)
    traj_refine = torch.cat([pred['loc_refine_pos'][..., :model.output_dim],
                             pred['scale_refine_pos'][..., :model.output_dim]], dim=-1)
    pi = pred['pi']

    eval_mask = data['agent']['category'] == 3  # focal track
    origin = data['agent']['position'][eval_mask, model.num_historical_steps - 1]
    theta = data['agent']['heading'][eval_mask, model.num_historical_steps - 1]
    cos, sin = theta.cos(), theta.sin()
    rot_mat = torch.zeros(int(eval_mask.sum()), 2, 2, device=device)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = sin
    rot_mat[:, 1, 0] = -sin
    rot_mat[:, 1, 1] = cos
    traj_world = torch.matmul(traj_refine[eval_mask, :, :, :2], rot_mat.unsqueeze(1)) + \
        origin[:, :2].reshape(-1, 1, 1, 2)
    pi_eval = F.softmax(pi[eval_mask], dim=-1)

    print(f'scenario {data["scenario_id"]}  city={data["city"]}  agents={data["agent"]["num_nodes"]}')
    print(f'focal agents: {int(eval_mask.sum())}   predicted traj shape: {tuple(traj_world.shape)}  (A,K,T,2)')
    for a in range(traj_world.size(0)):
        order = pi_eval[a].argsort(descending=True)
        print(f'  focal #{a}: mode probs (sorted) = ' +
              ', '.join(f'{pi_eval[a, k].item():.3f}' for k in order))
        best = order[0]
        ep = traj_world[a, best, -1]
        print(f'           best-mode endpoint (world) = ({ep[0].item():.2f}, {ep[1].item():.2f})')

    if args.out is not None:
        torch.save({'scenario_id': data['scenario_id'], 'traj_world': traj_world.cpu(),
                    'pi': pi_eval.cpu()}, args.out)
        print(f'saved -> {args.out}')


if __name__ == '__main__':
    main()
