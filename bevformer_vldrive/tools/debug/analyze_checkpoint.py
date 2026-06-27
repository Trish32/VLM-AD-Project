from pathlib import Path
from collections import Counter

import torch
"""
A quick script to analyze the checkpoint file and print a summary of the parameter counts by module prefix.
This can help verify that the expected parameters are present and identify any unexpected ones.
python analyze_checkpoint.py 2>/dev/null suppresses warnings about missing keys when loading the checkpoint, since we're only interested in the raw parameter keys here.
"""
# Resolve the checkpoint relative to the project root so the script runs from anywhere
# (it lives in tools/debug/, three levels below the repo root).
ROOT = Path(__file__).resolve().parent.parent.parent
ckpt = torch.load(ROOT / 'model/checkpoints/bevformer_tiny_fp16_epoch_24.pth', map_location='cpu')
raw = ckpt.get('state_dict', ckpt)
# Top-level prefixes (e.g. backbone, neck, pts_bbox_head, query_embedding, …) are a good first check for expected vs. unexpected parameter counts.
prefixes = Counter(k.split('.')[0] for k in raw.keys())
# Full module names (e.g. backbone.layer1, neck.fpn_convs, …) are also useful to check for unexpected parameter counts.
modules = Counter('.'.join(k.split('.')[:2]) for k in raw.keys())

# Print summary of top-level prefixes and second-level modules, sorted by count.
for p, n in sorted(prefixes.items()):
    print(f'{n:4d}  {p}')
print(f'total keys: {len(raw)}\n')
for m, n in sorted(modules.items()):
    if n > 5:  # only show modules with many parameters, to avoid overwhelming output
        print(f'{n:5d}  {m}')