"""Checkpoint loading for the pure-PyTorch BEVFusion-PP port.

The module names in `BEVF_FasterRCNN` were chosen to match the official
mmdet3d state_dict prefixes exactly, so loading is a direct (non-strict)
load_state_dict. This reports any missing/unexpected keys.
"""
from __future__ import annotations

import torch


def load_bevfusion_pp(model, ckpt_path, map_location="cpu", verbose=True):
    ck = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    sd = ck.get("state_dict", ck)
    info = model.load_state_dict(sd, strict=False)
    missing = [k for k in info.missing_keys]
    unexpected = [k for k in info.unexpected_keys]
    if verbose:
        print(f"[checkpoint] loaded {ckpt_path}")
        print(f"[checkpoint] missing keys   : {len(missing)}")
        for k in missing[:20]:
            print("    -", k)
        print(f"[checkpoint] unexpected keys: {len(unexpected)}")
        for k in unexpected[:20]:
            print("    +", k)
    return missing, unexpected
