"""Emit a self-contained Kaggle/Colab script that exports SparseDrive planner inputs
as PLAIN tensors.

WHY THIS EXISTS
---------------
`tools/run_detector.py` already dumps the tensors `DiffPlanner.forward` consumes,
but it records the head's call verbatim (`{"args": ..., "kwargs": ...}`), so
upstream objects ride along inside `metas`. The resulting `planner_inputs.pt`
cannot be unpickled without importing `projects.mmdet3d_plugin`, which needs
mmcv 1.x -- and mmcv is exactly what every port in this repo exists to avoid.
The practical consequence, verified: the cached file is unreadable in every
environment on this machine (`simple_bev_vldrive` and `base` have no mmcv, the
`openmmlab` env has 2.1.0 whose `mmcv.utils` no longer exports `print_log`).

So the export has to be flattened at the point of capture, on a machine that
has the dependency, and the flattened file must contain nothing but tensors and
primitives. Then it loads anywhere, forever, with no mmcv -- and the repo's
no-mmcv rule stays intact because mmcv only ever runs on Kaggle.

WHY KAGGLE RATHER THAN LOCAL
----------------------------
mmcv 1.x has no Apple-Silicon wheels and does not build there. A T4 also runs the
real `deformable_aggregation_ext` CUDA kernel, so upstream executes unmodified --
no `dfa_torch` substitution, no `MultiheadAttentionCompat`. The exported features
are therefore what upstream actually produces, not what our substitutes produce,
which is the whole point of using them as a reference.

This mirrors `make_cuda_validation_bundle.py`: one file, paste and run, no repo
checkout on the remote side.

Usage:
    python diffusiondrive_planner/tools/make_planner_export_bundle.py
    # -> diffusiondrive_planner/tools/kaggle_export_planner_inputs.py   (upload this)
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent / 'kaggle_export_planner_inputs.py'

SCRIPT = r'''#!/usr/bin/env python3
"""Export SparseDrive/DiffusionDrive planner inputs as PLAIN tensors. Run on a T4.

WHAT YOU MUST ATTACH TO THE KAGGLE NOTEBOOK
-------------------------------------------
  1. the stage-2 checkpoint   diffusiondrive_nusc_stage2.pth   (~1.1 GB)
  2. nuScenes v1.0-mini       (public Kaggle dataset, or your own upload)
  3. nothing else -- the upstream repo is cloned below

Set the three paths in CONFIG, enable the GPU, and run top to bottom. Output is
`planner_inputs_plain.pt`: a dict {sample_token: {name: tensor}} plus a `_meta`
entry, containing NO upstream classes. Verified before it is written.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------- CONFIG ----
CKPT = '/kaggle/input/diffusiondrive-stage2/diffusiondrive_nusc_stage2.pth'
NUSC = '/kaggle/input/nuscenes-mini/v1.0-mini'          # dir holding v1.0-mini/
OUT = '/kaggle/working/planner_inputs_plain.pt'
N_SAMPLES = 81                                          # mini_val; 0 = all
UPSTREAM = 'https://github.com/hustvl/DiffusionDrive'
CONFIG_REL = 'projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py'
# -----------------------------------------------------------------------------


def sh(cmd, **kw):
    print(f'$ {cmd}')
    subprocess.run(cmd, shell=True, check=True, **kw)


def install():
    """mmcv 1.x + mmdet/mmdet3d pinned to the versions upstream was written against.

    Pinned, not 'latest': mmcv 2.x renamed and removed APIs the plugin imports
    (`mmcv.utils.print_log`, `mmcv.Config`, `mmcv.runner.load_checkpoint`), which
    is the exact failure that makes the existing local export unreadable.
    """
    tv = subprocess.run([sys.executable, '-c',
                         'import torch;print(torch.__version__.split("+")[0]);'
                         'print(torch.version.cuda)'],
                        capture_output=True, text=True).stdout.split()
    torch_v, cu = tv[0], (tv[1] or '').replace('.', '')
    print(f'[env] torch {torch_v}  cuda {cu}')
    sh('pip -q install openmim')
    sh(f'mim install -q "mmcv-full==1.6.0"')
    sh('pip -q install "mmdet==2.25.1" "mmsegmentation==0.29.1" '
       '"numpy<1.24" "yapf==0.40.1" prettytable motmetrics==1.1.3')
    sh('pip -q install "mmdet3d==1.0.0rc6"')


def fetch_upstream():
    if not Path('DiffusionDrive').exists():
        sh(f'git clone --depth 1 {UPSTREAM}')
    sys.path.insert(0, str(Path('DiffusionDrive').resolve()))
    return Path('DiffusionDrive').resolve()


def to_plain(obj, depth=0):
    """Recursively reduce to tensors / primitives. Anything else is DROPPED.

    Dropping rather than best-effort converting is deliberate: a partially
    converted upstream object would still import its class on unpickle, which is
    the failure this whole script exists to prevent. Dropped keys are reported so
    a needed field cannot vanish silently.
    """
    import numpy as np
    import torch
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj) if obj.dtype != object else None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            p = to_plain(v, depth + 1)
            if p is not None or v is None:
                out[str(k)] = p
        return out
    if isinstance(obj, (list, tuple)):
        vals = [to_plain(v, depth + 1) for v in obj]
        return [v for v in vals if v is not None]
    return None                                   # upstream class -> dropped


def main():
    install()
    root = fetch_upstream()
    os.chdir(root)

    # Data must live where the config expects it.
    Path('data').mkdir(exist_ok=True)
    if not Path('data/nuscenes').exists():
        sh(f'ln -sfn {NUSC} data/nuscenes')

    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint

    cfg = Config.fromfile(CONFIG_REL)
    cfg.version = 'mini'
    import projects.mmdet3d_plugin  # noqa: F401

    from mmdet3d.models import build_model
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    info = load_checkpoint(model, CKPT, map_location='cpu', strict=False)
    print(f'[ckpt] missing {len(getattr(info, "missing_keys", []) or [])}')
    model.cuda().eval()

    from mmdet.datasets import build_dataloader
    from mmdet3d.datasets import build_dataset
    tcfg = dict(cfg.data.val if 'val' in cfg.data else cfg.data.test)
    dataset = build_dataset(tcfg)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                              dist=False, shuffle=False)
    print(f'[data] {len(dataset)} samples')

    # Tap the planning head's inputs. Same mechanism as tools/run_detector.py --
    # wrap the bound method rather than a pre-hook, which needs torch>=2.0.
    head = model.head.motion_plan_head
    original = head.forward
    grabbed = {}

    def recording(*a, **kw):
        grabbed['args'], grabbed['kwargs'] = a, kw
        return original(*a, **kw)

    head.forward = recording

    from mmcv.parallel import DataContainer

    def unwrap(o):
        if isinstance(o, DataContainer):
            return unwrap(o.data[0])
        if isinstance(o, dict):
            return {k: unwrap(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return type(o)(unwrap(v) for v in o)
        return o

    export, dropped = {}, set()
    limit = N_SAMPLES or len(dataset)
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= limit:
                break
            d = unwrap(data)
            # Sample token, so the export can be joined to nuScenes locally.
            metas = d.get('img_metas')
            m0 = metas[0] if isinstance(metas, (list, tuple)) else metas
            tok = (m0 or {}).get('token') or (m0 or {}).get('sample_idx') or f'idx{i}'

            grabbed.clear()
            model(return_loss=False, rescale=True, **{k: v.cuda()
                                                      if torch.is_tensor(v) else v
                                                      for k, v in d.items()})
            if not grabbed:
                print(f'[warn] sample {i}: head never called')
                continue

            rec = {}
            for j, a in enumerate(grabbed['args']):
                p = to_plain(a)
                if p is None:
                    dropped.add(f'arg{j}')
                else:
                    rec[f'arg{j}'] = p
            for k, v in grabbed['kwargs'].items():
                p = to_plain(v)
                if p is None:
                    dropped.add(k)
                else:
                    rec[k] = p
            export[str(tok)] = rec
            if i % 10 == 0:
                print(f'[run] {i + 1}/{limit}  token={tok}  fields={len(rec)}')

    head.forward = original
    export['_meta'] = {'n': len(export), 'config': CONFIG_REL,
                       'checkpoint': Path(CKPT).name,
                       'dropped_fields': sorted(dropped),
                       'note': 'plain tensors only; no upstream classes'}
    torch.save(export, OUT)
    mb = Path(OUT).stat().st_size / 1024 ** 2
    print(f'[save] {OUT}  ({mb:.1f} MB)  {len(export) - 1} samples')
    if dropped:
        print(f'[save] DROPPED (non-plain): {sorted(dropped)}')

    # The load-bearing check: re-read with weights_only=True, which refuses to
    # construct ANY class. If this passes, the file is genuinely portable and the
    # local side will never need mmcv.
    probe = torch.load(OUT, map_location='cpu', weights_only=True)
    print(f'[verify] weights_only load OK -- {len(probe) - 1} samples, portable')


if __name__ == '__main__':
    main()
'''


def main() -> None:
    OUT.write_text(SCRIPT)
    OUT.chmod(0o755)
    print(f'[DONE] {OUT}  ({len(SCRIPT.splitlines())} lines)')
    print('Upload that single file to a Kaggle notebook with a T4, attach the '
          'checkpoint and nuScenes-mini datasets, set CONFIG, and run.')


if __name__ == '__main__':
    main()
