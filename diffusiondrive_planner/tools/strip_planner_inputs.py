"""Read `planner_inputs.pt` WITHOUT mmcv, and re-save it as plain tensors.

WHY THIS REPLACES THE KAGGLE ROUTE
----------------------------------
`make_planner_export_bundle.py` assumed the cache could only be flattened on a
machine that has mmcv 1.x, because `tools/run_detector.py` pickled the head's
call verbatim and upstream objects ride inside it. That reasoning was wrong, and
the cost of being wrong was large: mmcv 1.x wants Python <= 3.11 AND x86 AND
CUDA, this Mac fails two of those and the Kaggle image fails the first
(`pkgutil.ImpImporter`, removed in 3.12), so the export sat blocked.

Tensors never pass through the missing classes. torch reconstructs them with
`_rebuild_tensor_v2` from storages in the zip archive -- a path that touches
nothing but torch. Only the CONTAINER objects need to resolve, and a stub class
satisfies the unpickler as well as the real one does. Measured on the actual
file: 110 tensors, and exactly THREE classes had to be stubbed
(SparseBox3DEncoder, SparseBox3DKeyPointsGenerator, mmcv's Linear wrapper).

VERIFICATION, AND WHY THE OBVIOUS ONE IS CIRCULAR
-------------------------------------------------
The natural check is "compare the stripped tensors against the original cache".
That cannot be run: reading the original cache is precisely what needs mmcv, so
the check needs the dependency it is supposed to prove unnecessary.

The non-circular form compares against the FILE rather than against a second
read of it. Every payload byte in a torch zip lives in an `archive/data/<n>`
entry. So: hash every such entry, hash the storage behind every recovered
tensor, and require a bijection. If each archive storage is byte-identical to a
recovered one and none is left over, nothing was dropped or altered -- proved
without a second reader, and a stronger claim than two readers agreeing.

This script REFUSES TO WRITE if that check fails.

What it does not prove: that the argument NAMES below match upstream's
signature. Those come from reading `DiffPlanner.forward`, and are asserted by
shape in `tests/test_strip_planner_inputs.py`.

Usage:
    python diffusiondrive_planner/tools/strip_planner_inputs.py
    # -> diffusiondrive_planner/data/planner_inputs_plain.pt
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import types
import zipfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent.parent
DEFAULT_IN = HERE / 'data' / 'planner_inputs.pt'
DEFAULT_OUT = HERE / 'data' / 'planner_inputs_plain.pt'

#: modules that will not import without mmcv 1.x. Anything under these is
#: replaced by a stub whose only job is to hold state so the walk can reach the
#: tensors inside it.
STUB_PREFIXES = ('projects.', 'mmcv', 'mmdet', 'mmdet3d', 'nuscenes',
                 'pyquaternion')


def _make_stub(module: str, name: str, registry: dict, hits: dict):
    key = f'{module}.{name}'
    hits[key] = hits.get(key, 0) + 1
    if key in registry:
        return registry[key]

    class _Stub:
        __module__ = module
        __qualname__ = name

        def __init__(self, *a, **k):
            self._args, self._kwargs = a, k

        def __setstate__(self, state):
            # nn.Module pickles its __dict__, which is where _parameters and
            # _modules live -- so the weights survive into the stub untouched.
            self.__dict__.update(state if isinstance(state, dict)
                                 else {'_state': state})

        def __repr__(self):
            return f'<stub {key}>'

    _Stub.__name__ = name
    registry[key] = _Stub
    return _Stub


def load_without_mmcv(path) -> tuple[object, dict]:
    """`torch.load` with every unresolvable class replaced by a stub."""
    registry: dict = {}
    hits: dict = {}

    class StubUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith(STUB_PREFIXES):
                return _make_stub(module, name, registry, hits)
            try:
                return super().find_class(module, name)
            except (ImportError, AttributeError):
                return _make_stub(module, name, registry, hits)

    shim = types.ModuleType('stubpickle')
    shim.Unpickler = StubUnpickler
    shim.Pickler = pickle.Pickler
    shim.load, shim.loads = pickle.load, pickle.loads
    shim.dump, shim.dumps = pickle.dump, pickle.dumps

    obj = torch.load(path, pickle_module=shim, map_location='cpu',
                     weights_only=False)
    return obj, hits


def flatten(obj, path: str = '', depth: int = 0, max_depth: int = 14) -> dict:
    """Every tensor and primitive leaf, keyed by its access path."""
    out: dict = {}
    if depth > max_depth:
        return out
    if torch.is_tensor(obj):
        out[path or 'tensor'] = obj.detach().clone()
    elif isinstance(obj, (str, int, float, bool, type(None))):
        out[path] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f'{path}.{k}' if path else str(k), depth + 1))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flatten(v, f'{path}[{i}]', depth + 1))
    elif hasattr(obj, '__dict__'):
        for k, v in vars(obj).items():
            out.update(flatten(v, f'{path}.{k}' if path else k, depth + 1))
    return out


def verify_against_archive(path, flat: dict) -> tuple[int, int, list]:
    """Bijection between archive storages and recovered tensor storages.

    Returns (matched, total_archive_entries, unaccounted_entry_names).
    """
    z = zipfile.ZipFile(path)
    archive = {}
    for n in z.namelist():
        if '/data/' in n:
            archive[hashlib.sha256(z.read(n)).hexdigest()] = n

    seen_ptr, recovered = set(), set()
    for t in flat.values():
        if not torch.is_tensor(t):
            continue
        st = t.untyped_storage() if hasattr(t, 'untyped_storage') else t.storage()
        if st.data_ptr() in seen_ptr:
            continue
        seen_ptr.add(st.data_ptr())
        view = torch.empty(0, dtype=torch.uint8).set_(st)   # uint8 view, no copy
        recovered.add(hashlib.sha256(view.numpy().tobytes()).hexdigest())

    hit = set(archive) & recovered
    missing = [archive[h] for h in set(archive) - recovered]
    return len(hit), len(archive), missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', type=Path, default=DEFAULT_IN)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()

    if not a.inp.exists():
        print(f'  missing {a.inp}')
        return 1

    obj, hits = load_without_mmcv(a.inp)
    print(f'  loaded {a.inp.name} with no mmcv')
    for k, v in sorted(hits.items(), key=lambda x: -x[1]):
        print(f'    stubbed {v:>3}x  {k}')

    flat = flatten(obj)
    tensors = {k: v for k, v in flat.items() if torch.is_tensor(v)}
    print(f'  {len(tensors)} tensors, {len(flat) - len(tensors)} primitive leaves')

    hit, total, missing = verify_against_archive(a.inp, flat)
    print(f'  storage check: {hit}/{total} archive entries byte-identical to a '
          f'recovered tensor')
    if missing:
        print(f'  REFUSING TO WRITE -- {len(missing)} storages unaccounted for:')
        for n in missing[:10]:
            print(f'    {n}')
        return 2

    payload = dict(flat)
    payload['_meta'] = {
        'source': a.inp.name,
        'n_tensors': len(tensors),
        'storages_verified': f'{hit}/{total}',
        'note': 'plain tensors only -- loads with no mmcv, no upstream classes',
    }
    torch.save(payload, a.out)

    # Re-read with a STOCK loader: the point of the exercise is that the output
    # needs no stubbing at all, so prove it rather than assume it.
    back = torch.load(a.out, map_location='cpu', weights_only=False)
    bad = [k for k, v in tensors.items()
           if not torch.equal(back[k].to(v.dtype), v)]
    if bad:
        print(f'  round-trip MISMATCH on {len(bad)} tensors: {bad[:5]}')
        return 3
    print(f'  wrote {a.out.name} ({a.out.stat().st_size / 1e6:.1f} MB), '
          f're-read with a stock torch.load and all {len(tensors)} tensors match')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
