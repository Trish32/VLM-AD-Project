"""Where nuScenes lives, resolved once instead of hard-coded in every tool.

Previously every entry point carried an absolute path from one developer's
machine as its `--dataroot` default. That works exactly on that machine and
nowhere else, and on a public repository it reads as an accident rather than a
default.

Resolution order:

  1. ``$NUSCENES_DATAROOT`` — set this once and every tool follows.
  2. ``~/Downloads/nuScenes_miniV1.0`` — the layout these tools were developed
     against, expanded from ``$HOME`` so no username appears in the source.
  3. ``./data/nuscenes`` — the conventional in-repo location, so a clone with a
     symlink there works with no configuration at all.

The first path that exists wins; if none do, (1) or (2) is returned anyway so
the error message names a real candidate rather than something invented.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = 'NUSCENES_DATAROOT'

_CANDIDATES = (
    '~/Downloads/nuScenes_miniV1.0',
    './data/nuscenes',
)


def default_dataroot() -> str:
    """Best guess at the nuScenes root. Always overridable with --dataroot."""
    env = os.environ.get(ENV_VAR)
    if env:
        return os.path.expanduser(env)

    for c in _CANDIDATES:
        p = Path(os.path.expanduser(c))
        if p.exists():
            return str(p)

    # Nothing found: name the documented default so the failure is actionable.
    return os.path.expanduser(_CANDIDATES[0])


def describe() -> str:
    """One line for argparse help, showing what would actually be used."""
    return (f'nuScenes root (default: ${ENV_VAR}, else '
            f'{default_dataroot()})')
