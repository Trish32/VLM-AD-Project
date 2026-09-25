"""Ablation: L2, output shrinkage, dropout, gating -- one-step AND rollout error.

One-step error is not what a world model is used for. A rollout compounds its
own mistakes, so a correction that helps marginally per step can still diverge
over a horizon, and one that helps per step may not survive integration. Both
are reported.

Usage:
    PYTHONPATH=. python e2e_pipeline/residual_ablation.py
"""
import os

import numpy as np
from nuscenes import NuScenes

from e2e_pipeline.tools.fit_residual import collect
from e2e_pipeline.planner.residual import ResidualDynamics

DT = 0.5


def rollout_rmse(model, X, Y, horizon=6, gate=True):
    """Error after integrating `horizon` steps of velocity correction.

    Y is (actual - physics) per step, so the physics rollout error accumulates
    as the running sum of Y, and the corrected rollout accumulates the residual
    of that. Position error is the integral of velocity error, hence the dt.
    """
    r = model.residual(X, gate=gate) if model is not None else np.zeros_like(Y)
    err_phys, err_corr = [], []
    for i in range(0, len(Y) - horizon, horizon):
        w = slice(i, i + horizon)
        err_phys.append(np.linalg.norm(np.cumsum(Y[w], axis=0)[-1] * DT))
        err_corr.append(np.linalg.norm(np.cumsum(Y[w] - r[w], axis=0)[-1] * DT))
    if not err_phys:
        return float('nan'), float('nan')
    return float(np.mean(err_phys)), float(np.mean(err_corr))


ARMS = [
    ('none (raw fit)',        dict(weight_decay=0.0,  shrink=0.0),   0.0, False),
    ('L2 only',               dict(weight_decay=1e-2, shrink=0.0),   0.0, False),
    ('output shrinkage only', dict(weight_decay=0.0,  shrink=1e-2),  0.0, False),
    ('dropout 0.2',           dict(weight_decay=0.0,  shrink=0.0),   0.2, False),
    ('dropout 0.5',           dict(weight_decay=0.0,  shrink=0.0),   0.5, False),
    ('L2 + shrink',           dict(weight_decay=1e-2, shrink=1e-2),  0.0, False),
    ('L2 + shrink + dropout', dict(weight_decay=1e-2, shrink=1e-2),  0.2, False),
    ('L2 + shrink + GATE',    dict(weight_decay=1e-2, shrink=1e-2),  0.0, True),
]


def main():
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'),
                    verbose=False)
    for tr, te, tag in (([0, 1, 2, 3, 4], [5, 6, 7, 8, 9], 'train 0-4 / test 5-9'),
                        ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4], 'train 5-9 / test 0-4')):
        Xtr, Ytr = collect(nusc, tr)
        Xte, Yte = collect(nusc, te)
        p1 = float(np.sqrt((Yte ** 2).mean()))
        pr, _ = rollout_rmse(None, Xte, Yte)
        print(f'\n  {tag}    physics: 1-step {p1:.4f} m/s | 3 s rollout {pr:.3f} m')
        print(f'  {"regulariser":24s}{"1-step":>10}{"Δ":>9}{"rollout":>10}{"Δ":>9}')
        for label, kw, drop, gate in ARMS:
            m = ResidualDynamics(dropout=drop).fit(Xtr, Ytr, epochs=400, **kw)
            e1 = float(np.sqrt(((Yte - m.residual(Xte, gate=gate)) ** 2).mean()))
            _, ro = rollout_rmse(m, Xte, Yte, gate=gate)
            print(f'  {label:24s}{e1:>10.4f}{1 - e1 / p1:>+8.1%}'
                  f'{ro:>10.3f}{1 - ro / pr:>+8.1%}')


if __name__ == '__main__':
    main()
