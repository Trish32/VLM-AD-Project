"""Online IDM-gain identification vs the offline learned residual.

Usage:
    PYTHONPATH=. python e2e_pipeline/fit_sysid.py
"""
import os

import numpy as np
from nuscenes import NuScenes

from e2e_pipeline.tools.fit_residual import collect
from e2e_pipeline.planner.residual import OnlineIDMGain, SysIDStats


def run_scene(X, Y, forgetting=0.98):
    """Causal evaluation: predict with the gain estimated from EARLIER steps only."""
    est = OnlineIDMGain(forgetting=forgetting)
    phys_err, adapt_err = [], []
    for x, y in zip(X, Y):
        idm_dv = -x[5] * 0.5                      # decel feature -> delta-v prior
        pred_prior = 0.0                          # residual target is already
        pred_adapt = (est.gain - 1.0) * idm_dv    # the prior's error
        phys_err.append(y[0] - pred_prior)
        adapt_err.append(y[0] - pred_adapt)
        est.update(idm_dv, y[0])                  # update AFTER predicting
    return (est.gain, float(np.sqrt(np.mean(np.square(phys_err)))),
            float(np.sqrt(np.mean(np.square(adapt_err)))))


def main():
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'),
                    verbose=False)
    print('  per-scene online identification (causal: gain from earlier steps only)')
    tot_p, tot_a = [], []
    for si in range(10):
        X, Y = collect(nusc, [si])
        if len(X) < 20:
            continue
        g, p, a = run_scene(X, Y)
        tot_p.append(p)
        tot_a.append(a)
        print(f'   scene {si}  n={len(X):5d}  gain {g:5.3f}  '
              f'physics {p:.4f} -> {a:.4f} m/s  {1 - a / p:+.1%}')
    print(f'\n  mean over scenes: physics {np.mean(tot_p):.4f} -> '
          f'{np.mean(tot_a):.4f} m/s  {1 - np.mean(tot_a) / np.mean(tot_p):+.1%}')


if __name__ == '__main__':
    main()
