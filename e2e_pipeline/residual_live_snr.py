"""Can the residual world model be fitted from real detections, and if not, why?

§22 recorded that `residual.py` "cannot be meaningfully re-fitted under live
perception" without saying what the obstruction was. Two candidates, with very
different implications:

  (a) the residual signal is real but buried in measurement noise -- a
      signal-to-noise problem, fixable with more data or smoothing; or
  (b) there is no paired data at all, because an agent cannot be followed from
      one frame to the next -- a plumbing problem, fixable only by fixing it.

It was (b), and (b) hid (a). The adapter's fallback track id hashed the frame
token, so 0.0% of ids survived to the next frame; `collect()` matches an agent
at t against the same track_id at t+1, so it was silently producing an empty or
near-empty training set. With nearest-neighbour association added (65% carried),
the pairs exist and (a) can finally be measured.

THE CEILING. The target is v_actual(t+1) - v_IDM(t). Under live perception both
terms are measured, and velocity error was measured at 2.46 m/s RMS -- flat
across detector score. If the residual the model is asked to learn is smaller
than that, no amount of regularisation recovers it, and the honest output is a
bound rather than a tuned model.

Usage:
    PYTHONPATH=. python e2e_pipeline/residual_live_snr.py
"""
from __future__ import annotations

import os

import numpy as np
from nuscenes import NuScenes

from e2e_pipeline.closed_loop import GTWorldModel, LivePerceptionWorldModel
from e2e_pipeline.fit_residual import idm_next_velocity
from e2e_pipeline.residual import ResidualDynamics, agent_features
from e2e_pipeline.residual_ablation import rollout_rmse
from e2e_pipeline.uncertainty import TrackCovarianceTracker

DT = 0.5
MEAS_VEL = TrackCovarianceTracker.MEAS_VEL_MPS
#: target RMS from the noise-free GT arm, used as the signal estimate for the
#: live arms. Filled by the first (control) call.
GT_SIGNAL: list = [None]


def collect(nusc, scenes, WorldCls, **kw):
    """(features, residual) pairs, and how many agents could be paired at all."""
    X, Y, seen, paired = [], [], 0, 0
    for si in scenes:
        w = WorldCls(nusc, scene_idx=si, **kw)
        for k in range(len(w.samples) - 1):
            fr, nx = w.samples[k], w.samples[k + 1]
            ego_v = float(np.linalg.norm(nx['xy'] - fr['xy']) / DT)
            now = {a.track_id: a for a in w.agents_at(k * DT, fr['xy'], fr['yaw'])}
            later = {a.track_id: a
                     for a in w.agents_at((k + 1) * DT, fr['xy'], fr['yaw'])}
            seen += len(now)
            for tid, a in now.items():
                b = later.get(tid)
                if b is None:
                    continue
                paired += 1
                pred_v, decel = idm_next_velocity(a, ego_v)
                X.append(agent_features(a, ego_v, decel))
                Y.append(np.asarray(b.vxy, dtype=np.float64) - pred_v)
    return (np.array(X), np.array(Y), seen, paired)


def arm(nusc, label, WorldCls, noisy: bool = True, **kw):
    tr, te = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
    Xtr, Ytr, s1, p1 = collect(nusc, tr, WorldCls, **kw)
    Xte, Yte, s2, p2 = collect(nusc, te, WorldCls, **kw)
    pair_rate = (p1 + p2) / max(s1 + s2, 1)
    print(f'\n  === {label} ===')
    print(f'  agents seen {s1 + s2}   paired across a step {p1 + p2} '
          f'({pair_rate:.0%})')
    if len(Xtr) < 50 or len(Xte) < 50:
        print('  too few pairs to fit -- association, not noise, is the '
              'obstruction')
        return

    sig = float(np.sqrt((Yte ** 2).mean()))
    print(f'  residual target RMS                   {sig:.3f} m/s')
    floor = None
    if noisy:
        # The naive floor is sqrt(2) x the per-box velocity error, since both
        # terms of the target are measured and independent. It comes out at
        # 3.48 m/s, which is FOUR TIMES the target RMS actually observed -- so
        # it is wrong for this subset, and saying so is the point.
        #
        # The reason is selection: association only pairs boxes that fall within
        # a 3 m gate two frames running, which preferentially keeps slow,
        # well-localised agents and discards exactly the fast ones carrying the
        # large velocity errors. The population constant does not describe the
        # paired subset.
        #
        # So derive the noise from the two arms instead. The GT arm measures the
        # same quantity with no measurement error, so treating signal and noise
        # as independent gives noise^2 = live^2 - gt^2 -- internally consistent,
        # and it needs no assumption the data contradicts.
        naive = MEAS_VEL * np.sqrt(2.0)
        print(f'  population velocity error (per box)   {MEAS_VEL:.3f} m/s'
              f'  -> naive floor {naive:.3f} m/s')
        if GT_SIGNAL[0] is not None and sig > GT_SIGNAL[0]:
            floor = float(np.sqrt(sig ** 2 - GT_SIGNAL[0] ** 2))
            print(f'  noise implied by GT vs LIVE variance  {floor:.3f} m/s '
                  f'-> SNR {GT_SIGNAL[0] / floor:.2f}')
            print(f'  (naive floor overstates it {naive / floor:.1f}x -- the '
                  f'3 m association gate keeps the clean agents)')
    else:
        GT_SIGNAL[0] = sig

    m = ResidualDynamics(hidden=64, dropout=0.0)
    m.fit(Xtr, Ytr, weight_decay=1e-2, shrink=1e-2)
    st = m.evaluate(Xte, Yte)
    p_phys, p_corr = rollout_rmse(m, Xte, Yte)
    print(f'  one-step  physics {st.physics_rmse:.4f} -> corrected '
          f'{st.residual_rmse:.4f}  ({st.improvement:+.1%})')
    print(f'  rollout   physics {p_phys:.4f} -> corrected {p_corr:.4f}  '
          f'({(p_phys - p_corr) / max(p_phys, 1e-9):+.1%})')
    if floor is not None and sig < floor:
        print('  VERDICT: target is BELOW the measurement noise floor; any '
              'apparent gain is fitting noise')


def main():
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'),
                    verbose=False)
    arm(nusc, 'GT annotations (control -- no measurement noise)',
        GTWorldModel, noisy=False)
    arm(nusc, 'LIVE detections, NO association (as shipped)',
        LivePerceptionWorldModel, associate=False)
    print('    ^ if this arm reports ~0% paired, §22 measured plumbing, '
          'not perception')
    arm(nusc, 'LIVE detections, nearest-neighbour association',
        LivePerceptionWorldModel, associate=True)


if __name__ == '__main__':
    main()
