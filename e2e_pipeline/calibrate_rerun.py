"""Is the risk price identifiable now that the covariance is measured?

`calibrate.py` replaces the hand-set W_RISK = 6.0 with a dual variable on a risk
BUDGET, and was found INERT: progress was identical at lam = 6.0, 5.2, 4.4 and
3.6. The diagnosis was structural, not numerical -- all six DiffusionDrive
anchors share an arc length, so `progress`, `offroad` and `clearance` are
constant across candidates and risk is the only discriminating term, making
argmax(-lam * risk) independent of lam for any lam > 0.

A recalibrated covariance changes the MAGNITUDE of the risk term but not that
argument, so the expected answer is "still inert". Worth re-running anyway,
because the diagnosis rests on a claim about candidate spread that was asserted
once and never re-measured, and because two things about the critic's risk term
have since turned out to be wrong:

  * it built its RiskModel with NO TRACKER, so the covariance never reached it;
  * and with NO CALIBRATOR, so it was in model units.

This measures the spread directly -- if `progress` really has zero spread across
candidates, no weight on it can matter and that is provable rather than argued.

Usage:
    PYTHONPATH=. python e2e_pipeline/calibrate_rerun.py
"""
from __future__ import annotations

import os

import numpy as np
from nuscenes import NuScenes

from e2e_pipeline.calibrate import CalibrationTrace, RiskBudget, calibrate_risk_price
from e2e_pipeline.closed_loop import (ClosedLoopRunner, LivePerceptionWorldModel,
                                      LoopConfig, diffusiondrive_anchor_planner)
from e2e_pipeline.safety_filter import FeasibilityLimits, SafetyFilter
from e2e_pipeline.scene import EgoState, SceneRepresentation
from e2e_pipeline.uncertainty import TrackCovarianceTracker
from e2e_pipeline.world_model import (AnalyticCritic, ReactiveWorldModel,
                                      plan_with_world_model)

A = 'diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy'


def term_spread(nusc, n_scenes=5, wired=True):
    """Spread of each critic term ACROSS candidates, per decision.

    A term with zero spread cannot change a ranking no matter what weight it
    carries. This is the claim the inertness diagnosis rests on.
    """
    tracker = TrackCovarianceTracker(calibrated_noise=True) if wired else None
    spreads = {k: [] for k in ('progress', 'risk', 'offroad', 'clearance',
                               'comfort', 'total')}
    for i in range(n_scenes):
        w = LivePerceptionWorldModel(nusc, scene_idx=i)
        cfg = LoopConfig(max_steps=20, initial_speed=w.initial_speed(),
                         calibrate_risk=True)
        r = ClosedLoopRunner(w, diffusiondrive_anchor_planner(A, cfg.dt), cfg,
                             safety=SafetyFilter(
                                 limits=FeasibilityLimits(max_risk=0.10)),
                             latent_model=ReactiveWorldModel())
        planner = diffusiondrive_anchor_planner(A, cfg.dt)
        critic = AnalyticCritic(dt=cfg.dt, tracker=tracker,
                                calibrator=r.calibrator if wired else None)
        for k in range(min(20, len(w.samples) - 1)):
            fr = w.samples[k]
            t = k * cfg.dt
            # Evaluate at the LOGGED pose: this measures candidate spread, a
            # property of the anchor set against a scene, and pinning the pose
            # keeps divergence out of it.
            agents = w.agents_at(t, fr['xy'], fr['yaw'])
            if tracker is not None:
                agents = tracker.update(agents, t)
            fs = w.freespace_at(t, fr['xy'], fr['yaw'])
            scene = SceneRepresentation(
                agents=agents, freespace=fs,
                ego=EgoState(speed=w.initial_speed(), length=cfg.ego_length,
                             width=cfg.ego_width, wheelbase=cfg.wheelbase),
                timestamp=t)
            cands, scores = planner(scene, 2)
            if cands is None or len(cands) < 2:
                continue
            ranked = plan_with_world_model(cands, scene,
                                           model=ReactiveWorldModel(),
                                           critic=critic, dt=cfg.dt)
            for key in spreads:
                v = np.array([getattr(rc.score, key) for rc in ranked], float)
                v = v[np.isfinite(v)]
                if len(v) >= 2:
                    spreads[key].append(float(v.max() - v.min()))
    return spreads


def main():
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'),
                    verbose=False)

    for wired in (False, True):
        tag = ('critic WIRED (measured covariance + calibrator)' if wired
               else 'critic AS SHIPPED (no tracker, no calibrator)')
        sp = term_spread(nusc, wired=wired)
        print(f'\n  === {tag} ===')
        print(f'  {"critic term":>12}{"n":>7}{"mean spread":>14}'
              f'{"max spread":>13}{"% decisions with spread>0":>28}')
        for key, v in sp.items():
            if not v:
                continue
            a = np.array(v)
            print(f'  {key:>12}{len(a):>7}{a.mean():>14.4f}{a.max():>13.4f}'
                  f'{np.mean(a > 1e-9):>28.0%}')
        print('  a term with zero spread cannot change a ranking at any weight')

    # Dual ascent, with the critic wired. If progress still has no spread the
    # dual has nothing to trade against and lam should wander without
    # converging to anything meaningful -- which is the honest output.
    print('\n  === dual ascent on the risk budget ===')
    from e2e_pipeline.calibration import PlattCalibrator
    r_cal = PlattCalibrator(a=0.457, b=-2.333, fitted=True, n_positive=11)
    tracker = TrackCovarianceTracker(calibrated_noise=True)

    def evaluate(lam):
        import e2e_pipeline.world_model as WM
        old = WM.W_RISK
        WM.W_RISK = float(lam)
        try:
            risks, comps = [], []
            for i in range(5):
                w = LivePerceptionWorldModel(nusc, scene_idx=i)
                cfg = LoopConfig(max_steps=20, initial_speed=w.initial_speed(),
                                 calibrate_risk=True, use_world_model=True)
                r = ClosedLoopRunner(
                    w, diffusiondrive_anchor_planner(A, cfg.dt), cfg,
                    safety=SafetyFilter(limits=FeasibilityLimits(max_risk=0.10)),
                    tracker=TrackCovarianceTracker(calibrated_noise=True),
                    latent_model=ReactiveWorldModel(),
                    critic=AnalyticCritic(dt=cfg.dt, tracker=tracker,
                                          calibrator=r_cal))
                _, m = r.run(command=2)
                risks += [x for x in r.risk_pred if x is not None]
                if m['route'].get('completion') is not None:
                    comps.append(m['route']['completion'])
            return (float(np.mean(risks)) if risks else 0.0,
                    float(np.mean(comps)) if comps else 0.0)
        finally:
            WM.W_RISK = old

    trace = CalibrationTrace()
    lam = calibrate_risk_price(evaluate, RiskBudget(max_mean_risk=0.05),
                               lam0=6.0, eta=40.0, iters=5, trace=trace)
    print(trace.describe())
    print(f'  converged lam {lam:.3f}')
    if len(set(f'{p:.4f}' for p in trace.progress)) == 1:
        print('  progress IDENTICAL at every lam -- the price is still '
              'unidentifiable, and for the structural reason above')


if __name__ == '__main__':
    main()
