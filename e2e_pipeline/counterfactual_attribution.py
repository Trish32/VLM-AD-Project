"""Does the measured covariance floor explain the much-worse tail?

§25 reported a per-decision counterfactual -- plan vs logged ego through
identical agents -- where 10% of live-perception decisions came out MUCH worse
(Δrisk > 0.10) against 2% under GT. The question is whether that tail is real
risk or a covariance artefact.

FIRST, A MEASUREMENT BUG. `divergence_report.py` built its counterfactual
RiskModel with NO TRACKER, so `constant_velocity_prediction` fell back to a
hardcoded (0.5 + 0.5 t)^2 spread. The metric that reports the tail never
consulted the score-dependent covariance at all -- it could not have responded
to a recalibration, and re-running it unchanged would have produced identical
numbers and looked like a null result. Fixed here by seeding a fresh tracker per
step, so each agent carries cov = R(score), the quantity actually calibrated.

THEN THE DECOMPOSITION. sigma = FLOOR + SLOPE / score has two parts and they
make different claims. Four arms separate them:

    reciprocal   0.500 / s            the original assumption
    full         0.609 + 0.466 / s    as measured
    slope only   0.466 / s            floor removed, shape kept
    floor only   0.609                score dependence removed

If the tail tracks `floor only` and not `slope only`, it is the irreducible
error at high score that drives it -- which would mean the tail is not the
detector being unsure, but the detector being confidently ~0.6 m off.

Usage:
    PYTHONPATH=. python e2e_pipeline/counterfactual_attribution.py
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
from nuscenes import NuScenes

from e2e_pipeline.closed_loop import (ClosedLoopRunner, GTWorldModel,
                                      LivePerceptionWorldModel, LoopConfig,
                                      diffusiondrive_anchor_planner)
from e2e_pipeline.safety_filter import FeasibilityLimits, SafetyFilter
from e2e_pipeline.scene import EgoState
from e2e_pipeline.uncertainty import RiskModel, TrackCovarianceTracker
from e2e_pipeline.world_model import ReactiveWorldModel

A = 'diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy'
DT = 0.5
MUCH_WORSE = 0.10


class _Arm(TrackCovarianceTracker):
    """Tracker with each calibrated term independently switchable."""

    def __init__(self, floor: float, slope: float, vel: float | None = None, **kw):
        super().__init__(calibrated_noise=True, **kw)
        self.MEAS_FLOOR_M = float(floor)
        self.MEAS_SLOPE_M = float(slope)
        if vel is not None:
            self.MEAS_VEL_MPS = float(vel)


ARMS = {
    'reciprocal (old)': lambda: TrackCovarianceTracker(
        calibrated_noise=False, pos_noise=0.5, vel_noise=1.0),
    'full (measured)': lambda: _Arm(0.609, 0.466),
    'slope only': lambda: _Arm(0.0, 0.466),
    'floor only': lambda: _Arm(0.609, 0.0),
    # Position is one term of the propagated variance and not obviously the
    # binding one. Under the CV model the velocity block enters as t^2 * P_vv,
    # so at a 3 s horizon it is multiplied by 9 while position is multiplied by
    # 1. These arms hold position at the measured curve and vary only velocity.
    'vel 0.2 (as shipped)': lambda: _Arm(0.609, 0.466, vel=0.2),
    'vel 1.0 (assumed)': lambda: _Arm(0.609, 0.466, vel=1.0),
    'vel 2.46 (measured)': lambda: _Arm(0.609, 0.466, vel=2.463),
}


def variance_decomposition(steps, horizon_s=3.0):
    """Which term of the propagated variance actually dominates at the horizon.

    var(t) = P_xx + 2 t P_xv + t^2 P_vv + q t^3 / 3. Reported so the claim
    "calibrating position cannot move the risk" is measured rather than argued.
    """
    tr = _Arm(0.609, 0.466)
    t = horizon_s
    pos, vel, proc = [], [], []
    for st in steps[:200]:
        for a in st['agents']:
            R = tr._R(a.score)
            pos.append(R[0, 0])
            vel.append((t ** 2) * R[2, 2])
            proc.append(tr._effective_accel_noise(a) * (t ** 3) / 3.0)
    if not pos:
        return
    p, v, q = np.mean(pos), np.mean(vel), np.mean(proc)
    tot = p + v + q
    print(f'\n  variance decomposition at t={t:.0f} s (mean over agents, m^2)')
    print(f'    position  P_xx       {p:8.2f}   {p / tot:5.1%}')
    print(f'    velocity  t^2 P_vv   {v:8.2f}   {v / tot:5.1%}')
    print(f'    process   q t^3/3    {q:8.2f}   {q / tot:5.1%}')
    print(f'    -> sigma {np.sqrt(tot):.2f} m; position contributes '
          f'{p / tot:.1%} of it')


def _logged_future(w, k, fr, n=6):
    out = []
    c, s = np.cos(-fr['yaw']), np.sin(-fr['yaw'])
    for h in range(1, n + 1):
        j = min(k + h, len(w.samples) - 1)
        d = w.samples[j]['xy'] - fr['xy']
        out.append([c * d[0] - s * d[1], s * d[0] + c * d[1]])
    return np.asarray(out, float)


def gather(nusc, WorldCls, n_scenes=10):
    """One rollout per scene, keeping the plan and the agents at each step."""
    steps = []
    for i in range(n_scenes):
        w = WorldCls(nusc, scene_idx=i)
        cfg = LoopConfig(max_steps=20, initial_speed=w.initial_speed(),
                         calibrate_risk=True)
        r = ClosedLoopRunner(w, diffusiondrive_anchor_planner(A, cfg.dt), cfg,
                             safety=SafetyFilter(
                                 limits=FeasibilityLimits(max_risk=0.10)),
                             latent_model=ReactiveWorldModel())
        recs, _ = r.run(command=2)
        for k, rec in enumerate(recs):
            if rec.planned_traj is None or not len(rec.planned_traj):
                continue
            fr = w.samples[min(k, len(w.samples) - 1)]
            agents = w.agents_at(k * DT, fr['xy'], fr['yaw'])
            if not agents:
                continue
            steps.append({
                'agents': agents, 'ego_v': rec.ego_v, 't': k * DT,
                'plan': np.asarray(rec.planned_traj, float),
                'logged': _logged_future(w, k, fr, len(rec.planned_traj)),
                'calibrator': r.calibrator,
            })
    return steps


def score_arm(steps, make_tracker):
    """Δrisk per decision under one covariance model, plus tail attribution."""
    deltas, tail = [], []
    for st in steps:
        tr = make_tracker()                       # fresh: cov = R(score)
        agents = st['agents']
        tr.update(agents, st['t'])
        rm = RiskModel(EgoState(speed=st['ego_v']), tracker=tr,
                       calibrator=st['calibrator'])
        try:
            rp = rm.evaluate(st['plan'], agents, dt=DT)
            rl = rm.evaluate(st['logged'], agents, dt=DT)
        except Exception:
            continue
        d = float(rp.total) - float(rl.total)
        deltas.append(d)
        if d > MUCH_WORSE:
            wa = {int(a.track_id): a for a in agents}.get(rp.worst_agent)
            if wa is not None:
                tail.append((float(wa.score),
                             float(np.linalg.norm(np.asarray(wa.xy, float))), d))
    d = np.asarray(deltas, float)
    return {
        'n': len(d),
        'mean': float(d.mean()) if len(d) else float('nan'),
        'worse': float(np.mean(d > 0)) if len(d) else float('nan'),
        'much_worse': float(np.mean(d > MUCH_WORSE)) if len(d) else float('nan'),
        'better': float(np.mean(d < 0)) if len(d) else float('nan'),
        'tail': tail,
    }


def main():
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'),
                    verbose=False)
    for label, cls in (('LIVE', LivePerceptionWorldModel), ('GT', GTWorldModel)):
        steps = gather(nusc, cls)
        print(f'\n  === {label} === {len(steps)} comparable decisions')
        variance_decomposition(steps)
        print(f'  {"covariance arm":>18}{"n":>6}{"mean Δrisk":>13}{"worse":>8}'
              f'{"much worse":>12}{"better":>9}')
        tails = {}
        for name, mk in ARMS.items():
            r = score_arm(steps, mk)
            tails[name] = r['tail']
            print(f'  {name:>18}{r["n"]:>6}{r["mean"]:>+13.4f}'
                  f'{r["worse"]:>8.0%}{r["much_worse"]:>12.0%}{r["better"]:>9.0%}')

        t = tails.get('full (measured)', [])
        if t:
            sc = np.array([x[0] for x in t])
            rg = np.array([x[1] for x in t])
            print(f'\n  tail attribution ({len(t)} much-worse decisions, '
                  f'full model): worst agent')
            print(f'    detector score   mean {sc.mean():.2f}  '
                  f'median {np.median(sc):.2f}  >0.7: {np.mean(sc > 0.7):.0%}')
            print(f'    range from ego   mean {rg.mean():.1f} m  '
                  f'median {np.median(rg):.1f} m  <20 m: {np.mean(rg < 20):.0%}')
            by = defaultdict(int)
            for s, _, _ in t:
                by[min(int(s * 5) / 5, 0.8)] += 1
            print('    score histogram  ' + '  '.join(
                f'{k:.1f}-{k + 0.2:.1f}:{v}' for k, v in sorted(by.items())))


if __name__ == '__main__':
    main()
