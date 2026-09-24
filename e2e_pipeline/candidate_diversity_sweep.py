"""Does candidate diversity make the occlusion prior trigger?

The occlusion prior has been inert through three separate investigations, always
for the same stated reason: 0 of 6 planned waypoints land in an unknown cell,
because the occlusion is LATERAL and the six anchors run straight down the
observed corridor. The same fixed-anchor limitation was blamed for W_RISK being
unidentifiable and for the critic having nothing to rank.

That diagnosis predicts something specific and falsifiable: widen the candidate
set laterally and the prior should start pricing. This runs it.

THE ARMS. Diversity has two available sources and they are not the same size:

  straight only     6 candidates, 0.5 m lateral endpoint spread  (current)
  all 3 commands   18 candidates, 19.9 m
  + diffusion      the real truncated-diffusion seed on top, which the schedule
                   says is worth ~0.1 m per delta -- included to show that it is
                   NOT where the diversity comes from, rather than assumed away

Each arm runs the same four-prior sweep as §26, reporting whether the prior
triggers (waypoints in unknown > 0), whether braking changes, and whether risk
changes.

Usage:
    PYTHONPATH=. python e2e_pipeline/candidate_diversity_sweep.py
"""
from __future__ import annotations

import os

import numpy as np
from nuscenes import NuScenes

from e2e_pipeline.closed_loop import (ClosedLoopRunner, FlashOccWorldModel,
                                      LoopConfig)
from e2e_pipeline.diffusion_sampler import multi_command_planner
from e2e_pipeline.safety_filter import FeasibilityLimits, SafetyFilter
from e2e_pipeline.world_model import ReactiveWorldModel

A = 'diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy'
PRIORS = (0.00, 0.02, 0.05, 0.10)

ARMS = (
    ('straight only (current)', dict(commands=(2,), n_diffusion_samples=0)),
    ('all 3 commands', dict(commands=(0, 1, 2), n_diffusion_samples=0)),
    ('3 commands + diffusion x2', dict(commands=(0, 1, 2),
                                       n_diffusion_samples=2)),
)


def run(nusc, prior, planner_kw, n=10):
    B = E = 0
    clears, risks, hits, unk, spread, ncand = [], [], [], [], [], []
    for i in range(n):
        w = FlashOccWorldModel(nusc, scene_idx=i, veto_unknown=False)
        cfg = LoopConfig(max_steps=20, initial_speed=w.initial_speed(),
                         calibrate_risk=True, unknown_prior=prior,
                         veto_unknown=False)
        planner = multi_command_planner(A, dt=cfg.dt, **planner_kw)
        r = ClosedLoopRunner(
            w, planner, cfg,
            safety=SafetyFilter(limits=FeasibilityLimits(max_risk=0.10)),
            latent_model=ReactiveWorldModel())
        recs, m = r.run(command=2)
        s = m['safety']
        B += s['emergency_brakes']
        E += s['n_collision_steps_ego_fault']
        if s['min_clearance_m'] is not None:
            clears.append(s['min_clearance_m'])
        risks += [x for x in r.risk_pred if x is not None]

        for k, rec in enumerate(recs):
            if rec.planned_traj is None or not len(rec.planned_traj):
                continue
            fs = w.freespace_at(k * cfg.dt, rec.ego_xy, rec.ego_yaw)
            if fs is None or fs.unknown is None:
                continue
            unk.append(float(fs.unknown.mean()))
            traj = np.asarray(rec.planned_traj, float)
            hits.append(float(np.mean(fs.unknown_at(traj))))

        # candidate-set geometry, measured on the real scenes rather than
        # inferred from the anchor file
        for k in range(0, min(20, len(w.samples) - 1), 5):
            fr = w.samples[k]
            from e2e_pipeline.scene import EgoState, SceneRepresentation
            sc_ = SceneRepresentation(
                agents=w.agents_at(k * cfg.dt, fr['xy'], fr['yaw']),
                freespace=w.freespace_at(k * cfg.dt, fr['xy'], fr['yaw']),
                ego=EgoState(speed=w.initial_speed()), timestamp=k * cfg.dt)
            c, _ = planner(sc_, 2)
            c = np.asarray(c, float)
            ncand.append(len(c))
            spread.append(float(c[:, -1, 1].max() - c[:, -1, 1].min()))

    return {
        'brakes': B, 'ego': E,
        'clear': float(np.mean(clears)) if clears else float('nan'),
        'risk': float(np.mean(risks)) if risks else float('nan'),
        'risk_p90': float(np.percentile(risks, 90)) if risks else float('nan'),
        'unknown_map': float(np.mean(unk)) if unk else float('nan'),
        'unknown_hit': float(np.mean(hits)) if hits else float('nan'),
        'hit_any': float(np.mean([h > 0 for h in hits])) if hits else float('nan'),
        'n_cand': float(np.mean(ncand)) if ncand else float('nan'),
        'spread': float(np.mean(spread)) if spread else float('nan'),
    }


def main():
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'),
                    verbose=False)
    for label, kw in ARMS:
        print(f'\n  === {label} ===')
        print(f'  {"prior":>7}{"cands":>7}{"lat spread":>12}{"brakes":>8}{"EGO":>5}'
              f'{"clear":>8}{"mean risk":>11}{"risk p90":>10}'
              f'{"map unk":>9}{"wp in unk":>11}{"steps hit":>11}')
        for p in PRIORS:
            r = run(nusc, p, kw)
            print(f'  {p:>7.2f}{r["n_cand"]:>7.0f}{r["spread"]:>11.1f}m'
                  f'{r["brakes"]:>8}{r["ego"]:>5}{r["clear"]:>7.2f}m'
                  f'{r["risk"]:>11.4f}{r["risk_p90"]:>10.4f}'
                  f'{r["unknown_map"]:>9.1%}{r["unknown_hit"]:>11.2%}'
                  f'{r["hit_any"]:>11.1%}')


if __name__ == '__main__':
    main()
