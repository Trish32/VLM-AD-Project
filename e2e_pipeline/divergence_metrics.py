"""Metrics that survive the divergence confound.

The undifferentiated collision count was measuring deviation from the
recording, not safety: pin the ego to the logged trajectory and collisions go to
zero under GT *and* live perception. Worse, it rewards parking -- the safest
possible policy by that number is to stop, which is also what got the ego struck
30 times.

Four replacements, each targeting a specific way the old metric lied.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DIVERGENCE_BUCKETS = ((0.0, 1.0), (1.0, 3.0), (3.0, 7.0), (7.0, np.inf))


@dataclass
class StepObs:
    """One step, with everything the four metrics need."""

    divergence: float
    collided: bool
    progress_m: float          # distance made good along the route this step
    ego_v: float
    logged_v: float
    risk_planner: float = 0.0  # collision prob of the plan actually chosen
    risk_logged: float = 0.0   # collision prob of the logged ego's own future


def bucketed_collision_rate(obs: list[StepObs]) -> list[dict]:
    """(1) Collision rate within matched divergence bands.

    Comparing GT against a detector across whole rollouts confounds two things:
    the detector is worse, AND the detector makes the ego drift further, and
    drift is what produces collisions. Conditioning on divergence holds the
    second constant so the first is visible.

    A bucket with too few samples is reported with its count rather than
    silently averaged -- the >7 m bucket in particular is dominated by whichever
    scene diverged most.
    """
    out = []
    for lo, hi in DIVERGENCE_BUCKETS:
        sel = [o for o in obs if lo <= o.divergence < hi]
        out.append({'lo': lo, 'hi': hi, 'n': len(sel),
                    'collision_rate': (float(np.mean([o.collided for o in sel]))
                                       if sel else float('nan'))})
    return out


def recovery_rate(divergence: list[float], threshold: float = 3.0,
                  recovered_at: float = 1.5) -> dict:
    """(2) Once behind, does the ego ever catch up?

    The distinction this draws is between marginal degradation and a trapped
    state. If divergence crosses `threshold` and never returns below
    `recovered_at`, the failure is a STATIONARY POINT -- the ego has entered a
    region it cannot leave, and a metric averaging over the rollout will report
    it as "somewhat worse" when it is actually "stuck".

    Reported as the fraction of excursions that recover, plus the longest
    unrecovered run, because one permanent trap and many brief dips give the
    same mean.
    """
    d = np.asarray(divergence, dtype=np.float64)
    excursions, i, n = [], 0, len(d)
    while i < n:
        if d[i] >= threshold:
            j = i
            while j < n and d[j] > recovered_at:
                j += 1
            excursions.append({'start': i, 'end': j, 'len': j - i,
                               'recovered': j < n})
            i = j
        else:
            i += 1
    if not excursions:
        return {'excursions': 0, 'recovery_rate': float('nan'),
                'longest_unrecovered': 0}
    rec = [e['recovered'] for e in excursions]
    unrec = [e['len'] for e in excursions if not e['recovered']]
    return {'excursions': len(excursions), 'recovery_rate': float(np.mean(rec)),
            'longest_unrecovered': max(unrec) if unrec else 0}


def counterfactual_safety(obs: list[StepObs]) -> dict:
    """(3) Per-decision: is the plan riskier than what the human actually did?

    Fixes the world at each logged frame and compares two trajectories through
    it -- the planner's choice and the logged ego's own future. No rollout, so
    no divergence: both are evaluated against the same agents from the same
    pose, and the difference is attributable to the decision alone.

    This is the only metric here that asks whether the PLANNER is unsafe, rather
    than whether the simulation drifted. `worse_rate` is the share of decisions
    where the plan carries more collision probability than the human's.
    """
    if not obs:
        return {'n': 0}
    d = np.array([o.risk_planner - o.risk_logged for o in obs])
    return {'n': len(d), 'mean_delta': float(d.mean()),
            'median_delta': float(np.median(d)),
            'worse_rate': float(np.mean(d > 0.01)),
            'much_worse_rate': float(np.mean(d > 0.10)),
            'better_rate': float(np.mean(d < -0.01))}


def progress_per_divergence(obs: list[StepObs]) -> dict:
    """(4) Progress bought per metre of drift -- so parking cannot win.

    A collision-only metric is maximised by stopping. This is a ratio, so a
    stationary policy scores zero rather than perfect: no progress, and whatever
    divergence accumulates from the world moving on without it.

    `stalled_fraction` is reported alongside because a high ratio achieved over
    three moving steps and seventeen stationary ones is not the same as a
    moderate ratio achieved throughout.
    """
    if not obs:
        return {'n': 0}
    prog = float(sum(o.progress_m for o in obs))
    # Final divergence of a CONCATENATED list is just the last scene's, which is
    # meaningless. Callers pass one scene at a time; `aggregate_scenes` below
    # combines them correctly.
    div = float(obs[-1].divergence)
    stalled = float(np.mean([o.ego_v < 0.5 for o in obs]))
    return {'n': len(obs), 'progress_m': prog, 'final_divergence_m': div,
            'progress_per_div': prog / max(div, 1.0),
            'stalled_fraction': stalled,
            'speed_ratio': float(np.mean([o.ego_v for o in obs]) /
                                 max(np.mean([o.logged_v for o in obs]), 1e-6))}


def aggregate_scenes(per_scene: list[list[StepObs]]) -> dict:
    """Progress-per-divergence summed correctly across scenes.

    Summing progress over a concatenated list and dividing by the last scene's
    final divergence is not a ratio of anything. Each scene contributes its own
    numerator and denominator, and the totals are divided at the end.
    """
    prog = sum(sum(o.progress_m for o in sc) for sc in per_scene)
    div = sum(sc[-1].divergence for sc in per_scene if sc)
    flat = [o for sc in per_scene for o in sc]
    return {'progress_m': float(prog), 'divergence_m': float(div),
            'progress_per_div': float(prog / max(div, 1.0)),
            'stalled_fraction': float(np.mean([o.ego_v < 0.5 for o in flat])),
            'speed_ratio': float(np.mean([o.ego_v for o in flat]) /
                                 max(np.mean([o.logged_v for o in flat]), 1e-6))}
