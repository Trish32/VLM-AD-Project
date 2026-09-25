"""Tests for the divergence-aware metrics, and for the two bugs they shipped with.

Both bugs produced output that looked like a result: Δrisk identically 0.0000
across 200 steps in both arms, and a progress ratio that favoured the worse
configuration. Neither raised anything.
"""
import numpy as np
import pytest

from e2e_pipeline.metrics.divergence_metrics import (
    DIVERGENCE_BUCKETS, StepObs, aggregate_scenes, bucketed_collision_rate,
    counterfactual_safety, progress_per_divergence, recovery_rate)


def _obs(div=0.0, collided=False, prog=1.0, v=5.0, lv=5.0, rp=0.0, rl=0.0):
    return StepObs(divergence=div, collided=collided, progress_m=prog,
                   ego_v=v, logged_v=lv, risk_planner=rp, risk_logged=rl)


# --- bug 1: StepRecord must carry the executed plan -------------------------


def test_step_record_has_planned_traj_field():
    """Without it the counterfactual compared the logged path against itself."""
    from e2e_pipeline.metrics.metrics import StepRecord
    assert 'planned_traj' in StepRecord.__dataclass_fields__


def test_runner_populates_planned_traj():
    """The field existing is not enough -- the runner must fill it."""
    from e2e_pipeline.tests.test_closed_loop import _runner
    recs, _ = _runner().run(command=2)
    filled = [r for r in recs if r.planned_traj is not None and len(r.planned_traj)]
    assert filled, 'no record carries a planned trajectory'
    assert np.asarray(filled[0].planned_traj).ndim == 2


def test_counterfactual_detects_a_riskier_plan():
    """Guards the identically-zero failure: a real difference must show up."""
    same = [_obs(rp=0.2, rl=0.2) for _ in range(10)]
    worse = [_obs(rp=0.5, rl=0.2) for _ in range(10)]
    assert counterfactual_safety(same)['worse_rate'] == 0.0
    assert counterfactual_safety(worse)['worse_rate'] == 1.0
    assert counterfactual_safety(worse)['much_worse_rate'] == 1.0
    assert counterfactual_safety(worse)['mean_delta'] == pytest.approx(0.3)


# --- bug 2: progress per divergence must aggregate within scenes ------------


def test_aggregate_scenes_sums_both_sides_per_scene():
    """The broken version divided total progress by the LAST scene's divergence.

    Two scenes: 100 m over 10 m of drift, and 20 m over 20 m. Correct answer is
    120/30 = 4.0. The bug gave 120/20 = 6.0 -- flattering, and wrong.
    """
    a = [_obs(div=d, prog=10.0) for d in np.linspace(1, 10, 10)]
    b = [_obs(div=d, prog=2.0) for d in np.linspace(2, 20, 10)]
    agg = aggregate_scenes([a, b])
    assert agg['progress_m'] == pytest.approx(120.0)
    assert agg['divergence_m'] == pytest.approx(30.0)
    assert agg['progress_per_div'] == pytest.approx(4.0)


def test_parking_scores_zero_not_perfect():
    """The property the whole metric exists for."""
    parked = [_obs(div=d, prog=0.0, v=0.0) for d in np.linspace(1, 15, 15)]
    agg = aggregate_scenes([parked])
    assert agg['progress_per_div'] == 0.0
    assert agg['stalled_fraction'] == 1.0


# --- the metrics themselves -------------------------------------------------


def test_buckets_partition_by_divergence():
    obs = ([_obs(div=0.5, collided=True)] * 2 + [_obs(div=0.5)] * 2
           + [_obs(div=9.0, collided=True)] * 3)
    rows = bucketed_collision_rate(obs)
    assert rows[0]['n'] == 4 and rows[0]['collision_rate'] == pytest.approx(0.5)
    assert rows[-1]['n'] == 3 and rows[-1]['collision_rate'] == pytest.approx(1.0)


def test_empty_bucket_reports_nan_not_zero():
    """An empty bucket is unknown, not safe -- zero would read as a good result."""
    rows = bucketed_collision_rate([_obs(div=0.2)])
    assert np.isnan(rows[-1]['collision_rate'])


def test_recovery_distinguishes_a_dip_from_a_trap():
    assert recovery_rate([0, 4, 2, 1, 0.5])['recovery_rate'] == 1.0
    trap = recovery_rate([0, 1, 4, 5, 6, 8, 9])
    assert trap['recovery_rate'] == 0.0
    assert trap['longest_unrecovered'] == 5


def test_no_excursion_reports_nan_not_perfect():
    """Never diverging is not a 100% recovery rate -- there was nothing to recover."""
    assert np.isnan(recovery_rate([0, 0.5, 1.0])['recovery_rate'])


def test_single_scene_progress_matches_aggregate_of_one():
    sc = [_obs(div=d, prog=3.0) for d in np.linspace(1, 6, 6)]
    assert (progress_per_divergence(sc)['progress_per_div']
            == pytest.approx(aggregate_scenes([sc])['progress_per_div']))


def test_counterfactual_is_gated_on_low_divergence():
    """It is only divergence-free where divergence is small.

    The metric transplants the human's next poses onto wherever the ego is. At
    22 m of drift that is not an alternative the human could have driven -- it
    teleports out of the agent cloud and scores as low risk for the wrong
    reason. Measured on scene 6 before the gate: +0.3879 mean delta, 80% worse,
    against -0.0309 and 18% across scenes at low divergence.
    """
    from e2e_pipeline.closed_loop import COUNTERFACTUAL_MAX_DIVERGENCE_M
    assert 0 < COUNTERFACTUAL_MAX_DIVERGENCE_M <= 5.0


def test_counterfactual_ignores_invalid_steps():
    """Placeholder zeros must not dilute the signal toward 'identical'."""
    from e2e_pipeline.metrics.metrics import StepRecord, divergence_metrics
    recs = []
    for i in range(6):
        r = StepRecord(t=i * 0.5, ego_xy=np.array([float(i), 0.0]), ego_yaw=0.0,
                       ego_v=5.0, accel=0.0, steer=0.0)
        r.divergence_m = 0.5
        # only half the steps carry a real comparison
        r.counterfactual_valid = i % 2 == 0
        r.risk_planner, r.risk_logged = (0.5, 0.1) if r.counterfactual_valid else (0.0, 0.0)
        recs.append(r)
    cf = divergence_metrics(recs)['counterfactual']
    assert cf['n'] == 3, 'invalid steps leaked into the counterfactual'
    assert cf['mean_delta'] == pytest.approx(0.4)


def test_unknown_prior_bounds_risk_through_occlusion():
    """The prior prices unobserved space, and is exactly zero where all is seen."""
    from e2e_pipeline.freespace import FreeSpace
    from e2e_pipeline.scene import EgoState
    from e2e_pipeline.uncertainty import RiskModel
    nx, ny = 200, 120
    def fs(unknown_beyond=None):
        u = np.zeros((nx, ny), bool)
        if unknown_beyond is not None:
            u[unknown_beyond:, :] = True
        return FreeSpace(traversable=np.ones((nx, ny), bool),
                         obstacle=np.zeros((nx, ny), bool), unknown=u,
                         esdf=np.full((nx, ny), 5.0, np.float32),
                         origin=(-10.0, -30.0), res=0.5)
    traj = np.stack([[6.0 * (t + 1), 0.0] for t in range(6)])
    ego = EgoState(speed=12.0)
    assert RiskModel(ego, unknown_prior=0.10).evaluate(
        traj, [], freespace=fs()).total == pytest.approx(0.0)
    assert RiskModel(ego, unknown_prior=0.10).evaluate(
        traj, [], freespace=fs(80)).total == pytest.approx(0.10)
    # and the prior BOUNDS it -- risk cannot exceed the stated prior on an
    # otherwise empty scene, which is the whole point of the conservative prior
    assert RiskModel(ego, unknown_prior=0.02).evaluate(
        traj, [], freespace=fs(80)).total == pytest.approx(0.02)


def test_ray_occlusion_marks_behind_obstacles_only():
    from e2e_pipeline.live_adapter import ray_occlusion
    obs = np.zeros((100, 100), bool)
    obs[70, 45:55] = True                      # a wall ahead of the ego
    u = ray_occlusion(obs, origin=(-20.0, -20.0), res=0.4)
    assert u[85, 50], 'cells behind the wall should be unknown'
    assert not u[60, 50], 'cells in front of the wall should be observed'


def test_calibrated_covariance_has_an_irreducible_floor():
    """A score-1.0 box still carries ~0.6 m of error; b/s cannot express that."""
    from e2e_pipeline.uncertainty import TrackCovarianceTracker
    t = TrackCovarianceTracker(calibrated_noise=True)
    sigma_best = float(np.sqrt(t._R(1.0)[0, 0]))
    assert sigma_best > 0.5, 'calibrated sigma lost the measured error floor'
    # and it must still shrink with confidence
    assert float(np.sqrt(t._R(0.3)[0, 0])) > sigma_best


def test_calibrated_covariance_tracks_measured_error():
    """Within 25% of measurement in every score bin (was 1.28-1.98x off)."""
    from e2e_pipeline.uncertainty import TrackCovarianceTracker
    t = TrackCovarianceTracker(calibrated_noise=True)
    for score, measured in ((0.32, 2.036), (0.62, 1.450), (0.93, 0.987)):
        pred = float(np.sqrt(t._R(score)[0, 0]))
        assert 0.75 < pred / measured < 1.25, f'score {score}: {pred:.2f} vs {measured:.2f}'


def test_uncalibrated_path_still_available():
    """The old model stays reachable so the comparison can be reproduced."""
    from e2e_pipeline.uncertainty import TrackCovarianceTracker
    old = float(np.sqrt(TrackCovarianceTracker(calibrated_noise=False)._R(0.93)[0, 0]))
    new = float(np.sqrt(TrackCovarianceTracker(calibrated_noise=True)._R(0.93)[0, 0]))
    assert old < new, 'the old model should be the over-confident one'
