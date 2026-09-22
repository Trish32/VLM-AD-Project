"""Tracking covariance and risk (item 3).

These tests pin the properties that make risk-aware selection meaningful: confidence
must move covariance in the right direction, uncertainty must grow with horizon, and
collision probability must respond to distance, spread and mode weight.
"""

import numpy as np
import pytest

from e2e_pipeline.scene import Agent, EgoState, TrajectoryDistribution
from e2e_pipeline.uncertainty import (RiskModel, TrackCovarianceTracker,
                                      constant_velocity_prediction)


def make_agent(track_id=1, xy=(10.0, 0.0), vxy=(0.0, 0.0), score=0.9,
               lwh=(4.5, 1.9, 1.6)) -> Agent:
    return Agent(track_id=track_id, xy=np.array(xy, dtype=float), yaw=0.0,
                 lwh=np.array(lwh, dtype=float), vxy=np.array(vxy, dtype=float),
                 score=score, label=0)


EGO = EgoState(speed=10.0)


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


def test_first_observation_seeds_covariance_from_measurement_noise():
    """A brand-new track knows exactly as much as its one measurement."""
    tr = TrackCovarianceTracker(pos_noise=0.5, vel_noise=1.0)
    a = make_agent(score=1.0)
    tr.update([a], timestamp=0.0)

    assert a.cov is not None and a.cov.shape == (4, 4)
    assert a.cov[0, 0] == pytest.approx(0.25)      # pos_noise^2
    assert a.cov[2, 2] == pytest.approx(1.0)       # vel_noise^2


def test_low_score_detections_widen_the_posterior():
    """Detector confidence must propagate into state uncertainty.

    A 0.2-score box should enter the filter as a vague observation, not as a
    confident one that drags the mean.
    """
    tr = TrackCovarianceTracker()
    confident, marginal = make_agent(track_id=1, score=1.0), make_agent(track_id=2, score=0.2)
    tr.update([confident, marginal], timestamp=0.0)
    assert marginal.cov[0, 0] > confident.cov[0, 0]


def test_repeated_observations_shrink_covariance():
    """The whole point of filtering: more evidence, tighter posterior."""
    tr = TrackCovarianceTracker()
    first = make_agent()
    tr.update([first], timestamp=0.0)
    p0 = first.cov[0, 0]

    p_last = p0
    for i in range(1, 6):
        a = make_agent()
        tr.update([a], timestamp=0.5 * i)
        p_last = a.cov[0, 0]

    assert p_last < p0, "covariance failed to contract under repeated observation"


def test_stale_tracks_are_dropped():
    """Unobserved tracks must age out, or the bank grows without bound."""
    tr = TrackCovarianceTracker(max_misses=2)
    tr.update([make_agent(track_id=7)], timestamp=0.0)
    assert 7 in tr._tracks
    for i in range(1, 5):
        tr.update([], timestamp=0.5 * i)
    assert 7 not in tr._tracks


def test_propagated_variance_grows_with_horizon():
    """Position uncertainty must increase the further ahead you look."""
    tr = TrackCovarianceTracker()
    a = make_agent(vxy=(5.0, 0.0))
    tr.update([a], timestamp=0.0)

    var = tr.propagated_position_var(a, np.array([0.5, 1.5, 3.0]))
    assert var.shape == (3, 2)
    assert var[0, 0] < var[1, 0] < var[2, 0]


def test_propagation_without_covariance_falls_back_not_to_zero():
    """An unfiltered agent must never present as perfectly known."""
    tr = TrackCovarianceTracker()
    var = tr.propagated_position_var(make_agent(), np.array([1.0]))
    assert var[0, 0] > 0.0


# ---------------------------------------------------------------------------
# Trajectory distribution
# ---------------------------------------------------------------------------


def test_laplace_scale_converts_to_gaussian_sigma():
    """Laplace(b) has variance 2b^2 — the conversion must not silently drop sqrt(2)."""
    d = TrajectoryDistribution(loc=np.zeros((1, 3, 2)), scale=np.full((1, 3, 2), 2.0),
                               probs=np.ones(1))
    assert d.sigma[0, 0, 0] == pytest.approx(2.0 * np.sqrt(2.0))


def test_mismatched_distribution_shapes_are_rejected():
    with pytest.raises(ValueError, match="must match"):
        TrajectoryDistribution(loc=np.zeros((2, 3, 2)), scale=np.zeros((2, 4, 2)),
                               probs=np.ones(2))


def test_constant_velocity_fallback_tracks_the_velocity():
    """The no-forecast path must still move the agent at its measured speed."""
    a = make_agent(xy=(0.0, 0.0), vxy=(10.0, 0.0))
    pred = constant_velocity_prediction(a, horizon=6, dt=0.5)

    assert pred.loc.shape == (1, 6, 2)
    assert pred.loc[0, -1, 0] == pytest.approx(30.0)      # 10 m/s * 3 s
    # And its spread must widen with time rather than staying flat.
    assert pred.sigma[0, -1, 0] > pred.sigma[0, 0, 0]


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


def test_distant_agent_is_low_risk():
    """A car 40 m off the plan must not veto anything."""
    rm = RiskModel(ego=EGO)
    traj = np.stack([np.arange(1, 7) * 5.0, np.zeros(6)], axis=1)
    report = rm.evaluate(traj, [make_agent(xy=(0.0, 40.0))])
    assert report.total < 0.01


def test_head_on_agent_is_high_risk():
    """An agent sitting on the planned path must produce large collision probability."""
    rm = RiskModel(ego=EGO)
    traj = np.stack([np.arange(1, 7) * 5.0, np.zeros(6)], axis=1)
    # Parked directly on the waypoint at t=3 (x = 15 m).
    report = rm.evaluate(traj, [make_agent(xy=(15.0, 0.0))])

    assert report.total > 0.5
    assert report.worst_agent == 1
    assert report.per_step.shape == (6,)


def test_risk_increases_as_prediction_spread_grows():
    """Wider forecasts must raise risk for an otherwise identical near-miss.

    This is the property that makes the Laplace scale worth plumbing: two agents at
    the same predicted location are not equally dangerous if one is far less certain.
    """
    rm = RiskModel(ego=EGO)
    traj = np.stack([np.arange(1, 7) * 5.0, np.zeros(6)], axis=1)

    def agent_with_spread(b: float) -> Agent:
        a = make_agent(xy=(15.0, 6.0))
        a.pred = TrajectoryDistribution(
            loc=np.tile(np.array([15.0, 6.0]), (1, 6, 1)),
            scale=np.full((1, 6, 2), b), probs=np.ones(1))
        return a

    tight = rm.evaluate(traj, [agent_with_spread(0.2)]).total
    loose = rm.evaluate(traj, [agent_with_spread(3.0)]).total
    assert loose > tight


def test_mode_probability_weights_the_dangerous_branch():
    """A low-probability collision mode must contribute less than a likely one.

    Collapsing QCNet to its argmax mode would make these two scenes identical, which
    is exactly the branch that hits you being discarded.
    """
    rm = RiskModel(ego=EGO)
    traj = np.stack([np.arange(1, 7) * 5.0, np.zeros(6)], axis=1)

    def bimodal(p_collide: float) -> Agent:
        a = make_agent(xy=(15.0, 0.0))
        on_path = np.tile(np.array([15.0, 0.0]), (6, 1))
        off_path = np.tile(np.array([15.0, 30.0]), (6, 1))
        a.pred = TrajectoryDistribution(
            loc=np.stack([on_path, off_path]),
            scale=np.full((2, 6, 2), 0.5),
            probs=np.array([p_collide, 1.0 - p_collide]))
        return a

    assert rm.evaluate(traj, [bimodal(0.1)]).total < rm.evaluate(traj, [bimodal(0.9)]).total


def test_multiple_agents_compound_risk():
    """Independent agents must combine, not overwrite each other."""
    rm = RiskModel(ego=EGO)
    traj = np.stack([np.arange(1, 7) * 5.0, np.zeros(6)], axis=1)

    one = rm.evaluate(traj, [make_agent(track_id=1, xy=(10.0, 3.5))]).total
    two = rm.evaluate(traj, [make_agent(track_id=1, xy=(10.0, 3.5)),
                             make_agent(track_id=2, xy=(20.0, 3.5))]).total
    assert two >= one


def test_empty_scene_is_zero_risk():
    rm = RiskModel(ego=EGO)
    traj = np.stack([np.arange(1, 7) * 5.0, np.zeros(6)], axis=1)
    assert rm.evaluate(traj, []).total == pytest.approx(0.0)


def test_risk_stays_a_probability():
    """Compounding many agents must saturate at 1, never exceed it."""
    rm = RiskModel(ego=EGO)
    traj = np.stack([np.arange(1, 7) * 2.0, np.zeros(6)], axis=1)
    crowd = [make_agent(track_id=i, xy=(2.0 * i, 0.0)) for i in range(1, 7)]
    total = rm.evaluate(traj, crowd).total
    assert 0.0 <= total <= 1.0
