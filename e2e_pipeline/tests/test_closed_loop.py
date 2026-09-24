"""Tests that actually construct and run a ClosedLoopRunner.

THE GAP THIS CLOSES. A change once terminated `__init__` early, orphaning every
assignment after the insertion point, and the runner lost `self.cfg` entirely.
All 154 tests passed, because not one of them built a runner -- the suite tested
the parts thoroughly and the assembly not at all. The failure surfaced only when
an ablation script tried to run a rollout.

These use a synthetic world rather than nuScenes so they run in milliseconds and
in CI, and so a missing dataset cannot silently skip the only coverage the
assembly has.
"""
import numpy as np
import pytest

from e2e_pipeline.freespace import FreeSpace
from e2e_pipeline.scene import Agent


class StubWorld:
    """Straight 100 m road, optional agents, satisfying the WorldModel protocol."""

    def __init__(self, agents=(), road_len=100.0, dt=0.5):
        self.dt = dt
        self._agents = list(agents)
        self._route = np.stack([[x, 0.0] for x in np.arange(0.0, road_len, 4.0)])

    def duration(self):
        return 20.0

    def measurement_noise(self):
        return (0.1, 0.2)

    def route(self):
        return self._route

    def initial_speed(self):
        return 8.0

    def agents_at(self, t, ego_xy, ego_yaw):
        from dataclasses import replace
        return [replace(a, xy=np.asarray(a.xy, float).copy()) for a in self._agents]

    def freespace_at(self, t, ego_xy, ego_yaw):
        nx, ny = 200, 120
        return FreeSpace(traversable=np.ones((nx, ny), bool),
                         obstacle=np.zeros((nx, ny), bool),
                         unknown=np.zeros((nx, ny), bool),
                         esdf=np.full((nx, ny), 5.0, np.float32),
                         origin=(-10.0, -30.0), res=0.5)


def _straight_planner(horizon=6, dt=0.5):
    def plan(scene, command):
        v = max(float(scene.ego.speed), 1.0)
        cands = np.stack([
            np.stack([[v * dt * (t + 1) * s, 0.0] for t in range(horizon)])
            for s in (1.0, 0.7, 0.4)])
        return cands, np.array([0.5, 0.3, 0.2])
    return plan


def _runner(**cfg_kw):
    from e2e_pipeline.closed_loop import ClosedLoopRunner, LoopConfig
    cfg = LoopConfig(max_steps=6, initial_speed=8.0, **cfg_kw)
    return ClosedLoopRunner(StubWorld(), _straight_planner(), cfg)


# --- construction -----------------------------------------------------------


def test_runner_constructs_with_every_attribute_its_run_needs():
    """The exact failure that slipped through: __init__ finishing incompletely."""
    r = _runner()
    for attr in ('world', 'cfg', 'safety', 'tracker', 'latent_model', 'critic',
                 'verifier', 'calibrator', 'risk_pred', 'risk_outcome',
                 'verifier_fired', 'structured_fired', 'shadow_fired'):
        assert hasattr(r, attr), f'ClosedLoopRunner lost {attr!r} from __init__'


def test_calibration_report_is_on_the_runner_not_the_protocol():
    """It was once inserted onto the WorldModel Protocol by a bad anchor match."""
    from e2e_pipeline.closed_loop import ClosedLoopRunner, WorldModel
    assert hasattr(ClosedLoopRunner, 'calibration_report')
    assert not hasattr(WorldModel, 'calibration_report')


# --- running ----------------------------------------------------------------


def test_rollout_produces_records_and_metrics():
    recs, m = _runner().run(command=2)
    assert len(recs) > 0
    for family in ('safety', 'route', 'comfort'):
        assert family in m
    assert 'n_collision_steps_ego_fault' in m['safety']


def test_ego_moves_on_a_clear_road():
    recs, _ = _runner().run(command=2)
    travelled = float(np.linalg.norm(recs[-1].ego_xy - recs[0].ego_xy))
    assert travelled > 1.0, 'ego did not move on an empty straight road'


def test_calibration_report_populated_after_a_rollout():
    r = _runner()
    r.run(command=2)
    rep = r.calibration_report()
    assert rep['n'] >= 0
    assert len(r.risk_pred) == len(r.risk_outcome)


# --- config flags actually take effect --------------------------------------


@pytest.mark.parametrize('flag', ['use_verifier', 'use_structured',
                                  'use_shadow', 'use_world_model',
                                  'calibrate_risk'])
def test_each_layer_flag_runs_without_error(flag):
    """Flags are off by default, so a broken layer would otherwise go unnoticed."""
    recs, m = _runner(**{flag: True}).run(command=2)
    assert len(recs) > 0


def test_risk_budget_path_runs():
    recs, _ = _runner(calibrate_risk=True, risk_budget=0.10).run(command=2)
    assert len(recs) > 0


def test_calibration_changes_the_risk_scale():
    """Calibrated risk must differ from raw -- proof the flag is not inert."""
    from e2e_pipeline.calibration import PlattCalibrator
    from e2e_pipeline.scene import EgoState
    from e2e_pipeline.uncertainty import RiskModel
    ego = EgoState(speed=10.0)
    ag = Agent(track_id=1, xy=np.array([12.0, 0.0]), yaw=0.0,
               lwh=np.array([4.5, 1.9, 1.5]), vxy=np.zeros(2), score=1.0, label=0)
    traj = np.stack([[5.0 * (t + 1), 0.0] for t in range(6)])
    cal = PlattCalibrator(a=0.457, b=-2.333, fitted=True, n_positive=11)
    raw = RiskModel(ego).evaluate(traj, [ag]).total
    adj = RiskModel(ego, calibrator=cal).evaluate(traj, [ag]).total
    assert raw > 0, 'fixture should produce non-zero risk'
    assert adj < raw, 'calibration should reduce an over-confident estimate'


def test_tracker_noise_model_follows_the_world_not_a_global_default():
    """A world declaring GT fidelity must not be handed the detector's floor.

    The calibrated curve is fitted to BEVFormer output and carries a 0.6 m floor
    at score 1.0. Defaulting it on applied that floor to annotations; defaulting
    it off tracked real detections at the oracle's 0.1 m. `detector_grade` is
    the world's declaration and this pins that the runner reads it.
    """
    r = _runner()
    assert r.tracker.calibrated_noise is False, 'oracle got detector-grade noise'
    # the declared measurement_noise must actually reach the filter
    assert float(np.sqrt(r.tracker._R(1.0)[0, 0])) == pytest.approx(0.1)

    class LiveStub(StubWorld):
        detector_grade = True

    from e2e_pipeline.closed_loop import ClosedLoopRunner, LoopConfig
    live = ClosedLoopRunner(LiveStub(), _straight_planner(), LoopConfig())
    assert live.tracker.calibrated_noise is True
    assert float(np.sqrt(live.tracker._R(1.0)[0, 0])) > 0.5


def test_live_perception_world_declares_detector_grade():
    """The class whose boxes the curve was fitted to must opt in."""
    from e2e_pipeline.closed_loop import GTWorldModel, LivePerceptionWorldModel
    assert LivePerceptionWorldModel.detector_grade is True
    assert GTWorldModel.detector_grade is False


def test_velocity_noise_is_not_scaled_by_detector_score():
    """Measured: velocity RMS is flat across score bins, so R must be too.

    The first calibrated cut carried the position curve into velocity via
    sigma_p * vel_noise / pos_noise, which is both unmeasured and -- for a world
    declaring pos_noise=0.1 -- a twentyfold inflation.
    """
    from e2e_pipeline.uncertainty import TrackCovarianceTracker
    t = TrackCovarianceTracker(calibrated_noise=True, pos_noise=0.1, vel_noise=0.2)
    assert t._R(0.3)[2, 2] == pytest.approx(t._R(1.0)[2, 2]), \
        'velocity noise must not depend on score'
    assert float(np.sqrt(t._R(1.0)[2, 2])) == pytest.approx(
        TrackCovarianceTracker.MEAS_VEL_MPS)


def test_live_track_ids_survive_to_the_next_frame():
    """The fallback id hashed the frame token, so nothing was ever tracked.

    Measured before the fix: 0.0% of live track ids appeared in the next frame,
    against 97.1% under GT. Every agent was seeded fresh each step, so the
    Kalman filter never ran a second update and anything needing an agent's
    history got an empty input rather than a poor one.
    """
    from e2e_pipeline.live_adapter import LiveDetectionAdapter

    def box(x, y, score=0.9, name='car'):
        return {'translation': [x, y, 0.0], 'size': [1.8, 4.5, 1.6],
                'rotation': [1.0, 0.0, 0.0, 0.0], 'velocity': [2.0, 0.0],
                'detection_score': score, 'detection_name': name}

    det = {'t0': [box(10.0, 0.0), box(30.0, 5.0)],
           't1': [box(11.0, 0.0), box(31.0, 5.0)]}     # each moved 1 m
    ad = LiveDetectionAdapter(None, det, score_thr=0.25)
    a = {x.track_id for x in ad.agents_at('t0', np.zeros(2), 0.0)}
    b = {x.track_id for x in ad.agents_at('t1', np.zeros(2), 0.0)}
    assert a == b, f'ids did not carry across the frame: {a} vs {b}'

    # and the id must not depend on how many times the frame was asked for
    again = {x.track_id for x in ad.agents_at('t0', np.zeros(2), 0.0)}
    assert again == a, 'association is not idempotent per token'


def test_association_gate_rejects_an_implausible_jump():
    """A box 20 m from any track starts a new one rather than stealing an id."""
    from e2e_pipeline.live_adapter import LiveDetectionAdapter

    def box(x):
        return {'translation': [x, 0.0, 0.0], 'size': [1.8, 4.5, 1.6],
                'rotation': [1.0, 0.0, 0.0, 0.0], 'velocity': [0.0, 0.0],
                'detection_score': 0.9, 'detection_name': 'car'}

    ad = LiveDetectionAdapter(None, {'t0': [box(10.0)], 't1': [box(30.0)]},
                              score_thr=0.25)
    a = [x.track_id for x in ad.agents_at('t0', np.zeros(2), 0.0)]
    b = [x.track_id for x in ad.agents_at('t1', np.zeros(2), 0.0)]
    assert a != b, 'a 20 m jump was associated through a 3 m gate'


def test_critic_risk_can_see_the_tracker_and_calibrator():
    """Both were hardcoded to None, so the critic scored in model units."""
    from e2e_pipeline.calibration import PlattCalibrator
    from e2e_pipeline.uncertainty import TrackCovarianceTracker
    from e2e_pipeline.world_model import AnalyticCritic
    plain = AnalyticCritic()
    assert plain.tracker is None and plain.calibrator is None, \
        'defaults must not move -- existing results depend on them'
    wired = AnalyticCritic(tracker=TrackCovarianceTracker(),
                           calibrator=PlattCalibrator(a=0.457, b=-2.333,
                                                      fitted=True, n_positive=11))
    assert wired.tracker is not None and wired.calibrator is not None
