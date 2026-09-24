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
