"""Tests for the action-conditioned world model and critic.

Each test isolates one claim the module makes, so a failure names the broken
property rather than just "planning changed".
"""
import numpy as np
import pytest

from e2e_pipeline.freespace import FreeSpace
from e2e_pipeline.scene import Agent, EgoState, SceneRepresentation
from e2e_pipeline.world_model import (Action, AnalyticCritic, KinematicWorldModel,
                                      actions_from_trajectory,
                                      plan_with_world_model, rollout)


def _freespace(drivable=True, clear=5.0, nx=140, ny=80, res=0.5):
    return FreeSpace(traversable=np.full((nx, ny), drivable, bool),
                     obstacle=np.zeros((nx, ny), bool),
                     unknown=np.zeros((nx, ny), bool),
                     esdf=np.full((nx, ny), clear, np.float32),
                     origin=(-10.0, -20.0), res=res)


def _scene(agents=(), speed=10.0, **fs):
    return SceneRepresentation(agents=list(agents), freespace=_freespace(**fs),
                               ego=EgoState(speed=speed))


def _agent(x, y, vx=0.0, vy=0.0, tid=1):
    return Agent(track_id=tid, xy=np.array([x, y], float), yaw=0.0,
                 lwh=np.array([4.5, 1.9, 1.5]), vxy=np.array([vx, vy], float),
                 score=1.0, label=0)


def _straight(T=6, step=5.0):
    return np.stack([[step * (t + 1), 0.0] for t in range(T)])


# --- dynamics ---------------------------------------------------------------


def test_zero_action_holds_speed_and_heading():
    wm = KinematicWorldModel()
    z = wm.encode(_scene(speed=8.0))
    out = rollout(wm, z, [Action(0.0, 0.0)] * 4, dt=0.5)
    assert out[-1].ego_v == pytest.approx(8.0)
    assert out[-1].ego_yaw == pytest.approx(0.0)
    # 4 steps x 0.5 s x 8 m/s straight ahead
    assert out[-1].ego_xy[0] == pytest.approx(16.0)
    assert out[-1].ego_xy[1] == pytest.approx(0.0, abs=1e-9)


def test_braking_reduces_speed_and_cannot_go_negative():
    wm = KinematicWorldModel()
    z = wm.encode(_scene(speed=2.0))
    out = rollout(wm, z, [Action(-6.0, 0.0)] * 5, dt=0.5)
    assert out[-1].ego_v == 0.0            # clamped, not reversed
    assert out[-1].ego_xy[0] < 2.0


def test_steering_turns_the_ego():
    wm = KinematicWorldModel()
    z = wm.encode(_scene(speed=10.0))
    out = rollout(wm, z, [Action(0.0, 0.2)] * 4, dt=0.5)
    assert out[-1].ego_yaw > 0.1           # left turn
    assert out[-1].ego_xy[1] > 0.5


def test_agents_advance_by_their_velocity():
    wm = KinematicWorldModel()
    z = wm.encode(_scene([_agent(20.0, 0.0, vx=-4.0)]))
    out = rollout(wm, z, [Action(0.0, 0.0)] * 4, dt=0.5)
    # 2 s of closing at 4 m/s
    assert out[-1].agents[0].xy[0] == pytest.approx(12.0)


def test_agent_variance_grows_monotonically_with_horizon():
    wm = KinematicWorldModel()
    z = wm.encode(_scene([_agent(20.0, 0.0)]))
    out = rollout(wm, z, [Action(0.0, 0.0)] * 5, dt=0.5)
    var = [float(s.agent_var[0]) for s in out]
    assert var[0] == 0.0
    assert all(b > a for a, b in zip(var, var[1:]))


def test_rollout_does_not_mutate_the_initial_state():
    wm = KinematicWorldModel()
    z0 = wm.encode(_scene([_agent(20.0, 0.0, vx=-4.0)]))
    before = z0.agents[0].xy.copy()
    rollout(wm, z0, [Action(1.0, 0.1)] * 4, dt=0.5)
    assert np.allclose(z0.agents[0].xy, before)
    assert z0.ego_xy.tolist() == [0.0, 0.0]


# --- action inversion -------------------------------------------------------


def test_actions_reproduce_the_trajectory_they_came_from():
    """The round trip is the contract: invert a path, roll it out, land on it."""
    wm = KinematicWorldModel()
    traj, v0, dt = _straight(), 10.0, 0.5
    acts = actions_from_trajectory(traj, v0, dt)
    out = rollout(wm, wm.encode(_scene(speed=v0)), acts, dt)
    got = np.array([s.ego_xy for s in out[1:]])
    assert np.allclose(got, traj, atol=0.05)


def test_constant_speed_path_needs_no_acceleration():
    acts = actions_from_trajectory(_straight(step=5.0), v0=10.0, dt=0.5)
    assert all(abs(a.accel) < 1e-6 for a in acts)
    assert all(abs(a.steer) < 1e-6 for a in acts)


# --- critic -----------------------------------------------------------------


def test_avoiding_an_obstacle_outranks_driving_into_it():
    sc = _scene([_agent(20.0, 0.0)])
    swerve = np.stack([[5.0 * (t + 1), 3.0 * min(1, (t + 1) / 3.0)]
                       for t in range(6)])
    ranked = plan_with_world_model(np.stack([_straight(), swerve]), sc, dt=0.5)
    assert ranked[0].index == 1
    # and for the stated reason, not by accident
    by_idx = {r.index: r.score for r in ranked}
    assert by_idx[0].risk > by_idx[1].risk


def test_offroad_is_penalised():
    on = plan_with_world_model(_straight()[None], _scene(), dt=0.5)[0]
    off = plan_with_world_model(_straight()[None], _scene(drivable=False),
                                dt=0.5)[0]
    assert off.score.offroad == pytest.approx(1.0)
    assert on.score.offroad == pytest.approx(0.0)
    assert off.score.total < on.score.total


def test_progress_rewards_covering_ground():
    far = plan_with_world_model(_straight(step=5.0)[None], _scene(), dt=0.5)[0]
    near = plan_with_world_model(_straight(step=1.0)[None], _scene(speed=2.0),
                                 dt=0.5)[0]
    assert far.score.progress > near.score.progress


def test_admissible_restricts_ranking_to_filter_survivors():
    """The critic ranks among admissible actions; it never resurrects one."""
    sc = _scene([_agent(20.0, 0.0)])
    swerve = np.stack([[5.0 * (t + 1), 3.0] for t in range(6)])
    ranked = plan_with_world_model(np.stack([_straight(), swerve]), sc,
                                   dt=0.5, admissible=[0])
    assert [r.index for r in ranked] == [0]


def test_score_breakdown_is_populated():
    r = plan_with_world_model(_straight()[None], _scene([_agent(25.0, 0.0)]),
                              dt=0.5)[0]
    for field in ('risk', 'offroad', 'clearance', 'comfort', 'progress'):
        assert isinstance(getattr(r.score, field), float)
    assert 'total' in r.score.describe()
