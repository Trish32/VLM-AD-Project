"""Tests for the independent verification layer.

One test per rule, each constructed so it fails for exactly that rule -- a
verifier whose tests pass for the wrong reason is worse than none.
"""
import numpy as np
import pytest

from e2e_pipeline.freespace import FreeSpace
from e2e_pipeline.scene import Agent, EgoState, SceneRepresentation
from e2e_pipeline.verifier import (MAX_CURVATURE, SPEED_LIMIT_MPS,
                                   TrajectoryVerifier, comfortable_stop)


def _scene(agents=(), speed=10.0):
    fs = FreeSpace(traversable=np.ones((200, 120), bool),
                   obstacle=np.zeros((200, 120), bool),
                   unknown=np.zeros((200, 120), bool),
                   esdf=np.full((200, 120), 5.0, np.float32),
                   origin=(-10.0, -30.0), res=0.5)
    return SceneRepresentation(agents=list(agents), freespace=fs,
                               ego=EgoState(speed=speed))


def _cruise(step=5.0, n=6):
    return np.stack([[step * (t + 1), 0.0] for t in range(n)])


def _agent(x, y, tid=1, vx=0.0):
    return Agent(track_id=tid, xy=np.array([x, y], float), yaw=0.0,
                 lwh=np.array([4.5, 1.9, 1.5]), vxy=np.array([vx, 0.0]),
                 score=1.0, label=0)


def _rules(rep):
    return {v.rule for v in rep.violations}


# --- the happy path must actually be happy ----------------------------------


def test_clean_plan_passes_untouched():
    rep = TrajectoryVerifier().verify(_cruise(), _scene())
    assert rep.passed and not rep.substituted
    assert rep.violations == []
    assert np.allclose(rep.trajectory, _cruise())


# --- (a) physical feasibility -----------------------------------------------


def test_excessive_acceleration_rejected():
    # 0 -> 20 m/s inside one 0.5 s step
    traj = np.stack([[10.0 * (t + 1), 0.0] for t in range(6)])
    rep = TrajectoryVerifier().verify(traj, _scene(speed=0.0))
    assert not rep.passed
    assert 'accel_limit' in _rules(rep)


def test_excessive_curvature_rejected():
    # a right-angle hook well inside the 2.9 m turning radius
    traj = np.array([[2.0, 0.0], [2.5, 0.0], [2.5, 1.5], [2.5, 3.0],
                     [2.5, 4.5], [2.5, 6.0]])
    rep = TrajectoryVerifier().verify(traj, _scene(speed=4.0))
    assert not rep.passed
    assert {'curvature_limit', 'lateral_accel'} & _rules(rep)


def test_gentle_plan_clears_the_dynamics_gates():
    rep = TrajectoryVerifier().verify(_cruise(step=2.0), _scene(speed=4.0))
    assert not ({'accel_limit', 'decel_limit', 'curvature_limit',
                 'lateral_accel'} & _rules(rep))


# --- (b) traffic rules ------------------------------------------------------


def test_red_light_with_a_rolling_plan_rejected():
    rep = TrajectoryVerifier().verify(_cruise(), _scene(), light='red')
    assert not rep.passed
    assert 'red_light' in _rules(rep)


def test_red_light_with_a_stopping_plan_passes():
    stopping = comfortable_stop(v0=10.0, horizon=6, dt=0.5)
    rep = TrajectoryVerifier().verify(stopping, _scene(), light='red')
    assert 'red_light' not in _rules(rep)


def test_green_light_does_not_require_stopping():
    rep = TrajectoryVerifier().verify(_cruise(), _scene(), light='green')
    assert rep.passed


def test_speed_limit_enforced():
    fast = np.stack([[12.0 * (t + 1), 0.0] for t in range(6)])
    rep = TrajectoryVerifier().verify(fast, _scene(speed=24.0))
    assert 'speed_limit' in _rules(rep)


def test_speed_limit_is_configurable_not_hardcoded():
    fast = np.stack([[12.0 * (t + 1), 0.0] for t in range(6)])
    v = TrajectoryVerifier(speed_limit_mps=30.0)
    assert 'speed_limit' not in _rules(v.verify(fast, _scene(speed=24.0)))


def test_decision_stop_contradicted_by_a_rolling_plan():
    """The gap no other component covered."""
    rep = TrajectoryVerifier().verify(_cruise(), _scene(), decision='STOP')
    assert not rep.passed
    assert 'decision_consistency' in _rules(rep)


def test_decision_slow_down_that_accelerates_only_warns():
    accel = np.stack([[3.0 * (t + 1) ** 1.4, 0.0] for t in range(6)])
    rep = TrajectoryVerifier().verify(accel, _scene(speed=3.0),
                                      decision='SLOW_DOWN')
    warns = [v for v in rep.violations if v.rule == 'decision_consistency']
    assert warns and all(v.severity == 'warn' for v in warns)


def test_decision_proceed_is_consistent_with_cruising():
    rep = TrajectoryVerifier().verify(_cruise(), _scene(), decision='PROCEED')
    assert rep.passed


# --- (c) collision ----------------------------------------------------------


def test_stationary_agent_in_path_rejected():
    rep = TrajectoryVerifier().verify(_cruise(), _scene([_agent(15.0, 0.0)]))
    assert not rep.passed
    assert 'collision' in _rules(rep)


def test_agent_well_off_to_the_side_is_fine():
    rep = TrajectoryVerifier().verify(_cruise(), _scene([_agent(15.0, 12.0)]))
    assert 'collision' not in _rules(rep)


def test_agents_are_propagated_not_frozen():
    """An agent that will cross our path must be caught before it is there."""
    # The ego reaches x=15 at step 3 (t=1.5 s). Put the agent at y=0, x=15 at
    # exactly that moment: 18 m of lateral travel in 1.5 s.
    crossing = Agent(track_id=2, xy=np.array([15.0, 18.0]), yaw=-np.pi / 2,
                     lwh=np.array([4.5, 1.9, 1.5]),
                     vxy=np.array([0.0, -12.0]), score=1.0, label=0)
    rep = TrajectoryVerifier().verify(_cruise(), _scene([crossing]))
    assert 'collision' in _rules(rep)


# --- fallback ---------------------------------------------------------------


def test_rejection_substitutes_a_decelerating_plan():
    rep = TrajectoryVerifier().verify(_cruise(), _scene(), decision='STOP')
    assert rep.substituted
    speeds = np.linalg.norm(np.diff(np.vstack([[0, 0], rep.trajectory]), axis=0),
                            axis=1) / 0.5
    assert speeds[-1] < speeds[0]


def test_substitute_false_reports_without_replacing():
    rep = TrajectoryVerifier().verify(_cruise(), _scene(), decision='STOP',
                                      substitute=False)
    assert not rep.passed and not rep.substituted
    assert np.allclose(rep.trajectory, _cruise())


def test_fallback_never_reverses():
    traj = comfortable_stop(v0=2.0, horizon=8, dt=0.5, decel=6.0)
    steps = np.diff(np.vstack([[0.0, 0.0], traj]), axis=0)[:, 0]
    assert (steps >= -1e-9).all()
