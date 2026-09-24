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


# --- shadow mode ------------------------------------------------------------


def _lead(x, tid=9):
    return Agent(track_id=tid, xy=np.array([x, 0.0]), yaw=0.0,
                 lwh=np.array([4.5, 1.9, 1.5]), vxy=np.zeros(2),
                 score=1.0, label=0)


def test_conservative_plan_is_never_flagged():
    """Asymmetry: going slower than the shadow is always acceptable."""
    from e2e_pipeline.verifier import DEGRADE_NONE, compare_to_shadow
    slow = np.stack([[1.0 * (t + 1), 0.0] for t in range(6)])
    rep = compare_to_shadow(slow, _scene())
    assert rep.excess_m < 0
    assert rep.action == DEGRADE_NONE


def test_lane_change_does_not_trigger_degradation():
    """The false-positive this design exists to avoid."""
    from e2e_pipeline.verifier import DEGRADE_NONE, compare_to_shadow
    lane = np.stack([[5.0 * (t + 1), 3.0 * min(1, (t + 1) / 3)] for t in range(6)])
    rep = compare_to_shadow(lane, _scene())
    assert rep.lateral_m == pytest.approx(3.0)      # deviation IS observed
    assert rep.action == DEGRADE_NONE               # and deliberately ignored


def test_closing_on_a_lead_car_triggers_deceleration():
    from e2e_pipeline.verifier import DEGRADE_DECEL, compare_to_shadow
    rep = compare_to_shadow(_cruise(), _scene([_lead(20.0)]))
    assert rep.excess_m > 0
    assert rep.action == DEGRADE_DECEL


def test_severe_overshoot_triggers_pull_over():
    from e2e_pipeline.verifier import DEGRADE_PULLOVER, compare_to_shadow
    rep = compare_to_shadow(_cruise(), _scene([_lead(10.0)]))
    assert rep.action == DEGRADE_PULLOVER


def test_degradation_is_ordered_by_severity():
    from e2e_pipeline.verifier import compare_to_shadow
    far = compare_to_shadow(_cruise(), _scene([_lead(25.0)])).excess_m
    near = compare_to_shadow(_cruise(), _scene([_lead(10.0)])).excess_m
    assert near > far


def test_decelerate_substitutes_the_shadow_itself():
    from e2e_pipeline.verifier import DEGRADE_DECEL, compare_to_shadow
    rep = compare_to_shadow(_cruise(), _scene([_lead(20.0)]))
    assert rep.action == DEGRADE_DECEL
    assert np.allclose(rep.trajectory, rep.shadow)


def test_shadow_converges_toward_the_speed_limit():
    """Approaches the limit at a comfortable rate; does not teleport to it.

    From 30 m/s the shadow cannot reach 16.7 m/s inside a 3 s horizon at 3 m/s^2
    -- that needs 4.4 s. Asserting it arrives would be asserting something
    physically impossible, so the contract is monotone approach.
    """
    from e2e_pipeline.verifier import SPEED_LIMIT_MPS, shadow_plan
    sh = shadow_plan(_scene(speed=30.0), horizon=6, dt=0.5)
    speeds = np.linalg.norm(np.diff(np.vstack([[0.0, 0.0], sh]), axis=0), axis=1) / 0.5
    assert all(b < a for a, b in zip(speeds, speeds[1:]))    # decelerating
    assert speeds[-1] < 30.0

    # and it does arrive, given enough horizon
    long_sh = shadow_plan(_scene(speed=30.0), horizon=14, dt=0.5)
    long_sp = np.linalg.norm(np.diff(np.vstack([[0.0, 0.0], long_sh]), axis=0),
                             axis=1) / 0.5
    assert long_sp[-1] <= SPEED_LIMIT_MPS + 1e-6


def test_shadow_slows_for_a_lead_vehicle():
    from e2e_pipeline.verifier import shadow_plan
    open_road = shadow_plan(_scene(), horizon=6, dt=0.5)
    blocked = shadow_plan(_scene([_lead(12.0)]), horizon=6, dt=0.5)
    assert blocked[-1, 0] < open_road[-1, 0]


def test_shadow_fires_when_the_plan_genuinely_overshoots():
    """Proof the ladder can trigger, not just that it stays quiet.

    Shadow mode never fired across 10 closed-loop scenes, which alone cannot
    distinguish "correctly silent" from "wired wrong". This constructs the case
    it exists for: a plan travelling further than the speed limit allows.
    """
    from e2e_pipeline.verifier import (DEGRADE_DECEL, DEGRADE_PULLOVER,
                                       SPEED_LIMIT_MPS, compare_to_shadow)
    over = np.stack([[(SPEED_LIMIT_MPS + 8.0) * 0.5 * (t + 1), 0.0]
                     for t in range(6)])
    rep = compare_to_shadow(over, _scene(speed=SPEED_LIMIT_MPS + 8.0))
    assert rep.excess_m > 0
    assert rep.action in (DEGRADE_DECEL, DEGRADE_PULLOVER)
    assert rep.trajectory is not rep.shadow or rep.action == DEGRADE_DECEL


# --- collision fault attribution --------------------------------------------


class _Rec:
    def __init__(self, v, yaw=0.0):
        self.ego_xy = np.zeros(2)
        self.ego_yaw = float(yaw)
        self.ego_v = float(v)


@pytest.mark.parametrize('name,v,pos,expect', [
    ('moving into a car ahead',        10.0, (8.0, 0.0),  True),
    ('struck from directly behind',    10.0, (-8.0, 0.0), False),
    ('moving into a car at 90 deg',    10.0, (0.0, 8.0),  True),
    ('stationary, struck from ahead',   0.0, (8.0, 0.0),  False),
    ('stationary, struck from behind',  0.0, (-8.0, 0.0), False),
    ('crawling below threshold',        0.4, (8.0, 0.0),  False),
])
def test_fault_attribution_known_cases(name, v, pos, expect):
    """Pinned because this rule reports 0 for one of its two categories.

    A metric that never produces one of its outcomes is indistinguishable from a
    broken one until the cases are enumerated. Measured on real rollouts: all 30
    contacts occur with the ego stationary, so 0 ego-fault is the rule behaving
    correctly, not failing to fire.
    """
    from e2e_pipeline.metrics import _ego_at_fault
    assert _ego_at_fault(_Rec(v), (np.array(pos),)) is expect
