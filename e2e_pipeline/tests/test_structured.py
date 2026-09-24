"""Tests for the structured intermediate representation and its TTC gate."""
import numpy as np
import pytest

from e2e_pipeline.freespace import FreeSpace
from e2e_pipeline.scene import Agent, EgoState, SceneRepresentation
from e2e_pipeline.structured import (TTC_CRITICAL_S, build_structured,
                                     classify_intent, structured_gate,
                                     time_to_collision, turn_observable)


def _scene(agents=(), speed=12.0):
    fs = FreeSpace(traversable=np.ones((200, 120), bool),
                   obstacle=np.zeros((200, 120), bool),
                   unknown=np.zeros((200, 120), bool),
                   esdf=np.full((200, 120), 5.0, np.float32),
                   origin=(-10.0, -30.0), res=0.5)
    return SceneRepresentation(agents=list(agents), freespace=fs,
                               ego=EgoState(speed=speed))


def _ag(x, y, vx=0.0, vy=0.0, yaw=0.0, tid=1):
    return Agent(track_id=tid, xy=np.array([x, y], float), yaw=yaw,
                 lwh=np.array([4.5, 1.9, 1.5]), vxy=np.array([vx, vy], float),
                 score=1.0, label=0)


def test_ttc_uses_closing_speed_not_raw_speed():
    """A fast car driving AWAY is not a conflict."""
    away, _, _ = time_to_collision(_ag(20.0, 0.0, vx=25.0), ego_speed=12.0)
    toward, _, _ = time_to_collision(_ag(20.0, 0.0, vx=-5.0), ego_speed=12.0)
    assert away == float('inf')
    assert np.isfinite(toward) and toward < 2.0


def test_stationary_agent_ahead_gives_finite_ttc():
    ttc, closing, dist = time_to_collision(_ag(24.0, 0.0), ego_speed=12.0)
    assert closing == pytest.approx(12.0)
    assert ttc == pytest.approx(2.0)
    assert dist == pytest.approx(24.0)


def test_intent_vocabulary_is_honest_without_forecasts():
    """left/right are unreachable with no forecast, and that is reported.

    The original classifier read a `yaw_rate` attribute Agent does not have, so
    the turn branch was dead and turning cars were labelled 'straight'. Measured
    over 200 closed-loop steps: 0 of 8335 agents were turn-observable.
    """
    a = _ag(20.0, 0.0, vx=5.0)
    assert turn_observable(a) is False
    assert classify_intent(a) in ('straight', 'lane_change', 'stationary')


def test_lateral_drift_reads_as_lane_change():
    assert classify_intent(_ag(30.0, 3.0, vx=5.0, vy=2.0)) == 'lane_change'


def test_slow_agent_is_stationary_not_straight():
    assert classify_intent(_ag(20.0, 0.0, vx=0.1)) == 'stationary'


def test_gate_fires_when_closing_fast_without_slowing():
    st = build_structured(_scene([_ag(15.0, 0.0)]))
    cruise = np.stack([[6.0 * (t + 1), 0.0] for t in range(6)])
    assert structured_gate(st, cruise, 12.0)


def test_gate_silent_when_the_plan_slows():
    st = build_structured(_scene([_ag(15.0, 0.0)]))
    braking = np.stack([[2.0 * (t + 1), 0.0] for t in range(6)])
    assert structured_gate(st, braking, 12.0) == []


def test_gate_silent_when_the_plan_is_already_stopped():
    """The false-positive class profiling caught before integration."""
    st = build_structured(_scene([_ag(6.0, 0.0)], speed=0.0))
    stopped = np.zeros((6, 2))
    assert structured_gate(st, stopped, 0.0) == []


def test_threshold_sits_below_the_observed_median():
    """3.0 s fired on >half of frames by construction; 1.5 s does not."""
    assert TTC_CRITICAL_S < 2.2
