"""Tests for the structured intermediate representation and its TTC gate."""
import numpy as np
import pytest

from e2e_pipeline.freespace import FreeSpace
from e2e_pipeline.scene import Agent, EgoState, SceneRepresentation
from e2e_pipeline.planner.structured import (
    TTC_CRITICAL_S, build_structured, classify_intent, structured_gate,
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
    assert structured_gate(st, cruise, 12.0, enabled=True)


def test_gate_silent_when_the_plan_slows():
    st = build_structured(_scene([_ag(15.0, 0.0)]))
    braking = np.stack([[2.0 * (t + 1), 0.0] for t in range(6)])
    assert structured_gate(st, braking, 12.0, enabled=True) == []


def test_gate_silent_when_the_plan_is_already_stopped():
    """The false-positive class profiling caught before integration."""
    st = build_structured(_scene([_ag(6.0, 0.0)], speed=0.0))
    stopped = np.zeros((6, 2))
    assert structured_gate(st, stopped, 0.0, enabled=True) == []


def test_threshold_sits_below_the_observed_median():
    """3.0 s fired on >half of frames by construction; 1.5 s does not."""
    assert TTC_CRITICAL_S < 2.2


# --- lane topology ----------------------------------------------------------


class _FakeMap:
    """Minimal map stand-in: two lanes, one feeding the other."""

    def __init__(self):
        self.ego_lane, self.feeder = 'LANE_EGO', 'LANE_FEED'

    def get_closest_lane(self, x, y, radius=3.0):
        if abs(y) < 2.0:
            return self.ego_lane
        if 2.0 <= abs(y) < 6.0:
            return self.feeder
        return ''

    def get_incoming_lane_ids(self, lane):
        return [self.feeder] if lane == self.ego_lane else []


def test_lane_facts_partition_agents():
    from e2e_pipeline.planner.structured import LaneContext, build_lane_facts
    scene = _scene([_ag(15.0, 0.0, tid=1),      # same lane
                    _ag(18.0, 4.0, tid=2),      # feeder lane
                    _ag(12.0, 20.0, tid=3)])    # off-lane
    ctx = LaneContext(_FakeMap(), np.zeros(2), 0.0)
    f = build_lane_facts(ctx, scene)
    assert f.ego_lane == 'LANE_EGO'
    assert f.same_lane == [1] and f.merging == [2] and f.unassigned == 1


def test_lane_layer_absent_without_a_map():
    """Degrades to None rather than reaching for a map itself."""
    from e2e_pipeline.planner.structured import build_lane_facts
    assert build_lane_facts(None, _scene()) is None


def test_merging_conflict_gate_can_fire():
    """Proof of a positive case -- the gate fired 0 times on real scenes.

    Zero firings alone cannot distinguish 'correctly silent' from 'wired wrong',
    the same ambiguity shadow mode had. This constructs the case it exists for:
    a vehicle in an incoming lane, closing, not yet in our path.
    """
    from e2e_pipeline.planner.structured import (LaneContext, build_lane_facts,
                                         lane_conflict_gate)
    # Must CONVERGE: under CPA an agent holding 4 m of lateral offset is
    # correctly not a conflict, however fast it closes along the sight line.
    merging = _ag(12.0, 4.0, vx=-6.0, vy=-3.0, tid=2)
    scene = _scene([merging], speed=10.0)
    ctx = LaneContext(_FakeMap(), np.zeros(2), 0.0)
    facts = build_lane_facts(ctx, scene)
    assert facts.merging == [2]
    assert lane_conflict_gate(facts, build_structured(scene))


def test_merging_gate_silent_when_not_closing():
    from e2e_pipeline.planner.structured import (LaneContext, build_lane_facts,
                                         lane_conflict_gate)
    scene = _scene([_ag(40.0, 4.0, vx=12.0, tid=2)], speed=10.0)
    ctx = LaneContext(_FakeMap(), np.zeros(2), 0.0)
    facts = build_lane_facts(ctx, scene)
    assert lane_conflict_gate(facts, build_structured(scene)) == []


def test_gate_is_retired_by_default():
    """Retired after two rounds of diagnosis; opt-in only. See structured_gate."""
    st = build_structured(_scene([_ag(15.0, 0.0)]))
    cruise = np.stack([[6.0 * (t + 1), 0.0] for t in range(6)])
    assert structured_gate(st, cruise, 12.0) == []
    assert structured_gate(st, cruise, 12.0, enabled=True)


def test_cpa_ignores_oncoming_traffic_in_the_opposite_lane():
    """The logic error the gate's 36 useless firings traced to.

    Range/range-rate reads an oncoming car one lane over as an imminent head-on,
    because closing speed along the sight line is the SUM of both speeds.
    Closest point of approach asks whether the paths actually meet.
    """
    opposite = _ag(23.0, 3.5, vx=-14.0)
    same_lane = _ag(23.0, 0.0, vx=-14.0)
    assert time_to_collision(opposite, 15.3)[0] == float('inf')
    assert np.isfinite(time_to_collision(same_lane, 15.3)[0])
    # closing speeds are nearly identical; only the miss distance differs
    assert abs(time_to_collision(opposite, 15.3)[1]
               - time_to_collision(same_lane, 15.3)[1]) < 1.0
