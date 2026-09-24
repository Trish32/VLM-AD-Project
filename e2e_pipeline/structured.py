"""(7) Structured intermediate representation, and gating on it.

A planner that emits only a trajectory is hard to argue with: when it does
something odd there is no intermediate quantity to point at. This module makes
the implicit explicit -- per-object intent and time-to-collision, scene-level
drivable fraction and light state, a scalar risk level -- and then gates the
plan against those structured facts rather than against raw geometry.

WHAT THIS IS NOT
----------------
The usual way to get these is an auxiliary DECODER HEAD trained on the shared
backbone, supervised with intent labels and TTC targets. That is not what this
is, and the difference is not cosmetic: nothing in this repo is trained, so an
untrained head would emit noise. Every field here is COMPUTED from quantities
the pipeline already produces -- velocities, yaw, forecasts, occupancy rasters --
by explicit formulas you can read and check.

The practical trade is: no learned priors (a head could learn that a car slowing
near a junction is probably turning, this cannot), but also no training data, no
label noise, and no silent distribution shift. For a validation layer that
second property matters more than the first -- a gate you cannot audit is not
much of a gate.

WHAT WAS ALREADY COVERED
------------------------
Two of the three requested gates already exist and are deliberately NOT
duplicated here:

  * red light vs. non-stopping plan  -> `verifier._check_rules` (`red_light`,
    and `decision_consistency` for the VLM-said-STOP case)
  * risk score over threshold -> degrade  -> `verifier.compare_to_shadow`

What is new is the TTC gate: an object closing fast enough to matter while the
plan does not slow down. Nothing previously connected those two facts -- the
safety filter checks whether a trajectory *collides*, not whether it *responds*.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scene import SceneRepresentation

# Intent classification. Thresholds are geometric, not learned.
YAW_RATE_TURN = 0.15        # rad/s sustained -> turning rather than drifting
LATERAL_DRIFT = 0.8         # m/s sideways with low yaw rate -> lane change
MOVING_MPS = 0.6            # below this, intent is 'stationary' not 'straight'

# TTC gating.
# Set from the MEASURED min-TTC distribution over 200 closed-loop steps
# (p10 0.9 / p50 2.2 / p90 7.8 s), not from a plausible-sounding round number.
# The original 3.0 sat ABOVE the median, so it fired on more than half of all
# frames by construction -- a threshold above p50 guarantees noise.
TTC_CRITICAL_S = 1.5        # closing inside this demands a response
TTC_MIN_CLOSING = 0.5       # m/s; slower closure is not a conflict
DECEL_EXPECTED = -0.3       # m/s^2; the plan must be at least this negative
STATIONARY_PLAN_M = 0.5     # total plan arc below this: nothing to respond with

INTENTS = ('stationary', 'straight', 'left', 'right', 'lane_change')


@dataclass
class ObjectFact:
    """One agent, as the structured layer sees it."""

    track_id: int
    intent: str
    ttc_s: float                    # inf when not closing
    closing_mps: float
    distance_m: float

    def describe(self) -> str:
        ttc = 'inf' if not np.isfinite(self.ttc_s) else f'{self.ttc_s:.1f}s'
        return (f'track {self.track_id}: {self.intent}  ttc {ttc}  '
                f'closing {self.closing_mps:+.1f} m/s  at {self.distance_m:.1f} m')


@dataclass
class SceneFacts:
    drivable_fraction: float
    light: str
    n_agents: int


@dataclass
class StructuredScene:
    objects: list[ObjectFact] = field(default_factory=list)
    scene: SceneFacts | None = None
    risk_level: float = 0.0         # 0 = clear, 1 = imminent

    @property
    def min_ttc(self) -> float:
        vals = [o.ttc_s for o in self.objects if np.isfinite(o.ttc_s)]
        return min(vals) if vals else float('inf')

    def describe(self) -> str:
        head = (f'risk {self.risk_level:.2f}  min-ttc '
                f'{"inf" if not np.isfinite(self.min_ttc) else f"{self.min_ttc:.1f}s"}')
        if self.scene:
            head += (f'  light {self.scene.light}  drivable '
                     f'{self.scene.drivable_fraction:.0%}  agents {self.scene.n_agents}')
        return '\n'.join([head] + ['  ' + o.describe() for o in self.objects[:6]])


def classify_intent(agent) -> str:
    """Straight / left / right / lane_change / stationary, from kinematics.

    Uses yaw rate where a forecast supplies one, else infers from the angle
    between heading and velocity: a vehicle whose velocity points off its own
    axis is changing lane, one whose heading is rotating is turning. Crude
    compared to a trained classifier, but every decision is inspectable, and the
    cases it confuses (slow turn vs. lane change) are ones where the downstream
    gate treats both the same way anyway.
    """
    v = np.asarray(agent.vxy, dtype=np.float64)
    speed = float(np.linalg.norm(v))
    if speed < MOVING_MPS:
        return 'stationary'

    # Angle between the agent's heading and where it is actually going.
    heading = float(agent.yaw)
    course = float(np.arctan2(v[1], v[0]))
    slip = (course - heading + np.pi) % (2 * np.pi) - np.pi

    # Turn detection needs a heading RATE, which a single frame cannot supply.
    # `Agent` has no yaw_rate field, so the original `getattr(agent,'yaw_rate',0)`
    # was always 0.0 and left/right were unreachable -- a four-way classifier
    # that was structurally three-way, silently. Measured over 200 steps: 5167
    # stationary, 3111 straight, 57 lane_change, 0 left, 0 right.
    #
    # The rate is now taken from the forecast when one is attached, which is the
    # only place in this pipeline that knows where an agent is going. Without a
    # forecast the vocabulary is honestly three-way, and `turn_observable` says
    # so rather than reporting 'straight' for a turning car.
    yaw_rate = _forecast_yaw_rate(agent)
    if yaw_rate is not None and abs(yaw_rate) > YAW_RATE_TURN:
        return 'left' if yaw_rate > 0 else 'right'

    lateral = speed * np.sin(slip)
    if abs(lateral) > LATERAL_DRIFT:
        return 'lane_change'
    return 'straight'


def _forecast_yaw_rate(agent, dt: float = 0.5) -> float | None:
    """Heading rate from the attached forecast, or None when there is none."""
    pred = getattr(agent, 'pred', None)
    loc = getattr(pred, 'loc', None) if pred is not None else None
    if loc is None:
        return None
    arr = np.asarray(loc, dtype=np.float64)
    if arr.ndim == 3:                      # (modes, T, 2) -> most likely mode
        probs = getattr(pred, 'probs', None)
        arr = arr[int(np.argmax(probs))] if probs is not None else arr[0]
    if arr.ndim != 2 or len(arr) < 3:
        return None
    seg = np.diff(arr, axis=0)
    head = np.arctan2(seg[:, 1], seg[:, 0])
    dh = (np.diff(head) + np.pi) % (2 * np.pi) - np.pi
    return float(np.mean(dh) / dt)


def turn_observable(agent) -> bool:
    """Whether left/right can be distinguished for this agent at all."""
    return _forecast_yaw_rate(agent) is not None


def time_to_collision(agent, ego_speed: float) -> tuple[float, float, float]:
    """(ttc_s, closing_mps, distance_m) along the line joining ego and agent.

    Closing speed is the relative velocity projected onto the line of sight,
    which is the quantity that actually determines whether a gap shrinks. Using
    raw speed instead would flag a fast car driving away, and miss a slow one
    reversing toward us.
    """
    p = np.asarray(agent.xy, dtype=np.float64)
    dist = float(np.linalg.norm(p))
    if dist < 1e-6:
        return 0.0, 0.0, 0.0
    los = p / dist

    v_agent = np.asarray(agent.vxy, dtype=np.float64)
    v_ego = np.array([float(ego_speed), 0.0])          # ego frame: +x forward
    closing = float((v_ego - v_agent) @ los)           # + means gap shrinking

    if closing < TTC_MIN_CLOSING:
        return float('inf'), closing, dist
    return dist / closing, closing, dist


def build_structured(scene: SceneRepresentation, light: str = 'none'
                     ) -> StructuredScene:
    """Derive the structured view from a scene the pipeline already produced."""
    objects = []
    for ag in scene.agents:
        ttc, closing, dist = time_to_collision(ag, scene.ego.speed)
        objects.append(ObjectFact(track_id=int(ag.track_id),
                                  intent=classify_intent(ag),
                                  ttc_s=ttc, closing_mps=closing,
                                  distance_m=dist))
    objects.sort(key=lambda o: o.ttc_s)

    fs = getattr(scene, 'freespace', None)
    drivable = float(fs.traversable.mean()) if fs is not None else 1.0
    facts = SceneFacts(drivable_fraction=drivable, light=light,
                       n_agents=len(scene.agents))

    # Risk level saturates as the nearest TTC approaches zero. A ratio rather
    # than a probability: it is a gating scalar, and calling it a probability
    # would imply a calibration nothing here establishes.
    mt = min((o.ttc_s for o in objects), default=float('inf'))
    risk = 0.0 if not np.isfinite(mt) else float(np.clip(1.0 - mt / TTC_CRITICAL_S,
                                                         0.0, 1.0))
    return StructuredScene(objects=objects, scene=facts, risk_level=risk)


@dataclass
class Intervention:
    reason: str
    detail: str

    def __str__(self) -> str:
        return f'[INTERVENE] {self.reason}: {self.detail}'


def structured_gate(struct: StructuredScene, traj: np.ndarray, ego_speed: float,
                    dt: float = 0.5) -> list[Intervention]:
    """The gate this module exists for: does the plan RESPOND to the facts?

    The safety filter asks whether a trajectory collides. This asks something
    different and previously unchecked: whether a trajectory *reacts*. A plan
    that threads past a vehicle closing at 2 s TTC without shedding any speed may
    be geometrically clear and still wrong, because it has no margin for that
    vehicle doing anything other than exactly what was predicted.
    """
    out = []
    if len(traj) < 2:
        return out

    # Same guard the verifier needed: when the plan is already stopped, "it does
    # not decelerate" is vacuous and the substitute would be the plan itself.
    # Profiling caught this BEFORE integration -- unguarded it fired on 46% of
    # steps at a firing median ego speed of 0.07 m/s, reproducing the verifier's
    # original false-positive population exactly.
    arc = float(np.linalg.norm(np.diff(np.vstack([[0.0, 0.0], np.asarray(traj)]),
                                       axis=0), axis=1).sum())
    if arc < STATIONARY_PLAN_M:
        return out

    pts = np.vstack([[0.0, 0.0], np.asarray(traj, dtype=np.float64)])
    speeds = np.linalg.norm(np.diff(pts, axis=0), axis=1) / dt
    accel = (speeds[-1] - float(ego_speed)) / (len(speeds) * dt)

    mt = struct.min_ttc
    if np.isfinite(mt) and mt < TTC_CRITICAL_S and accel > DECEL_EXPECTED:
        worst = struct.objects[0]
        out.append(Intervention(
            'ttc_without_response',
            f'min TTC {mt:.1f}s (track {worst.track_id}, {worst.intent}, '
            f'closing {worst.closing_mps:.1f} m/s) but plan accelerates '
            f'{accel:+.2f} m/s^2'))
    return out


# --- lane topology ----------------------------------------------------------
#
# Intent inferred from bare kinematics cannot mean much: "left turn" is a claim
# about a lane graph, and without one the classifier can only say "drifting
# sideways". This adds the graph, from the nuScenes map expansion.
#
# It is OPTIONAL and supplied by the caller rather than reached for.
# `SceneRepresentation` is ego-frame and map-free by design -- that is what makes
# the detector swappable -- so pulling a map into it would couple the scene
# contract to one dataset. `LaneContext` carries the map and the global ego pose
# alongside, and everything degrades to None when it is absent.

LANE_RADIUS_M = 3.0          # lane assignment tolerance


@dataclass
class LaneContext:
    """What the structured layer needs to query a map, supplied externally."""

    nusc_map: object
    ego_xy_global: np.ndarray
    ego_yaw_global: float


@dataclass
class LaneFacts:
    ego_lane: str | None = None
    same_lane: list[int] = field(default_factory=list)      # track ids
    merging: list[int] = field(default_factory=list)        # feed into ego's lane
    unassigned: int = 0                                     # off-lane agents

    def describe(self) -> str:
        return (f'lane {self.ego_lane[:8] if self.ego_lane else "none"}  '
                f'same {self.same_lane}  merging {self.merging}  '
                f'off-lane {self.unassigned}')


def build_lane_facts(ctx: LaneContext | None, scene) -> LaneFacts | None:
    """Lane membership and merge relationships, or None without a map.

    An agent is 'merging' when its lane is an INCOMING edge of the ego's lane --
    it will join our lane without currently being in it. That is the case
    neither the safety filter nor the TTC gate can see: the filter checks where
    things are, TTC checks the line of sight, and a vehicle about to merge is
    conflicting on neither measure until it already has.
    """
    if ctx is None or ctx.nusc_map is None:
        return None
    m = ctx.nusc_map
    ex, ey = float(ctx.ego_xy_global[0]), float(ctx.ego_xy_global[1])
    ego_lane = m.get_closest_lane(ex, ey, radius=LANE_RADIUS_M) or None

    facts = LaneFacts(ego_lane=ego_lane)
    if ego_lane is None:
        facts.unassigned = len(scene.agents)
        return facts

    try:
        incoming = set(m.get_incoming_lane_ids(ego_lane))
    except Exception:
        incoming = set()

    c, s = np.cos(ctx.ego_yaw_global), np.sin(ctx.ego_yaw_global)
    for a in scene.agents:
        gx = ex + c * float(a.xy[0]) - s * float(a.xy[1])
        gy = ey + s * float(a.xy[0]) + c * float(a.xy[1])
        lane = m.get_closest_lane(gx, gy, radius=LANE_RADIUS_M) or None
        if lane is None:
            facts.unassigned += 1
        elif lane == ego_lane:
            facts.same_lane.append(int(a.track_id))
        elif lane in incoming:
            facts.merging.append(int(a.track_id))
    return facts


def lane_conflict_gate(facts: LaneFacts | None, struct: StructuredScene,
                       ttc_s: float = 4.0) -> list[Intervention]:
    """A vehicle merging into our lane while closing, before it is in our path.

    Deliberately a LONGER horizon than the TTC gate (4 s vs 1.5 s): a merge is
    foreseeable earlier than a rear-end, and the whole point is to react before
    the geometry makes it obvious. Firing on the same 1.5 s would mean the
    vehicle is already alongside, at which point the ordinary gates see it too.
    """
    out = []
    if facts is None or not facts.merging:
        return out
    by_id = {o.track_id: o for o in struct.objects}
    for tid in facts.merging:
        o = by_id.get(tid)
        if o is not None and np.isfinite(o.ttc_s) and o.ttc_s < ttc_s:
            out.append(Intervention(
                'merging_conflict',
                f'track {tid} in an incoming lane, ttc {o.ttc_s:.1f}s, '
                f'closing {o.closing_mps:.1f} m/s -- not yet in our path'))
    return out
