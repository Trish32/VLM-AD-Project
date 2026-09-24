"""Action-conditioned world model: plan before acting.

The pipeline up to here is reactive. The VLA emits candidate actions, the safety
filter rejects the infeasible ones, and the best survivor executes -- but every
gate evaluates a candidate against the scene *as it is now*. Nothing asks what
the scene becomes if the candidate is taken, so a candidate that is clear at t=0
and boxed in at t=2s scores identically to one that stays clear.

This module adds that step:

    z_0 = encode(scene)
    z_{t+1} = f(z_t, a_t)            for each candidate action a
    score  = critic(z_0..z_N)
    execute argmax score

so the ordering reflects where a candidate LEADS rather than where it starts.

WHAT z IS, AND WHY IT IS NOT A NEURAL EMBEDDING
-----------------------------------------------
`LatentState` is a structured, inspectable scene encoding: ego pose and speed,
agent poses and velocities, and the positional variance of each agent. It is
"latent" in the sense that matters here -- a compact state the dynamics can be
iterated on -- but every field has units and meaning.

A learned encoder would be the usual choice. It is not used because this project
trains nothing; every component is a port, an integration or an evaluation. An
opaque 256-d vector produced by an untrained network would be noise wearing the
costume of a world model, and the rollouts would be meaningless while looking
sophisticated. The `WorldModel` Protocol below is the seam a learned encoder
drops into when there is something trained to put there.

The same applies to the critic. `AnalyticCritic` composes terms the repo already
measures -- collision risk, drivable-area violation, comfort, progress -- with
stated weights. It is a scoring function, NOT a learned value function, and it is
named accordingly so the distinction cannot be lost downstream.

WHY THIS IS NOT REDUNDANT WITH THE SAFETY FILTER
------------------------------------------------
The filter answers "is this candidate admissible?" as a hard gate at t=0. The
critic answers "of the admissible ones, which leads somewhere better?" over a
horizon. They compose: filter first (cheap, veto), then roll out only survivors
(expensive, ranking). Running the critic on rejected candidates would waste the
rollout, and letting the critic overrule the filter would put a soft score in
front of a hard safety gate -- which is the arrangement this whole stack exists
to avoid.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, Sequence

import numpy as np

from .scene import Agent, EgoState, SceneRepresentation, ego_footprint_corners
from .uncertainty import RiskModel

# Critic term weights. Stated here rather than buried so the ranking is auditable:
# a change in behaviour should be traceable to a number someone chose.
W_RISK = 6.0          # expected collisions over the horizon
W_OFFROAD = 4.0       # fraction of rollout steps outside drivable space
W_CLEARANCE = 1.0     # reward for keeping distance
W_COMFORT = 0.4       # |accel| and |jerk|
W_PROGRESS = 1.0      # distance made good along the candidate
W_PRIOR = 3.0         # the upstream planner's own score for this candidate


@dataclass
class LatentState:
    """z -- the scene encoding the dynamics iterate on.

    Ego is in the ORIGINAL ego frame (the frame z_0 was encoded in), not a
    re-centred one: rolling out must not move the coordinate system, or agent
    positions from different steps would be incomparable and the critic would
    integrate apples with oranges.
    """

    t: float                              # seconds since z_0
    ego_xy: np.ndarray                    # (2,) in z_0's ego frame
    ego_yaw: float
    ego_v: float
    agents: list[Agent] = field(default_factory=list)
    agent_var: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ego_accel: float = 0.0                # what the last step commanded

    def copy(self) -> "LatentState":
        return LatentState(
            t=self.t, ego_xy=self.ego_xy.copy(), ego_yaw=self.ego_yaw,
            ego_v=self.ego_v, agents=[replace(a, xy=a.xy.copy()) for a in self.agents],
            agent_var=self.agent_var.copy(), ego_accel=self.ego_accel)


@dataclass
class Action:
    """a_t -- one control the dynamics consume.

    Kept as (accel, steer) rather than a waypoint so `f` is a true dynamics
    model: a waypoint already presupposes the motion that reaches it, which
    would make the world model a resampler rather than a predictor.
    """

    accel: float                          # m/s^2, longitudinal
    steer: float                          # rad, front-wheel angle


class WorldModel(Protocol):
    """z_{t+1} = f(z_t, a_t). Swap in a learned model behind this."""

    def encode(self, scene: SceneRepresentation) -> LatentState: ...

    def step(self, z: LatentState, a: Action, dt: float) -> LatentState: ...


class KinematicWorldModel:
    """Analytic f: bicycle model for ego, constant velocity for agents.

    Deliberately the same kinematics the closed-loop simulator uses, so a rollout
    predicts what the controller will actually produce rather than a different
    model's idea of it. Prediction error against the real simulator is then zero
    for the ego and comes entirely from the agents -- which is the honest place
    for it, since agent intent is the part nobody can know.

    Agent uncertainty grows with the horizon via the same speed-scaled process
    noise `TrackCovarianceTracker` uses, so a rollout's later steps are correctly
    less trusted than its early ones.
    """

    def __init__(self, wheelbase: float = 2.85, accel_noise: float = 0.6) -> None:
        self.wheelbase = float(wheelbase)
        self.accel_noise = float(accel_noise)

    def encode(self, scene: SceneRepresentation) -> LatentState:
        return LatentState(
            t=0.0, ego_xy=np.zeros(2), ego_yaw=0.0, ego_v=float(scene.ego.speed),
            agents=[replace(a, xy=np.asarray(a.xy, dtype=np.float64).copy())
                    for a in scene.agents],
            agent_var=np.zeros(len(scene.agents)))

    def step(self, z: LatentState, a: Action, dt: float) -> LatentState:
        out = z.copy()
        # --- ego: kinematic bicycle -------------------------------------
        v = max(0.0, z.ego_v + a.accel * dt)
        yaw = z.ego_yaw + (v / self.wheelbase) * np.tan(a.steer) * dt
        mid = 0.5 * (z.ego_yaw + yaw)
        out.ego_xy = z.ego_xy + v * dt * np.array([np.cos(mid), np.sin(mid)])
        out.ego_yaw = float(yaw)
        out.ego_v = float(v)
        out.ego_accel = float(a.accel)

        # --- agents: constant velocity, variance grows ------------------
        for ag in out.agents:
            ag.xy += np.asarray(ag.vxy, dtype=np.float64) * dt
        # var(t) for a constant-acceleration noise model integrates as
        # sigma_a^2 * t^4 / 4; differencing it per step keeps rollouts of
        # different lengths consistent with a single closed-form evaluation.
        t0, t1 = z.t, z.t + dt
        growth = (self.accel_noise ** 2) * (t1 ** 4 - t0 ** 4) / 4.0
        out.agent_var = z.agent_var + growth
        out.t = t1
        return out


def actions_from_trajectory(traj: np.ndarray, v0: float, dt: float,
                            wheelbase: float = 2.85) -> list[Action]:
    """Invert a candidate waypoint path into the (accel, steer) that tracks it.

    The planner emits waypoints; the world model consumes controls. Recovering
    controls from the path -- rather than teleporting the ego along it -- is what
    makes the rollout a dynamics prediction: a path demanding more acceleration
    or steering than the ego can deliver produces a rollout that FALLS BEHIND it,
    and the critic sees the shortfall.
    """
    pts = np.vstack([[0.0, 0.0], np.asarray(traj, dtype=np.float64)])
    seg = np.diff(pts, axis=0)
    dist = np.linalg.norm(seg, axis=1)
    speeds = dist / dt
    head = np.arctan2(seg[:, 1], seg[:, 0])

    out, v_prev = [], float(v0)
    for i in range(len(seg)):
        accel = (speeds[i] - v_prev) / dt
        dyaw = head[i] - (head[i - 1] if i else 0.0)
        dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi      # wrap to [-pi, pi]
        v_ref = max(speeds[i], 1e-3)
        steer = float(np.arctan(dyaw * wheelbase / (v_ref * dt)))
        out.append(Action(accel=float(accel), steer=steer))
        v_prev = float(speeds[i])
    return out


def rollout(model: WorldModel, z0: LatentState, actions: Sequence[Action],
            dt: float) -> list[LatentState]:
    """z_0 .. z_N under an action sequence. Returns N+1 states including z_0."""
    states = [z0]
    z = z0
    for a in actions:
        z = model.step(z, a, dt)
        states.append(z)
    return states


@dataclass
class CriticScore:
    """Why a candidate ranked where it did -- never just the scalar.

    A bare score is unusable in review: two candidates 0.1 apart could differ by
    a hair of comfort or by a near-collision traded against progress, and the
    scalar cannot tell you which.
    """

    total: float
    risk: float
    offroad: float
    clearance: float
    comfort: float
    progress: float

    def describe(self) -> str:
        return (f"total {self.total:+.3f}  risk {self.risk:.3f}  "
                f"offroad {self.offroad:.2f}  clear {self.clearance:.2f}m  "
                f"comfort {self.comfort:.2f}  progress {self.progress:.2f}m")


class SafetyCritic(Protocol):
    """Score a rollout. Swap in a learned value function behind this."""

    def score(self, states: Sequence[LatentState],
              scene: SceneRepresentation) -> CriticScore: ...


class AnalyticCritic:
    """Weighted sum of measured terms. NOT a learned value function.

    Every term is something the repo already computes, so the score is auditable
    end to end rather than being a number a network asserts. The weights are
    module constants and are the honest tuning surface -- they were chosen for
    scale compatibility, not fitted, and no result in this repo depends on their
    exact values.
    """

    def __init__(self, dt: float = 0.5, tracker=None, calibrator=None) -> None:
        # RiskModel is bound to an EgoState, so it is built per scene in score()
        # rather than held here -- caching one across scenes would silently
        # evaluate new geometry against a stale ego footprint.
        self.dt = float(dt)
        # Both default to None, which is the behaviour this always had: the
        # critic's risk term was computed with no tracker (so the agent
        # covariance fell through to constant_velocity_prediction's hardcoded
        # spread) and no calibrator (so it was in model units, ~6x
        # over-confident). Injectable so that can be measured rather than
        # assumed; the defaults are unchanged so no existing result moves.
        self.tracker = tracker
        self.calibrator = calibrator

    def _drivable(self, scene: SceneRepresentation, xy: np.ndarray) -> bool:
        fs = getattr(scene, 'freespace', None)
        if fs is None:
            return True
        ix = int((xy[0] - fs.origin[0]) / fs.res)
        iy = int((xy[1] - fs.origin[1]) / fs.res)
        if not (0 <= ix < fs.traversable.shape[0] and
                0 <= iy < fs.traversable.shape[1]):
            return False                      # off-grid is not known-drivable
        return bool(fs.traversable[ix, iy])

    def _clearance(self, scene: SceneRepresentation, xy: np.ndarray) -> float:
        fs = getattr(scene, 'freespace', None)
        if fs is None:
            return float('inf')
        ix = int((xy[0] - fs.origin[0]) / fs.res)
        iy = int((xy[1] - fs.origin[1]) / fs.res)
        if not (0 <= ix < fs.esdf.shape[0] and 0 <= iy < fs.esdf.shape[1]):
            return 0.0
        return float(fs.esdf[ix, iy])

    def score(self, states: Sequence[LatentState],
              scene: SceneRepresentation) -> CriticScore:
        if len(states) < 2:
            return CriticScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        path = np.array([s.ego_xy for s in states[1:]])

        # Risk uses the FINAL state's agents and variances, which already carry
        # the horizon's uncertainty growth -- evaluating per step and summing
        # would double-count the same encounter across adjacent steps.
        final = states[-1]
        report = RiskModel(scene.ego, tracker=self.tracker,
                           calibrator=self.calibrator).evaluate(
            path, final.agents, dt=self.dt)
        risk = float(report.expected_collisions)

        offroad = 1.0 - np.mean([self._drivable(scene, p) for p in path])
        clearance = float(np.min([self._clearance(scene, p) for p in path]))

        acc = np.array([s.ego_accel for s in states[1:]])
        jerk = np.diff(acc) if len(acc) > 1 else np.zeros(1)
        comfort = float(np.mean(np.abs(acc)) + np.mean(np.abs(jerk)))

        progress = float(np.linalg.norm(path[-1] - states[0].ego_xy))

        total = (W_PROGRESS * progress
                 + W_CLEARANCE * min(clearance, 5.0)
                 - W_RISK * risk
                 - W_OFFROAD * offroad
                 - W_COMFORT * comfort)
        return CriticScore(total=total, risk=risk, offroad=float(offroad),
                           clearance=clearance, comfort=comfort,
                           progress=progress)


@dataclass
class RankedCandidate:
    index: int
    trajectory: np.ndarray
    score: CriticScore
    states: list[LatentState]


def plan_with_world_model(candidates: np.ndarray, scene: SceneRepresentation,
                          model: WorldModel | None = None,
                          critic: SafetyCritic | None = None,
                          dt: float = 0.5,
                          admissible: Sequence[int] | None = None,
                          prior: Sequence[float] | None = None
                          ) -> list[RankedCandidate]:
    """Roll out each candidate and rank by critic score, best first.

    `admissible` restricts the rollout to candidates the safety filter already
    passed. Ranking is a preference among admissible actions; it never promotes
    one the filter rejected, so the hard gate stays in front of the soft score.

    `prior` is the upstream planner's own score per candidate -- the VLM's intent
    preference, expressed through DiffusionDrive's anchor scoring. It is ADDED to
    the critic rather than replaced by it. Omitting it was a measured mistake:
    a critic that re-ranks on rollout terms alone throws away the intent signal
    the whole VLA stage exists to produce, and the closed loop got worse on every
    metric. The rollout should tell you which admissible action leads somewhere
    better, not re-litigate which action was wanted.
    """
    model = model or KinematicWorldModel()
    critic = critic or AnalyticCritic()
    cands = np.asarray(candidates, dtype=np.float64)
    idxs = range(len(cands)) if admissible is None else list(admissible)

    z0 = model.encode(scene)
    out = []
    for i in idxs:
        acts = actions_from_trajectory(cands[i], scene.ego.speed, dt)
        states = rollout(model, z0, acts, dt)
        sc = critic.score(states, scene)
        if prior is not None and i < len(prior):
            sc = replace(sc, total=sc.total + W_PRIOR * float(prior[i]))
        out.append(RankedCandidate(index=int(i), trajectory=cands[i],
                                   score=sc, states=states))
    out.sort(key=lambda r: r.score.total, reverse=True)
    return out


# --- reactive agents --------------------------------------------------------
#
# The constant-velocity model above cannot explain why rollout ranking lost to
# the safety filter: it re-derives quantities the filter's swept-footprint check
# already has. A critic only earns its cost if it sees something the gate cannot,
# and the obvious candidate is INDUCED BEHAVIOUR -- that cutting in front of a
# vehicle makes it brake, which a footprint sweep against frozen agents can never
# represent.

IDM_A_MAX = 1.5        # m/s^2, comfortable acceleration
IDM_B = 2.0            # m/s^2, comfortable deceleration
IDM_T = 1.5            # s, desired time headway
IDM_S0 = 2.0           # m, minimum standstill gap
LANE_HALF_WIDTH = 1.8  # m, lateral band within which the ego is "in the way"


class ReactiveWorldModel(KinematicWorldModel):
    """Agents brake for the ego instead of ignoring it.

    Longitudinal response only, via IDM: an agent whose heading puts the ego
    inside `LANE_HALF_WIDTH` of its path, and ahead of it, decelerates toward a
    safe headway. Lateral evasion is deliberately NOT modelled -- predicting that
    a driver swerves rather than brakes is a claim about intent that a two-line
    heuristic has no business making, and assuming evasion would make dangerous
    candidates look safe. Braking-only is the conservative half.

    This is what makes the rollout non-trivial: with frozen agents, f(z, a) is
    just the ego's kinematics replayed, and the critic reports what the filter
    already knows. With reaction, two candidates that are both admissible now can
    lead to visibly different futures.
    """

    def __init__(self, wheelbase: float = 2.85, accel_noise: float = 0.6,
                 react: bool = True) -> None:
        super().__init__(wheelbase=wheelbase, accel_noise=accel_noise)
        self.react = bool(react)

    def step(self, z: LatentState, a: Action, dt: float) -> LatentState:
        out = super().step(z, a, dt)
        if not self.react:
            return out

        ego = out.ego_xy
        for ag in out.agents:
            v = np.asarray(ag.vxy, dtype=np.float64)
            speed = float(np.linalg.norm(v))
            if speed < 0.5:
                continue                       # parked: nothing to slow down
            fwd = v / speed
            lat = np.array([-fwd[1], fwd[0]])
            rel = ego - np.asarray(ag.xy, dtype=np.float64)
            gap = float(rel @ fwd)             # + means ego is ahead of the agent
            offset = abs(float(rel @ lat))
            if gap <= 0.0 or offset > LANE_HALF_WIDTH:
                continue                       # ego is behind, or out of its lane

            # IDM: desired gap grows with speed and closing rate.
            closing = speed - out.ego_v
            s_star = (IDM_S0 + max(0.0, speed * IDM_T +
                                   speed * closing / (2 * np.sqrt(IDM_A_MAX * IDM_B))))
            decel = IDM_A_MAX * (s_star / max(gap, 0.5)) ** 2
            new_speed = max(0.0, speed - min(decel, IDM_B * 3.0) * dt)
            ag.vxy = fwd * new_speed
        return out
