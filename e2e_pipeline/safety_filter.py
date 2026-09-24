"""
(2) Safety / feasibility filter:  DiffusionDrive  ->  best feasible trajectory.

A learned planner produces trajectories that are *likely*, not trajectories that are
*permissible*.  Those differ whenever the training distribution is thin — which is
precisely the situation where you need the guarantee.  This module is the last stage
that can say no, and it says no for four independent reasons:

    1. drivable area   — every swept cell is on road surface, not sidewalk or void
    2. collision       — the swept footprint keeps clearance from occupied space
    3. dynamics        — curvature, acceleration and lateral load are reachable by
                         the vehicle the controller actually drives
    4. risk            — probabilistic collision with uncertain agents stays under
                         threshold  (see uncertainty.RiskModel)

Gates 1 and 2 come from the FlashOcc branch and are class-agnostic: they reject a
plan that drives into an unlabelled obstacle the object detector never reported,
which is the specific hole a purely object-centric stack cannot close.  Gate 4 comes
from the Sparse4D + QCNet branch and is identity-aware.  The two are complementary
and neither is redundant.

WHY RE-RANKING IS LEGITIMATE HERE
---------------------------------
This matters and is easy to get wrong.  In our SparseDrive `EgoPlanner`
(`sparse4d_vldrive/.../motion_planning.py`) `ego_fut_mode=3` and the three modes *are*
the driving commands (left / straight / right) — re-ranking across them would silently
override the navigation intent and turn a commanded left turn into a straight-ahead.

DiffusionDrive is different: it has `3 commands x 6 anchors = 18` plan queries, and
the command *selects* which set of 6 to use (see `diffusiondrive_planner/DESIGN.md`).
The 6 candidates handed to this filter are therefore genuine alternatives under one
fixed intent, and choosing among them is a safety decision, not a routing decision.

If every candidate fails, the filter does not return the "least bad" plan — it
returns an explicit emergency-brake trajectory and flags the frame.  A planner that
silently degrades is worse than one that admits it is stuck.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .freespace import FreeSpace
from .scene import EgoState, SceneRepresentation, ego_footprint_corners, yaw_from_waypoints
from .uncertainty import RiskModel, RiskReport


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeasibilityLimits:
    """Dynamic envelope of the ego vehicle.

    Defaults are matched to `simulator/kbm.py` (wheelbase 2.85 m, max steer 0.6 rad)
    so that anything this filter passes is actually executable by the controller
    downstream.  Keeping them in one frozen dataclass rather than scattered constants
    means a change to the vehicle model cannot silently desync the two.
    """

    wheelbase: float = 2.85
    max_steer_rad: float = 0.6                       # ~34 deg, matches KBM clamp
    max_accel: float = 3.0                           # m/s^2, comfortable launch
    max_decel: float = 6.0                           # m/s^2, hard but not ABS-limit
    max_lat_accel: float = 4.0                       # m/s^2, ~0.4 g
    min_clearance: float = 0.5                       # m to nearest occupied cell
    max_risk: float = 0.05                           # reject above 5% collision prob
    allow_unknown: bool = False                      # may plans cross unobserved space

    # --- three-valued unknown handling ------------------------------------
    #
    # The clearance gate used to be two-valued: a footprint either cleared an
    # obstacle or it did not, and unknown space was a separate hard veto at
    # gate 1. That collapses two different statements -- "there IS something
    # here" and "I do not know what is here" -- into the same rejection, and a
    # plan cannot be graded on the second.
    #
    #   footprint meets an OBSERVED OBSTACLE -> hard reject (unchanged)
    #   footprint meets OBSERVED FREE        -> pass       (unchanged)
    #   footprint meets UNKNOWN              -> admissible, but constrained and
    #                                           penalised rather than refused
    #
    # `unknown_speed_limit` is the speed above which traversing unobserved space
    # is penalised: you may drive into what you cannot see, slowly.
    three_valued_unknown: bool = False
    unknown_speed_limit: float = 5.0                 # m/s through unobserved space
    unknown_penalty: float = 2.0                     # ranking cost per metre of it
    unknown_sight_margin_m: float = 1.0              # slack on the stopping check

    def __post_init__(self) -> None:
        """Reject configurations that cannot do what they claim.

        `three_valued_unknown` and `allow_unknown` are two different answers to
        the same question and setting both hides which one applied. And a
        three-valued gate whose penalty is zero is a two-valued gate wearing a
        different name -- it admits unknown space and then prices it at nothing,
        which is exactly the silent no-op this whole line of work has been
        tripping over.
        """
        if self.three_valued_unknown and self.allow_unknown:
            raise ValueError(
                'three_valued_unknown and allow_unknown both set: the first '
                'grades unknown space, the second waves it through. Pick one.')
        if self.three_valued_unknown and self.unknown_penalty <= 0.0:
            raise ValueError(
                'three_valued_unknown with unknown_penalty=0 admits unobserved '
                'space and prices it at nothing, which is allow_unknown with '
                'extra steps. Set a positive penalty.')
        if self.unknown_speed_limit <= 0.0:
            raise ValueError('unknown_speed_limit must be positive')

    @property
    def max_curvature(self) -> float:
        """Tightest turn the steering clamp permits: kappa = tan(delta_max) / L."""
        return float(np.tan(self.max_steer_rad) / self.wheelbase)


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


@dataclass
class CandidateVerdict:
    """Why one candidate passed or failed — the filter's audit trail.

    Deliberately records *all* violated gates rather than short-circuiting on the
    first.  When every candidate is rejected you need to know whether the scene is
    geometrically impossible (all failed on clearance) or the planner is proposing
    undriveable curvature (all failed on dynamics); those call for different fixes.
    """

    index: int
    feasible: bool
    reasons: list[str] = field(default_factory=list)
    min_clearance: float = float("inf")
    off_road_steps: int = 0
    unknown_steps: int = 0
    max_curvature: float = 0.0
    max_accel: float = 0.0
    max_lat_accel: float = 0.0
    risk: RiskReport | None = None
    cost: float = float("inf")
    planner_score: float = 0.0
    unknown_depth_m: float = 0.0      # path length spent in unobserved space
    unknown_entry_m: float = float("inf")   # arc length to the first unknown cell
    unknown_verdict: str = "PASS"     # PASS | PENALIZE | REJECT

    def describe(self) -> str:
        tag = "OK " if self.feasible else "REJ"
        why = ",".join(self.reasons) if self.reasons else "-"
        risk = f"{self.risk.total:.3f}" if self.risk is not None else "n/a"
        return (f"[{tag}] cand {self.index}: clear={self.min_clearance:.2f}m "
                f"risk={risk} kappa={self.max_curvature:.3f} "
                f"cost={self.cost:.3f} ({why})")


@dataclass
class FilterResult:
    """What the filter hands the controller."""

    trajectory: np.ndarray                           # (T, 2) chosen plan, ego frame
    verdicts: list[CandidateVerdict]
    chosen_index: int | None                         # None when braking
    emergency: bool = False

    @property
    def feasible_count(self) -> int:
        return sum(v.feasible for v in self.verdicts)

    def report(self) -> str:
        head = (f"SafetyFilter: {self.feasible_count}/{len(self.verdicts)} feasible"
                + ("  [EMERGENCY BRAKE]" if self.emergency else
                   f"  chose {self.chosen_index}"))
        return "\n".join([head] + ["  " + v.describe() for v in self.verdicts])


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


class SafetyFilter:
    """Gate and rank DiffusionDrive's candidate set.

    Parameters
    ----------
    limits : dynamic envelope and thresholds.
    dt : seconds per planning step (0.5 for the 2 Hz / 3 s horizon this stack uses).
    w_risk / w_clearance / w_planner : cost weights for ranking the *feasible*
        survivors.  The planner term keeps the learned preference in play — among
        equally safe options we want the one DiffusionDrive actually liked, not the
        one that hugs the centre of the widest gap.

        `w_risk` DEFAULTED TO 10.0 AND WAS MEASURED, NOT CHOSEN, ONLY LATER.
        Across feasible candidates the calibrated `risk.total` spreads 0.0073
        while the planner term spreads 0.0095, so risk overtakes the prior at
        w_risk ~ 1.3 and everything above that is the same argmin. 10.0 sat an
        order of magnitude inside that saturated region, which is why three
        successive sweeps over 6-25 reported the weight as inert.

        Swept downward (10 scenes, 20 steps, live perception, derived commands):

            w_risk  EGO  other  brakes  clearance  completion  jerk  mean risk
               0.0    0     27      97     1.18 m       40.0%     -      0.0517
               1.0    0     20      94     1.83 m       39.8%  1.32      0.0507
              10.0    1     21     105     1.37 m       38.3%  1.24      0.0549

        1.0 is better on safety, braking, clearance, completion AND mean risk;
        the single cost is 6.5% more jerk. It is NOT a safety-for-progress
        trade, which is how the 10.0 result was first described here and was
        wrong. Default changed to 1.0; results committed before 0f5f147 used
        10.0 and are labelled as such in EXPERIMENT.md.
    footprint_lattice : (n_long, n_lat) sample points across the ego rectangle used
        for the swept-volume check.  3x2 corners-plus-centre is enough at 0.4 m grid
        resolution; raise it for finer grids.
    """

    def __init__(self,
                 limits: FeasibilityLimits | None = None,
                 dt: float = 0.5,
                 w_risk: float = 1.0,
                 w_clearance: float = 1.0,
                 w_planner: float = 2.0,
                 footprint_lattice: tuple[int, int] = (5, 3),
                 multiplicative: bool = False) -> None:
        self.limits = limits or FeasibilityLimits()
        self.dt = float(dt)
        self.w_risk = float(w_risk)
        self.w_clearance = float(w_clearance)
        self.w_planner = float(w_planner)
        self.footprint_lattice = footprint_lattice
        #: score = clearance * (1 - calibrated_risk) instead of a weighted sum.
        #: Default off so no committed result moves; see `_evaluate` for why the
        #: additive form loses risk discrimination to calibration.
        self.multiplicative = bool(multiplicative)

    # -- geometry -----------------------------------------------------------

    def _footprint_samples(self, xy: np.ndarray, yaw: float,
                           ego: EgoState) -> np.ndarray:
        """(n, 2) world points covering the ego rectangle at one pose.

        Sampling a lattice rather than rasterising the polygon keeps this exact and
        allocation-free; at 0.4 m cells a 5x3 lattice over a 4.6x1.8 m body leaves no
        gap wide enough to hide an obstacle cell.
        """
        n_l, n_w = self.footprint_lattice
        ls = np.linspace(-0.5 * ego.length, 0.5 * ego.length, n_l)
        ws = np.linspace(-0.5 * ego.width, 0.5 * ego.width, n_w)
        local = np.stack(np.meshgrid(ls, ws, indexing="ij"), axis=-1).reshape(-1, 2)
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + np.asarray(xy, dtype=np.float64)

    def _sweep(self, traj: np.ndarray, ego: EgoState) -> np.ndarray:
        """(T, n, 2) footprint samples for every waypoint of a plan."""
        yaws = yaw_from_waypoints(traj, initial_yaw=0.0)
        return np.stack([self._footprint_samples(traj[t], yaws[t], ego)
                         for t in range(len(traj))], axis=0)

    # -- kinematics ---------------------------------------------------------

    def _profile(self, traj: np.ndarray, ego: EgoState
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Speed (T,), longitudinal accel (T,), curvature (T,) implied by waypoints.

        The planner emits positions only, so every dynamic quantity here is a finite
        difference of those positions.  That makes the check slightly pessimistic at
        the endpoints — a real controller smooths across the boundary — which is the
        right direction to err for a safety gate.
        """
        wp = np.asarray(traj, dtype=np.float64)
        T = len(wp)

        # Prepend the ego's current position so the first segment is measured against
        # where the vehicle actually is, not against the first waypoint.
        pts = np.vstack([np.zeros((1, 2)), wp])                          # (T+1, 2)
        seg = np.diff(pts, axis=0)                                       # (T, 2)
        speed = np.linalg.norm(seg, axis=1) / self.dt                    # (T,)

        accel = np.diff(np.concatenate([[ego.speed], speed])) / self.dt  # (T,)

        # Menger curvature over consecutive triples; endpoints reuse their neighbour.
        kappa = np.zeros(T, dtype=np.float64)
        for i in range(1, T):
            a, b, c = pts[i - 1], pts[i], pts[i + 1]
            ab, bc, ca = (np.linalg.norm(b - a), np.linalg.norm(c - b),
                          np.linalg.norm(a - c))
            denom = ab * bc * ca
            if denom < 1e-6:
                continue
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            kappa[i] = abs(2.0 * cross) / denom
        if T > 1:
            kappa[0] = kappa[1]
        return speed, accel, kappa

    # -- unknown space, graded rather than vetoed ---------------------------

    @staticmethod
    def unknown_geometry(traj: np.ndarray, unknown_per_step: np.ndarray
                         ) -> tuple[float, float]:
        """(arc length to the first unknown step, total arc length inside it).

        Both measured along the path, in metres, so they are comparable with a
        stopping distance. `unknown_per_step` is (T,) bool -- whether the swept
        footprint at that step touches an unobserved cell.
        """
        p = np.vstack([np.zeros((1, 2)), np.asarray(traj, float)])
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)              # (T,)
        u = np.asarray(unknown_per_step, bool)
        n = min(len(seg), len(u))
        seg, u = seg[:n], u[:n]
        if not u.any():
            return float("inf"), 0.0
        first = int(np.argmax(u))
        entry = float(seg[:first].sum())
        depth = float(seg[u].sum())
        return entry, depth

    def unknown_feasibility(self, entry_m: float, depth_m: float,
                            ego_speed: float) -> str:
        """PASS / PENALIZE / REJECT for a plan that enters unobserved space.

        THE STOPPING CHECK USES DISTANCE TO THE UNKNOWN, NOT DEPTH INTO IT, and
        that is a deliberate departure from the specification this implements.
        The spec compared stopping distance against penetration depth. Depth is
        the wrong side of the geometry: a hidden obstacle can be anywhere in the
        unobserved region, so the worst case is one sitting at its NEAR edge.
        Safety therefore requires being able to stop before reaching that edge --
        `stopping_distance <= entry`. Comparing against depth makes a plan safer
        the further it commits into the unknown, which inverts the constraint:
        clipping 0.5 m of an occlusion corner would REJECT while ploughing 40 m
        through it would PASS.

        Both quantities are recorded on the verdict so the alternative reading is
        measurable rather than argued away.

        Depth still matters, but as EXPOSURE rather than as a hard limit: it sets
        the ranking penalty, so among plans that can all stop in time the one
        spending least time blind is preferred.
        """
        if not np.isfinite(entry_m):
            return "PASS"
        stopping = float(ego_speed) ** 2 / (2.0 * max(self.limits.max_decel, 1e-6))
        if stopping > entry_m + self.limits.unknown_sight_margin_m:
            return "REJECT"          # cannot halt before the first blind cell
        if float(ego_speed) > self.limits.unknown_speed_limit:
            return "PENALIZE"
        return "PENALIZE" if depth_m > 0.0 else "PASS"

    # -- per-candidate evaluation ------------------------------------------

    def _evaluate(self, index: int, traj: np.ndarray, scene: SceneRepresentation,
                  risk_model: RiskModel | None, planner_score: float
                  ) -> CandidateVerdict:
        lim = self.limits
        fs: FreeSpace = scene.freespace
        ego = scene.ego
        v = CandidateVerdict(index=index, feasible=True, planner_score=planner_score)

        sweep = self._sweep(traj, ego)                                   # (T, n, 2)
        flat = sweep.reshape(-1, 2)

        # --- gate 1: drivable area ----------------------------------------
        unknown = fs.unknown_at(flat).reshape(sweep.shape[:2])           # (T, n)
        v.unknown_steps = int(unknown.any(axis=1).sum())
        three = lim.three_valued_unknown
        if v.unknown_steps and not (lim.allow_unknown or three):
            v.feasible = False
            v.reasons.append(f"unobserved@{v.unknown_steps}steps")

        # `freespace` strips unobserved cells from `traversable` — it cannot vouch for
        # road it never saw.  So when the caller explicitly opts into planning through
        # unknown space we have to re-admit those cells here, or `allow_unknown` would
        # be unreachable: the unknown gate would pass and this one would still reject.
        on_road = fs.traversable_at(flat).reshape(sweep.shape[:2])
        if lim.allow_unknown or three:
            on_road = on_road | unknown
        v.off_road_steps = int((~on_road.all(axis=1)).sum())
        if v.off_road_steps:
            v.feasible = False
            v.reasons.append(f"off-road@{v.off_road_steps}steps")

        # --- gate 1b: unknown, graded ---------------------------------------
        if three and v.unknown_steps:
            v.unknown_entry_m, v.unknown_depth_m = self.unknown_geometry(
                traj, unknown.any(axis=1))
            v.unknown_verdict = self.unknown_feasibility(
                v.unknown_entry_m, v.unknown_depth_m, ego.speed)
            if v.unknown_verdict == "REJECT":
                v.feasible = False
                v.reasons.append(
                    f"blind-stop{v.unknown_entry_m:.1f}<"
                    f"{ego.speed ** 2 / (2 * lim.max_decel):.1f}")

        # --- gate 2: collision / clearance --------------------------------
        # Three-valued: clearance is a statement about OBSERVED obstacles, so
        # unknown cells must not be allowed to fail it. With the geometric mask
        # they always did -- unknown sat directly behind obstacles, so the ESDF
        # there was ~0 and every occlusion-entering candidate died here rather
        # than at the gate meant to judge it (measured: 98.8%, median 0.00 m).
        clear = fs.clearance_at(flat)
        if three:
            observed = ~fs.unknown_at(flat)
            clear = clear[observed] if observed.any() else np.array([np.inf])
        v.min_clearance = float(clear.min())
        if v.min_clearance < lim.min_clearance:
            v.feasible = False
            v.reasons.append(f"clearance{v.min_clearance:.2f}<{lim.min_clearance}")

        # --- gate 3: dynamic feasibility ----------------------------------
        speed, accel, kappa = self._profile(traj, ego)
        v.max_curvature = float(kappa.max())
        v.max_accel = float(np.abs(accel).max())
        v.max_lat_accel = float((speed ** 2 * kappa).max())

        if v.max_curvature > lim.max_curvature:
            v.feasible = False
            v.reasons.append(f"kappa{v.max_curvature:.3f}>{lim.max_curvature:.3f}")
        if accel.max() > lim.max_accel:
            v.feasible = False
            v.reasons.append(f"accel{accel.max():.2f}>{lim.max_accel}")
        if -accel.min() > lim.max_decel:
            v.feasible = False
            v.reasons.append(f"decel{-accel.min():.2f}>{lim.max_decel}")
        if v.max_lat_accel > lim.max_lat_accel:
            v.feasible = False
            v.reasons.append(f"lat{v.max_lat_accel:.2f}>{lim.max_lat_accel}")

        # --- gate 4: probabilistic risk -----------------------------------
        if risk_model is not None:
            # `freespace` was not passed here, so `RiskModel.unknown_prior` was
            # silently dropped for every candidate the filter ever scored. The
            # prior was reachable only from the counterfactual call in
            # closed_loop, which does not gate anything -- so the conservative
            # unknown-space prior could not affect a decision at any value.
            v.risk = risk_model.evaluate(traj, scene.agents, dt=self.dt,
                                         freespace=fs)
            if v.risk.total > lim.max_risk:
                v.feasible = False
                v.reasons.append(f"risk{v.risk.total:.3f}>{lim.max_risk}")

        # --- ranking cost (only meaningful for survivors) ------------------
        risk_term = self.w_risk * (v.risk.total if v.risk is not None else 0.0)
        # Clearance beyond a couple of metres buys nothing, so saturate it rather
        # than rewarding plans that hug the middle of an empty road.
        clear_term = -self.w_clearance * min(v.min_clearance, 2.0)
        plan_term = -self.w_planner * planner_score
        # Exposure, not a veto: among plans that can all stop before the first
        # blind cell, prefer the one spending least path length unable to see.
        # This is where penetration depth belongs -- it grades a plan, it does
        # not decide whether the plan is admissible.
        unknown_term = self.limits.unknown_penalty * v.unknown_depth_m

        if self.multiplicative:
            # WHY ADDITIVE RANKING LOSES TO CALIBRATION.
            #
            # Platt maps every raw risk into [0.0884, 0.1328] -- a band 4.4
            # points wide. In an additive score the risk term can therefore
            # only ever move the total by w_risk * 0.044, while clearance
            # spreads 0.893 across candidates. Calibrating the risk model, which
            # is the correct thing to do to its NUMBERS, destroys its influence
            # on the RANKING: measured, wiring the calibrator into the critic
            # cut risk spread 0.516 -> 0.081, a 6.4x loss of discrimination.
            #
            # Multiplying instead of adding makes risk a FRACTION of the value
            # of a plan rather than a fixed subtraction from it:
            #
            #     score = clearance * (1 - calibrated_risk)
            #
            # A 10% collision probability then costs 10% of whatever the plan
            # was worth, which is scale-free -- it does not matter that the
            # calibrated range is narrow, because the multiplier acts on a term
            # that is not. It also has the right limit behaviour: risk -> 1
            # zeroes the plan's value however much clearance it has, whereas an
            # additive score lets a roomy trajectory buy its way past danger.
            value = (min(v.min_clearance, 2.0) * self.w_clearance
                     + self.w_planner * planner_score)
            risk_p = float(np.clip(v.risk.total if v.risk is not None else 0.0,
                                   0.0, 1.0))
            v.cost = float(-value * (1.0 - risk_p) + unknown_term)
        else:
            v.cost = float(risk_term + clear_term + plan_term + unknown_term)
        return v

    # -- fallback -----------------------------------------------------------

    def emergency_brake(self, ego: EgoState, horizon: int) -> np.ndarray:
        """Straight-ahead maximum-decel stop, (T, 2).

        Emitted when no candidate survives.  Deliberately simple and deliberately not
        a plan — if the filter has rejected everything, the honest action is to shed
        speed along the current heading and let the next frame re-plan, not to invent
        an evasive manoeuvre from the same distribution that just failed.
        """
        v0, a = ego.speed, self.limits.max_decel
        t = np.arange(1, horizon + 1) * self.dt
        stop_t = v0 / a if a > 0 else 0.0
        tc = np.minimum(t, stop_t)
        x = v0 * tc - 0.5 * a * tc ** 2
        return np.stack([x, np.zeros_like(x)], axis=1)

    # -- main entry point ---------------------------------------------------

    def __call__(self, candidates: np.ndarray, scene: SceneRepresentation,
                 planner_scores: np.ndarray | None = None,
                 risk_model: RiskModel | None = None) -> FilterResult:
        """Filter and rank a candidate set.

        Parameters
        ----------
        candidates : (K, T, 2) plans in the ego frame — DiffusionDrive's `plan_reg`
            for the commanded mode.
        scene : the fused representation; supplies free space, agents and ego state.
        planner_scores : (K,) the planner's own preference (`plan_cls` softmaxed).
            Normalised internally; uniform if omitted.
        risk_model : enables gate 4.  Omit to run geometry-only filtering.

        Returns
        -------
        FilterResult with the chosen trajectory and a per-candidate audit trail.
        """
        cand = np.asarray(candidates, dtype=np.float64)
        if cand.ndim != 3 or cand.shape[-1] != 2:
            raise ValueError(f"expected (K, T, 2) candidates, got {cand.shape}")
        K, T, _ = cand.shape

        if planner_scores is None:
            scores = np.full(K, 1.0 / K)
        else:
            s = np.asarray(planner_scores, dtype=np.float64).ravel()
            if s.shape[0] != K:
                raise ValueError(f"planner_scores has {s.shape[0]} entries, need {K}")
            total = s.sum()
            scores = s / total if total > 0 else np.full(K, 1.0 / K)

        verdicts = [self._evaluate(i, cand[i], scene, risk_model, float(scores[i]))
                    for i in range(K)]

        survivors = [v for v in verdicts if v.feasible]
        if not survivors:
            return FilterResult(
                trajectory=self.emergency_brake(scene.ego, T),
                verdicts=verdicts, chosen_index=None, emergency=True)

        best = min(survivors, key=lambda v: v.cost)
        return FilterResult(trajectory=cand[best.index], verdicts=verdicts,
                            chosen_index=best.index, emergency=False)
