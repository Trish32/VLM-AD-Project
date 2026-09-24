"""
(3) Uncertainty-aware prediction  ->  risk-aware trajectory selection.

    3D detection -> tracking + covariance -> trajectory distribution
                 -> diffusion trajectory samples -> risk-aware selection

The default stack treats every predicted agent trajectory as a point that is exactly
where the model says it will be.  That is wrong in a way that matters asymmetrically:
being wrong about a distant parked car costs nothing, being wrong about a cyclist
closing on your path costs everything.  Ranking plans by expected collision
*probability* rather than by nearest-point distance lets that asymmetry express
itself.

WHERE THE UNCERTAINTY COMES FROM
--------------------------------
Four sources, two of which already exist in the stack and were simply never read:

  position / velocity   Sparse4D emits boxes and stable track ids but no covariance —
                        it is a detector with an instance bank, not a filter.  We run
                        a constant-velocity Kalman filter over the association it
                        already provides, which costs nothing and yields a real 4x4 P.
                        Detection score feeds the measurement noise, so a marginal
                        0.3-score box is admitted as a *vague* observation rather
                        than a confident one.

  prediction spread     QCNet's decoder already regresses a Laplace scale per mode,
                        per timestep, per dim (`to_scale_refine_pos` in
                        modules/qcnet_decoder.py).  The checkpoint predicts it, the
                        loss trains it, and nothing downstream ever looked at it.

  mode probability      QCNet's `pi` over K modes — an agent that might turn left or
                        go straight is genuinely bimodal, and collapsing to the
                        argmax throws away exactly the branch that hits you.

  plan multiplicity     DiffusionDrive emits 6 anchors under the commanded mode, so
                        there is a real candidate set to select *among* rather than
                        a single trajectory to accept or reject.

COMBINING THEM
--------------
For agent a, mode k, horizon step t the predicted position is Gaussian-equivalent
with variance

    sigma^2_total(t) = sigma^2_pred(a, k, t)  +  sigma^2_track(a, t)

the first from QCNet's Laplace (var = 2b^2), the second from propagating the Kalman
covariance forward under the constant-velocity model.  They are independent in the
sense that matters here: one is "where will it choose to go", the other is "where is
it right now and how fast", and neither informs the other.

Collision probability against a deterministic ego waypoint is then the probability
that a 2D Gaussian lands inside a disc of radius r_ego + r_agent, which is exactly a
non-central chi-squared CDF with 2 degrees of freedom — no sampling required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scene import (Agent, EgoState, TrajectoryDistribution,
                    yaw_from_waypoints)

try:
    from scipy.stats import ncx2 as _ncx2
except ImportError:                                  # pragma: no cover
    _ncx2 = None


# ---------------------------------------------------------------------------
# Tracking + covariance
# ---------------------------------------------------------------------------


@dataclass
class _TrackState:
    x: np.ndarray                                    # (4,)  [x, y, vx, vy]
    P: np.ndarray                                    # (4, 4)
    last_seen: float
    misses: int = 0


class TrackCovarianceTracker:
    """Constant-velocity Kalman filter keyed by Sparse4D's track ids.

    Sparse4D v3 does association internally (its headline is end-to-end tracking with
    no separate tracker), so this deliberately does *not* re-do data association — it
    only supplies the second-order statistics that the detector does not produce.
    Feeding it the same track id twice in one frame is a caller error.

    Parameters
    ----------
    accel_noise : spectral density of the acceleration process noise, m^2/s^3.  This
        is the "how hard can this thing manoeuvre" knob; 2.0 covers ordinary traffic,
        raise it for agents you expect to brake or swerve.
    pos_noise / vel_noise : base measurement standard deviations at score 1.0, metres
        and m/s.  Divided by the detection score so low-confidence boxes widen the
        posterior instead of dragging the mean.
    score_floor : clamp on that division, so a 0.05-score box does not produce a
        20x noise blow-up.
    max_misses : drop a track after this many consecutive unobserved frames.
    """

    def __init__(self,
                 accel_noise: float = 2.0,
                 pos_noise: float = 0.5,
                 vel_noise: float = 1.0,
                 score_floor: float = 0.1,
                 calibrated_noise: bool = True,
                 max_misses: int = 3) -> None:
        self.accel_noise = float(accel_noise)
        self.pos_noise = float(pos_noise)
        self.vel_noise = float(vel_noise)
        self.score_floor = float(score_floor)
        self.calibrated_noise = bool(calibrated_noise)
        self.max_misses = int(max_misses)
        self._tracks: dict[int, _TrackState] = {}

    # -- model matrices -----------------------------------------------------

    @staticmethod
    def _F(dt: float) -> np.ndarray:
        return np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.float64)

    def _Q(self, dt: float) -> np.ndarray:
        """Standard CV process noise for a white-acceleration model."""
        q = self.accel_noise
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        return q * np.array([[dt4 / 4, 0, dt3 / 2, 0],
                             [0, dt4 / 4, 0, dt3 / 2],
                             [dt3 / 2, 0, dt2, 0],
                             [0, dt3 / 2, 0, dt2]], dtype=np.float64)

    # Measured against nuScenes annotations over 11,730 matched BEVFormer
    # detections: sigma = FLOOR + SLOPE / score fits 2.7x better than the pure
    # reciprocal this used to assume (weighted RMS residual 0.084 m vs 0.225 m).
    #
    # The reciprocal was wrong three ways, and only the first was obvious:
    #   * MAGNITUDE -- it under-estimated error by 1.28x to 1.98x across every
    #     score bin, so the tracker was over-confident about every box and every
    #     risk number downstream inherited that.
    #   * SHAPE -- the under-estimate GREW with score (1.28 low, 1.98 high), so
    #     the curve was too steep, not merely too low.
    #   * FLOOR -- a score-1.0 detection still carries ~0.6 m of position error.
    #     A pure b/s model cannot express an irreducible floor at any b.
    #
    # Direction was right, though: score genuinely predicts error, 2.04 m in the
    # lowest bin against 0.99 m in the highest.
    #
    # Reproduce with `python e2e_pipeline/covariance_calibration.py`.
    MEAS_FLOOR_M = 0.609
    MEAS_SLOPE_M = 0.466

    # VELOCITY IS NOT PREDICTED BY SCORE, and the first cut assumed it was.
    # Measured against `box_velocity` over the same 11,730 matches:
    #
    #   score bin   0.25-0.40  0.40-0.55  0.55-0.70  0.70-0.85  0.85-1.01
    #   vel RMS       2.219      2.133      2.867      3.018      1.945   m/s
    #
    # The least-squares fit is sigma_v = 2.463 - 0.075 / score: the slope is
    # NEGATIVE and two orders below the floor, i.e. flat. Confidence in "there
    # is an object here" says nothing about "and this is how fast it is going",
    # which is unsurprising once stated -- the score is a classification
    # logit and velocity comes from a separate regression branch.
    #
    # So the score shape is applied to position ONLY. Carrying it into velocity
    # (as this did, via sigma_p * vel_noise / pos_noise) was an unmeasured
    # extrapolation that also silently coupled the velocity term to a position
    # scale the calibrated branch no longer uses -- for a world declaring
    # pos_noise=0.1 that inflated sigma_v more than twentyfold.
    MEAS_VEL_MPS = 2.463

    def _R(self, score: float) -> np.ndarray:
        """Measurement covariance for a detection of confidence `score`.

        NOTE under `calibrated_noise` the constructor's `pos_noise`/`vel_noise`
        are IGNORED -- the curve is fitted to BEVFormer output and carries its
        own scale. Pass `calibrated_noise=False` for a world whose boxes do not
        come from that detector (the GT oracle), or its declared fidelity is
        silently discarded.
        """
        s = max(float(score), self.score_floor)
        if self.calibrated_noise:
            sp = (self.MEAS_FLOOR_M + self.MEAS_SLOPE_M / s) ** 2
            sv = self.MEAS_VEL_MPS ** 2
        else:
            sp = (self.pos_noise / s) ** 2
            sv = (self.vel_noise / s) ** 2
        return np.diag([sp, sp, sv, sv])

    # -- main entry point ---------------------------------------------------

    def update(self, agents: list[Agent], timestamp: float) -> list[Agent]:
        """Attach a 4x4 `cov` to every agent, filtering in place across frames.

        Returns the same list, mutated.  Agents whose track id is new are initialised
        from their own measurement with the measurement noise as the prior — an
        honest "we know nothing more than what we just saw".
        """
        seen: set[int] = set()

        for a in agents:
            tid = int(a.track_id)
            seen.add(tid)
            z = np.concatenate([np.asarray(a.xy, dtype=np.float64),
                                np.asarray(a.vxy, dtype=np.float64)])
            R = self._R(a.score)

            tr = self._tracks.get(tid)
            if tr is None:
                self._tracks[tid] = _TrackState(x=z.copy(), P=R.copy(),
                                                last_seen=timestamp)
                a.cov = R.copy()
                continue

            dt = max(timestamp - tr.last_seen, 1e-3)
            F = self._F(dt)
            x_pred = F @ tr.x
            P_pred = F @ tr.P @ F.T + self._Q(dt)

            # Full-state measurement: Sparse4D gives position *and* velocity, so H = I
            # and the update is the textbook form with no projection.
            S = P_pred + R
            K = P_pred @ np.linalg.inv(S)
            tr.x = x_pred + K @ (z - x_pred)
            tr.P = (np.eye(4) - K) @ P_pred
            tr.last_seen = timestamp
            tr.misses = 0

            a.cov = tr.P.copy()

        for tid in list(self._tracks):               # age out unobserved tracks
            if tid not in seen:
                self._tracks[tid].misses += 1
                if self._tracks[tid].misses > self.max_misses:
                    del self._tracks[tid]

        return agents

    # A parked car will not accelerate 4 m in 3 s, but a single `accel_noise`
    # applied to every agent says it might. In a parking lot with 30-50 mostly
    # static agents that inflates each one's collision probability, and because
    # agent risks compound as 1 - prod(1 - p), the total saturates near 1
    # regardless of geometry — the closed loop hit exactly this, rejecting every
    # candidate on every step at 2.4-5.6 m of clearance.
    #
    # Manoeuvre uncertainty genuinely scales with how fast something is already
    # going, so the process noise does too: a stationary agent gets a small
    # fraction, ramping to the full value by `full_noise_speed`.
    STATIC_NOISE_FRACTION = 0.08
    FULL_NOISE_SPEED = 5.0                           # m/s

    def _effective_accel_noise(self, agent: Agent) -> float:
        ramp = min(1.0, agent.speed / self.FULL_NOISE_SPEED)
        scale = self.STATIC_NOISE_FRACTION + (1.0 - self.STATIC_NOISE_FRACTION) * ramp
        return self.accel_noise * scale

    def propagated_position_var(self, agent: Agent, horizon_s: np.ndarray) -> np.ndarray:
        """Position variance (T, 2) from propagating `agent.cov` forward.

        Under the CV model the position block grows as

            var_x(t) = P_xx + 2 t P_xv + t^2 P_vv  +  q t^3 / 3

        where the last term is the integrated acceleration noise.  Agents with no
        covariance yet (first frame, or tracking disabled) fall back to the base
        measurement noise so the risk model never silently sees zero uncertainty.
        """
        t = np.asarray(horizon_s, dtype=np.float64).reshape(-1, 1)      # (T, 1)
        if agent.cov is None:
            base = self.pos_noise ** 2
            P = np.diag([base, base, self.vel_noise ** 2, self.vel_noise ** 2])
        else:
            P = agent.cov

        pos = np.array([P[0, 0], P[1, 1]])[None, :]                     # (1, 2)
        cross = np.array([P[0, 2], P[1, 3]])[None, :]
        vel = np.array([P[2, 2], P[3, 3]])[None, :]
        proc = self._effective_accel_noise(agent) * (t ** 3) / 3.0
        return pos + 2.0 * t * cross + (t ** 2) * vel + proc            # (T, 2)


# ---------------------------------------------------------------------------
# Fallback prediction for agents QCNet does not forecast
# ---------------------------------------------------------------------------


def constant_velocity_prediction(agent: Agent, horizon: int, dt: float,
                                 tracker: TrackCovarianceTracker | None = None,
                                 ) -> TrajectoryDistribution:
    """Single-mode CV rollout, used when no learned forecast is available.

    Not every agent is worth running QCNet on — it is ~12 s/scene at batch size 1 —
    and parked cars in particular do not need a learned multi-modal future.  A CV
    rollout with honestly growing covariance is the right floor: it never claims more
    certainty than the filter supports.
    """
    t = np.arange(1, horizon + 1, dtype=np.float64) * dt                # (T,)
    loc = agent.xy[None, :] + np.outer(t, agent.vxy)                    # (T, 2)

    if tracker is not None:
        var = tracker.propagated_position_var(agent, t)                 # (T, 2)
    else:
        var = (0.5 + 0.5 * t[:, None]) ** 2 * np.ones((1, 2))
    # TrajectoryDistribution stores a Laplace scale; invert var = 2 b^2.
    scale = np.sqrt(np.maximum(var, 1e-6) / 2.0)

    return TrajectoryDistribution(
        loc=loc[None, ...],                                             # (1, T, 2)
        scale=scale[None, ...],
        probs=np.ones(1, dtype=np.float64),
        dt=dt,
    )


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


class _Required:
    """Sentinel distinguishing "no free space here" from "forgot to pass it".

    `freespace=None` meant both, and the difference is the whole bug: six
    separate call sites omitted the argument while `unknown_prior` was set, and
    every one silently computed a risk with the prior switched off. None of them
    raised, none of them logged, and the resulting "the prior changes nothing"
    was reported three times as a finding about occlusion.

    A default that silently disables a safety term is the wrong default. With
    this sentinel, omitting `freespace` while a prior is configured is an
    explicit error; passing `None` is the deliberate "this caller genuinely has
    no free-space raster" and still works.
    """

    def __repr__(self) -> str:                       # pragma: no cover
        return '<required>'

    def __bool__(self) -> bool:                      # pragma: no cover
        return False


_REQUIRED = _Required()


@dataclass
class RiskReport:
    """Per-candidate risk breakdown, kept for diagnosis rather than just a scalar.

    A single risk number tells you a plan was rejected but not what to do about it;
    `per_step` and `worst_agent` tell you whether the problem is one aggressive
    oncoming vehicle at t=3 s or a diffuse crowd at t=0.5 s.
    """

    total: float                                     # max over horizon, in [0, 1]
    per_step: np.ndarray                             # (T,) collision prob per step
    worst_agent: int | None                          # track_id driving `total`
    worst_step: int

    @property
    def expected_collisions(self) -> float:
        """Sum over the horizon — a softer aggregate than the max, useful for ranking
        candidates that all sit below the rejection threshold."""
        return float(self.per_step.sum())


class RiskModel:
    """Collision probability between a deterministic ego plan and uncertain agents.

    Parameters
    ----------
    ego : supplies the footprint radius for the disc-disc approximation.
    tracker : used to propagate state covariance; may be None to use prediction
        spread alone.
    inflate_m : extra margin added to the collision radius, metres.  Covers the gap
        between the true rectangle-rectangle geometry and the circumscribed discs
        this model uses; set to 0 if you want raw disc overlap.
    """

    def __init__(self,
                 ego: EgoState,
                 tracker: TrackCovarianceTracker | None = None,
                 inflate_m: float = 0.0,
                 calibrator=None,
                 unknown_prior: float = 0.0) -> None:
        """`calibrator` maps the raw probability onto observed frequencies.

        Without it this returns a MODEL's probability, measured over 966 rollout
        steps to be over-confident by roughly 6x and worst in the tail -- it
        predicted 0.520 where 0.015 occurred. Any threshold applied to that
        output is therefore in model units, not real ones, which is why
        max_risk=0.05 behaved like 0.008 and why relaxing it improved safety.

        With a calibrator fitted and validated across disjoint scenes (Platt,
        a=0.457 b=-2.333, held-out ECE 0.11-0.18 -> 0.02-0.03), the number means
        what it says and a threshold can be stated as an actual collision
        probability.
        """
        self.ego = ego
        self.tracker = tracker
        self.inflate_m = float(inflate_m)
        self.calibrator = calibrator
        # Probability that an unobserved cell the ego sweeps contains an
        # obstacle. Zero reproduces the old behaviour, which is not "no risk"
        # but UNQUANTIFIED risk: this model is agent-only, so a plan driving
        # into a region no camera ever saw scored exactly as safely as one
        # driving down an empty observed road.
        #
        # The drivable-area gate already excludes unknown cells
        # (`traversable &= ~unknown`), but that is a hard veto on a different
        # axis -- it cannot express "this plan is probably fine and might not
        # be", so the probability the filter thresholds stayed silent about
        # occlusion entirely.
        #
        # A prior makes the missed cases VISIBLE rather than correct. It does
        # not discover what is behind the occlusion; it bounds how confident the
        # number is allowed to be, which converts an unbounded unknown into a
        # bounded estimate. Expect more braking -- that is the trade, not a
        # side effect.
        self.unknown_prior = float(unknown_prior)

    @staticmethod
    def _support_radius(half_len: np.ndarray, half_wid: np.ndarray,
                        heading: np.ndarray, bearing: np.ndarray) -> np.ndarray:
        """Half-extent of a rectangle along `bearing` — its support function.

        Replaces the circumscribed disc, which is badly wrong for the case that
        dominates real scenes. A 4.6 x 1.8 m car has a circumscribed radius of
        2.47 m, so two of them "collide" at 4.94 m of centre separation. Along
        their long axes that is nearly right (true 4.60 m); BROADSIDE it is
        wrong by 3.14 m, because the true distance is 1.80 m.

        Cars parked along a road are broadside to the ego's path, so every one
        of them carried a ~3 m phantom margin. With 30-50 of them each
        contributing a few percent, `1 - prod(1 - p)` saturated near 1 before
        geometry got a say, and the safety filter emergency-braked on scenes
        with 3+ m of real clearance.

        a|cos t| + b|sin t| is exact for a rectangle, and using it along the
        centre-line makes this the separating-axis test on that one axis — still
        conservative (another axis may separate the boxes) but far tighter.
        """
        t = bearing - heading
        return half_len * np.abs(np.cos(t)) + half_wid * np.abs(np.sin(t))

    # -- core probability ---------------------------------------------------

    @staticmethod
    def _disc_probability(mu: np.ndarray, var: np.ndarray, r: float) -> np.ndarray:
        """P(||X|| < r) for X ~ N(mu, diag(var)), vectorised over leading dims.

        Isotropised by averaging the two per-axis variances, which turns the integral
        into a non-central chi-squared with 2 dof:

            ||X||^2 / sigma^2  ~  ncx2(df=2, nc=||mu||^2 / sigma^2)

        The isotropic step is an approximation — a long thin prediction ellipse is
        treated as a disc of equal area — and it is conservative in the direction
        that matters, since the elongation is almost always *along* the agent's path
        where the ego is least likely to be.
        """
        mu = np.asarray(mu, dtype=np.float64)
        var = np.asarray(var, dtype=np.float64)
        sigma2 = np.maximum(var.mean(axis=-1), 1e-9)
        lam2 = (mu ** 2).sum(axis=-1)

        if _ncx2 is not None:
            return _ncx2.cdf((r ** 2) / sigma2, df=2, nc=lam2 / sigma2)

        # Fallback: Rayleigh-style bound, exact when the means coincide (lam2 = 0)
        # and conservative elsewhere.  pragma: no cover
        return np.exp(-np.maximum(lam2 - r ** 2, 0.0) / (2.0 * sigma2))

    # -- main entry point ---------------------------------------------------

    def evaluate(self, ego_traj: np.ndarray, agents: list[Agent],
                 dt: float = 0.5, freespace=_REQUIRED) -> RiskReport:
        """Collision risk of one candidate plan.

        Parameters
        ----------
        ego_traj : (T, 2) planned waypoints in the ego frame.
        agents : scene agents; those carrying a `pred` use it, the rest fall back to
            a constant-velocity rollout.
        dt : seconds per planning step.

        Returns
        -------
        RiskReport whose `total` is the max per-step probability, calibrated
        when a calibrator was supplied.  We take the max
        rather than 1 - prod(1 - p_t) because consecutive steps of the same encounter
        are strongly correlated — chaining them as independent events inflates a
        single close pass into near-certain collision.
        """
        if freespace is _REQUIRED:
            if self.unknown_prior > 0.0:
                raise TypeError(
                    'RiskModel was built with unknown_prior='
                    f'{self.unknown_prior:g} but evaluate() was called without '
                    '`freespace`. The prior would be silently ignored, which is '
                    'how six call sites came to report "the occlusion prior '
                    'changes nothing". Pass freespace=<FreeSpace> to price '
                    'unobserved space, or freespace=None to state explicitly '
                    'that this caller has none.')
            freespace = None

        traj = np.asarray(ego_traj, dtype=np.float64)
        T = traj.shape[0]
        horizon = np.arange(1, T + 1, dtype=np.float64) * dt
        # Ego heading per waypoint: the support radius depends on how the ego is
        # oriented relative to each agent, not just where it is.
        ego_headings = yaw_from_waypoints(traj)

        # Survival product across agents: independence *between* agents is a much
        # safer assumption than independence across time for one agent.
        survival = np.ones(T, dtype=np.float64)
        per_agent_peak: dict[int, float] = {}

        for a in agents:
            pred = a.pred
            # QCNet's Laplace scale is *prediction* spread only — where the agent
            # might choose to go — so the filter's state covariance still has to be
            # added.  The CV fallback below already propagates that covariance
            # internally, so adding it again would double-count and inflate every
            # agent's risk by sqrt(2) in sigma.
            add_state_cov = pred is not None
            if pred is None:
                pred = constant_velocity_prediction(a, T, dt, self.tracker)

            steps = min(T, pred.loc.shape[1])
            if steps == 0:
                continue

            # (K, steps, 2) offsets from each predicted mode to the ego waypoint.
            delta = traj[None, :steps, :] - pred.loc[:, :steps, :]

            # Orientation-aware collision distance along each centre-line.
            bearing = np.arctan2(delta[..., 1], delta[..., 0])
            ego_yaw = ego_headings[None, :steps]
            r = (self._support_radius(0.5 * self.ego.length, 0.5 * self.ego.width,
                                      ego_yaw, bearing)
                 + self._support_radius(0.5 * float(a.lwh[0]), 0.5 * float(a.lwh[1]),
                                        float(a.yaw), bearing)
                 + self.inflate_m)
            var = pred.sigma[:, :steps, :] ** 2                          # (K, steps, 2)

            if self.tracker is not None and add_state_cov:
                var = var + self.tracker.propagated_position_var(
                    a, horizon[:steps])[None, ...]

            p_mode = self._disc_probability(delta, var, r)               # (K, steps)
            p_agent = (pred.probs[:, None] * p_mode).sum(axis=0)         # (steps,)

            survival[:steps] *= (1.0 - np.clip(p_agent, 0.0, 1.0))
            per_agent_peak[int(a.track_id)] = float(p_agent.max())

        per_step = 1.0 - survival
        worst_step = int(np.argmax(per_step))
        worst_agent = max(per_agent_peak, key=per_agent_peak.get) if per_agent_peak else None

        # Calibrate the scalar the filter thresholds on, and the per-step curve
        # with it, so both are in the same units. Applied here rather than at
        # each call site: a threshold comparing against an uncalibrated number
        # somewhere would silently reintroduce the 6x scale error.
        # Occlusion hazard: the swept path through unobserved space, priced at
        # `unknown_prior` per step. Combined as independent survival with the
        # agent term rather than max(), because "an unseen obstacle" and "a
        # tracked agent" are different events and either can end the rollout.
        if self.unknown_prior > 0.0 and freespace is not None and T:
            unk = np.asarray(freespace.unknown_at(ego_traj), dtype=np.float64)
            per_step = 1.0 - (1.0 - per_step) * (1.0 - self.unknown_prior * unk)

        raw_total = float(per_step.max()) if T else 0.0
        if self.calibrator is not None:
            total = float(self.calibrator(raw_total))
            per_step = np.asarray(self.calibrator(per_step), dtype=np.float64)
        else:
            total = raw_total

        return RiskReport(
            total=total,
            per_step=per_step,
            worst_agent=worst_agent,
            worst_step=worst_step,
        )
