"""The bridge: VLM emits a driving intent, DiffusionDrive turns it into trajectories.

    cameras -> [Qwen2.5-VL]  -> DrivingIntent {command, target_speed, light, hazard}
                                      |
                        DiffusionDrive anchors for that command
                                      |
                              safety / feasibility filter
                                      |
                                 controller -> KBM

WHY THIS SHAPE
--------------
The interface already existed and nobody was using it. DiffusionDrive carries
`3 commands x 6 anchors` and the *command selects the anchor set*, so a VLM that
emits one of three words plugs straight into a trained head with no retraining.
Everything else here is about making that one word trustworthy.

Three properties make it safe to let a 7B model steer at all:

1. The VLM is UPSTREAM of the safety filter. It chooses among candidates; the
   filter decides which candidates exist. A hallucinated intent cannot produce a
   collision because the geometric and risk gates sit downstream of it — the
   worst it can do is pick a worse feasible plan, or get overruled into a brake.

2. Intent is a SLOWLY-VARYING signal, so the VLM's ~7.5 s latency stops being a
   defect. `IntentCache` holds the last intent while the planner and filter run
   at 20-55 Hz. What was "this model is too slow to drive" becomes "this model
   runs at the cadence its output actually changes".

3. Intents are VALIDATED before use. Asked for a driving intent on a red-light
   frame, the model returned `hazard: "red_traffic_light_ahead"` together with
   `target_speed_mps: 15` — it identified the hazard and then ignored it. A
   self-contradictory intent is clamped here rather than being handed to a
   planner and left for the safety filter to catch downstream.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .scene import SceneRepresentation

# DiffusionDrive's own ordering, from tools/gen_plan_anchors.py:
#   CMD_RIGHT, CMD_LEFT, CMD_STRAIGHT = 0, 1, 2
# NOT the (right, straight, left) used by the SparseDrive EgoPlanner. Mixing the
# two silently swaps left turns for straight-aheads, so the mapping is explicit.
COMMAND_INDEX = {'right': 0, 'left': 1, 'straight': 2}
INDEX_COMMAND = {v: k for k, v in COMMAND_INDEX.items()}

#: Upstream's threshold on the FINAL future waypoint's lateral offset, metres.
#: nuscenes_converter.py:386 -- `if ego_fut_trajs[-1][0] >= 2: Turn Right`, in the
#: LiDAR frame where x is lateral-right. This package is (forward, left), so the
#: sign flips; the magnitude and the "final step only" rule are upstream's.
COMMAND_LATERAL_THRESHOLD_M = 2.0


def command_from_future(future_xy) -> int:
    """Drive command implied by an ego-frame future trajectory.

    nuScenes carries no navigation command, so `gt_ego_fut_cmd` is DERIVED from
    where the ego actually went -- upstream thresholds the final future
    waypoint's lateral offset at +-2 m. This reproduces that rule so the
    candidate set is conditioned the same way the anchors were clustered
    (`tools/kmeans/kmeans_plan.py` buckets by `gt_ego_fut_cmd` before running
    k-means, so anchors for command c only describe trajectories of that class).

    Parameters
    ----------
    future_xy : (T, 2) waypoints in the ego frame, (forward, left) metres.
    """
    import numpy as _np
    f = _np.asarray(future_xy, dtype=_np.float64)
    if f.ndim != 2 or f.shape[0] == 0:
        return COMMAND_INDEX['straight']
    lateral_left = float(f[-1, 1])
    if lateral_left <= -COMMAND_LATERAL_THRESHOLD_M:
        return COMMAND_INDEX['right']
    if lateral_left >= COMMAND_LATERAL_THRESHOLD_M:
        return COMMAND_INDEX['left']
    return COMMAND_INDEX['straight']

# Structured output: Ollama constrains generation to this schema, which removes
# the prose-parsing failure mode entirely. Verified working on qwen2.5vl:7b.
INTENT_SCHEMA = {
    'type': 'object',
    'properties': {
        'command': {'type': 'string', 'enum': ['left', 'straight', 'right']},
        'target_speed_mps': {'type': 'number'},
        'hazard': {'type': 'string'},
        'confidence': {'type': 'number'},
        'reasoning': {'type': 'string'},
    },
    'required': ['command', 'target_speed_mps', 'hazard', 'confidence'],
}


@dataclass
class DrivingIntent:
    """What the VLM decided, in a form a planner can consume."""

    command: str = 'straight'
    target_speed_mps: float = 0.0
    light: str = 'none'
    hazard: str = ''
    confidence: float = 0.0
    reasoning: str = ''
    clamped: list[str] = field(default_factory=list)   # validation audit trail
    stale_frames: int = 0
    latency_ms: float = 0.0

    @property
    def command_index(self) -> int:
        return COMMAND_INDEX.get(self.command, COMMAND_INDEX['straight'])

    def summary(self) -> str:
        s = (f"{self.command}@{self.target_speed_mps:.1f}m/s "
             f"light={self.light} conf={self.confidence:.2f}")
        if self.hazard:
            s += f" hazard={self.hazard}"
        if self.clamped:
            s += f"  [clamped: {','.join(self.clamped)}]"
        if self.stale_frames:
            s += f"  [stale {self.stale_frames}]"
        return s


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_intent(intent: DrivingIntent, ego_speed: float,
                    max_speed: float = 16.0,
                    max_accel: float = 3.0, horizon_s: float = 3.0
                    ) -> DrivingIntent:
    """Clamp an intent into something coherent, recording every change.

    The model is capable of naming a hazard and then ignoring it in the same
    breath, so this is not defensive padding — it is the observed failure mode.
    Every clamp is recorded in `clamped` so a silently-corrected intent is
    visible in the logs instead of looking like the model got it right.
    """
    out = DrivingIntent(**{**intent.__dict__, 'clamped': list(intent.clamped)})

    if out.command not in COMMAND_INDEX:
        out.clamped.append(f'command:{out.command}->straight')
        out.command = 'straight'

    # A red or yellow light governing this lane means stop, whatever speed the
    # model asked for. This is the exact contradiction observed in practice.
    if out.light in ('red', 'yellow') and out.target_speed_mps > 0.5:
        out.clamped.append(f'light_{out.light}:speed{out.target_speed_mps:.1f}->0')
        out.target_speed_mps = 0.0

    if not np.isfinite(out.target_speed_mps) or out.target_speed_mps < 0:
        out.clamped.append('speed:nonfinite->0')
        out.target_speed_mps = 0.0
    elif out.target_speed_mps > max_speed:
        out.clamped.append(f'speed:{out.target_speed_mps:.1f}->{max_speed}')
        out.target_speed_mps = max_speed

    # Do not request a speed change the vehicle cannot deliver in the horizon.
    reach_lo = max(0.0, ego_speed - max_accel * horizon_s)
    reach_hi = ego_speed + max_accel * horizon_s
    if not (reach_lo <= out.target_speed_mps <= reach_hi):
        clamped = float(np.clip(out.target_speed_mps, reach_lo, reach_hi))
        out.clamped.append(f'unreachable:{out.target_speed_mps:.1f}->{clamped:.1f}')
        out.target_speed_mps = clamped

    if not np.isfinite(out.confidence):
        out.confidence = 0.0
    out.confidence = float(np.clip(out.confidence, 0.0, 1.0))
    return out


# ---------------------------------------------------------------------------
# Querying the VLM
# ---------------------------------------------------------------------------


_INTENT_PROMPT = """You are the high-level planner of a self-driving car.

IMAGE 1 — top-down BEV of the scene. IMAGE 2 — the forward camera.
{light_line}
MEASURED DETECTIONS (authoritative — from the 3-D detector, not the pictures.
Do NOT estimate distances or speeds from the images):
{detections}

Current speed: {speed:.1f} m/s.

Emit a driving INTENT, not a trajectory. Another module turns it into a path.
  command          : which way the route goes from here
  target_speed_mps : what speed to hold over the next 3 seconds (0 = stop)
  hazard           : the single most relevant hazard, or "none"
  confidence       : 0-1, how sure you are
{light_rule}"""


class VLMIntentPlanner:
    """Query Qwen2.5-VL for a structured driving intent.

    Reuses the two-stage light read from `bevformer_vldrive/tools/vis_infer.py`
    (camera alone, plus a map-projected crop) because that stage is measurably
    more reliable in isolation: a single call carrying detection text answers
    "no traffic lights visible" on frames with an obvious red.
    """

    def __init__(self, model: str = 'qwen2.5vl:7b',
                 base_url: str = 'http://localhost:11434',
                 timeout: int = 180,
                 vis_infer_dir: str | None = None) -> None:
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        # Imported lazily and by path: e2e_pipeline must stay usable without the
        # BEVFormer project present.
        import sys
        d = vis_infer_dir or str(Path(__file__).resolve().parent.parent
                                 / 'bevformer_vldrive' / 'tools')
        if d not in sys.path:
            sys.path.insert(0, d)

    def _post(self, prompt: str, images: list[str], schema: dict) -> dict:
        payload = json.dumps({
            'model': self.model, 'prompt': prompt, 'images': images,
            'stream': False, 'format': schema,
            'options': {'temperature': 0.0, 'num_predict': 160},
        }).encode('utf-8')
        req = urllib.request.Request(f'{self.base_url}/api/generate', data=payload,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(json.loads(r.read())['response'])

    def query(self, bev_b64: str, cam_b64: str | None, detections_text: str,
              ego_speed: float, light: str = 'none') -> DrivingIntent:
        """One intent for the current frame. Never raises — returns a safe default."""
        t0 = time.perf_counter()
        images = [bev_b64] + ([cam_b64] if cam_b64 else [])
        light_line = (f'\nTRAFFIC LIGHT GOVERNING YOUR LANE: **{light.upper()}** '
                      f'(read from the camera; treat as fact).\n'
                      if light not in ('none', '') else '\n')
        light_rule = ('\nA red or yellow light means target_speed_mps = 0, even if '
                      'the road ahead is clear.' if light in ('red', 'yellow') else '')
        prompt = _INTENT_PROMPT.format(light_line=light_line,
                                       detections=detections_text or '  (none)',
                                       speed=ego_speed, light_rule=light_rule)
        try:
            obj = self._post(prompt, images, INTENT_SCHEMA)
            intent = DrivingIntent(
                command=str(obj.get('command', 'straight')).lower(),
                target_speed_mps=float(obj.get('target_speed_mps', 0.0)),
                light=light, hazard=str(obj.get('hazard', '')),
                confidence=float(obj.get('confidence', 0.0)),
                reasoning=str(obj.get('reasoning', '')))
        except Exception as exc:
            # A failed intent must not read as permission to proceed.
            intent = DrivingIntent(command='straight', target_speed_mps=0.0,
                                   light=light, hazard=f'vlm_error:{exc}',
                                   confidence=0.0)
            intent.clamped.append('query_failed->stop')
        intent.latency_ms = (time.perf_counter() - t0) * 1000
        return validate_intent(intent, ego_speed)


# ---------------------------------------------------------------------------
# Rate decoupling
# ---------------------------------------------------------------------------


class IntentCache:
    """Hold the last intent while the fast path runs, and decay trust in it.

    The VLM takes ~7.5 s; the planner and safety filter take ~25 ms. Blocking the
    control loop on the VLM would be absurd, and it is also unnecessary — intent
    changes on the scale of seconds, which is exactly the cadence the VLM can
    sustain. This is what turns the latency from a defect into an architecture.

    Staleness is not free, though: an intent nobody has refreshed is evidence
    about a scene that has moved on, so beyond `max_stale` the cache decays
    toward a stop rather than continuing to assert a speed it can no longer
    justify.
    """

    def __init__(self, max_stale: int = 6, decay: float = 0.6) -> None:
        self.max_stale = int(max_stale)
        self.decay = float(decay)
        self._intent: DrivingIntent | None = None
        self._age = 0

    def put(self, intent: DrivingIntent) -> None:
        self._intent = intent
        self._age = 0

    def get(self) -> DrivingIntent:
        if self._intent is None:
            # Nothing has been decided yet: hold station rather than roll.
            return DrivingIntent(command='straight', target_speed_mps=0.0,
                                 hazard='no_intent_yet', confidence=0.0)
        out = DrivingIntent(**{**self._intent.__dict__,
                               'clamped': list(self._intent.clamped)})
        out.stale_frames = self._age
        if self._age > self.max_stale:
            n = self._age - self.max_stale
            out.target_speed_mps *= self.decay ** n
            out.confidence *= self.decay ** n
            out.clamped.append(f'stale{self._age}:speed_decayed')
        self._age += 1
        return out

    @property
    def age(self) -> int:
        return self._age


# ---------------------------------------------------------------------------
# Intent -> DiffusionDrive candidates
# ---------------------------------------------------------------------------


def intent_conditioned_planner(anchor_npy: str, dt: float = 0.5,
                               max_accel: float = 3.0):
    """Turn a `DrivingIntent` into DiffusionDrive candidate trajectories.

    The intent picks the anchor set (`command`) and the speed the anchors are
    rescaled to (`target_speed_mps`). The safety filter then chooses among them,
    so the VLM narrows the search and never has the last word.

    SPEED CONDITIONING, DONE ON THE FIRST STEP.  An earlier attempt rescaled the
    horizon-average speed and changed nothing, because the binding constraint is
    per-step: at dt=0.5 s and 3 m/s^2 the first waypoint can only move +-1.5 m/s
    worth of distance from the current speed, so an anchor whose *opening* step
    is too aggressive fails the dynamics gate no matter how reasonable its
    endpoint looks. Conditioning therefore blends the anchor's own speed profile
    toward the target while pinning the first step to what is reachable.

    Frame: anchors are DiffusionDrive's (lateral, forward); this package uses
    (forward, left). Converted once at load.
    """
    raw = np.load(anchor_npy)
    if raw.ndim != 4 or raw.shape[0] != 3 or raw.shape[-1] != 2:
        raise ValueError(f'expected (3, K, T, 2) anchors, got {raw.shape}')
    anchors = np.stack([raw[..., 1], -raw[..., 0]], axis=-1)   # -> (fwd, left)

    def plan(scene: SceneRepresentation, intent: DrivingIntent):
        cands = anchors[intent.command_index].astype(np.float64).copy()
        K, T, _ = cands.shape
        v0 = scene.ego.speed
        v_target = intent.target_speed_mps

        out = np.empty_like(cands)
        for i in range(K):
            path = cands[i]
            step = np.linalg.norm(np.diff(np.vstack([[0.0, 0.0], path]), axis=0),
                                  axis=1)
            total = float(step.sum())
            if total < 1e-6:
                out[i] = path
                continue

            # Desired speed profile: first step pinned to what is reachable from
            # v0, ramping to the intent's target over the horizon.
            v_first = float(np.clip(v_target, v0 - max_accel * dt,
                                    v0 + max_accel * dt))
            speeds = np.linspace(v_first, v_target, T)
            arc = np.cumsum(np.maximum(speeds, 0.0) * dt)
            if arc[-1] < 1e-6:
                out[i] = np.zeros_like(path)          # commanded stop
                continue

            # Same shape, new length, resampled at the desired arc lengths. The
            # lateral profile is preserved; only the timing along it changes.
            scaled = path * (arc[-1] / total)
            cum = np.cumsum(step) / total             # fraction along the shape
            for t in range(T):
                out[i, t] = _interp_along(scaled, cum, arc[t] / arc[-1])

        # Prefer anchors whose realised endpoint speed lands nearest the intent.
        implied = np.linalg.norm(out[:, -1, :], axis=1) / (T * dt)
        scores = 1.0 / (1.0 + np.abs(implied - max(v_target, 0.0)))
        return out, scores / scores.sum()
    return plan


def _interp_along(path: np.ndarray, cum: np.ndarray, frac: float) -> np.ndarray:
    """Point at normalised arc-length `frac` along `path` (cum in [0, 1])."""
    frac = float(np.clip(frac, 0.0, 1.0))
    j = int(np.searchsorted(cum, frac))
    if j <= 0:
        return path[0] * (frac / max(cum[0], 1e-9))
    if j >= len(path):
        return path[-1]
    lo, hi = cum[j - 1], cum[j]
    w = 0.0 if hi - lo < 1e-9 else (frac - lo) / (hi - lo)
    return path[j - 1] + w * (path[j] - path[j - 1])
