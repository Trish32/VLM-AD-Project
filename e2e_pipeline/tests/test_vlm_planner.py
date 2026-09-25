"""The VLM -> DiffusionDrive bridge.

The interesting surface is not the query (that is one HTTP call) but everything
guarding it: the model is capable of naming a hazard and then ignoring it, and a
7B model is only safe to put upstream of a planner if its output is validated,
bounded and allowed to go stale gracefully.
"""

import numpy as np
import pytest

from e2e_pipeline.scene import EgoState, SceneRepresentation
from e2e_pipeline.planner.vlm_planner import (
    COMMAND_INDEX, DrivingIntent, IntentCache, intent_conditioned_planner,
    validate_intent)

ANCHORS = 'diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy'


class _Scene:
    """Minimal stand-in: the planner only reads ego speed."""

    def __init__(self, speed):
        self.ego = EgoState(speed=speed)


# ---------------------------------------------------------------------------
# Command convention
# ---------------------------------------------------------------------------


def test_command_ordering_matches_diffusiondrive_not_sparsedrive():
    """0=right, 1=left, 2=straight, per tools/gen_plan_anchors.py.

    The SparseDrive EgoPlanner uses (right, straight, left). Mixing the two
    silently turns commanded left turns into straight-aheads — no error, just a
    car that does not turn.
    """
    assert COMMAND_INDEX == {'right': 0, 'left': 1, 'straight': 2}
    assert DrivingIntent(command='left').command_index == 1
    assert DrivingIntent(command='straight').command_index == 2


def test_unknown_command_falls_back_to_straight():
    assert DrivingIntent(command='u-turn').command_index == COMMAND_INDEX['straight']


# ---------------------------------------------------------------------------
# Validation — the observed failure modes
# ---------------------------------------------------------------------------


def test_red_light_forces_a_stop_even_when_speed_requested():
    """The exact contradiction observed: hazard named, then ignored.

    Asked for an intent on a red-light frame the model returned
    hazard="red_traffic_light_ahead" alongside target_speed_mps=15.
    """
    i = validate_intent(DrivingIntent(command='straight', target_speed_mps=15.0,
                                      light='red', hazard='red_traffic_light_ahead'),
                        ego_speed=8.0)
    assert i.target_speed_mps == 0.0
    assert any('light_red' in c for c in i.clamped)


def test_yellow_light_also_stops():
    i = validate_intent(DrivingIntent(target_speed_mps=10.0, light='yellow'),
                        ego_speed=8.0)
    assert i.target_speed_mps == 0.0


def test_green_light_does_not_force_a_stop():
    i = validate_intent(DrivingIntent(target_speed_mps=8.0, light='green'),
                        ego_speed=8.0)
    assert i.target_speed_mps == pytest.approx(8.0)
    assert not i.clamped


def test_unreachable_speed_is_clamped_to_the_dynamic_envelope():
    """Asking a car at 2 m/s for 30 m/s is not an intent, it is a wish."""
    i = validate_intent(DrivingIntent(target_speed_mps=30.0), ego_speed=2.0)
    assert i.target_speed_mps <= 2.0 + 3.0 * 3.0 + 1e-9
    assert any('unreachable' in c or 'speed:' in c for c in i.clamped)


def test_negative_and_nonfinite_speeds_resolve_to_stop():
    for bad in (-5.0, float('nan'), float('inf')):
        assert validate_intent(DrivingIntent(target_speed_mps=bad),
                               ego_speed=5.0).target_speed_mps == 0.0


def test_confidence_is_bounded():
    assert validate_intent(DrivingIntent(confidence=7.0), 5.0).confidence == 1.0
    assert validate_intent(DrivingIntent(confidence=-1.0), 5.0).confidence == 0.0


def test_every_clamp_is_recorded():
    """A silently-corrected intent must not look like the model got it right."""
    i = validate_intent(DrivingIntent(command='sideways', target_speed_mps=99.0,
                                      light='red'), ego_speed=5.0)
    assert len(i.clamped) >= 2
    assert 'clamped' in i.summary()


def test_validation_does_not_mutate_the_input():
    original = DrivingIntent(target_speed_mps=15.0, light='red')
    validate_intent(original, ego_speed=5.0)
    assert original.target_speed_mps == 15.0 and not original.clamped


# ---------------------------------------------------------------------------
# Rate decoupling
# ---------------------------------------------------------------------------


def test_cache_holds_the_last_intent_between_slow_queries():
    """~7.5 s VLM against a ~25 ms planner: the fast path must not block."""
    c = IntentCache()
    c.put(DrivingIntent(command='left', target_speed_mps=6.0, confidence=0.9))
    for expected_age in range(4):
        got = c.get()
        assert got.command == 'left'
        assert got.stale_frames == expected_age


def test_no_intent_yet_holds_station_rather_than_rolling():
    got = IntentCache().get()
    assert got.target_speed_mps == 0.0
    assert got.hazard == 'no_intent_yet'


def test_a_stale_intent_decays_toward_a_stop():
    """An unrefreshed intent is evidence about a scene that has moved on."""
    c = IntentCache(max_stale=2, decay=0.5)
    c.put(DrivingIntent(command='straight', target_speed_mps=8.0, confidence=1.0))
    speeds = [c.get().target_speed_mps for _ in range(7)]
    assert speeds[:3] == pytest.approx([8.0, 8.0, 8.0])   # within max_stale
    assert speeds[-1] < 1.0                                # decayed past it
    assert all(b <= a + 1e-9 for a, b in zip(speeds, speeds[1:]))


def test_a_fresh_intent_resets_staleness():
    c = IntentCache(max_stale=1, decay=0.5)
    c.put(DrivingIntent(target_speed_mps=8.0))
    for _ in range(4):
        c.get()
    c.put(DrivingIntent(target_speed_mps=8.0))
    assert c.get().target_speed_mps == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Intent -> candidates
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def planner():
    import os
    if not os.path.exists(ANCHORS):
        pytest.skip('DiffusionDrive anchors not generated')
    return intent_conditioned_planner(ANCHORS)


def test_command_selects_the_anchor_set(planner):
    """Left turns go +y (left) in this package's frame, right turns -y."""
    left, _ = planner(_Scene(5.0), DrivingIntent(command='left', target_speed_mps=5.0))
    right, _ = planner(_Scene(5.0), DrivingIntent(command='right', target_speed_mps=5.0))
    assert left[:, -1, 1].mean() > 0
    assert right[:, -1, 1].mean() < 0


def test_straight_stays_straight(planner):
    c, _ = planner(_Scene(5.0), DrivingIntent(command='straight', target_speed_mps=5.0))
    assert abs(c[:, -1, 1]).max() < 2.0


def test_first_step_is_always_dynamically_reachable(planner):
    """The constraint that matters is per-step, not horizon-average.

    An earlier version rescaled the endpoint speed and changed nothing, because
    at dt=0.5 s and 3 m/s^2 the opening waypoint can only move +-1.5 m/s worth of
    distance — an anchor whose first step is too aggressive fails the dynamics
    gate however sensible its endpoint looks.
    """
    for v0, target in [(5.0, 0.0), (5.0, 15.0), (0.0, 12.0), (12.0, 0.0)]:
        c, _ = planner(_Scene(v0), DrivingIntent(target_speed_mps=target))
        first = np.linalg.norm(c[:, 0, :], axis=1) / 0.5
        # 1e-2 m/s of slack: resampling interpolates linearly along a curved
        # path, so a chord sits marginally inside the arc. The physical claim is
        # "within the reachable band", not "to machine precision".
        assert first.max() <= v0 + 3.0 * 0.5 + 1e-2, (v0, target, first.max())
        assert first.min() >= max(0.0, v0 - 3.0 * 0.5) - 1e-2, (v0, target, first.min())


def test_commanded_stop_produces_a_zero_trajectory(planner):
    c, _ = planner(_Scene(0.0), DrivingIntent(target_speed_mps=0.0))
    assert np.abs(c).max() == pytest.approx(0.0, abs=1e-6)


def test_lateral_shape_survives_speed_conditioning(planner):
    """Re-timing must change how far, not which way."""
    slow, _ = planner(_Scene(5.0), DrivingIntent(command='left', target_speed_mps=3.0))
    fast, _ = planner(_Scene(5.0), DrivingIntent(command='left', target_speed_mps=6.5))
    assert np.linalg.norm(fast[:, -1], axis=1).mean() > \
           np.linalg.norm(slow[:, -1], axis=1).mean()
    assert slow[:, -1, 1].mean() > 0 and fast[:, -1, 1].mean() > 0


def test_scores_are_a_normalised_preference(planner):
    _, s = planner(_Scene(5.0), DrivingIntent(target_speed_mps=5.0))
    assert s.sum() == pytest.approx(1.0)
    assert (s >= 0).all()


def test_candidates_are_consumable_by_the_safety_filter(planner):
    """The whole point: the VLM narrows the set, the filter still decides."""
    from e2e_pipeline.freespace import FREE_CLASS, FreeSpaceExtractor, GridConfig
    grid = GridConfig(x=(-10.0, 50.0, 0.4), y=(-20.0, 20.0, 0.4), z=(-1.0, 5.4, 0.4))
    sem = np.full(grid.shape, FREE_CLASS, dtype=np.int64)
    sem[:, :, grid.z_band_indices(-0.4, 0.2)[0]] = 11
    fs = FreeSpaceExtractor(grid)(sem)

    from e2e_pipeline.planner.safety_filter import SafetyFilter
    ego = EgoState(speed=5.0)
    cands, scores = planner(_Scene(5.0), DrivingIntent(target_speed_mps=5.0))
    scene = SceneRepresentation(agents=[], freespace=fs, ego=ego)
    result = SafetyFilter()(cands, scene, scores)
    assert result.trajectory.shape[1] == 2
    assert len(result.verdicts) == len(cands)
