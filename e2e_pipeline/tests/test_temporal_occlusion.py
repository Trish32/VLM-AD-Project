"""Unknown as "not observed lately" rather than "geometrically shadowed"."""
import numpy as np
import pytest

from e2e_pipeline.temporal_occlusion import TemporalOcclusionMemory, visible_mask


def test_visible_mask_stops_at_the_obstacle_and_includes_it():
    obs = np.zeros((100, 100), bool)
    obs[70, 45:55] = True
    vis = visible_mask(obs, origin=(-20.0, -20.0), res=0.4)
    assert vis[60, 50], 'clear cell in front of the wall should be visible'
    assert vis[70, 50], 'the wall itself is seen'
    assert not vis[85, 50], 'behind the wall is not seen'


def test_memory_shrinks_unknown_as_the_ego_moves():
    """The whole point: a geometric shadow never shrinks, a memory does."""
    obs = np.zeros((60, 60), bool)
    obs[30, 25:35] = True
    origin, res = (-12.0, -12.0), 0.4
    mem = TemporalOcclusionMemory(bounds=((0.0, 40.0), (-5.0, 5.0)), res=res,
                                  horizon_frames=100)

    first = None
    for k in range(6):
        ego = np.array([4.0 * k, 0.0])
        mem.observe(obs, origin, res, ego, 0.0, frame=k)
        u = mem.unknown_ego(obs.shape, origin, res, ego, 0.0, frame=k)
        if first is None:
            first = float(u.mean())
    assert float(u.mean()) < first, 'unknown did not shrink as the ego drove'


def test_memory_expires_stale_observations():
    obs = np.zeros((40, 40), bool)
    origin, res = (-8.0, -8.0), 0.4
    mem = TemporalOcclusionMemory(bounds=((0.0, 10.0), (-5.0, 5.0)), res=res,
                                  horizon_frames=3)
    ego = np.zeros(2)
    mem.observe(obs, origin, res, ego, 0.0, frame=0)
    fresh = mem.unknown_ego(obs.shape, origin, res, ego, 0.0, frame=1)
    stale = mem.unknown_ego(obs.shape, origin, res, ego, 0.0, frame=50)
    assert fresh.mean() < stale.mean(), 'observations never expired'
    assert stale.all(), 'everything should be unknown long after the last look'


def test_unknown_is_reachable_unlike_the_geometric_shadow():
    """A shadow sits behind an obstacle; unvisited space does not.

    This is the property the geometric definition lacked and the reason the
    occlusion prior measured exactly zero: reaching a shadow meant driving
    through the object casting it.
    """
    from e2e_pipeline.live_adapter import ray_occlusion
    obs = np.zeros((80, 80), bool)
    obs[40, 35:45] = True
    origin, res = (-16.0, -16.0), 0.4
    shadow = ray_occlusion(obs, origin, res)
    mem = TemporalOcclusionMemory(bounds=((0.0, 10.0), (-5.0, 5.0)), res=res,
                                  horizon_frames=10)
    mem.observe(obs, origin, res, np.zeros(2), 0.0, frame=0)
    temporal = mem.unknown_ego(obs.shape, origin, res, np.zeros(2), 0.0, frame=0)
    # straight ahead along +x on the centre line, past the wall
    line = [(i, 40) for i in range(45, 70)]
    assert all(shadow[i, j] for i, j in line), 'fixture: the line is shadowed'
    # the temporal mask also calls it unknown at frame 0 -- the difference is
    # that it stops doing so once observed, which the shadow never does
    assert temporal[line[0]]


# --- three-valued clearance gate -------------------------------------------


def _filter(**kw):
    from e2e_pipeline.planner.safety_filter import FeasibilityLimits, SafetyFilter
    return SafetyFilter(limits=FeasibilityLimits(three_valued_unknown=True, **kw))


def test_unknown_geometry_measures_entry_and_depth_along_the_path():
    f = _filter()
    traj = np.stack([[4.0 * (t + 1), 0.0] for t in range(6)])   # 4 m per step
    per_step = np.array([False, False, True, True, False, False])
    entry, depth = f.unknown_geometry(traj, per_step)
    assert entry == pytest.approx(8.0)     # two clear steps before the first
    assert depth == pytest.approx(8.0)     # two steps inside


def test_stopping_check_uses_distance_to_unknown_not_depth_into_it():
    """The spec compared stopping distance against DEPTH, which inverts it.

    A hidden obstacle can sit at the near edge of the unobserved region, so the
    binding constraint is being able to halt before reaching that edge. Under
    the depth reading, clipping a corner would be rejected and ploughing
    straight through would pass.
    """
    f = _filter(unknown_sight_margin_m=0.0)
    # stopping distance at 10 m/s with 6 m/s^2 is 8.33 m
    assert f.unknown_feasibility(entry_m=20.0, depth_m=0.5, ego_speed=10.0) != 'REJECT'
    assert f.unknown_feasibility(entry_m=2.0, depth_m=40.0, ego_speed=10.0) == 'REJECT'


def test_unknown_is_penalised_not_refused_when_stoppable():
    f = _filter(unknown_speed_limit=5.0, unknown_sight_margin_m=0.0)
    # slow and far from the unknown: admissible
    assert f.unknown_feasibility(entry_m=30.0, depth_m=4.0, ego_speed=3.0) == 'PENALIZE'
    # same geometry, too fast through it
    assert f.unknown_feasibility(entry_m=30.0, depth_m=4.0, ego_speed=12.0) == 'PENALIZE'
    # never touches unknown at all
    assert f.unknown_feasibility(entry_m=float('inf'), depth_m=0.0,
                                 ego_speed=12.0) == 'PASS'


def test_three_valued_gate_does_not_fail_clearance_on_unknown_cells():
    """Clearance is a claim about OBSERVED obstacles.

    With the geometric mask, unknown sat directly behind obstacles, so the ESDF
    there was ~0 and 98.8% of occlusion-entering candidates died on clearance
    rather than at the gate meant to judge them.
    """
    from e2e_pipeline.freespace import FreeSpace
    from e2e_pipeline.scene import EgoState, SceneRepresentation
    nx, ny = 200, 120
    unknown = np.zeros((nx, ny), bool)
    unknown[120:, :] = True
    esdf = np.full((nx, ny), 5.0, np.float32)
    esdf[120:, :] = 0.0                      # unknown cells read as zero clearance
    scene = SceneRepresentation(
        agents=[], ego=EgoState(speed=3.0), timestamp=0.0,
        freespace=FreeSpace(traversable=np.ones((nx, ny), bool),
                            obstacle=np.zeros((nx, ny), bool), unknown=unknown,
                            esdf=esdf, origin=(-10.0, -30.0), res=0.5))
    traj = np.stack([[8.0 * (t + 1), 0.0] for t in range(6)])
    two = _filter().limits.__class__(three_valued_unknown=False)
    from e2e_pipeline.planner.safety_filter import SafetyFilter
    v2 = SafetyFilter(limits=two)._evaluate(0, traj, scene, None, 0.5)
    v3 = _filter()._evaluate(0, traj, scene, None, 0.5)
    assert v2.min_clearance == pytest.approx(0.0), 'fixture: two-valued sees 0 m'
    assert v3.min_clearance > 0.0, 'three-valued must ignore unknown cells here'
    assert v3.unknown_depth_m > 0.0, 'depth should be recorded for the penalty'


# --- make the omission impossible rather than silent ------------------------


def test_omitting_freespace_with_a_prior_is_an_error_not_a_no_op():
    """The exact defect, six times over: a keyword-only arg defaulting to None.

    `freespace=None` meant both "this caller has no raster" and "I forgot", so
    every omission silently disabled the prior and the resulting null was
    reported as a finding about occlusion.
    """
    from e2e_pipeline.scene import EgoState
    from e2e_pipeline.uncertainty import RiskModel
    traj = np.stack([[5.0 * (t + 1), 0.0] for t in range(6)])
    rm = RiskModel(EgoState(speed=8.0), unknown_prior=0.10)
    with pytest.raises(TypeError, match='freespace'):
        rm.evaluate(traj, [])


def test_explicit_none_is_still_allowed():
    """A caller that genuinely has no free space must still be able to say so."""
    from e2e_pipeline.scene import EgoState
    from e2e_pipeline.uncertainty import RiskModel
    traj = np.stack([[5.0 * (t + 1), 0.0] for t in range(6)])
    assert RiskModel(EgoState(speed=8.0), unknown_prior=0.10).evaluate(
        traj, [], freespace=None).total >= 0.0


def test_no_prior_means_freespace_stays_optional():
    """Most call sites have no prior and must not be forced to pass a raster."""
    from e2e_pipeline.scene import EgoState
    from e2e_pipeline.uncertainty import RiskModel
    traj = np.stack([[5.0 * (t + 1), 0.0] for t in range(6)])
    assert RiskModel(EgoState(speed=8.0)).evaluate(traj, []).total >= 0.0


def test_limits_reject_contradictory_unknown_configuration():
    from e2e_pipeline.planner.safety_filter import FeasibilityLimits
    with pytest.raises(ValueError, match='Pick one'):
        FeasibilityLimits(three_valued_unknown=True, allow_unknown=True)
    with pytest.raises(ValueError, match='extra steps'):
        FeasibilityLimits(three_valued_unknown=True, unknown_penalty=0.0)
    with pytest.raises(ValueError, match='unknown_speed_limit'):
        FeasibilityLimits(unknown_speed_limit=0.0)
    FeasibilityLimits(three_valued_unknown=True)          # the valid one


# --- detection vs tracking submissions --------------------------------------


def test_load_detections_normalises_a_tracking_submission(tmp_path):
    """Both submission flavours must arrive downstream in one shape.

    A nuScenes DETECTION submission has detection_name/detection_score and no
    identity; a TRACKING submission has tracking_name/tracking_score/
    tracking_id. The tracker was designed for the second and fed the first.
    """
    import json

    from e2e_pipeline.live_adapter import load_detections
    box = {'translation': [1.0, 2.0, 0.0], 'size': [1.8, 4.5, 1.6],
           'rotation': [1.0, 0.0, 0.0, 0.0], 'velocity': [0.0, 0.0],
           'tracking_name': 'car', 'tracking_score': 0.8, 'tracking_id': '3_17'}
    p = tmp_path / 'track.json'
    p.write_text(json.dumps({'meta': {}, 'results': {'tok': [box]}}))
    got = load_detections(p)['tok'][0]
    assert got['detection_name'] == 'car'
    assert got['detection_score'] == pytest.approx(0.8)
    assert got['tracking_id'] == '3_17', 'identity must survive normalisation'


def test_real_track_ids_are_used_in_preference_to_association():
    """Associating on top of a real tracker discards a measured AMOTA 0.627."""
    from e2e_pipeline.live_adapter import LiveDetectionAdapter

    def box(x, tid):
        return {'translation': [x, 0.0, 0.0], 'size': [1.8, 4.5, 1.6],
                'rotation': [1.0, 0.0, 0.0, 0.0], 'velocity': [0.0, 0.0],
                'detection_name': 'car', 'detection_score': 0.9,
                'tracking_id': tid}

    # the same object jumps 20 m, which the 3 m association gate would reject --
    # a real tracker says it is the same object and must win
    det = {'t0': [box(5.0, 'a')], 't1': [box(25.0, 'a')]}
    ad = LiveDetectionAdapter(None, det, score_thr=0.25)
    a = [x.track_id for x in ad.agents_at('t0', np.zeros(2), 0.0)]
    b = [x.track_id for x in ad.agents_at('t1', np.zeros(2), 0.0)]
    assert a == b, 'real tracking_id was overridden by nearest-neighbour'
    assert ad.used_real_ids is True


def test_missing_source_file_names_the_export_command():
    """A silent empty detection set would look like a perception failure."""
    from e2e_pipeline.closed_loop import LivePerceptionWorldModel
    assert 'sparse4d' in LivePerceptionWorldModel.SOURCES
    assert 'bevformer' in LivePerceptionWorldModel.SOURCES
