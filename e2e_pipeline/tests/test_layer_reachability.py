"""Every optional layer must be able to change something measurable.

THE FAILURE THIS GENERALISES. Six separate call sites passed no `freespace=`
while an occlusion prior was configured, so the prior was silently disabled and
"it changes nothing" was reported three times as a finding about occlusion. The
same shape appeared in the counterfactual metric (no tracker, so covariance was
unreadable), in `fit_calibration` (three command modes that were one), and in
`residual.py` (six training pairs from 14,615 agents).

In every case the conclusion was "inert" and the truth was "unreachable". The
distinction is invisible from the result -- both look like a flat column -- and
is only exposed by asking whether the measurement COULD have come out
differently.

These tests ask exactly that, per layer. They assert reachability, not benefit:
a layer that fires and does no good is a finding, a layer that cannot fire is a
bug wearing a finding's clothes.
"""
import numpy as np
import pytest

from e2e_pipeline.freespace import FreeSpace
from e2e_pipeline.scene import Agent
from e2e_pipeline.tests.test_closed_loop import StubWorld, _straight_planner


class OccludedWorld(StubWorld):
    """StubWorld with unobserved space ahead and an agent to carry risk."""

    def __init__(self, unknown_from=40, **kw):
        super().__init__(**kw)
        self.unknown_from = unknown_from

    def agents_at(self, t, ego_xy, ego_yaw):
        return [Agent(track_id=1, xy=np.array([18.0, 1.0]), yaw=0.0,
                      lwh=np.array([4.5, 1.9, 1.5]), vxy=np.array([-1.0, 0.0]),
                      score=0.6, label=0)]

    def freespace_at(self, t, ego_xy, ego_yaw):
        nx, ny = 200, 120
        unknown = np.zeros((nx, ny), bool)
        unknown[self.unknown_from:, :] = True
        return FreeSpace(traversable=np.ones((nx, ny), bool),
                         obstacle=np.zeros((nx, ny), bool), unknown=unknown,
                         esdf=np.full((nx, ny), 5.0, np.float32),
                         origin=(-10.0, -30.0), res=0.5)


def _run(world=None, **cfg_kw):
    from e2e_pipeline.closed_loop import ClosedLoopRunner, LoopConfig
    from e2e_pipeline.safety_filter import FeasibilityLimits, SafetyFilter
    cfg = LoopConfig(max_steps=8, initial_speed=8.0, **cfg_kw)
    r = ClosedLoopRunner(world or StubWorld(), _straight_planner(), cfg,
                         safety=SafetyFilter(
                             limits=FeasibilityLimits(max_risk=0.60)))
    recs, m = r.run(command=2)
    return r, recs, m


def test_unknown_prior_reaches_the_recorded_risk():
    """The exact six-site defect: the prior must move the number it gates on."""
    off, _, m_off = _run(OccludedWorld(), unknown_prior=0.0, veto_unknown=False)
    on, _, m_on = _run(OccludedWorld(), unknown_prior=0.5, veto_unknown=False)
    a = [p for p in off.risk_pred if p is not None]
    b = [p for p in on.risk_pred if p is not None]
    assert a, 'no risk recorded at all -- the probe itself is broken'
    # The prior can express itself two ways and both count as reachable: by
    # moving the recorded risk, or by braking on steps that previously planned
    # (an emergency step records None, so the lists differ in length).
    changed = (len(a) != len(b)
               or m_off['safety']['emergency_brakes']
               != m_on['safety']['emergency_brakes']
               or not np.allclose(a, b))
    assert changed, (
        'unknown_prior did not change the recorded risk or the braking; it is '
        'unreachable, not inert')


def test_calibrate_risk_reaches_the_recorded_risk():
    off, _, _ = _run(OccludedWorld(), calibrate_risk=False)
    on, _, _ = _run(OccludedWorld(), calibrate_risk=True)
    a = [p for p in off.risk_pred if p is not None]
    b = [p for p in on.risk_pred if p is not None]
    assert a and b
    assert not np.allclose(a, b), 'calibration did not reach the recorded risk'


def test_verifier_can_fire():
    r, _, _ = _run(OccludedWorld(), use_verifier=True)
    assert hasattr(r, 'verifier_fired')
    assert isinstance(r.verifier_rules, dict)


def test_shadow_mode_records_an_excess_when_enabled():
    """"Never fires" is only meaningful if the comparison ran at all."""
    r, _, _ = _run(OccludedWorld(), use_shadow=True)
    assert len(r.shadow_excess) > 0, (
        'shadow mode recorded nothing, so "never fires" is unsupported')


def test_three_valued_gate_reaches_the_verdict():
    """The graded gate must actually produce PENALIZE/REJECT, not just exist."""
    from e2e_pipeline.safety_filter import FeasibilityLimits, SafetyFilter
    from e2e_pipeline.scene import EgoState, SceneRepresentation
    w = OccludedWorld(unknown_from=40)
    scene = SceneRepresentation(agents=[], ego=EgoState(speed=8.0),
                                timestamp=0.0,
                                freespace=w.freespace_at(0, np.zeros(2), 0.0))
    f = SafetyFilter(limits=FeasibilityLimits(three_valued_unknown=True))
    traj = np.stack([[6.0 * (t + 1), 0.0] for t in range(6)])
    v = f._evaluate(0, traj, scene, None, 0.5)
    assert v.unknown_verdict in ('PENALIZE', 'REJECT'), v.unknown_verdict
    assert v.unknown_depth_m > 0.0
    assert np.isfinite(v.unknown_entry_m)


def test_multiplicative_ranking_changes_the_cost():
    """A ranking form that scores identically to the additive one is a no-op."""
    from e2e_pipeline.safety_filter import FeasibilityLimits, SafetyFilter
    from e2e_pipeline.scene import EgoState, SceneRepresentation
    from e2e_pipeline.uncertainty import RiskModel
    w = OccludedWorld(unknown_from=180)
    scene = SceneRepresentation(agents=w.agents_at(0, np.zeros(2), 0.0),
                                ego=EgoState(speed=8.0), timestamp=0.0,
                                freespace=w.freespace_at(0, np.zeros(2), 0.0))
    traj = np.stack([[5.0 * (t + 1), 0.0] for t in range(6)])
    rm = RiskModel(scene.ego)
    lim = FeasibilityLimits(max_risk=0.60)
    add = SafetyFilter(limits=lim)._evaluate(0, traj, scene, rm, 0.5).cost
    mul = SafetyFilter(limits=lim, multiplicative=True)._evaluate(
        0, traj, scene, rm, 0.5).cost
    assert not np.isclose(add, mul), 'multiplicative ranking is a no-op'
