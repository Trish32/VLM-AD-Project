"""Calibrate the critic's risk price by dual ascent, instead of choosing it.

W_RISK = 6.0 was a number I picked for scale compatibility. That is the wrong
kind of constant to have in a safety-relevant objective: it encodes a
risk/progress trade-off nobody stated, and anything fitted against it inherits
the preference silently.

The fix is to stop choosing a weight and start stating a BUDGET:

    maximise   progress
    subject to realised risk <= budget

and let the weight be the dual variable of that constraint, updated by

    lam <- max(0, lam + eta * (measured - budget))

At convergence lam is not a preference, it is a PRICE -- the exchange rate at
which the risk constraint is exactly tight. Report it as "the risk budget was
B and it cost lam units of progress per unit of risk", which is checkable, where
"W_RISK = 6.0" is not.

WHY W_PROGRESS STAYS 1.0. It is not a free parameter: fixing one coefficient
sets the units. Progress is the numeraire and every other term is priced in
metres of progress forgone. A scale-invariant objective has one redundant degree
of freedom, and spending it here makes the remaining weights interpretable.

MEASURED FIRST, AND IT CHANGED THE PROBLEM. Running this against the current
critic showed progress identical (62.5%) at lam = 6.0, 5.2, 4.4 and 3.6 -- the
weight does not affect behaviour at all. The reason is structural: all six
DiffusionDrive anchors share an arc length (45.96 m), so `progress`, `offroad`
and `clearance` are CONSTANT across candidates and risk is the only
discriminating term. argmax(-lam * risk) is independent of lam for any lam > 0.

So W_RISK cannot bias the ranking, and calibrating it would have produced a
confident number for a knob connected to nothing. The weights only become
identifiable once candidates differ on more than one term -- which needs a
scene-adaptive candidate generator (the DiffusionDrive denoiser), not a better
calibrator.

This module is therefore correct and currently INERT. It is kept because the
formulation is the right one for when candidates do differ, and because the
alternative -- quietly leaving a hand-set 6.0 in place and presenting it as
principled -- is worse than an honest no-op.

NO GRADIENTS THROUGH THE SIMULATOR. Dual ascent needs only a scalar measurement
per iteration, so the simulator, the safety filter and the collision test all
stay non-differentiable and the hard gate stays terminal. This deliberately does
not make the stack end-to-end trainable -- a policy trained to maximise a scalar
learns to avoid whatever that scalar punishes, and in this project caution is
what it punishes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RiskBudget:
    """The constraint that replaces a chosen weight."""

    max_mean_risk: float = 0.02      # mean per-step risk of executed plans
    max_ego_fault: int = 0           # ego-caused collision steps, hard


@dataclass
class CalibrationTrace:
    lam: list[float] = field(default_factory=list)
    measured: list[float] = field(default_factory=list)
    progress: list[float] = field(default_factory=list)

    def describe(self) -> str:
        rows = [f'  iter {i}: lam {l:7.3f}  risk {m:.4f}  progress {p:.1%}'
                for i, (l, m, p) in enumerate(zip(self.lam, self.measured,
                                                  self.progress))]
        return '\n'.join(rows)


def calibrate_risk_price(evaluate, budget: RiskBudget, lam0: float = 1.0,
                         eta: float = 40.0, iters: int = 8,
                         trace: CalibrationTrace | None = None) -> float:
    """Dual ascent on the risk constraint. `evaluate(lam) -> (risk, progress)`.

    `eta` is a step size on the dual, not a tuning knob on behaviour: it changes
    how fast lam converges, not what it converges to. A budget that cannot be
    met drives lam upward without bound, which is the correct signal that the
    constraint is infeasible rather than a number to quietly clip.
    """
    trace = trace if trace is not None else CalibrationTrace()
    lam = float(lam0)
    for _ in range(iters):
        risk, progress = evaluate(lam)
        trace.lam.append(lam)
        trace.measured.append(float(risk))
        trace.progress.append(float(progress))
        lam = max(0.0, lam + eta * (float(risk) - budget.max_mean_risk))
    return lam
