"""Calibration for the risk model: ECE as the measure, Platt scaling as the fix.

`risk_calibration.py` showed RiskModel is over-confident -- it predicts 0.292
where 0.050 occurs and 0.595 where nothing occurs. That was a reliability table;
this reduces it to a single number and provides the correction.

WHY THIS AND NOT A RETRAINED MODEL. The analytic risk computation is sound
geometry (rectangle support function, non-central chi-squared over Kalman
covariance); what is wrong is its SCALE. A post-hoc map fixes scale without
touching the geometry, needs two parameters rather than a network, and leaves
the safety filter thresholding something that finally means what it says.

TEMPERATURE VS FULL PLATT. Temperature alone (a single scalar on the logit)
cannot move a systematically biased model onto the diagonal -- it rotates about
p = 0.5 and leaves the base rate wrong. The bias term is what corrects a model
whose events are rarer than it thinks, which is exactly this case. Both are
provided; `TemperatureCalibrator` is the constrained special case.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def expected_calibration_error(pred, label, n_bins: int = 10,
                               strategy: str = 'uniform') -> dict:
    """ECE, MCE and the per-bin table behind them.

    ECE is the count-weighted mean gap between predicted confidence and observed
    frequency. It is reported WITH the bin table because the scalar hides where
    the error lives: a model that is accurate on the 90% of samples near zero and
    wildly wrong on the rare high-risk tail has a small ECE and is dangerous, and
    the tail is the only part a safety filter ever thresholds on.

    `strategy='quantile'` puts equal counts per bin instead of equal width, which
    matters when predictions pile up near zero -- uniform bins then leave the
    upper bins nearly empty and their gaps dominated by noise.
    """
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(label, dtype=np.float64)
    n = len(p)
    if n == 0:
        return {'ece': float('nan'), 'mce': float('nan'), 'n': 0, 'bins': []}

    if strategy == 'quantile':
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    rows, ece, mce = [], 0.0, 0.0
    for lo, hi in zip(edges, edges[1:]):
        m = (p >= lo) & (p < hi) if hi < edges[-1] else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        conf, freq, w = float(p[m].mean()), float(y[m].mean()), int(m.sum())
        gap = abs(conf - freq)
        ece += (w / n) * gap
        mce = max(mce, gap)
        rows.append({'lo': float(lo), 'hi': float(hi), 'n': w,
                     'confidence': conf, 'frequency': freq, 'gap': gap})
    return {'ece': float(ece), 'mce': float(mce), 'n': n,
            'base_rate': float(y.mean()), 'bins': rows}


def format_reliability(report: dict) -> str:
    head = (f"ECE {report['ece']:.4f}   MCE {report['mce']:.4f}   "
            f"n {report['n']}   base rate {report.get('base_rate', float('nan')):.3f}")
    rows = [f"  [{b['lo']:.3f},{b['hi']:.3f})  n={b['n']:4d}  "
            f"pred {b['confidence']:.3f}  obs {b['frequency']:.3f}  "
            f"gap {b['gap']:+.3f}" for b in report['bins']]
    return '\n'.join([head] + rows)


@dataclass
class PlattCalibrator:
    """p' = sigmoid(a * logit(p) + b), fitted by Newton steps on the BCE.

    Two parameters, fitted by maximum likelihood -- the standard Platt map. `a`
    is the inverse temperature (sharpness), `b` shifts the base rate.
    """

    a: float = 1.0
    b: float = 0.0
    fitted: bool = False
    n_positive: int = 0

    def __call__(self, p):
        return _sigmoid(self.a * _logit(p) + self.b)

    def fit(self, pred, label, iters: int = 100, lr: float = 0.5
            ) -> "PlattCalibrator":
        x = _logit(pred)
        y = np.asarray(label, dtype=np.float64)
        self.n_positive = int(y.sum())
        a, b = 1.0, 0.0
        for _ in range(iters):
            q = _sigmoid(a * x + b)
            ga = float(((q - y) * x).mean())
            gb = float((q - y).mean())
            a -= lr * ga
            b -= lr * gb
        self.a, self.b, self.fitted = float(a), float(b), True
        return self

    @property
    def identifiable(self) -> bool:
        """Whether the fit rests on enough positives to mean anything.

        A Platt map fitted on a handful of positive events reproduces those
        events and generalises to nothing, while looking like a correction. Ten
        is a floor, not a recommendation -- it is the point below which the fit
        is obviously meaningless rather than the point above which it is
        trustworthy.
        """
        return self.fitted and self.n_positive >= 10


@dataclass
class TemperatureCalibrator(PlattCalibrator):
    """Platt with the bias pinned at 0 -- a single temperature T = 1/a."""

    def fit(self, pred, label, iters: int = 100, lr: float = 0.5
            ) -> "TemperatureCalibrator":
        x = _logit(pred)
        y = np.asarray(label, dtype=np.float64)
        self.n_positive = int(y.sum())
        a = 1.0
        for _ in range(iters):
            q = _sigmoid(a * x)
            a -= lr * float(((q - y) * x).mean())
        self.a, self.b, self.fitted = float(a), 0.0, True
        return self

    @property
    def temperature(self) -> float:
        return 1.0 / self.a if self.a else float('inf')
