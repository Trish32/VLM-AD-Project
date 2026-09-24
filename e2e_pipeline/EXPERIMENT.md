# e2e_pipeline — experiment log

Every change here was measured before and after. Results that got worse are kept
with the same prominence as results that got better, and five causal hypotheses
that failed their own tests are recorded as retractions rather than edited out.

Unless stated, the setup is: **10 nuScenes-mini scenes × 20 steps = 200 closed-loop
steps**, DiffusionDrive anchors, ego seeded from the logged frame-0 speed.

---

## Summary: what actually moved the numbers

| # | change | headline effect |
|---|---|---|
| 1 | seed the ego from the log, not a constant | brakes 12→3, route 7.3%→67.8% on scene-0796 |
| 2 | guard degenerate routes | a parked car stopped scoring 94.3% completion |
| 3 | rectangle support function for collision geometry | risk 1.000→0.43 on scene-0061 |
| 4 | move the world model *before* the safety filter | removed a regression; then a net gain |
| 5 | reactive agents in the simulator | brakes 143→131, collisions 35→30 |
| 6 | split collisions by fault | revealed the metric was counting rear-ends |
| 7 | relax `max_risk` 0.05→0.60 | collisions 30→6, completion 29.5%→43.2% |
| 8 | calibrate the risk model (Platt) | held-out ECE 0.135→0.015 |
| 9 | risk as a soft signal, not a veto | brakes →85, best comfort; safety mixed |

---

## 1. Ego seeding

The rollout began at a constant 5 m/s regardless of scene. Logged frame-0 speeds
in nuScenes-mini span **0 to 15.3 m/s**, so the ego started at the wrong speed,
left the drivable corridor on step one, and the filter correctly rejected
everything.

| scene | brakes/24 | route completion | collisions |
|---|---|---|---|
| 0796 | 12 → **3** | 7.3% → **67.8%** | 2 → **0** |
| 0061 | 24 → **12** | 7.4% → **55.5%** | 0 → 0 |
| 0655 | 11 → 11 | 17.8% → **39.5%** | 2 → **0** |

Collisions went to zero on every scene where the ego actually drives. The
"conservative filter" reading of the old numbers was wrong — the filter was
reacting correctly to a badly initialised ego.

## 2. Degenerate route guard

scene-0553's logged "route" is **4 cm of GPS jitter** (a parked car). A `total > 0`
guard divided by it, so a run that emergency-braked all 24 steps scored **94.3%
completion**. Routes under `MIN_ROUTE_M = 5.0` now report `None`.

## 3. Collision geometry

A circumscribed disc over-reports broadside separation by **3.14 m** for a
4.6 × 1.8 m vehicle. Replacing it with the rectangle support function
`a|cos t| + b|sin t|`:

- scene-0061 risk **1.000 → 0.43**; 37/52 agents became exactly zero
- scene-0655 brakes **14 → 8**, completion **7.9% → 16.4%**

## 4. World model placement

Placed *after* the filter it only ever saw the **28.5%** of steps where something
feasible remained (57 rankable of 200).

| placement | brakes | collisions | completion | clearance |
|---|---|---|---|---|
| baseline (no world model) | 143 | 35 | 27.6% | 1.37 m |
| after the filter | 149 | 39 | 26.1% | 0.92 m |
| **before, constant-velocity** | 146 | 35 | 27.0% | 1.31 m |
| **before, reactive agents** | **144** | **35** | **27.4%** | **1.37 m** |

The regression was an artefact of ordering, not of the rollout. It prunes rather
than overrules — the hard gate still runs on survivors.

## 5. Reactive agents in the simulator

`GTWorldModel` re-read annotations every step, so agents followed the paths they
took around the ego that *actually* drove. A simulated ego that brakes gets
driven through.

| harness | brakes | collisions | completion | clearance |
|---|---|---|---|---|
| log replay | 143 | 35 | 27.6% | 1.37 m |
| **reactive (IDM)** | **131** | **30** | **29.5%** | **1.39 m** |

Better on every metric — that is the replay penalty quantified.

## 6. Collision fault attribution

The single most consequential change. Splitting contacts into ego-caused and
other-caused, across a caution sweep:

| planned distance travelled | total | **ego-fault** | other-fault | completion |
|---|---|---|---|---|
| 100% | 30 | **0** | 30 | 29.5% |
| 80% | 35 | **0** | 35 | 15.5% |
| 60% | 29 | **0** | 29 | 10.4% |
| 40% | 32 | **0** | 32 | 9.2% |

**Every collision is the ego being run into.** Instrumenting further: all 30
contacts occur with the ego **stationary**, and 11 are agents driving into the
*front* of a parked ego.

The failure mode was never "the planner drives into things". It was "the planner
stops, and then things hit it" — and the safety number had been measuring what
happens to a parked car in traffic.

## 7. The risk gate was the pathology

Ablating each filter gate over 200 steps:

| gate disabled | steps with ≥1 feasible candidate |
|---|---|
| — (baseline) | 69/200 (34%) |
| off-road | 34% (+0 pp) |
| clearance | 36% (+1 pp) |
| dynamics | 34% (+0 pp) |
| **risk** | **68% (+34 pp)** |

Sweeping the threshold, with safety split by fault:

| `max_risk` | brakes | ego-fault | other-fault | completion | clearance |
|---|---|---|---|---|---|
| **0.05** (original) | 131 | 0 | **30** | 29.5% | 1.39 m |
| 0.15 | 121 | 0 | 24 | 32.4% | 1.41 m |
| 0.30 | 110 | 0 | 19 | 33.5% | 1.43 m |
| **0.60** | **102** | 0 | **6** | **43.2%** | **1.52 m** |
| 1.01 (off) | 100 | **1** | 8 | 43.4% | 1.47 m |

The conservative gate was **causing** the collisions, not trading safety for
progress. Ego-fault stays 0 until the gate is fully disabled.

## 8. Re-baseline: every layer against an unjammed pipeline

All earlier layer measurements used `max_risk = 0.05`, i.e. a pipeline that
barely drove. Re-run at 0.60 with reactive agents and separated reporting:

| config | ego-fault | other | clearance | brakes | completion | jerk | interventions |
|---|---|---|---|---|---|---|---|
| baseline | 0 | 6 | 1.52 m | 102 | 43.2% | 1.57 | 0 |
| + verifier | 0 | 6 | 1.50 m | 98 | 43.0% | 1.57 | 6 |
| + TTC gate | **1** | **20** | 1.19 m | 98 | **29.3%** | 1.71 | 27 |
| + shadow mode | 0 | 6 | 1.52 m | 102 | 43.2% | 1.57 | 0 |
| **+ world model** | 0 | 6 | **1.66 m** | 101 | **43.5%** | **1.49** | 0 |
| + all | 1 | 21 | 1.22 m | 95 | 28.8% | 1.63 | 32 |

- **World model**: best clearance and comfort, slightly better completion. Its
  earlier "no improvement" verdict was made against the jammed baseline and is
  withdrawn.
- **TTC gate**: genuinely harmful, not an artefact. Re-baselining was meant to
  exonerate it and did the opposite.
- **Verifier**: near-neutral, the expected steady state for redundancy.
- **Shadow mode**: never fires at logged speeds (median longitudinal excess
  −7.9 m). Proven able to fire at 20 m/s (4 decelerate events).

## 9. Risk-model calibration

The collision probability is computed analytically and had never been checked
against outcomes. Reliability over 966 (prediction, outcome) pairs collected
across 10 scenes × 3 commands × 3 initial speeds, outcome = clearance < 1.0 m:

| predicted bin | n | mean predicted | **observed** |
|---|---|---|---|
| [0.000, 0.002) | 192 | 0.000 | 0.000 |
| [0.002, 0.027) | 192 | 0.010 | 0.016 |
| [0.027, 0.146) | 195 | 0.072 | 0.031 |
| [0.146, 0.332) | 192 | 0.220 | 0.047 |
| [0.332, 0.814) | 195 | **0.520** | **0.015** |

Over-confident by ~6×, and worst in the tail — the only region a safety filter
thresholds on. Fitting a two-parameter Platt map (`a = 0.457, b = −2.333`,
11 training positives, `identifiable = True`):

| held-out half | ECE | MCE |
|---|---|---|
| uncalibrated | 0.1352 | 0.5047 |
| **Platt** | **0.0152** | **0.0407** |

**Temperature alone is the wrong tool** and the fit shows it: the fitted
`T = 0.820` would *sharpen* an already over-confident model. The correction comes
from the bias term, which shifts a model whose events are rarer than it believes.

This explains §7: a 0.05 threshold on a 6×-inflated input behaves like 0.008.

## 10. Risk as a soft signal

The filter treated risk as a veto whose only fallback was a hard stop, so the
risk model controlled the parking brake directly. Restored separation: the model
scores, a graded response decides speed, the filter vetoes only at 0.85.

| config | ego-fault | other | clearance | brakes | completion | jerk |
|---|---|---|---|---|---|---|
| hard veto @0.05 (original) | 0 | 30 | 1.39 m | 131 | 29.5% | 1.20 |
| hard veto @0.60 (tuned) | 0 | **6** | **1.52 m** | 102 | **43.2%** | 1.57 |
| **soft + emergency-only @0.85** | 0 | 16 | 1.44 m | **85** | 35.4% | **1.18** |

Fewest emergency brakes and best comfort — the brake is closer to an exception
than an operating mode. But the response curve (`RISK_FREE`, `RISK_SATURATE`,
`MAX_REDUCTION`) is hand-set, and it loses to the tuned veto on collisions and
completion. **A hand-set threshold was replaced with a hand-set curve.**

Against the *original* (the fair comparison — both uncalibrated) the structural
change wins on every axis. Against a threshold swept on the same 10 scenes it is
evaluated on, it loses.

## 11. Response curve removed, not fitted

Fitting the curve the way the Platt map was fitted is not possible: Platt had
labels (observed collision frequency is a ground truth to regress onto), a
response curve has none -- there is no "correct" speed scale in the data. Grid
searching `RISK_FREE` / `RISK_SATURATE` / `MAX_REDUCTION` over the same ten
scenes they are evaluated on is how `max_risk = 0.60` was chosen, and that is
precisely why 0.60 is not trustworthy.

So the curve was removed. With risk calibrated, the response follows by
inverting it: bisect for the largest speed scale whose calibrated risk stays
within a budget epsilon. Three hand-set constants become one stated requirement.

| config | ego-fault | other | clearance | brakes | completion | jerk |
|---|---|---|---|---|---|---|
| no response (filter only) | 0 | 8 | 1.47 m | 99 | 43.5% | 1.60 |
| hand-set curve | 0 | 16 | 1.44 m | 85 | 35.4% | **1.18** |
| risk budget ε=0.02 | 0 | **34** | 1.06 m | 96 | **17.3%** | 1.30 |
| risk budget ε=0.05 | 0 | 18 | **1.57 m** | 90 | 28.4% | 1.81 |
| **risk budget ε=0.10** | 0 | **5** | 1.47 m | 90 | **43.8%** | 1.34 |
| risk budget ε=0.20 | 0 | 8 | 1.47 m | 99 | 43.5% | 1.54 |

**A tighter risk budget produces 7× more collisions.** ε=0.02 gives 34
other-fault collisions and 17.3% completion; ε=0.10 gives 5 and 43.8%. This is
the clearest statement of the pathology in the whole log: caution, applied to a
policy already too cautious, is not merely wasteful but actively unsafe, because
a slow ego in traffic gets struck.

The sweep is reported as a trade-off curve rather than a recommendation.
Declaring ε=0.10 "best" because it tops this table would be selecting a constant
on the same ten scenes it is scored on -- the `max_risk = 0.60` mistake again.
What ε *should* be is a policy statement about acceptable collision probability;
what this table shows is what each choice costs, and that the relationship is not
monotone in the direction intuition expects.

## 12. Scene-level calibration validation

The §9 split was random over *steps*, so all ten scenes appeared in both halves
and scene structure leaked. The honest test trains and tests on disjoint scenes,
run in both directions so neither half is cherry-picked:

| split | train positives | identifiable | fitted a | fitted b | ECE | MCE |
|---|---|---|---|---|---|---|
| train 0-4 / test 5-9 | 15 | yes | 0.624 | −2.203 | 0.1109 → **0.0182** | 0.4725 → **0.0765** |
| train 5-9 / test 0-4 | 6 | no | 0.560 | −2.185 | 0.1793 → **0.0338** | 0.4820 → **0.0772** |

**The correction generalises.** ECE improves 6.1× and 5.3× on scenes the fit
never saw, and the fitted bias is near-identical across the two directions
(−2.203 vs −2.185) despite one fit having 2.5× the positives of the other.

That stability has a cause worth stating: the bias term is essentially set by the
base rate, which 966 samples determine well even when positives are scarce. The
slope needs positives and is the shakier of the two (0.624 vs 0.560).

**The fragility is in the scene distribution, not the fit.** Positives per scene:

    scene    0   1   2   3   4   5   6   7   8   9
    pos      0   6   0   0   9   0   6   0   0   0

Only **3 of 10 scenes contain any positive event at all**. A split placing all
three in the test half would leave the training half with zero positives and no
fit possible. The validation passed, but it rests on three scenes, and that is a
property of nuScenes-mini rather than of the method.

---

## Retractions

Five causal explanations were committed and then refuted by their own
measurements:

1. **"mAP ≈ 0 means perception is broken."** It was a benchmark artefact: three
   of ten classes have zero GT instances in the split, and the devkit averages
   their vacuous AP = 0 into the mean.
2. **"Off-road 0.83 contradicts the filter passing those candidates."** The
   filter had not passed them — `feasible 0/6, emergency=True`. I asserted a
   contradiction without running the filter.
3. **"The 10 m synthetic corridor is the binding constraint."** Real map drivable
   area gave 2.6× the coverage and bought **one fewer emergency brake in 200**.
4. **"The 40% verifier firing rate traces to constant-velocity propagation."**
   Violations were uniform across the horizon, not clustered late. All 80 firings
   were a stationary ego.
5. **"Non-reactive agents are why no defensive layer can show benefit."** Under
   reactive agents the TTC gate still degraded.

The pattern in all five: a real defect was found, correctly identified as real,
and then over-credited with the observed symptom.

## Open

- The soft response curve is hand-set and should be fitted the way the Platt map was.
- The calibration split is random over *steps*; all 10 scenes appear in both
  halves. A scene-level split is the stricter test and is not yet run.
- The TTC gate should leave the default stack.
- The critic remains a weighted sum, not the constrained program
  (`max progress s.t. P(collision) ≤ ε`) it should be.
- 21 positives from 10 correlated scenes is weaker than 21 from 850 independent
  ones.

## Reproduce

```bash
PYTHONPATH=. python e2e_pipeline/risk_sweep.py        # §7
PYTHONPATH=. python e2e_pipeline/rebaseline.py        # §8
PYTHONPATH=. python e2e_pipeline/fit_calibration.py   # §9
PYTHONPATH=. python e2e_pipeline/soft_risk_ab.py      # §10
```
