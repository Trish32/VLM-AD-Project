# e2e_pipeline — experiment log

Every change here was measured before and after. Results that got worse are kept
with the same prominence as results that got better, and five causal hypotheses
that failed their own tests are recorded as retractions rather than edited out.

Unless stated, the setup is: **10 nuScenes-mini scenes × 20 steps = 200 closed-loop
steps**, DiffusionDrive anchors, ego seeded from the logged frame-0 speed.

### Defaults changed, and when

Anything measured before the listed commit used the old value. These are the
only behavioural defaults that have moved; everything else added by this log
ships off by default.

| default | old | new | commit | why |
|---|---|---|---|---|
| `SafetyFilter.w_risk` | 10.0 | **1.0** | §29 | 10.0 sat in the saturated region of the ranking; 1.0 is better on six metrics, worse on jerk |
| `TrackCovarianceTracker.calibrated_noise` | — | **on for detector worlds** | §26 | fitted to 11,730 detections; GT worlds keep their declared noise |
| `LiveDetectionAdapter.associate` | — | **True** | §26 | the fallback track id hashed the frame token, so 0.0% of tracks survived a step |
| `diffusiondrive_anchor_planner` command | hardcoded `'straight'` | **honours the argument** | §27 | it discarded the command it was passed; caused every collision in the project |

> ## ⚠ Read §19 before §§6–18
>
> **Every collision count taken under a simulated ego measures deviation from the
> recording, not driving quality.** §19 pins the ego to the logged trajectory and
> collisions fall to **zero** — under ground truth *and* under live perception.
> The recorded agents drove around a car that followed the logged path; put the
> ego back on it and nothing hits it, however it perceives.
>
> This affects the headline number of §§6–18 and §20, including three
> conclusions I argued from it: `max_risk` 0.05→0.60 "reducing collisions 30→6"
> (§7), the TTC gate "costing 12 collisions" (§8, §15), and "a tighter risk
> budget produces 7× more collisions" (§11).
>
> **The rankings mostly survive; the stated reasons often do not.** `max_risk =
> 0.60` really is better than 0.05 — not because it reduces collisions, but
> because it brakes less, so the ego stays near the corridor the scene was
> recorded around. §§23–24 found two *correct* fixes that this metric had
> dismissed.
>
> **Not affected:** anything measured with the ego pinned (§19, §23, §24 pinned
> rows), all calibration numbers (ECE/MCE compare a prediction to a per-step
> outcome and never integrate a trajectory — §22 shows they transfer across
> perception sources), §16's prediction RMSE, and §§1–3.
>
> §25 replaces the metric. As of `evaluate()` today, every rollout reports
> divergence-bucketed collision rates, recovery rate and progress-per-drift
> *above* the unconditioned count.

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

## 13. Residual dynamics — learned, and it does not generalise

`z_{t+1} = f_physics(z, a) + g_theta(z, a)`, with the physics prior kept and a
small MLP learning only what IDM gets wrong about agent motion. Supervised
directly from logs: the target is *what the agent actually did* minus what the
prior predicted, so no counterfactuals are needed.

Scene-level split, both directions:

| split | train | **held-out scenes** |
|---|---|---|
| train 0–4 / test 5–9 | +5.7% | **−1.4%** |
| train 5–9 / test 0–4 | +6.5% | **−9.5%** |

It fits the training scenes and makes prediction **worse** on unseen ones, in
both directions. The physics prior alone is the better predictor out of sample.

The cause is visible in the baseline itself: physics RMSE is **0.128 m/s** on
scenes 0–4 and **0.279 m/s** on 5–9, a 2.2× difference. The scenes are
heterogeneous enough that one group's residual structure does not describe the
other's, so a model fitted on either learns something locally true and globally
wrong.

**Why calibration generalised and this did not.** The Platt map corrects a
global scalar bias — essentially the base rate — which is one number and stable
across scenes. The residual is a 6→2 function that must capture scene-specific
agent behaviour. The simpler correction transferred; the richer one overfitted
on the same data. That is an argument about data volume, not about the method:
10 scenes support estimating a base rate and do not support learning a dynamics
correction.

Kept, disabled, and honest about it: `ResidualDynamics.residual()` returns
exactly zero when untrained, so the object degrades to the physics prior rather
than to a random initialisation.

## 14. Calibration wired into the decision path

§9 and §12 fitted and validated the Platt map, and then nothing used it -- the
risk model still emitted raw, 6×-inflated probabilities and the filter still
thresholded them. Integrated at three points: inside `RiskModel` (so no call
site can compare against an uncalibrated number), as the input to the
risk-budget speed solve, and as in-loop ECE monitoring on the runner.

| config | ego-fault | other | clearance | brakes | completion | jerk | in-loop ECE |
|---|---|---|---|---|---|---|---|
| raw @0.60 (previous best) | 0 | 6 | 1.52 m | 102 | 43.2% | 1.57 | 0.153 |
| raw @0.05 (original) | 0 | 30 | 1.39 m | 131 | 29.5% | 1.20 | 0.012 |
| calibrated @0.05 | 0 | 23 | 1.50 m | 119 | 33.8% | 1.31 | 0.023 |
| **calibrated @0.10** | 0 | **6** | **1.60 m** | 104 | **43.6%** | **1.47** | 0.034 |
| calibrated @0.20 | 0 | 8 | 1.55 m | 103 | 43.8% | 1.59 | 0.043 |
| calibrated + budget ε=0.10 | **1** | 7 | 1.55 m | 104 | 43.8% | 1.50 | 0.068 |

**Calibrated @0.10 matches or beats the previous best on every axis** — same
collisions, better clearance (1.60 vs 1.52 m), better completion, better
comfort — and the threshold is now a real collision probability rather than a
number in model units.

**The 6× factor appears exactly where predicted.** Calibrated 0.10 behaves like
raw 0.60. That is the over-confidence measured in §9, now visible as an
operating-point equivalence rather than an inferred one.

**Calibration helps even without re-tuning.** At the original 0.05 threshold,
simply calibrating the input takes collisions 30 → 23 and completion 29.5% →
33.8%. The threshold was never the whole story; the input's scale was.

### A caveat about the ECE column

Raw @0.05 shows the *lowest* in-loop ECE (0.012) and is the *worst* configuration
by every safety and progress measure. That is not a contradiction — at that
threshold the ego barely moves, predictions collapse toward zero, and nothing
happens, so "predict ≈0, observe ≈0" is trivially well-calibrated. ECE measures
agreement between prediction and outcome, not usefulness, and on a degenerate
distribution it rewards a model that has stopped saying anything. Read it
alongside the operating point, never alone.

## 15. TTC gate: logic error found, fixed, then retired anyway

Diagnosed in two rounds, because the second problem only became visible once the
first was fixed.

### Round 1 — the logic was wrong, not the threshold

Every firing recorded with its TTC, ego speed, agent speed and whether a
collision followed:

| scene | ttc | ego v | agent v | "closing" | realised min clearance | collided |
|---|---|---|---|---|---|---|
| 5 | 0.9 s | 15.3 | 14.0 | **25.3** | 4.16 m | no |
| 5 | 1.4 s | 15.3 | 14.2 | **28.2** | 4.48 m | no |
| 7 | 1.3 s | 3.3 | 18.3 | 15.0 | 2.58 m | no |

**0 of 36 firings preceded a collision.** The closing speeds give it away: 25.3
m/s is the *sum* of 15.3 and 14.0, not the difference. These are oncoming
vehicles in the opposite lane, approaching along the sight line and passing
comfortably 4 m apart.

`time_to_collision` used range ÷ range-rate, which ignores whether the paths
intersect at all. Replaced with **closest point of approach**:

    t* = -(p · v_rel) / |v_rel|²        miss = |p + v_rel · t*|

and a conflict requires `miss` under a vehicle width. Discrimination check:

| case | closing | old | **CPA** |
|---|---|---|---|
| oncoming, opposite lane | +29.0 | 0.8 s | **inf** |
| oncoming, our lane | +29.3 | 0.8 s | **0.8 s** |
| parked car beside us | +15.0 | 1.3 s | **inf** |
| stopped car in our lane | +15.3 | 1.3 s | **1.3 s** |

Near-identical closing speeds, opposite verdicts — only the miss distance
differs. Firings fell **36 → 3**, each a genuinely stationary obstacle in the
ego's own path.

### Round 2 — the response is still wrong, and redundant

| config | ego-fault | other | clearance | brakes | completion | firings |
|---|---|---|---|---|---|---|
| baseline | 0 | 6 | 1.60 m | 104 | 43.6% | 0 |
| + TTC gate (CPA) | 0 | **18** | 1.51 m | 109 | **36.7%** | 4 |

**4 firings cost 12 additional collisions and 6.9 points of completion.** Each
one brakes an already-admissible plan, and braking is what gets a vehicle
rear-ended — the mechanism §6 and §7 established.

So the detection was repairable and the response is not. It is also redundant:
the safety filter's risk gate already covers these conflicts, and covers them by
**filtering candidates** rather than decelerating a chosen one. Two mechanisms
for one responsibility, and the better-placed one already exists.

Retired rather than deleted — `time_to_collision` is now correct and feeds the
structured representation, and `enabled=True` reproduces both measurements.

### Why filtering candidates beats decelerating a chosen one

Both mechanisms respond to the same input — a collision probability — so the
question is only *where in the pipeline* the response belongs. Three reasons it
belongs before selection, not after.

**1. The action space is K-way before, and 1-D after.** The risk gate sees six
candidates and removes the unacceptable ones; whatever the planner then prefers
among the survivors is still a plan it endorsed. A post-selection gate has
exactly one lever — slow the chosen plan down — because switching to a different
trajectory *is* re-running selection. Six options collapse to one option with a
scalar knob.

**2. Shape and speed profile are coupled, and decelerating breaks that.**
DiffusionDrive's anchors are not paths with a free speed parameter; each is a
manoeuvre whose geometry assumes a particular speed. Suppose candidate A is fast
and straight but risky, and candidate B is slower with a lane shift and safe.
Filtering returns B — a coherent manoeuvre. The post-hoc gate can only produce
"A, slower", which is a lane-holding geometry executed at a speed it was not
chosen for. Neither the planner nor the filter ever evaluated that object. §13's
`decelerate_along` exists precisely because the naive version of this — swapping
in a straight-line stop — could brake a plan *into* the obstacle it was steering
around.

**3. Slowing has a side effect that filtering does not.** This is the empirical
part, and it is the whole story of §6 and §7: in traffic, a decelerating ego
gets rear-ended. Every collision in this project occurs with the ego
**stationary**, and a tighter risk budget produces **7× more** of them (§11).
Filtering avoids risk by choosing differently; decelerating avoids risk by
becoming an obstacle. One of those has a failure mode and the other does not.

The general form: **a gate that can only subtract speed should sit where it can
still subtract options instead.** Once a plan has been selected, the only
remaining authority is braking — which is why the emergency brake is the right
thing to keep at the end of the pipeline, and a graded response is not.

## 16. Making the residual generalise: regularisation, gating, and rollout error

§13 found the learned residual overfits across scenes. Three remedies tried in
order, each measured on a scene-level split **and over a rollout**, because
one-step error is not what a world model is used for — it compounds.

### L2, shrinkage, dropout, gating

Trained on scenes 5–9, tested on 0–4 (physics baseline: 1-step 0.128 m/s,
3 s rollout 0.141 m):

| regulariser | 1-step | Δ | **rollout** | **Δ** |
|---|---|---|---|---|
| none (raw fit) | 0.1457 | −13.5% | 0.191 | **−35.7%** |
| L2 only | 0.1284 | −0.0% | 0.144 | −2.0% |
| output shrinkage only | 0.1392 | −8.4% | 0.185 | **−30.8%** |
| dropout 0.2 | 0.1290 | −0.5% | 0.145 | −3.0% |
| dropout 0.5 | 0.1291 | −0.6% | 0.145 | −3.2% |
| L2 + shrink | 0.1284 | −0.0% | 0.143 | −1.4% |
| L2 + shrink + dropout | 0.1284 | +0.0% | 0.143 | −1.0% |
| L2 + shrink + gate | 0.1285 | −0.1% | 0.144 | −2.1% |

**The rollout column is the one that matters, and it is far worse.** A 13.5%
one-step degradation compounds to **35.7%** over 3 s. Measuring only per-step
error would have understated the damage by 2.6×.

**No arm is positive.** L2 is the most effective regulariser and dropout close
behind, but both work by driving the correction to zero — the model becomes the
prior. Output shrinkage alone is insufficient (−30.8% rollout). Gating adds
nothing once the output is already ≈0, because there is nothing left to gate.

The pattern is exhaustive rather than suggestive: **every setting strong enough
to prevent harm is strong enough to erase the effect.** There is no middle
ground on ten scenes.

### Online system identification instead

If the error differs *by scene* rather than by feature — physics RMSE is
0.128 m/s on scenes 0–4 and 0.279 on 5–9 — then stop fitting a function and
estimate a parameter that tracks the current scene. `OnlineIDMGain` does RLS on
one number, IDM's deceleration gain, evaluated causally (gain from earlier steps
only):

| scene | n | gain | physics → adapted | Δ |
|---|---|---|---|---|
| 0 | 4203 | 0.988 | 0.0870 → 0.0870 | −0.0% |
| 3 | 1939 | 0.810 | 0.0848 → 0.0845 | +0.4% |
| 4 | 564 | **0.383** | 0.1680 → 0.1666 | +0.9% |
| 5 | 494 | **0.307** | 0.5071 → 0.5042 | +0.6% |
| 7 | 591 | 1.000 | 0.8626 → 0.8626 | +0.0% |
| **mean** | | | 0.2540 → 0.2535 | **+0.2%** |

It identifies real per-scene structure — gains of 0.31 and 0.38 mean IDM
over-brakes by 3× in those scenes — and converts it into **+0.2%**. The
parameter is right and the leverage is not.

### Verdict

Learned residuals are abandoned. Both `ResidualDynamics` and `OnlineIDMGain` are
kept, disabled, and honest: `residual()` returns exactly zero untrained, and
`gain = 1` recovers the unmodified prior. The physics prior is the better
predictor and nothing measured here beats it.

What would change the answer is data, not method: the offline fit needs scenes
that share error structure, and the online estimator needs a scene where IDM is
wrong enough for a 3× gain correction to matter.

## 17. GT oracle replaced with live detector output

Every closed-loop number in this log was produced against `GTWorldModel` —
perfect boxes, perfect velocities, perfect recall. None of them said anything
about the pipeline under perception it would actually have.
`LivePerceptionWorldModel` substitutes real BEVFormer output keyed by sample
token (404 samples, all 10 scenes) and changes nothing else, so the difference
is attributable to perception alone.

| world model | ego-fault | other | clearance | brakes | completion | jerk | agents/step | divergence |
|---|---|---|---|---|---|---|---|---|
| GT oracle | 0 | **7** | **1.58 m** | **105** | **43.5%** | 1.47 | 42.2 | — |
| live @ score 0.25 | 0 | 26 | 1.12 m | 125 | 34.7% | 1.29 | 39.5 | 7.7 m |
| live @ score 0.40 | 0 | 23 | 1.16 m | 131 | 35.3% | 1.14 | 16.6 | 7.9 m |

**Perception costs 3.7× the collisions** (7 → 26), 8.8 points of completion, and
0.46 m of clearance. **(Superseded by §19: at zero divergence the collision cost
is zero, and the real cost is 64% more emergency braking.)** Ego-fault stays at 0 — real perception does not make the
planner drive into things; it makes it stop more (105 → 125 brakes) and get
struck more, which is the mechanism §6 established, now driven by detector error
rather than a bad threshold.

Raising the score threshold to 0.40 more than halves the agent count (39.5 →
16.6) and barely moves the outcome: collisions 26 → 23, completion 34.7% →
35.3%, brakes 125 → 131. The threshold trades false positives for misses at
roughly a wash, which says the damage is not dominated by spurious detections.

### The confound, stated

Mean divergence between the simulated and logged ego is **7.7 m**. The frame
transform is correct — detections are world-frame boxes rotated into the
*simulated* ego frame — so this is not a coordinate error. It is a
**field-of-view mismatch**: the detector only saw what was visible from the
logged pose, so an object near the simulated ego but occluded or out of range
from the logged one is simply absent.

That is unfixable without running perception on sensor data from a pose the car
never occupied, which nuScenes cannot provide. So the 3.7× is an upper bound on
the cost of *detector error* and includes an unquantified share of viewpoint
error. It is the honest number available, not a clean one.

### Scope

Only the object branch is substituted. Free space still comes from the logged
corridor rather than FlashOcc, deliberately: replacing both at once would leave
the difference unattributable between them, and only the detector has saved
output covering all ten scenes.

## 18. Calibration ablation re-run under live perception

§14 measured the calibration against the GT oracle, and the Platt map was
*fitted* on GT-based rollouts. Live detections have a different risk
distribution, so the map could easily have been out of distribution. Re-run:

| config | ego-fault | other | clearance | brakes | completion | ECE |
|---|---|---|---|---|---|---|
| GT raw @0.60 | 0 | 7 | 1.60 m | 106 | 43.4% | 0.162 |
| GT calibrated @0.10 | 0 | 7 | 1.58 m | 105 | 43.5% | **0.038** |
| LIVE raw @0.60 | 0 | 26 | 1.12 m | 125 | 34.7% | 0.183 |
| LIVE raw @0.05 | 0 | 35 | 0.83 m | 163 | 20.6% | 0.013 |
| LIVE calibrated @0.10 | 0 | 26 | 1.12 m | 125 | 34.7% | **0.042** |
| LIVE calibrated @0.05 | 0 | 35 | **1.37 m** | 140 | 26.7% | 0.022 |
| **LIVE calibrated @0.20** | 0 | **25** | 1.25 m | **124** | **35.9%** | 0.047 |

**The calibration transfers.** In-loop ECE falls 0.183 → 0.042 under live
detections, a 4.4× improvement almost identical to the 4.3× seen under GT. A map
fitted on oracle rollouts corrects a detector's over-confidence too, which says
the over-confidence is a property of the analytic risk computation rather than
of the perception feeding it.

**The 6× equivalence holds exactly.** LIVE calibrated @0.10 and LIVE raw @0.60
produce *identical* results on every column — 26 collisions, 1.12 m, 125 brakes,
34.7%. The same threshold correspondence measured under GT in §14.

**It helps most where the threshold is tight.** At 0.05, calibrating the input
alone takes clearance 0.83 → **1.37 m** (+65%), completion 20.6% → 26.7% and
brakes 163 → 140, at unchanged collisions. That is the §14 finding reproduced
under harder conditions: the threshold was never the whole story, the input's
scale was.

### What survives and what does not

§14's *relative* conclusions survive contact with real perception. The *absolute*
numbers do not: 7 collisions become 26, completion 43.5% → 34.7%. Every figure
in §§7–16 is an oracle figure and should be read as optimistic.

The ECE caveat from §14 repeats more starkly here: LIVE raw @0.05 has the best
ECE in the table (0.013) and is the worst configuration in it — 35 collisions,
0.83 m clearance, 163 brakes, 20.6% completion. A pipeline braking itself into
paralysis predicts near-zero risk and is correct, which is exactly why ECE must
never be read alone.

## 19. Detector replayed at the logged pose — the 3.7× was mostly confound

§17 measured live perception costing 3.7× the collisions, and flagged a 7.7 m
divergence between the simulated and logged ego as an unquantified confound.
`LoopConfig.follow_logged_ego` pins the ego to the recorded trajectory, driving
divergence to zero so the detector sees the scene from the viewpoint its
detections were actually computed at. Any GT-vs-live gap in that mode is
**detector error alone**.

| config | ego-fault | other | clearance | brakes | agents/step | divergence |
|---|---|---|---|---|---|---|
| GT, simulated ego | 0 | 7 | 1.58 m | 105 | 42.2 | — |
| LIVE, simulated ego | 0 | **26** | 1.12 m | 125 | 39.5 | 7.68 m |
| **GT, logged pose** | 0 | **0** | **1.94 m** | **44** | 42.4 | — |
| **LIVE, logged pose** | 0 | **0** | 1.73 m | 72 | 40.0 | **0.00 m** |

### Every collision in this project came from ego divergence

Pinned to the logged trajectory, collisions fall to **zero under both GT and
live perception** — 7 → 0 and 26 → 0. Not reduced: eliminated.

The recorded agents drove around a car that followed the logged path. Put the
ego back on that path and nothing hits it, regardless of how it perceives. So
the collision counts throughout §§6–18 were measuring **deviation from the
recording**, not unsafe driving and not detector quality.

### The detector's real cost is braking, not collisions

Isolated at zero divergence, live perception versus GT costs:

| | GT | live | Δ |
|---|---|---|---|
| collisions | 0 | 0 | — |
| clearance | 1.94 m | 1.73 m | −0.21 m |
| emergency brakes | 44 | 72 | **+64%** |

Detector error makes the pipeline **brake 64% more often** and hold 0.21 m less
clearance. It does not make it crash. §17's headline — "perception costs 3.7×
the collisions" — was dominated by viewpoint mismatch and is **withdrawn**.

### And the simulated ego is expensive even with perfect perception

GT simulated (105 brakes) against GT logged-pose (44) is a 2.4× difference with
*identical* perception. Simulating the ego drifts it into states the planner
handles badly, and that cost exceeds the detector's.

### What the pinned mode is not

It is not a closed loop for the ego: the planner's output no longer affects
where the car goes, so route completion is trivially the logged route and is
omitted from the table. Safety and clearance stay meaningful because they are
evaluated against the agents the pipeline actually perceived. The mode isolates
perception; it cannot evaluate planning.

## 20. Ego ODD: where the collisions are, and where the 7.7 m comes from

### All 7 collisions are one scene, one failure mode

| scene | step | along-track | cross-track | ego_v | logged_v |
|---|---|---|---|---|---|
| 6 | 13 | −11.7 m | 0.9 m | **0.0** | 4.2 |
| 6 | 14 | −13.8 m | −1.0 m | **0.0** | 4.0 |
| 6 | 15 | −15.4 m | −3.7 m | **0.0** | 3.9 |
| 6 | 16 | −16.3 m | −6.8 m | **0.0** | 3.9 |
| 6 | 17 | −16.1 m | −10.7 m | **0.0** | 4.1 |
| 6 | 18 | −14.9 m | −14.7 m | **0.0** | 4.4 |
| 6 | 19 | −13.6 m | −17.8 m | **0.0** | 4.6 |

Every collision is **scene 6, steps 13–19, ego speed 0.0**, sitting 11–16 m
behind where the recording has it while the logged car continues at ~4 m/s. Nine
scenes contribute nothing.

So the headline safety number is not a distributed property of the planner. It
is **one sustained stop in one scene**, counted once per step for seven
consecutive steps.

### The 7.7 m divergence is longitudinal, and it is self-inflicted

| | mean | \|mean\| | p95 |
|---|---|---|---|
| along-track | −2.54 m | **5.96 m** | 19.9 m |
| cross-track | −1.26 m | 1.65 m | 5.7 m |

**78% of the divergence is along-track.** It is not a path-following error and
not a controller problem — the ego is on the right line, in the wrong place
along it.

Growth over the rollout says why:

| step | \|along\| | \|cross\| | ego_v | logged_v | braking |
|---|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 5.7 | 5.7 | 30% |
| 2 | 0.19 | 0.16 | 5.7 | 5.3 | 30% |
| 5 | 1.43 | 0.50 | 5.5 | 5.6 | 40% |
| 10 | 5.75 | 1.72 | 4.7 | 5.6 | 60% |
| 15 | 10.66 | 2.74 | 2.9 | 4.9 | 60% |
| 19 | **15.27** | 4.06 | **2.1** | 5.0 | **80%** |

The ego starts matched (5.7 vs 5.7 m/s) and brakes itself to a standstill while
the recording holds ~5 m/s. Emergency braking climbs **30% → 80%**, speed falls
5.7 → 2.1, and the gap compounds to 15 m.

### The whole causal chain, finally

    over-braking → falls behind the log → recorded agents drive into it
         → "collisions" → read as a safety problem → more caution added
         → more braking

Every symptom this document chased traces to the first arrow. §19 proved the
last one is an artefact (pin the ego, collisions go to zero); this shows the
first one is the actual defect.

### Priority

1. **Reduce the emergency-brake rate.** It causes the divergence, which causes
   the collisions. Everything else is downstream.
2. Scene 6 specifically — it is the only scene producing collisions, and a
   single-scene failure is far cheaper to diagnose than a distributed one.
3. **Not** lateral tracking, **not** the controller, **not** the detector. Cross-track
   divergence is 1.65 m mean and contributes 22%.

## 21. The brake rate is driven by divergence, not by scene content

§20 showed emergency braking climbing 30% → 80% across a rollout. A rate that
*grows* is not simple over-caution, which would be roughly constant — it
suggests a feedback loop. `follow_logged_ego` pins divergence to zero by
construction, so any surviving step-dependence must be scene content.

### Pinning divergence removes the climb

| step | simulated | **pinned (div = 0)** |
|---|---|---|
| 0 | 30% | 40% |
| 2 | 30% | 30% |
| 5 | 40% | 20% |
| 8 | 40% | 20% |
| 11 | 60% | 10% |
| 14 | 80% | 50% |
| 17 | 80% | 10% |
| 19 | **80%** | **20%** |
| **all steps** | **52%** | **22%** |

With the ego on the logged trajectory the rate is flat and noisy (10–50%, no
trend) and **less than half** the simulated rate. The climb is entirely an
artefact of the ego drifting.

### Divergence predicts braking, controlling for time

Both divergence and step index grow together, so the table above cannot separate
them alone. Splitting late steps (10–19) at the median divergence does:

| steps 10–19 | n | brake rate |
|---|---|---|
| below median divergence (5.0 m) | 50 | 60% |
| **above median** | 50 | **80%** |

At the same point in the rollout, more accumulated divergence means 20 points
more braking. Binned across all steps:

| divergence | n | brake rate |
|---|---|---|
| [0.0, 0.5) | 58 | 57% |
| [0.5, 2.0) | 25 | 20% |
| [2.0, 5.0) | 59 | 44% |
| [5.0, 10.0) | 22 | 68% |
| [10.0, ∞) | 36 | **72%** |

Monotone above 0.5 m. The first bin breaks the pattern because it is dominated
by step 0, where divergence is zero by construction and the ego is often already
braking from its initial state — a confound of the binning, not a
counterexample.

### The feedback loop, confirmed

    brake → fall behind the log → corridor and agent geometry stop matching
      where the ego is → fewer candidates survive → brake

This is the first hypothesis in this document to survive its own test. It also
explains why threshold tuning kept producing partial fixes: every one of them
weakened the loop without breaking it, so the rate fell but the shape stayed.

**What this means for the fix.** The lever is not a threshold. It is either
keeping the ego near the corridor the scene was recorded around, or making the
scene representation robust to an ego that has drifted — free space and agent
geometry that stay valid off the logged path. The second is the real
requirement; the first is what the pinned mode does artificially.

## 22. Re-validating the three retired components against live perception

`calibration/calibrate.py` (inert), the TTC gate (retired), and `planner/residual.py` (abandoned)
were all judged under the GT oracle. §19 showed the oracle was hiding real
costs, so each verdict deserved re-testing against live detections.

### TTC gate — now harmless, and still useless

| config (live perception) | ego-fault | other | clearance | brakes | **firings** |
|---|---|---|---|---|---|
| simulated, baseline | 0 | 26 | 1.12 m | 125 | — |
| simulated, + TTC gate | 0 | 26 | 1.12 m | 125 | **0** |
| pinned, baseline | 0 | 0 | 1.73 m | 72 | — |
| pinned, + TTC gate | 0 | 0 | 1.73 m | 72 | **0** |

Under GT the CPA gate fired 4 times and cost 12 collisions. Under live
detections it fires **zero** times, in both ego modes, so the loop is identical
to baseline on every column.

The reason is noise: detector position and velocity error inflates the
closest-approach miss distance, and CPA requires a miss under a vehicle width.
Real perception rarely produces an agent whose predicted path passes that close
with a TTC under 1.5 s.

That is not a rehabilitation. A gate that never fires cannot help, and the one
configuration where it *did* fire it did harm. Verdict unchanged: retired.

### calibrate.py — still inert, and structurally so

`W_RISK` swept over a 120× range (0.5 → 60) under live detections: argmax is
candidate 3 at every value. The weight cannot affect the ranking.

The reason is unchanged by perception, because it was never about perception:
all six anchors share an arc length, so progress, off-road and clearance are
constant across candidates and risk is the only discriminating term — and
`argmax(−λ·risk)` is independent of λ. Swapping the detector changes the risk
*values*, not the fact that they are the sole discriminator.

### residual.py — cannot be meaningfully re-fitted under live perception

Not measured, because the measurement would be meaningless, and that is worth
stating rather than silently skipping.

The residual's target is *actual agent motion minus the IDM prediction*. Under
GT, "actual" is the nuScenes annotation — a real observation. Under live
perception there is no such ground truth: the only available "actual" is the
detector's own output at the next step. Fitting to that trains the model to
predict **detector noise**, not physics error, and a good fit would mean the
residual had learned to reproduce the detector's mistakes.

Refitting would produce a number; it would not produce knowledge. Abandoned
verdict stands, for a different and stronger reason than §16's.

### What the round-trip established

All three verdicts survive live perception, reached independently of the oracle
that §19 discredited. That is worth more than the individual results: it means
§§14–16's conclusions about these components were not artefacts of GT, even
though §§17–21 showed the oracle was distorting the absolute numbers around
them.

## 23. The map work re-measured — §12's null was a metric artefact

§12 replaced the 10 m synthetic corridor with nuScenes drivable-area polygons,
got 2.6× the coverage, and measured "one fewer emergency brake in 200". That
conclusion predates both the fault split (§6) and the divergence loop (§21), and
§21 predicted a specific signature the original measurement was not looking for:
a **flatter brake-rate curve**, not a lower average.

### Simulated ego — the signature appears

| config | ego | other | clearance | brakes | divergence | brake rate @ step 0/5/10/15/19 | slope |
|---|---|---|---|---|---|---|---|
| 10 m corridor | 0 | 7 | 1.58 m | 105 | 6.6 m | 30/40/60/60/80% | +2.5 pp/step |
| **real map** | 0 | 10 | 1.34 m | **86** | 7.3 m | 30/40/50/40/50% | **+0.8 pp/step** |

The slope flattens **3×** and braking falls 18%. Collisions rise 7 → 10 and
clearance drops — but §19 established those are divergence artefacts under a
simulated ego, and divergence is indeed slightly higher (6.6 → 7.3 m) precisely
*because* the ego brakes less and therefore travels further from the recording.

### Pinned ego — divergence controlled, and it is unambiguous

| config (divergence = 0) | ego | other | clearance | brakes |
|---|---|---|---|---|
| 10 m corridor | 0 | 0 | 1.94 m | 44 |
| **real map** | 0 | 0 | **1.94 m** | **27** |

**−39% emergency braking at identical safety** — same zero collisions, same
clearance to two decimals. With the confound removed there is no trade-off at
all; the map is simply better.

### Why §12 missed it

Three reasons, all of which apply to other sections of this log:

1. It measured the **average** brake rate, where the effect is in the *slope*.
   §21's loop is a compounding process, so the right signature is curvature.
2. Its safety comparison used the undifferentiated collision count, which §6
   showed was counting rear-ends and §19 showed was counting divergence.
3. It ran only in simulated-ego mode, where the benefit (driving more) inflates
   the artefact (divergence) and the two partly cancel.

**§12's conclusion — "the harness is not the binding constraint" — is
withdrawn.** The map was the right fix and the measurement was wrong, which is
the inverse of the usual failure in this document.

## 24. Free-space error isolated: FlashOcc in place of the synthetic corridor

`GTWorldModel` synthesises a band around the logged route — a stand-in that
knows the road because it knows where the car went. `FlashOccWorldModel` replays
real FlashOcc occupancy (404 frames cached, 0.19 s/frame) through the same
`FreeSpaceExtractor`, with **objects left as ground truth** so the delta belongs
to the dense branch alone.

| config | ego-fault | other | clearance | brakes | brake @ 0/5/10/15/19 | slope |
|---|---|---|---|---|---|---|
| **simulated** corridor | 0 | 7 | 1.58 m | 105 | 30/40/60/60/80% | +2.5 |
| **simulated** FlashOcc | **1** | **15** | **0.78 m** | 87 | 40/20/50/60/60% | +1.7 |
| **pinned** corridor | 0 | 0 | 1.94 m | 44 | 40/20/20/30/20% | −0.6 |
| **pinned** FlashOcc | 0 | 0 | **1.94 m** | **35** | 40/10/10/20/30% | −0.3 |

### Real occupancy is better where it can be read, worse where it cannot

**Pinned (divergence = 0):** brakes 44 → **35** (−20%) at identical safety —
same zero collisions, same 1.94 m clearance. Real occupancy beats the synthetic
corridor outright, the same direction and similar magnitude as the real map in
§23 (−39%).

**Simulated:** clearance collapses 1.58 → 0.78 m, collisions 7 → 15, and the
**first ego-fault collision in this entire document** appears. Braking still
falls (105 → 87) and the slope still flattens (+2.5 → +1.7).

### Why the simulated case degrades, and why it is not FlashOcc's fault

A voxel grid cannot be transformed to a new viewpoint. Detector boxes can —
they are world-frame objects, rotated into whatever ego frame you like, which
is why §17's live detections stayed usable at 7.7 m divergence. Occupancy is
inferred *in a frame*, and reading it from a pose 7.7 m away means reading the
wrong cells, with no way to resample what was never observed.

So the simulated row measures **occupancy misalignment**, not occupancy quality,
and the pinned row measures quality. They disagree because they measure
different things.

That ego-fault collision is worth noting precisely: it is the pipeline driving
into something, and it appears only when free space is both real *and*
misaligned. A corridor that is wrong-but-smooth degrades gracefully; a voxel
grid that is right-but-shifted puts drivable surface where an obstacle is.

### A near-miss worth recording

The first attempt reused `GTWorldModel`'s grid — x[−20,60] y[−30,30] →
(200,150,16) — against FlashOcc's native (200,200,16) and **raised a shape
error**. That was the good outcome. Had the shapes matched by coincidence, every
voxel would have been offset 10 m in y with no error at all, and the ablation
would have produced a confident number for a silently misaligned map.

### Scope

Objects remain GT throughout, as specified. Whether free-space and object errors
interact is a separate question and deliberately unmeasured here — combining
them would leave any difference unattributable between the two branches, which
is the confound §17 introduced and §19 spent two sections removing.

## 25. Replacing the collision metric with divergence-aware measures

§19 showed the collision count measures deviation from the recording, and it
rewards parking — the safest policy by that number is to stop, which is also
what got the ego struck. Four replacements:

### (1) Collision rate within matched divergence buckets

| divergence | GT n | GT rate | LIVE n | LIVE rate |
|---|---|---|---|---|
| 0–1 m | 71 | **0.0%** | 71 | **2.8%** |
| 1–3 m | 32 | 0.0% | 28 | 0.0% |
| 3–7 m | 53 | **0.0%** | 38 | **5.3%** |
| 7+ m | 44 | **15.9%** | 63 | **34.9%** |

This decomposes the perception cost that §17's headline could not. Live
perception is worse **two ways, independently**: higher collision rate *within*
every populated bucket, and more time spent *in* the worst bucket (63 vs 44
steps above 7 m). The old whole-rollout number multiplied these together and
attributed the product to the detector.

Collisions are near-absent below 7 m of divergence under GT. That is the §19
finding made quantitative.

### (2) Recovery rate — the failure is a stationary point

| | excursions past 3 m | recovery rate | longest unrecovered |
|---|---|---|---|
| GT | 9 | **0%** | 16 steps |
| LIVE | 8 | **0%** | 16 steps |

**Not one excursion recovers.** Once the ego falls 3 m behind it never returns
below 1.5 m, in any scene, under either perception source. The longest trap runs
16 of 20 steps.

This is the single most important number in this document. Divergence is not
marginal degradation that averaging can summarise — it is an **absorbing
state**. Every metric that reports a mean over a rollout is averaging across a
boundary the ego crosses once and never recrosses.

### (3) Counterfactual safety — per-decision, no rollout

Fixing the world at each logged frame and comparing the planner's chosen
trajectory against the logged ego's own future through the same agents:

| | n | mean Δrisk | worse | much worse (>0.10) | better |
|---|---|---|---|---|---|
| GT | 200 | **−0.0004** | 14% | 2% | 26% |
| LIVE | 200 | **+0.0041** | 24% | 10% | 26% |

Under GT the planner is, on average, **very slightly safer than the human** and
is better more often than worse (26% vs 14%). Under live perception it tips
negative: 24% worse, and much-worse cases rise 2% → 10%.

This is the only metric here that evaluates the **planner** rather than the
simulation, because both trajectories are scored against identical agents from
an identical pose. It says the planning stage is sound and the perception stage
costs roughly 1 decision in 10 becoming materially riskier.

### (4) Progress per unit divergence

| | progress | divergence | ratio | stalled | speed vs logged |
|---|---|---|---|---|---|
| GT | 416 m | 169 m | **2.5** | 33% | 81% |
| LIVE | 369 m | 188 m | **2.0** | 44% | 72% |

A stationary policy scores zero here rather than perfect. Live perception buys
20% less progress per metre of drift, stalls 44% of steps against 33%, and runs
at 72% of the logged speed against 81%.

### Two measurement bugs caught before reporting

The first run gave **Δrisk = 0.0000 exactly** for both arms, and 0%/0%/0%.
`StepRecord` had no `planned_traj` field, so the code fell back to comparing the
logged trajectory against itself. A null result that is *identically* zero is
almost always a plumbing failure rather than a finding.

The second: progress-per-divergence summed progress over all scenes and divided
by the **last scene's** final divergence, giving LIVE a better ratio than GT
(369 vs 124) by accident. Fixed to accumulate numerator and denominator per
scene.

---

## 26. The covariance re-measured, and four places it never reached

§25's covariance work fitted `sigma = 0.609 + 0.466 / score` to 11,730 matched
detections and stopped there. Six follow-ups were run against it. Two produced
the expected answer; four found that the quantity being calibrated was not
reaching the thing it was supposed to inform, which made several earlier
conclusions measurements of plumbing rather than of perception.

### Velocity is not predicted by score, and the first cut assumed it was

The same 11,730 matches, now against `box_velocity`:

| score bin | n | measured position RMS | measured **velocity** RMS |
|---|---|---|---|
| 0.25–0.40 | 5860 | 2.036 m | 2.219 m/s |
| 0.40–0.55 | 1866 | 1.742 m | 2.133 m/s |
| 0.55–0.70 | 891 | 1.450 m | 2.867 m/s |
| 0.70–0.85 | 1128 | 1.269 m | 3.018 m/s |
| 0.85–1.01 | 1985 | 0.987 m | 1.945 m/s |

The least-squares fit is `sigma_v = 2.463 − 0.075 / score` — the slope is
**negative** and two orders below the floor, i.e. flat. Position error halves
across the score range; velocity error does not move. Unsurprising once stated:
the score is a classification logit and velocity comes from a separate
regression branch.

The first calibrated cut carried the position curve into velocity anyway, via
`sigma_p * vel_noise / pos_noise`. That was unmeasured, and it also coupled
velocity to a position scale the calibrated branch no longer uses — for a world
declaring `pos_noise=0.1` it inflated `sigma_v` more than twentyfold.

### And position is not the term that matters

Propagated variance at a 3 s horizon, averaged over agents:

| term | LIVE | share | GT | share |
|---|---|---|---|---|
| position `P_xx` | 3.44 | **5.5%** | 1.16 | **1.9%** |
| velocity `t² P_vv` | 54.60 | **86.9%** | 54.60 | **90.6%** |
| process `q t³/3` | 4.78 | 7.6% | 4.52 | 7.5% |

Under the CV model the velocity block is multiplied by `t²` and position by 1,
so at 3 s velocity dominates by more than an order of magnitude. **The position
curve — the entire subject of §25's calibration — moves 5.5% of the number.**
That is the single most useful result here: the careful part was the part that
could not matter.

### The metric reporting the tail never read the covariance

`divergence_report.py` built its counterfactual `RiskModel` with **no tracker**,
so `constant_velocity_prediction` fell through to a hardcoded `(0.5 + 0.5 t)²`
spread. The much-worse tail in §25 was therefore a property of that constant.
Re-running it unchanged would have produced identical numbers and read as a
clean null.

Wired, with a fresh tracker per step (`cov = R(score)`, the quantity calibrated):

| covariance arm | mean Δrisk | worse | **much worse** | better |
|---|---|---|---|---|
| reciprocal (old) | +0.0069 | 56% | **3%** | 44% |
| full (measured) | +0.0027 | 56% | **0%** | 44% |
| slope only (no floor) | +0.0030 | 56% | 0% | 44% |
| floor only (no slope) | +0.0031 | 56% | 0% | 44% |
| **σv = 0.2 (as shipped)** | +0.0098 | 55% | **4%** | 44% |
| **σv = 1.0 (assumed)** | +0.0057 | 56% | **2%** | 44% |
| **σv = 2.46 (measured)** | +0.0027 | 56% | **0%** | 44% |

The floor does **not** explain the tail, and neither does the slope — the three
position arms agree to ±0.0004. The velocity arms reproduce it exactly:
4% → 2% → 0% as `sigma_v` goes 0.2 → 1.0 → 2.46. The tail was the filter being
over-confident about how fast things were moving.

### Live detections were never tracked at all

The adapter's fallback track id hashed `(token, rounded xy)`. The token in that
tuple made every id unique to its frame:

| | ids carried to the next frame |
|---|---|
| GT annotations | **97.1%** |
| live detections | **0.0%** |

So the Kalman filter never ran a second update on any agent under live
perception — every covariance stayed at its seed forever. The comment beside
that hash warned against exactly the behaviour it caused. Greedy
nearest-neighbour association (3 m gate, class-constrained, CV-predicted) takes
it to **65%**.

`LivePerceptionWorldModel` also inherited `GTWorldModel.measurement_noise()`, so
real detections were declared at the oracle's 0.1 m. Both arms were wrong in
opposite directions once `calibrated_noise` defaulted on; the world now declares
`detector_grade` and the runner reads it.

### Where the 22% orphan rate lives

| range | n | orphan | of those, class-confused | **true FP** | mean score | position err |
|---|---|---|---|---|---|---|
| 0–10 m | 1446 | 6% | 57% | **3%** | 0.69 | 1.06 m |
| 10–20 m | 3590 | 9% | 34% | **6%** | 0.60 | 1.38 m |
| 20–30 m | 4013 | 17% | 24% | 13% | 0.49 | 1.75 m |
| 30–40 m | 3153 | 30% | 13% | 26% | 0.40 | 2.11 m |
| 40+ m | 2787 | 44% | 11% | **40%** | 0.34 | 2.23 m |

"22% false positives" is a far-field average. Inside 10 m — where a planner acts
— the true false-positive rate is **3%**, and most of what remains is a real
obstacle under the wrong class label rather than a hallucination. The headline
number and the planning-relevant number differ by 13×.

### Risk gate: 0.10 is now the 90th percentile of realised risk

Pinned to the logged ego, so divergence is not in the measurement:

| max_risk | EGO-fault | brakes | clearance | completion | risk p50 | risk p90 |
|---|---|---|---|---|---|---|
| 0.05 | 0 | 147 | 1.73 m | 52.5% | 0.0352 | 0.0459 |
| **0.10** | 0 | **75** | 1.73 m | 52.5% | 0.0561 | 0.0862 |
| 0.20 | 0 | **40** | 1.73 m | 52.5% | 0.0662 | 0.1164 |
| 0.40 | 0 | 40 | 1.73 m | 52.5% | 0.0662 | 0.1164 |

At zero divergence the threshold changes **nothing except how often it brakes** —
same collisions (zero), same clearance, same completion. 0.10 costs 35 extra
interventions per 200 steps against 0.20 with no measured benefit.

The covariance fix roughly doubled the risk scale (p50 0.031 → 0.063), so a gate
that used to sit well above the bulk now sits at about the p90. Free-running,
the §7 direction reproduces: looser is better on everything — clearance
0.64 → 1.07 m, completion 17.5% → 36.6%, other-fault collisions 36 → 24.

**Recommendation: 0.20.** Caveat stated plainly — zero collisions in 200 pinned
steps is weak evidence of *no* safety benefit, not proof of it.

### residual.py: the obstruction was association, not noise

| arm | agents seen | paired across a step | target RMS | rollout Δ |
|---|---|---|---|---|
| GT (control) | 16,215 | 15,713 (**97%**) | 0.279 m/s | −0.1% |
| LIVE, no association | 14,615 | **6 (0%)** | — | — |
| LIVE, associated | 14,631 | 9,419 (**64%**) | 0.878 m/s | −1.5% |

§22's "cannot be meaningfully re-fitted under live perception" was measuring
**six training pairs**. With association the data exists, and the fit still does
nothing — but now for a measured reason rather than an empty input.

The naive noise floor (`√2 × 2.463 = 3.48 m/s`) is 4× larger than the observed
target RMS, so it is wrong for this subset: the 3 m association gate keeps slow,
well-localised agents and discards the fast ones carrying the large velocity
errors. Deriving the noise from the two arms instead — `noise² = live² − gt²` —
gives **0.833 m/s against a 0.279 m/s signal, SNR 0.33**. The GT control settles
it: even with the noise removed entirely the residual is not learnable from
these six features (+0.0% one-step, −0.1% rollout).

### calibrate.py: still inert, and the stated reason was wrong

§22 explained the inertness by claiming all six anchors share an arc length, so
"`progress`, `offroad` and `clearance` are CONSTANT across candidates and risk is
the only discriminating term". Measured spread across candidates, per decision:

| critic term | mean spread | max spread | % decisions with spread > 0 |
|---|---|---|---|
| progress | 0.0054 | 0.0074 | 100% |
| **clearance** | **0.6744** | **2.8000** | 77% |
| offroad | 0.0200 | 0.3333 | 10% |
| risk (as shipped) | 0.2623 | 1.5437 | 100% |
| risk (critic wired) | 0.0492 | 0.3348 | 100% |

Clearance is **not** constant — it is the most discriminating term in the
objective, out-spreading risk 14× once the critic is wired. The real reason the
price is unidentifiable is that **progress, the numeraire the dual trades
against, has 125× less spread than clearance**. Dual ascent confirms it:
progress is 52.1% at every λ from 6.0 to 9.8, and λ rises monotonically because
the 0.05 budget is infeasible at a realised 0.069 — which is the correct signal,
not a number to clip.

Wiring the critic's own `RiskModel` (it had **no tracker and no calibrator**,
a third instance of the same gap) *reduces* risk spread 5.3×, because the Platt
map compresses. It makes risk even less able to discriminate.

### Occlusion prior: still exactly inert, and now provably geometric

| prior | brakes | clearance | mean risk | map unknown | **waypoints in unknown** |
|---|---|---|---|---|---|
| 0.00 | 87 | 0.78 m | 0.0376 | 84.0% | **0.0%** |
| 0.02 | 87 | 0.78 m | 0.0376 | 84.0% | **0.0%** |
| 0.05 | 87 | 0.78 m | 0.0376 | 84.0% | **0.0%** |
| 0.10 | 87 | 0.78 m | 0.0376 | 84.0% | **0.0%** |

**84% of the map is unknown and 0.0% of planned waypoints enter it.** A stress
arm forcing detector-grade covariance onto the GT world (mean risk 0.0376 →
0.0581, EGO-fault 1 → 7) is identical across priors too, which rules out the
prior being swamped rather than idle. The obstruction is geometric and no
covariance change can touch it — the occlusion is lateral, the six fixed anchors
run straight down the observed corridor.

### What this round established

Four components were correct and disconnected: the covariance never reached the
counterfactual metric, the critic, or the live tracker, and the live tracker had
no tracks to filter. Three earlier conclusions rested on those gaps. The
recurring shape is the one already noted about the Platt calibration — the most
carefully measured component is the one nothing consumes — and the fix each time
was wiring rather than modelling.

The one substantive modelling result is inverted from the expected direction:
position covariance, measured carefully in §25, governs 5.5% of the propagated
variance, while velocity — assumed, never measured, and wrong by 12× — governs
87%.

---

## 27. The drive command was hardcoded, and it caused every collision

`diffusiondrive_anchor_planner` accepted a `command` argument and built
`DrivingIntent(command='straight')` regardless. Invisible for the usual reason:
every caller passed `2`, and `2` IS straight, so the hardcode agreed with the
argument by coincidence.

It matters because the anchors are clustered PER COMMAND upstream
(`kmeans_plan.py` buckets on `gt_ego_fut_cmd` before k-means), so command 2's six
anchors describe only trajectories that went straight — **0.5 m of lateral
endpoint spread against 19.9 m for the full vocabulary.** The pipeline was
choosing among six near-identical straight lines.

### How often that was wrong

Deriving the command from the logged ego future by upstream's own rule (±2 m
lateral offset at the final waypoint, `nuscenes_converter.py:386`):

| | steps | share |
|---|---|---|
| straight | 159 | 79.5% |
| left | 21 | 10.5% |
| right | 20 | 10.0% |

**20.5% of steps were commanded wrongly.** Concentrated, not spread: scene 6 is
16/20 right-turn, scene 0 is 10/20 left.

### Scene 6 is the scene with all the collisions

| arm | EGO | other | brakes | clearance | completion | divergence |
|---|---|---|---|---|---|---|
| hardcoded straight, scene 6 | 0 | **7** | 12 | 0.00 m | 18.2% | 8.0 m |
| derived per step, scene 6 | 0 | **0** | **1** | **1.05 m** | **44.6%** | **2.1 m** |
| hardcoded straight, all 10 | 0 | **7** | 105 | 1.58 m | 43.5% | 6.6 m |
| derived per step, all 10 | 0 | **0** | **71** | **2.03 m** | **49.9%** | **4.4 m** |

§20 established that all 7 collisions were scene 6, steps 13–19, ego speed 0.0.
Scene 6 is 80% right-turn and was planned as straight throughout. Giving it the
right command takes collisions to **zero**, brakes 12 → 1, clearance 0.00 →
1.05 m and completion 18.2% → 44.6%.

Across all ten scenes: collisions 7 → 0, brakes −32%, clearance +28%,
divergence −33%.

### This re-roots §§19–21 rather than contradicting them

The mechanism in §20 was right — over-braking → falls behind → recorded agents
drive into the stopped ego. What was missing is why it braked. Commanded
straight through a right turn, every candidate ran off the drivable corridor
(measured min-clearance 0.00 m), the filter rejected all six, and the emergency
brake fired until the car stopped. Divergence was the mechanism; the command was
the cause.

So "every collision came from ego divergence" (§19) stands, and "the divergence
is self-inflicted over-braking" (§20) stands, but neither is the root. The
absorbing-state result (§25, 0% recovery) was measured on rollouts that were
being steered off-route by construction and should be re-run.

### What this is not

`command_at` reads the LOGGED future, so it is an oracle. A real stack takes the
command from a navigation layer. The honest claim is not "the planner is good
now" — it is that **the planner was being given the wrong instruction on a fifth
of all steps**, and that the previous default was also an oracle and
additionally a wrong one. `run(command=None)` opts in; an explicit int keeps the
old behaviour so no committed result moves silently.

Also caught: `fit_calibration.py` sweeps `cmd in (0, 1, 2)` to widen the Platt
calibration set. All three produced identical rollouts, so the fit saw a third
of the variation it was written to sample, each configuration triplicated. The
calibration should be refitted.

---

## 28. Auditing every "inert" conclusion for reachability

Six call sites passed no `freespace=` while an occlusion prior was configured.
Each silently disabled the prior, and the resulting flat column was reported
three separate times as a finding about occlusion. That is enough repetitions of
one mistake to require going back over every null result in this document and
asking a different question of it:

> Not "did it change anything", but **"could it have?"**

A flat column has two causes that look identical from the outside — the
mechanism is inert, or the measurement cannot see it — and only the second is a
bug. The audit:

| claim | could the measurement have moved? | verdict |
|---|---|---|
| occlusion prior inert (×3) | No — six independent blockers | **unreachable** |
| `traversable &= ~unknown` (§25) | No — `mask_camera` never supplied, 0.0% unknown | **unreachable** |
| counterfactual much-worse tail (§25) | No — risk model built with no tracker | **unreachable** |
| W_RISK unidentifiable (§22) | No — swept 6.0–9.5, reorders above ~11 | **unreachable** |
| critic "nothing to rank" (§22) | No — command hardcoded, 0.5 m candidate spread | **unreachable** |
| residual unfittable under live (§22) | No — 6 training pairs from 14,615 agents | **unreachable** |
| §12 map work null | No — metric artefact; signature appears pinned (§23) | **unreachable** |
| TTC gate harmful (§8) | Yes — 27 interventions fired and metrics degraded | genuine |
| verifier near-neutral (§8) | Yes — 6 interventions in 200 | genuine |
| shadow never fires at logged speed (§8) | Yes — proven able to fire at 20 m/s | genuine |
| residual unlearnable *after* the fix (§26) | Yes — 9,419 pairs, and a GT control | genuine |

**Seven of eleven were unreachable measurements, not inert mechanisms.** The
four that survive share a property the seven lack: each was accompanied by a
non-zero firing count, which is direct evidence the path executed. Every one of
the seven reported a mechanism's effect without ever reporting that the
mechanism ran.

### Making the failure impossible rather than rare

`freespace=None` meant both "this caller has no raster" and "I forgot". Those
need different handling and the signature could not tell them apart, which is
why six omissions were silent. `RiskModel.evaluate` now takes a `_REQUIRED`
sentinel: omitting `freespace` while `unknown_prior > 0` raises, while
`freespace=None` remains the explicit "no raster here" and still works.

`FeasibilityLimits.__post_init__` rejects configurations that cannot do what
they claim — `three_valued_unknown` together with `allow_unknown` (two answers
to one question), and a three-valued gate with `unknown_penalty=0` (which is
`allow_unknown` with extra steps).

`tests/test_layer_reachability.py` asserts, per layer, that toggling it changes
something recorded. It deliberately tests reachability rather than benefit: a
layer that fires and does no good is a finding; a layer that cannot fire is a
bug wearing a finding's clothes.

## 29. Calibration compresses discrimination, and additive ranking cannot absorb it

The Platt map fitted in §9 is `sigmoid(0.457 · raw − 2.333)`. Its **entire
output range** is:

| raw | 0.00 | 0.05 | 0.10 | 0.30 | 0.60 | 1.00 |
|---|---|---|---|---|---|---|
| calibrated | 0.0884 | 0.0903 | 0.0922 | 0.1001 | 0.1132 | **0.1328** |

Every possible risk lands in **[0.0884, 0.1328]** — a band 4.4 points wide. Three
consequences, all of which had already been observed separately without the
common cause being identified:

1. **`max_risk = 0.10` is raw risk 0.297.** A threshold stated in calibrated
   units is 3× looser than it reads. This is why §26 found the gate saturating
   at 0.20 — above 0.133 nothing can ever be rejected.
2. **The occlusion prior was 3× too small for its own gate.** A prior of 0.10
   raw shifts calibrated risk by 0.0038. §27's sweep found the crossover between
   0.30 and 0.60, matching the 0.297 prediction.
3. **Calibrating the critic destroyed its risk discrimination.** Measured:
   wiring the calibrator cut risk spread across candidates **0.516 → 0.081**,
   6.4×. Doing the correct thing to the *numbers* made them powerless over the
   *ranking*.

The third is the structural one. In a weighted sum the risk term can only move
the total by `w_risk × 0.044`, while clearance spreads 0.893 across candidates —
so risk is outvoted by an order of magnitude no matter how well calibrated it is.

### W_RISK is identifiable after all — in [0, 2], not [6, 25]

Two predictions made from the above, both refuted by their own measurement, and
recorded because the second refutation is what produced the answer.

**Prediction 1: λ reorders around 11.** Swept to 25 — `w_risk` 6, 9, 11, 13, 15,
25 gave *identical* results on every metric. Refuted.

**Prediction 2: the ranking is rarely exercised, so no weight can matter.** The
feasible set has ≥2 candidates on **44.5%** of decisions and all six on 25%.
Refuted.

Measuring the cost terms directly across *feasible* candidates instead of
guessing a third time:

| term | mean spread | median | non-zero on |
|---|---|---|---|
| `risk.total` | **0.0073** | 0.0036 | 98% |
| clearance, raw | 1.2309 | 1.1266 | 88% |
| clearance, **saturated at 2 m** | 0.2383 | **0.0000** | **34%** |
| `planner_score` | 0.0047 | 0.0002 | 100% |

Two corrections fall out. The 0.081 figure quoted above is the *critic's*
`expected_collisions`, a different quantity from the filter's calibrated
`risk.total`, whose spread is 0.0073 — 11× smaller. And clearance **saturates**:
`min(clearance, 2.0)` is tied across all candidates on 66% of decisions, so on
those the ranking reduces to risk against the planner prior.

That fixes the crossover at `w_risk × 0.0073 = 0.0095`, i.e. **w_risk ≈ 1.3** —
and the entire swept range 6–25 was above it, in the region where risk already
dominates and scaling it further cannot change an argmin. Sweeping *downward*:

| w_risk | EGO | other | brakes | clearance | completion | jerk | mean risk |
|---|---|---|---|---|---|---|---|
| 0.0 | 0 | 27 | 97 | 1.18 m | 40.0% | — | 0.0517 |
| **0.5** | **0** | **20** | **94** | **1.83 m** | 39.9% | — | 0.0513 |
| **1.0** | **0** | **20** | **94** | **1.83 m** | **39.8%** | 1.32 | **0.0507** |
| 2.0 | 1 | 21 | 105 | 1.43 m | 38.3% | — | 0.0545 |
| 4.0 | 1 | 21 | 105 | 1.38 m | 38.3% | — | 0.0550 |
| 10.0 *(was shipped)* | 1 | 21 | 105 | 1.37 m | 38.3% | **1.24** | 0.0549 |

The transition lands between 1.0 and 2.0, as the arithmetic predicted. **The
shipped `w_risk = 10.0` was an order of magnitude above the identifiable
region.**

**This is not a safety-for-progress trade, and describing it as one was an
error.** `w_risk = 1.0` is better than 10.0 on ego-fault (0 vs 1), other-fault
(20 vs 21), braking (94 vs 105), clearance (1.83 vs 1.37 m), completion
(39.8% vs 38.3%) **and** mean risk (0.0507 vs 0.0549). The first write-up of
this table called the 1.5 pp completion difference a cost; it is a gain, the
sign was inverted.

The one metric that does get worse is jerk, **1.24 → 1.32 (+6.5%)** — a comfort
cost, not a safety one, and it was missing from the sweep's output until the
dominance claim was checked rather than asserted. Six metrics better, one worse.

**Default changed to `w_risk = 1.0`.** Every closed-loop result committed before
0f5f147 was produced at 10.0.

So "W_RISK is inert" was the fourth unreachable-measurement result in this
document, and the most expensive: three separate explanations were offered for
it (constant terms, then λ too small, then ranking unexercised) before anyone
measured the term spreads that settle it.

### Multiplicative ranking, measured

| form | EGO | other | brakes | clearance | completion | jerk | mean risk |
|---|---|---|---|---|---|---|---|
| additive | 1 | 21 | 105 | 1.37 m | 38.3% | 1.24 | 0.0549 |
| multiplicative | 1 | 21 | 105 | **1.43 m** | 38.3% | 1.29 | 0.0545 |

Marginally better clearance, marginally worse jerk, everything else identical —
a wash at `w_risk = 10`. That is consistent with the diagnosis rather than
against it: at a weight that already saturates the ordering, changing the
*form* of the combination cannot reorder either. The multiplicative form's value
is that it has no weight to mis-set, which is worth more than the 0.06 m.

### Multiplicative combination

    score = clearance · (1 − calibrated_risk)

Risk becomes a **fraction of a plan's value** rather than a fixed subtraction
from it, which is scale-free: a narrow calibrated range still expresses itself
proportionally, because the multiplier acts on a term that is not narrow. It
also has the right limit behaviour — risk → 1 zeroes the plan however much
clearance it has, whereas an additive score lets a roomy trajectory buy its way
past danger.

Available as `SafetyFilter(multiplicative=True)`, default off.

---

## 30. End-to-end re-baseline under the current defaults

Four defaults moved and each was measured alone, which is right for attribution
and wrong for a headline: the combination had never been run. Canonical config,
ten scenes × 20 steps, derived commands, `w_risk = 1.0`.

| config | EGO | other | brakes | clearance | completion | jerk | divergence | risk | lat p50 | lat p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| GT, pinned | **0** | **0** | 41 | 1.94 m | 52.5% | 1.13 | 0.0 m | 0.0236 | 23.7 ms | 63.2 ms |
| GT, free | **0** | **0** | 78 | **2.05 m** | 49.2% | **1.00** | 4.5 m | 0.0229 | 23.4 ms | 67.9 ms |
| LIVE, pinned | **0** | **0** | 70 | 1.73 m | 52.5% | 1.27 | 0.0 m | 0.0540 | 27.9 ms | 69.0 ms |
| LIVE, free | **0** | 20 | 94 | 1.83 m | 39.8% | 1.32 | 5.2 m | 0.0507 | 27.0 ms | 69.1 ms |
| *GT, free, cmd=straight* | 0 | 7 | 106 | 1.60 m | 43.7% | 1.36 | 6.6 m | 0.0287 | 23.5 ms | 68.0 ms |
| *LIVE, free, cmd=straight* | 0 | 25 | 125 | 1.24 m | 34.2% | 1.42 | 7.6 m | 0.0551 | 25.9 ms | 69.3 ms |

Against the old default, holding everything else at current values:

| | GT free | LIVE free |
|---|---|---|
| other-fault collisions | **7 → 0** | 25 → 20 (−20%) |
| emergency brakes | 106 → 78 (−26%) | 125 → 94 (−25%) |
| clearance | 1.60 → 2.05 m (+28%) | 1.24 → 1.83 m (+48%) |
| route completion | 43.7% → 49.2% (+5.5 pp) | 34.2% → 39.8% (+5.6 pp) |
| jerk | 1.36 → 1.00 (−26%) | 1.42 → 1.32 (−7%) |
| divergence | 6.6 → 4.5 m (−32%) | 7.6 → 5.2 m (−32%) |

**Zero ego-fault collisions in every configuration, and zero collisions of any
kind under ground truth.** Under the pinned ego — the mode that removes the
deviation confound §19 identified — both perception sources are clean.

Latency is 23–28 ms p50 and 63–69 ms p95 for the full stack, so the planning
loop runs comfortably inside its 2 Hz budget with ~15× headroom at p50.

### The absorbing state survives

| | excursions past 3 m | recovery rate | longest unrecovered |
|---|---|---|---|
| GT | 8 | **0%** | 16 steps |
| LIVE | 8 | **0%** | 16 steps |

This is the one §25 result the command fix does **not** rescue, and it was
plausible that it would — divergence was being driven by planning straight
through turns, so a correct command might have made excursions recoverable. It
did not. Excursions are 33% rarer and 32% shorter, and **still not one of them
returns below 1.5 m**. Divergence remains a one-way boundary; what changed is
how often the ego crosses it, not what happens after.

That is the honest limit of this stack: it now drives without hitting anything
under ground truth, and it still cannot recover once it falls behind.

---

## 31. Real track ids, finally supplied — and they bought nothing

The architecture names **Sparse4D v3** as the object branch because it propagates
identity through its temporal instance bank, and `TrackCovarianceTracker` is
built on that: it deliberately does no association of its own. The live arm has
always replayed **BEVFormer-tiny** instead (NDS 0.2255, mAP 0.2334 on mini-val,
against the official tiny's 0.252), because those were the only saved per-sample
detections covering all ten scenes.

A nuScenes *detection* submission carries no `tracking_id`. So the tracker's
precondition was never met, the adapter synthesised ids, and §26 measured the
consequence: 0.0% of ids survived a frame, and the Kalman filter never ran a
second update on any agent.

`eval_track.py --export` now writes Sparse4D's unfiltered per-sample output —
`run_inference` already covered every scene and the split filter was discarding
it, so this costs one file write and no new inference. Re-running reproduced
AMOTA 0.627 / MOTA 0.631.

### Identity is fixed

| source | agents/frame | ids carried to next frame |
|---|---|---|
| GT annotations | 41.2 | **96.9%** |
| BEVFormer-tiny, token hash (as shipped) | 37.1 | **0.1%** |
| BEVFormer-tiny, NN association (§26 fix) | 37.1 | 64.4% |
| **Sparse4D v3, real instance-bank ids** | **19.9** | **91.0%** |

Real ids land within 6 points of the ground-truth ceiling, and 27 points above
the nearest-neighbour stand-in. The stated problem is solved.

### And it does not help

Class-matched, because Sparse4D submits only the 7 tracking classes and drops
boxes with no assigned track:

| arm | mode | EGO | other | brakes | clearance | completion | agents |
|---|---|---|---|---|---|---|---|
| BEVFormer-tiny 10-class + NN | pinned | 0 | 0 | 70 | 1.73 m | 52.5% | 40.0 |
| BEVFormer-tiny 10-class + NN | free | 0 | 20 | 94 | 1.83 m | 39.8% | 39.8 |
| BEVFormer-tiny **7-class** + NN | pinned | 0 | 0 | 69 | 1.73 m | 52.5% | 33.8 |
| **BEVFormer-tiny 7-class + NN** | free | **0** | **8** | 88 | **2.04 m** | 47.0% | 33.6 |
| Sparse4D 7-class, **real ids** | pinned | 0 | 0 | **46** | **2.03 m** | 52.5% | 20.8 |
| Sparse4D 7-class, **real ids** | free | **1** | 15 | 83 | 1.74 m | 47.8% | 20.7 |

**Against the class-matched baseline, real identity is worse on free-running
safety** — 15 other-fault against 8, plus the only ego-fault step in the table.
Its pinned numbers do look best (46 brakes against 69, clearance 2.03 m), but it
is seeing **20.7 agents against 33.6**: recall 0.709 and every untracked box
discarded. So even the class-matched comparison is confounded by count.

The dominant variable is **how many agents you see, not whether you can follow
them**. That is worth stating plainly because the entire exercise was premised on
the opposite, and the premise came from this document — §26 identified the
association failure, correctly, and then assumed fixing it properly would pay.

### The 7-class restriction is the actual finding, and it is suspicious

Dropping `barrier`, `traffic_cone` and `construction_vehicle` from BEVFormer-tiny
takes other-fault collisions **20 → 8** and completion 39.8% → 47.0%, at no cost
anywhere in the table. Those three classes were *causing* collisions.

The mechanism is the §19 chain with a named trigger: false positives on static
roadside furniture inflate risk, the gate over-brakes, the ego falls behind the
recording, and the recorded traffic drives into it. Removing the classes removes
the false positives.

**Not shipped as a default.** "Delete the obstacle classes and the safety metric
improves" is the shape of a measurement artefact, not an improvement — barriers
and cones are real obstacles, and a stack that drives better without seeing them
is telling you something about the risk model rather than about the classes. The
right follow-up is to measure the per-class false-positive rate inside 10 m
(§26's range table did this in aggregate: 3% true FP inside 10 m against 40%
beyond 40 m) and fix the precision, not the taxonomy.

### What the export is still worth

It closes a stated architectural assumption that had been false for the whole
project, and it converts "we cannot test this" into a measured null. Both live
sources are kept and selectable — `LivePerceptionWorldModel(source='bevformer'
| 'sparse4d')` — with `bevformer` remaining the default.

---

## Retractions

Twelve causal explanations were committed and then refuted by their own
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
6. **"22% of detections above 0.25 are false positives."** A far-field average.
   Inside 10 m the true false-positive rate is **3%**, and 57% of the orphans
   there are a real obstacle under the wrong class label (§26).
7. **"Live perception costs roughly 1 decision in 10 becoming materially
   riskier."** §25's much-worse tail was computed by a counterfactual that built
   its risk model with no tracker, so it never read the covariance at all. Wired,
   it is **0%** under every position arm (§26).
8. **"residual.py cannot be meaningfully re-fitted under live perception."**
   True, but not for the reason implied: the live adapter produced **6 training
   pairs from 14,615 agents** because track ids were unique per frame. That
   measured plumbing, not perception (§26).
9. **"All six anchors share an arc length, so progress, offroad and clearance
   are constant across candidates and risk is the only discriminating term."**
   Clearance spreads 0.674 on 77% of decisions — it is the *most* discriminating
   term, out-spreading risk 14×. The price is unidentifiable because *progress*
   has 125× less spread than clearance, which is a different claim (§26).

10. **"The planner never proposes driving into an occlusion."** It proposes them
    on **20% of steps** with the original six anchors and 84% with the full
    vocabulary. The "0 of 6 waypoints in unknown" that supported this across
    three investigations was measured on the *post-filter selected plan*, so it
    described gate 1 and not the planner (§27).
11. **"W_RISK is unidentifiable."** Third explanation offered for it, and also
    wrong. It is identifiable in [0, 2]; the sweeps ran over 6–25, entirely
    inside the region where risk already dominates the argmin (§29).
12. **"`w_risk = 1.0` buys safety for 1.5 pp of completion."** A sign error.
    Completion is 39.8% at 1.0 against 38.3% at 10.0 — 1.5 pp *better*. 1.0 wins
    on six metrics and loses only on jerk, +6.5% (§29).

The pattern in the first five: a real defect was found, correctly identified as
real, and then over-credited with the observed symptom. The pattern in 6–11 is
different and worse — a number was reported from a path that could not have
produced a different answer, so the null result carried no information. Each was
caught only by asking what would have to change for the measurement to move.

12 is a third kind and the most embarrassing: the measurement was reachable, it
ran, it produced the right numbers, and the write-up inverted the sign of a
difference it had computed correctly. No amount of instrumentation catches that
one — only reading the table against the sentence describing it.

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
PYTHONPATH=. python -m e2e_pipeline.tools.risk_sweep        # §7
PYTHONPATH=. python -m e2e_pipeline.tools.rebaseline        # §8
PYTHONPATH=. python -m e2e_pipeline.tools.fit_calibration   # §9
PYTHONPATH=. python -m e2e_pipeline.tools.soft_risk_ab      # §10

PYTHONPATH=. python -m e2e_pipeline.tools.covariance_calibration      # §26 covariance + orphans
PYTHONPATH=. python -m e2e_pipeline.tools.counterfactual_attribution  # §26 tail attribution
PYTHONPATH=. python -m e2e_pipeline.tools.risk_gate_recheck           # §26 max_risk
PYTHONPATH=. python -m e2e_pipeline.tools.residual_live_snr           # §26 residual SNR
PYTHONPATH=. python -m e2e_pipeline.tools.calibrate_rerun             # §26 critic spread
PYTHONPATH=. python -m e2e_pipeline.tools.occlusion_prior_sweep       # §26 occlusion prior
```
