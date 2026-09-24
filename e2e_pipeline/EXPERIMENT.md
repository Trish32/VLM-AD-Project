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
