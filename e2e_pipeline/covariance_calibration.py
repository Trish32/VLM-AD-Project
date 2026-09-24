"""Does detector score actually predict box error?

`TrackCovarianceTracker._R` assumes sigma_pos = pos_noise / score, so a
0.25-score detection is treated as four times noisier than a 1.0. That is a
modelling ASSUMPTION and has never been checked. If it is wrong, every
downstream risk number inherits a covariance that does not describe reality --
the same failure as the uncalibrated collision probability, one level lower.

Matches live detections to nuScenes annotations (nearest same-class within
MATCH_RADIUS) and compares predicted sigma against measured error, binned by
score. A reliability plot for covariance rather than for probability.

Usage:
    PYTHONPATH=. python e2e_pipeline/covariance_calibration.py
"""
import os
from collections import defaultdict

import numpy as np
from nuscenes import NuScenes
from nuscenes.eval.detection.utils import category_to_detection_name
from pyquaternion import Quaternion

from e2e_pipeline.live_adapter import DATA, load_detections
from e2e_pipeline.uncertainty import TrackCovarianceTracker

MATCH_RADIUS = 4.0
SCORE_BINS = ((0.25, 0.4), (0.4, 0.55), (0.55, 0.7), (0.7, 0.85), (0.85, 1.01))
RANGE_BINS = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, 1e9))


def ego_xy(nusc, token):
    """Ego translation for a sample, so detections can be binned by range."""
    sd = nusc.get('sample_data', nusc.get('sample', token)['data']['LIDAR_TOP'])
    return np.asarray(nusc.get('ego_pose', sd['ego_pose_token'])['translation'][:2],
                      dtype=float)


def _rbin(r):
    for lo, hi in RANGE_BINS:
        if lo <= r < hi:
            return (lo, hi)
    return RANGE_BINS[-1]


def gt_boxes(nusc, token):
    out = []
    for a in nusc.get('sample', token)['anns']:
        ann = nusc.get('sample_annotation', a)
        name = category_to_detection_name(ann['category_name'])
        if name:
            v = nusc.box_velocity(a)[:2]
            out.append((name, np.asarray(ann['translation'][:2], float),
                        np.asarray(ann['size'], float),
                        Quaternion(ann['rotation']).yaw_pitch_roll[0],
                        np.asarray(v, float)))
    return out


def main():
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'),
                    verbose=False)
    det = load_detections(DATA / 'results_mini_train.json',
                          DATA / 'results_mini_val.json')
    tr = TrackCovarianceTracker()

    per_bin = defaultdict(list)
    per_range = defaultdict(lambda: {'matched': 0, 'orphan': 0, 'confused': 0,
                                     'err': [], 'score': []})
    unmatched = matched = 0
    for token, boxes in det.items():
        try:
            gts = gt_boxes(nusc, token)
            exy = ego_xy(nusc, token)
        except Exception:
            continue
        for b in boxes:
            s = float(b.get('detection_score', 0.0))
            if s < SCORE_BINS[0][0]:
                continue
            name = b.get('detection_name')
            p = np.asarray(b['translation'][:2], float)
            rb = per_range[_rbin(float(np.linalg.norm(p - exy)))]
            best, bd = None, MATCH_RADIUS
            for gname, gp, gsz, gyaw, gv in gts:
                if gname != name:
                    continue
                d = float(np.linalg.norm(gp - p))
                if d < bd:
                    best, bd = (gp, gsz, gyaw, gv), d
            if best is None:
                unmatched += 1
                rb['orphan'] += 1
                # An orphan with a DIFFERENT-class annotation sitting on it is a
                # label error, not a hallucinated object. Separating the two
                # decides whether the 22% is a recall/precision problem or a
                # taxonomy one, and they have opposite implications for risk:
                # a class-confused box still marks a real obstacle.
                if any(float(np.linalg.norm(gp - p)) < MATCH_RADIUS
                       for _, gp, _, _, _ in gts):
                    rb['confused'] += 1
                continue
            matched += 1
            rb['matched'] += 1
            rb['err'].append(bd)
            rb['score'].append(s)
            dv = float('nan')
            if np.all(np.isfinite(best[3])) and b.get('velocity') is not None:
                dv = float(np.linalg.norm(np.asarray(b['velocity'][:2], float)
                                          - best[3]))
            for lo, hi in SCORE_BINS:
                if lo <= s < hi:
                    per_bin[(lo, hi)].append((bd, s,
                                              abs(float(np.linalg.norm(b['size'][:2])
                                                        - np.linalg.norm(best[1][:2]))),
                                              dv))
                    break

    print(f'  matched {matched}  unmatched {unmatched} '
          f'({unmatched / max(matched + unmatched, 1):.0%} false positives)')
    print(f'\n  {"score bin":>14}{"n":>7}{"predicted σ":>13}{"measured RMS":>14}'
          f'{"ratio":>8}{"dim err":>9}{"pred σv":>9}{"vel RMS":>9}{"v ratio":>9}')
    rows, vrows = [], []
    for k in SCORE_BINS:
        v = per_bin.get(k, [])
        if len(v) < 20:
            continue
        errs = np.array([x[0] for x in v])
        s_mean = float(np.mean([x[1] for x in v]))
        dim = float(np.mean([x[2] for x in v]))
        pred_sigma = float(np.sqrt(tr._R(s_mean)[0, 0]))
        meas = float(np.sqrt(np.mean(errs ** 2)))
        dvs = np.array([x[3] for x in v])
        dvs = dvs[np.isfinite(dvs)]
        # Velocity was never measured -- vel_noise=1.0 m/s was assumed, and the
        # calibrated branch tied it to pos_noise, which is meaningless for a
        # world that declares a position scale it no longer uses. Measure it.
        vmeas = float(np.sqrt(np.mean(dvs ** 2))) if len(dvs) >= 20 else float('nan')
        pred_v = float(np.sqrt(tr._R(s_mean)[2, 2]))
        rows.append((k, len(v), pred_sigma, meas, meas / pred_sigma, dim))
        if np.isfinite(vmeas):
            vrows.append((s_mean, vmeas, len(dvs)))
        print(f'  {f"{k[0]:.2f}-{k[1]:.2f}":>14}{len(v):>7}{pred_sigma:>13.3f}'
              f'{meas:>14.3f}{meas / pred_sigma:>8.2f}{dim:>9.3f}'
              f'{pred_v:>9.3f}{vmeas:>9.3f}{vmeas / pred_v:>9.2f}')

    if len(vrows) >= 2:
        # least squares for sigma_v = floor + slope / score
        A = np.stack([np.ones(len(vrows)), 1.0 / np.array([r[0] for r in vrows])], 1)
        y = np.array([r[1] for r in vrows])
        w = np.sqrt(np.array([r[2] for r in vrows], float))
        coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        resid = float(np.sqrt(np.average((A @ coef - y) ** 2, weights=w ** 2)))
        print(f'\n  velocity fit  σv = {coef[0]:.3f} + {coef[1]:.3f} / score   '
              f'(weighted RMS residual {resid:.3f} m/s)')

    if len(rows) >= 2:
        r = [x[4] for x in rows]
        print(f'\n  ratio spread {min(r):.2f}..{max(r):.2f}  '
              f'(1.00 = covariance matches error)')
        hi_s, lo_s = rows[-1][3], rows[0][3]
        print(f'  measured error, lowest-score bin {lo_s:.2f} m vs '
              f'highest {hi_s:.2f} m  -> score '
              f'{"IS" if lo_s > hi_s * 1.2 else "is NOT"} predictive of error')

    # -- where the orphans are ------------------------------------------------
    #
    # 22% of detections above 0.25 have no same-class GT within 4 m. Reported as
    # one number that is uninterpretable: it could be uniform hallucination, or
    # it could be entirely the far field where the detector is guessing and the
    # annotator stopped labelling. Those want different responses -- the first
    # is a precision problem, the second is an ODD boundary -- so bin by range.
    print(f'\n  {"range":>12}{"n":>7}{"orphan":>9}{"class-conf":>12}'
          f'{"true FP":>10}{"mean score":>12}{"meas err":>10}')
    for k in RANGE_BINS:
        v = per_range.get(k)
        if not v:
            continue
        n = v['matched'] + v['orphan']
        if n < 20:
            continue
        lo, hi = k
        lbl = f'{lo:.0f}-{hi:.0f} m' if hi < 1e8 else f'{lo:.0f}+ m'
        err = float(np.sqrt(np.mean(np.square(v['err'])))) if v['err'] else float('nan')
        sc = float(np.mean(v['score'])) if v['score'] else float('nan')
        print(f'  {lbl:>12}{n:>7}{v["orphan"] / n:>9.0%}'
              f'{v["confused"] / max(v["orphan"], 1):>12.0%}'
              f'{(v["orphan"] - v["confused"]) / n:>10.0%}{sc:>12.2f}{err:>10.2f}')
    print('  (class-conf = share of orphans with a DIFFERENT-class GT within 4 m;\n'
          '   those are real obstacles under a wrong label, not hallucinations)')


if __name__ == '__main__':
    main()


# --- track stability --------------------------------------------------------

def track_stability(nusc, det, score_thr=0.25, match_radius=2.0):
    """ID-switch and fragmentation rates, as a planner confidence signal.

    An unstable track is worse than a missing one: the Kalman filter resets its
    covariance on every new id, so a fragmenting track looks freshly uncertain
    forever and never accumulates the confidence that would let a planner
    commit. Feeding the rates to the planner lets it distrust the track list as
    a whole when association is struggling.

    IMPORTANT LIMIT. These detections carry no `tracking_id` -- the adapter
    synthesises one by hashing rounded position, so identity is *defined* by
    location and an ID switch is not directly observable. What is measured here
    is the GT-side proxy: how often a ground-truth instance that was detected in
    one frame has no detection near it in the next. That is fragmentation of
    coverage, which drives the same covariance reset, but it is not an MOTA
    id-switch count and should not be reported as one.
    """
    from collections import defaultdict
    seen = defaultdict(list)
    for si in range(len(nusc.scene)):
        tok = nusc.scene[si]['first_sample_token']
        k = 0
        while tok:
            boxes = [b for b in det.get(tok, [])
                     if float(b.get('detection_score', 0)) >= score_thr]
            pts = np.array([b['translation'][:2] for b in boxes]) if boxes else np.zeros((0, 2))
            for a in nusc.get('sample', tok)['anns']:
                ann = nusc.get('sample_annotation', a)
                if category_to_detection_name(ann['category_name']) is None:
                    continue
                g = np.asarray(ann['translation'][:2], float)
                hit = bool(len(pts)) and bool(
                    (np.linalg.norm(pts - g, axis=1) < match_radius).any())
                seen[ann['instance_token']].append((k, hit))
            tok = nusc.get('sample', tok)['next']
            k += 1

    frag, total, covered = 0, 0, 0
    for inst, hist in seen.items():
        hits = [h for _, h in hist]
        if not any(hits):
            continue
        covered += 1
        # a fragmentation is a detected->missed transition after first detection
        first = hits.index(True)
        frag += sum(1 for i in range(first + 1, len(hits))
                    if hits[i - 1] and not hits[i])
        total += max(len(hits) - first - 1, 0)
    return {'instances': len(seen), 'ever_detected': covered,
            'coverage': covered / max(len(seen), 1),
            'fragmentations': frag, 'transitions': total,
            'fragmentation_rate': frag / max(total, 1)}
