#!/usr/bin/env python3
"""Build a traffic-light annotation set for nuScenes-mini.

nuScenes carries traffic-light *locations* in the map expansion but no *state*:
each fixture lists its full red/yellow/green bulb stack with per-lamp heights, and
there is no timestamp or sample linkage anywhere in the layer.  So the only way to
score a light-reading model is to hand-label, and this generates the material.

What it does:
  1. Projects every map `traffic_light` fixture into CAM_FRONT using ego pose and
     the camera calibration, keeping frames where one actually lands in view.
  2. Crops a window around the projected fixture — reading state off a 300 px crop
     is far faster and more reliable than scanning a 1600x900 frame.
  3. Computes a *hint* for the hard question, "does this light govern my lane?",
     by testing whether the ego stands on the road block the fixture points from.
  4. Emits a manifest plus a self-contained HTML annotator (see annotate.html).

Frames are emitted in scene/frame order, not by range, because signals change on a
~30-60 s cycle while a scene is only 20 s — so most scenes contain at most one
phase change.  Annotating in temporal order lets you mark transitions and extend
runs rather than judging each frame independently.

Usage:
    conda run -n simple_bev_vldrive python tools/make_light_annotation.py
    open bev_outputs/light_annotation/annotate.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / 'bev_outputs' / 'light_annotation'

# Fixture height is not in the map (nodes are 2-D), but every record carries a
# `pose.tz`; fall back to a typical signal-head height when it is missing.
FALLBACK_TZ = 5.2
CROP_HALF = 170          # px in the native frame, at CROP_REF_RANGE
CROP_REF_RANGE = 25.0    # m — range at which CROP_HALF is used verbatim
CROP_OUT = 340           # upscaled crop size written to disk
CONTEXT_W = 800          # downscaled full frame for lane context


def fixture_records(dataroot: str, location: str) -> list[dict]:
    """Every traffic-light fixture with a global (x, y, z) and its road block."""
    d = json.load(open(Path(dataroot) / 'maps' / 'expansion' / f'{location}.json'))
    nodes = {n['token']: (n['x'], n['y']) for n in d['node']}
    lines = {l['token']: l['node_tokens'] for l in d['line']}
    out = []
    for tl in d.get('traffic_light', []):
        pose = tl.get('pose') or {}
        xy = [nodes[t] for t in lines.get(tl['line_token'], []) if t in nodes]
        # Position comes from the LINE GEOMETRY, not `pose`.  The `pose.tx/ty`
        # fields are populated only on boston-seaport; on all three Singapore maps
        # every fixture has tx = ty = 0, which silently parks 327 lights at the map
        # origin where they never project into any camera.  Where pose IS real it
        # agrees with the node centroid to ~3.7 m, well inside the crop window, so
        # the centroid is simply the better primary source.  `pose.tz` is populated
        # everywhere and is still the best height estimate.
        if not xy:
            continue
        x, y = np.asarray(xy, float).mean(axis=0)
        out.append({'token': tl['token'], 'xy': (float(x), float(y)),
                    'z': float(pose.get('tz') or FALLBACK_TZ),
                    'from_road_block': tl.get('from_road_block_token')})
    return out


def project(p_global: np.ndarray, ego_pose: dict, cam_cs: dict
            ) -> tuple[float, float, float] | None:
    """Global point -> (u, v, depth) in CAM_FRONT, or None if behind the camera."""
    R_e = Quaternion(ego_pose['rotation']).rotation_matrix
    t_e = np.asarray(ego_pose['translation'])
    R_c = Quaternion(cam_cs['rotation']).rotation_matrix
    t_c = np.asarray(cam_cs['translation'])
    p_cam = R_c.T @ (R_e.T @ (p_global - t_e) - t_c)
    if p_cam[2] < 1.0:
        return None
    uv = np.asarray(cam_cs['camera_intrinsic']) @ p_cam
    return float(uv[0] / uv[2]), float(uv[1] / uv[2]), float(p_cam[2])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', default='/Users/trish/Downloads/nuScenes_miniV1.0')
    ap.add_argument('--out-dir', default=str(DEFAULT_OUT))
    ap.add_argument('--max-range', type=float, default=60.0,
                    help='Skip fixtures further than this (m); far ones are unreadable')
    ap.add_argument('--margin', type=int, default=60,
                    help='Require the projection this far inside the image border')
    ap.add_argument('--include-negatives', action='store_true',
                    help='Also emit keyframes with NO fixture in view. These are the '
                         'only way to measure false positives — the model has been '
                         'seen inventing "LIGHT: red" with no camera attached — and '
                         'they also catch fixtures the map is missing.')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / 'crop').mkdir(parents=True, exist_ok=True)
    (out_dir / 'full').mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version='v1.0-mini', dataroot=args.dataroot, verbose=False)
    fixtures: dict[str, list[dict]] = {}
    maps: dict[str, NuScenesMap] = {}
    frames: list[dict] = []

    for scene in nusc.scene:
        loc = nusc.get('log', scene['log_token'])['location']
        if loc not in fixtures:
            fixtures[loc] = fixture_records(args.dataroot, loc)
            try:
                maps[loc] = NuScenesMap(dataroot=args.dataroot, map_name=loc)
            except Exception:
                maps[loc] = None
        if not fixtures[loc] and not args.include_negatives:
            continue

        tok, fidx = scene['first_sample_token'], 0
        while tok:
            s = nusc.get('sample', tok)
            sd = nusc.get('sample_data', s['data']['CAM_FRONT'])
            ep = nusc.get('ego_pose', sd['ego_pose_token'])
            cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])

            # Which road block is the ego standing on? Used only as a hint.
            ego_block = ''
            if maps.get(loc) is not None:
                try:
                    ego_block = maps[loc].layers_on_point(
                        ep['translation'][0], ep['translation'][1]).get('road_block', '')
                except Exception:
                    ego_block = ''

            best = None
            for fx in fixtures[loc]:
                p = np.array([fx['xy'][0], fx['xy'][1],
                              ep['translation'][2] + fx['z']])
                pr = project(p, ep, cs)
                if pr is None:
                    continue
                u, v, depth = pr
                if not (args.margin <= u < 1600 - args.margin
                        and args.margin <= v < 900 - args.margin):
                    continue
                if depth > args.max_range:
                    continue
                if best is None or depth < best[2]:
                    best = (u, v, depth, fx)

            if best is None and args.include_negatives:
                # Negative: no fixture projects into view. Context frame only —
                # there is nothing to zoom to — and the expected label is `none`,
                # but it still needs eyes on it in case the map missed a fixture.
                img = cv2.imread(str(Path(args.dataroot) / sd['filename']))
                if img is not None:
                    idx = len(frames)
                    ctx = cv2.resize(img, (CONTEXT_W, int(900 * CONTEXT_W / 1600)))
                    cv2.imwrite(str(out_dir / 'full' / f'{idx:03d}.jpg'), ctx,
                                [cv2.IMWRITE_JPEG_QUALITY, 82])
                    cv2.imwrite(str(out_dir / 'crop' / f'{idx:03d}.jpg'),
                                cv2.resize(ctx, (CROP_OUT, CROP_OUT)),
                                [cv2.IMWRITE_JPEG_QUALITY, 82])
                    frames.append({
                        'idx': idx, 'sample_token': tok, 'scene': scene['name'],
                        'frame': fidx, 'range_m': None, 'fixture_token': None,
                        'road_block_match': False, 'governs_hint': False,
                        'negative': True,
                    })

            if best is not None:
                u, v, depth, fx = best
                img = cv2.imread(str(Path(args.dataroot) / sd['filename']))
                if img is not None:
                    idx = len(frames)
                    # Apparent fixture size falls as 1/range, so scale the native
                    # crop window the same way — otherwise a light at 50 m is a
                    # handful of pixels in the same 340 px tile where a light at
                    # 15 m fills the frame, and the far ones are unlabellable.
                    half = int(np.clip(CROP_HALF * CROP_REF_RANGE / max(depth, 1.0),
                                       70, 300))
                    x0 = int(np.clip(u - half, 0, 1600 - 2 * half))
                    y0 = int(np.clip(v - half, 0, 900 - 2 * half))
                    crop = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
                    crop = cv2.resize(crop, (CROP_OUT, CROP_OUT),
                                      interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(str(out_dir / 'crop' / f'{idx:03d}.jpg'), crop,
                                [cv2.IMWRITE_JPEG_QUALITY, 92])

                    ctx = cv2.resize(img, (CONTEXT_W, int(900 * CONTEXT_W / 1600)))
                    # Mark where the fixture projects, so lane context is unambiguous.
                    cv2.circle(ctx, (int(u * CONTEXT_W / 1600),
                                     int(v * CONTEXT_W / 1600)), 16, (0, 235, 235), 2)
                    cv2.imwrite(str(out_dir / 'full' / f'{idx:03d}.jpg'), ctx,
                                [cv2.IMWRITE_JPEG_QUALITY, 85])

                    frames.append({
                        'idx': idx, 'sample_token': tok, 'scene': scene['name'],
                        'frame': fidx, 'range_m': round(depth, 1),
                        'fixture_token': fx['token'],
                        # A WEAK cue, not a default.  `from_road_block_token` names
                        # the one specific approach block a fixture points from,
                        # while the ego usually sits on a neighbouring block of the
                        # same approach — so exact equality holds on only ~3% of
                        # frames even though both fields are fully populated.
                        # Surfaced for the annotator to weigh; the default for
                        # `governs_ego` is True (a fixture you are driving toward
                        # usually does govern you) and `e` flips it.
                        'road_block_match': bool(
                            ego_block and ego_block == fx['from_road_block']),
                        'governs_hint': True,
                        'negative': False,
                    })
            tok, fidx = s['next'], fidx + 1

    (out_dir / 'manifest.json').write_text(json.dumps(frames, indent=1))
    _write_html(out_dir, frames)

    by_scene: dict[str, int] = {}
    for f in frames:
        by_scene[f['scene']] = by_scene.get(f['scene'], 0) + 1
    print(f'[DONE] {len(frames)} candidate frames across {len(by_scene)} scenes')
    for k, v in sorted(by_scene.items()):
        print(f'   {k}: {v}')
    hinted = sum(f['governs_hint'] for f in frames)
    print(f'   governs-ego hint true on {hinted}/{len(frames)}')
    print(f'\n   open {out_dir / "annotate.html"}')


def _write_html(out_dir: Path, frames: list[dict]) -> None:
    (out_dir / 'annotate.html').write_text(
        _HTML.replace('__MANIFEST__', json.dumps(frames)))


_HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>nuScenes-mini traffic-light annotation</title>
<style>
 :root { color-scheme: dark; }
 body { margin:0; background:#111; color:#ddd;
        font:14px/1.45 ui-monospace,Menlo,Consolas,monospace; }
 header { padding:8px 14px; background:#000; display:flex; gap:18px;
          align-items:center; flex-wrap:wrap; position:sticky; top:0; z-index:5; }
 #bar { height:6px; background:#222; }
 #fill { height:100%; background:#3a8; width:0; }
 main { display:flex; gap:14px; padding:14px; align-items:flex-start; }
 img.crop { width:340px; height:340px; image-rendering:auto; border:1px solid #333; }
 img.ctx  { width:min(800px,52vw); border:1px solid #333; }
 .k { display:inline-block; padding:2px 7px; border:1px solid #555;
      border-radius:4px; margin-right:4px; }
 .red{color:#e55} .yellow{color:#dc4} .green{color:#5d8} .none{color:#999}
 .unknown{color:#c8a}
 table { border-collapse:collapse; margin:10px 14px; font-size:12px; }
 td,th { border:1px solid #333; padding:2px 8px; text-align:left; }
 tr.cur { background:#243; }
 #meta { margin-left:auto; color:#8aa; }
 button { background:#222; color:#ddd; border:1px solid #555; padding:6px 12px;
          border-radius:4px; cursor:pointer; font:inherit; }
 button:hover { background:#333; }
 button.lab { border-width:2px; font-weight:600; }
 button.lab.red{border-color:#e55;color:#e55} button.lab.yellow{border-color:#dc4;color:#dc4}
 button.lab.green{border-color:#5d8;color:#5d8} button.lab.none{border-color:#888;color:#aaa}
 button.lab.unknown{border-color:#c8a;color:#c8a}
</style>
<header>
  <b id="pos"></b>
  <span id="scene"></span>
  <span id="btns">
    <button class="lab red"     onclick="click_('red')">r &nbsp;red</button>
    <button class="lab yellow"  onclick="click_('yellow')">y &nbsp;yellow</button>
    <button class="lab green"   onclick="click_('green')">g &nbsp;green</button>
    <button class="lab none"    onclick="click_('none')">n &nbsp;none</button>
    <button class="lab unknown" onclick="click_('unknown')">u &nbsp;unknown</button>
  </span>
  <span>
    <button onclick="same()">space &nbsp;same as prev</button>
    <button onclick="back()">b &nbsp;back</button>
    <button onclick="toggleGoverns()">e &nbsp;governs-ego</button>
  </span>
  <button onclick="exportJSONL()">Export JSONL</button>
  <span id="meta"></span>
</header>
<div id="bar"><div id="fill"></div></div>
<main>
  <div>
    <img class="crop" id="crop">
    <div id="hint" style="padding:6px 2px;color:#8aa"></div>
  </div>
  <img class="ctx" id="ctx">
</main>
<table id="recent"></table>
<div id="err" style="display:none;background:#601;color:#fdd;padding:10px 14px;
     font-weight:600"></div>
<script>
window.onerror = function (msg, src, line) {
  const d = document.getElementById('err');
  d.style.display = 'block';
  d.textContent = 'PAGE ERROR (keys will not respond): ' + msg + '  @line ' + line
                + (location.protocol === 'file:'
                   ? '  — you are on file://; serve over http:// instead.' : '');
  return false;
};
if (location.protocol === 'file:') {
  window.onerror('opened via file:// — localStorage is blocked in most browsers, '
               + 'so autosave and key handling may fail', '', 0);
}
const F = __MANIFEST__;
const KEY = 'nusc_light_labels_v1';
let storageOK = true;
function load() {
  try { return JSON.parse(localStorage.getItem(KEY) || '{}'); }
  catch (e) { storageOK = false; return {}; }
}
function save(o) {
  try { localStorage.setItem(KEY, JSON.stringify(o)); }
  catch (e) {
    if (storageOK) { storageOK = false;
      alert('Autosave unavailable (localStorage blocked).\n'
          + 'Serve over http:// instead of opening the file directly, '
          + 'and export often.'); }
  }
}
let labels = load();
let i = 0;
// Resume where the last session stopped rather than restarting at 0.
while (i < F.length && labels[F[i].sample_token]) i++;
if (i >= F.length) i = F.length - 1;

function render() {
  const f = F[i];
  document.getElementById('crop').src = `crop/${String(f.idx).padStart(3,'0')}.jpg`;
  document.getElementById('ctx').src  = `full/${String(f.idx).padStart(3,'0')}.jpg`;
  document.getElementById('pos').textContent = `${i+1} / ${F.length}`;
  document.getElementById('scene').textContent =
      `${f.scene}  frame ${f.frame}  ` +
      (f.negative ? '— NO LIGHT IN VIEW (expect: none)' : `${f.range_m} m`);
  document.getElementById('crop').style.opacity = f.negative ? 0.35 : 1;
  const lab = labels[f.sample_token];
  const g = lab ? lab.governs_ego : f.governs_hint;
  document.getElementById('hint').innerHTML =
      `governs ego lane: <b>${g ? 'YES' : 'no'}</b> ` +
      `<span style="color:#666">(road-block match: ${f.road_block_match}` +
      ` — weak cue only; press e to flip)</span>` +
      (lab ? ` &nbsp;|&nbsp; labelled <b class="${lab.light}">${lab.light}</b>` : '');
  const done = Object.keys(labels).length;
  document.getElementById('meta').textContent =
      `${done} labelled` + (storageOK ? '' : '  ⚠ AUTOSAVE OFF — export often');
  document.getElementById('fill').style.width = (100*done/F.length) + '%';

  let rows = '<tr><th>#</th><th>scene</th><th>frame</th><th>m</th>'
           + '<th>light</th><th>governs</th></tr>';
  for (let j = Math.max(0,i-6); j <= Math.min(F.length-1,i+3); j++) {
    const x = F[j], l = labels[x.sample_token];
    rows += `<tr class="${j===i?'cur':''}"><td>${j+1}</td><td>${x.scene}</td>`
         +  `<td>${x.frame}</td><td>${x.range_m}</td>`
         +  `<td class="${l?l.light:''}">${l?l.light:'-'}</td>`
         +  `<td>${l?(l.governs_ego?'yes':'no'):'-'}</td></tr>`;
  }
  document.getElementById('recent').innerHTML = rows;
}

function setLabel(light, governs) {
  const f = F[i];
  labels[f.sample_token] = {
    sample_token: f.sample_token, scene: f.scene, frame: f.frame,
    light, governs_ego: governs, range_m: f.range_m,
    negative: !!f.negative,
    fixture_token: f.fixture_token, ts: new Date().toISOString()
  };
  save(labels);                                        // autosave every keypress
  if (i < F.length - 1) i++;
  render();
}

function curGoverns() {
  const f = F[i], c = labels[f.sample_token];
  return c ? c.governs_ego : f.governs_hint;
}
function click_(light) { setLabel(light, curGoverns()); }
function same() {
  const prev = i > 0 ? labels[F[i-1].sample_token] : null;
  if (prev) setLabel(prev.light, prev.governs_ego);
}
function back() { i = Math.max(0, i-1); render(); }
function toggleGoverns() {
  const f = F[i], c = labels[f.sample_token];
  if (c) { c.governs_ego = !c.governs_ego; save(labels); }
  else { f.governs_hint = !f.governs_hint; }
  render();
}

document.addEventListener('keydown', e => {
  const f = F[i];
  const cur = labels[f.sample_token];
  const g = cur ? cur.governs_ego : f.governs_hint;
  const map = {r:'red', y:'yellow', g:'green', n:'none', u:'unknown'};
  if (map[e.key]) { setLabel(map[e.key], g); e.preventDefault(); return; }
  if (e.key === ' ') {
    // Run-extension: signals hold for many frames, so copying the previous label
    // is the common case and should cost one keystroke.
    const prev = i > 0 ? labels[F[i-1].sample_token] : null;
    if (prev) setLabel(prev.light, prev.governs_ego);
    e.preventDefault(); return;
  }
  if (e.key === 'b') { i = Math.max(0, i-1); render(); e.preventDefault(); return; }
  if (e.key === 'e') {
    if (cur) { cur.governs_ego = !cur.governs_ego; save(labels); }
    else { F[i].governs_hint = !F[i].governs_hint; }
    render(); e.preventDefault();
  }
});

function exportJSONL() {
  const lines = F.filter(f => labels[f.sample_token])
                 .map(f => JSON.stringify(labels[f.sample_token])).join('\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lines], {type:'application/x-ndjson'}));
  a.download = 'light_labels.jsonl';
  a.click();
}
render();
</script>
"""


if __name__ == '__main__':
    main()
