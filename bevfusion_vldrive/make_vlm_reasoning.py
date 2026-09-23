#!/usr/bin/env python3
"""Generate the VLM reasoning track that the demo GIFs overlay on CAM_BACK.

Turns each port's detection demo into a detector -> VLM -> decision demo. The
detections come from the port's ALREADY-SAVED `results_nusc.json`, so neither
BEVFusion checkpoint is re-run here: the boxes in the GIF and the boxes the VLM
reasons about come from the same evaluated predictions, which is also what makes
the two ports' GIFs a fair comparison rather than two independent runs.

The VLM stage is imported wholesale from `bevformer_vldrive/tools` and receives
the SAME FOUR CHANNELS BEVFormer's pipeline sends -- BEV raster, forward camera,
map-projected traffic-light crop (stage 1), and structured detections as text --
so the only thing that differs between the three ports is which detector produced
the boxes. Reimplementing any of it here would let the two drift and quietly turn
a detector comparison into a prompt comparison.

The raster is produced by BEVFormer's own `build_scene_canvas`, fed through
`boxes_to_bevformer_tensors` below, rather than by a lookalike renderer.

Usage:
    conda run -n simple_bev_vldrive python make_vlm_reasoning.py \
        --results bevfusion_vl/eval_out_det/results_nusc.json \
        --out bevfusion_vl/viz_out/vlm_reasoning.jsonl --max-frames 15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEVFORMER_TOOLS = HERE.parent / 'bevformer_vldrive' / 'tools'
sys.path.insert(0, str(BEVFORMER_TOOLS))
sys.path.insert(0, str(BEVFORMER_TOOLS.parent))

from nuscenes import NuScenes

import math

import numpy as np
import torch
from pyquaternion import Quaternion

# The global->BEVFormer-tensor inverse lives next to the decode it inverts,
# so the two cannot drift.
from compare_detectors import (boxes_to_bevformer_tensors, boxes_to_items,
                               load_results)
from dataroot import default_dataroot
from vis_infer import (CLASS_NAMES, _encode_image, _format_detection_rows,
                       _parse_response, _query_ollama_streaming, build_prompt,
                       front_camera_b64, light_crop_b64, query_light)
from visualizer import PC_RANGE, build_scene_canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True, help='port results_nusc.json')
    ap.add_argument('--out', required=True, help='JSONL the visualiser reads')
    ap.add_argument('--dataroot', default=default_dataroot())
    ap.add_argument('--split', default='mini_val')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--max-frames', type=int, default=15)
    ap.add_argument('--score-thr', type=float, default=0.25)
    ap.add_argument('--max-rows', type=int, default=5)
    ap.add_argument('--ollama-model', default='qwen2.5vl:7b')
    ap.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    ap.add_argument('--ollama-timeout', type=int, default=180)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    res = load_results(args.results)
    nusc = NuScenes(version='v1.0-mini', dataroot=args.dataroot, verbose=False)

    # Same ordering the visualisers use: split order, then [start:start+n].
    from nuscenes.utils import splits
    want = set(getattr(splits, args.split.replace('mini_', 'mini_')))
    tokens = []
    for sc in nusc.scene:
        if sc['name'] not in want:
            continue
        tok = sc['first_sample_token']
        while tok:
            tokens.append(tok)
            tok = nusc.get('sample', tok)['next']
    tokens = [t for t in tokens if t in res][args.start:args.start + args.max_frames]
    print(f'[INFO] {len(tokens)} frames from {args.results}')

    # Map layer per scene location, so the raster carries the same road context
    # BEVFormer's does. None would silently drop the layer and change the image.
    from nuscenes.map_expansion.map_api import NuScenesMap
    maps, lidar_yaw = {}, {}

    def scene_of(tok):
        return nusc.get('scene', nusc.get('sample', tok)['scene_token'])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_png = out.parent / '_bev_payload.png'
    with out.open('w') as f:
        prev = None
        for i, tok in enumerate(tokens):
            sample = nusc.get('sample', tok)
            lid_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ego = nusc.get('ego_pose', lid_sd['ego_pose_token'])
            sc = scene_of(tok)
            loc = nusc.get('log', sc['log_token'])['location']
            if loc not in maps:
                maps[loc] = NuScenesMap(dataroot=args.dataroot, map_name=loc)
            if sc['token'] not in lidar_yaw:
                cs = nusc.get('calibrated_sensor', lid_sd['calibrated_sensor_token'])
                lidar_yaw[sc['token']] = Quaternion(cs['rotation']).yaw_pitch_roll[0]

            occ = boxes_to_bevformer_tensors(
                res.get(tok, []), float(ego['translation'][0]),
                float(ego['translation'][1]),
                Quaternion(ego['rotation']).yaw_pitch_roll[0],
                lidar_yaw[sc['token']])
            canvas = build_scene_canvas(
                occ, ego, maps[loc],
                patch_origin=(float(ego['translation'][0]),
                              float(ego['translation'][1])),
                score_thr=args.score_thr,
                lidar2ego_yaw=lidar_yaw[sc['token']],
                heading_up=True)
            import cv2 as _cv2
            _cv2.imwrite(str(tmp_png), _cv2.cvtColor(canvas, _cv2.COLOR_RGB2BGR))
            bev = _encode_image(str(tmp_png))

            cam = front_camera_b64(nusc, tok, args.dataroot)
            crop = light_crop_b64(nusc, tok, args.dataroot)
            light = query_light(cam, args.ollama_model, args.ollama_url,
                                args.ollama_timeout, crop_b64=crop)

            items = [it for it in boxes_to_items(nusc, tok, res.get(tok, []))
                     if it[5] >= args.score_thr]
            det = _format_detection_rows(items, max_rows=args.max_rows)
            # Same four channels BEVFormer sends, in the same order: BEV raster
            # first, forward camera second, light state as text (from the
            # map-projected crop in stage 1), detections as the metric block.
            raw = _query_ollama_streaming(
                [bev, cam], args.ollama_model, args.ollama_url, args.ollama_timeout,
                on_update=lambda _t: None,
                # prev IS carried here: this renders a continuous clip, where the
                # shipped pipeline's hysteresis behaviour is what a viewer should
                # see. The ablations deliberately drop it to keep frames
                # independent; a demo is not an ablation.
                prompt=build_prompt(det, with_front_cam=True,
                                    light_state=light, prev=prev),
                temperature=args.temperature, seed=args.seed)
            reasoning, decision, _ = _parse_response(raw)
            prev = {'decision': decision, 'light': light}

            rec = {'token': tok, 'light': light, 'decision': decision,
                   'reasoning': reasoning, 'n_det': len(items)}
            f.write(json.dumps(rec) + '\n')
            f.flush()
            print(f'  [{i + 1}/{len(tokens)}] light={light:6s} {decision:10s} '
                  f'({len(items)} det)')
    print(f'[DONE] -> {out}')


if __name__ == '__main__':
    main()
