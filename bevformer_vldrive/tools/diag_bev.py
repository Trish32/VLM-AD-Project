#!/usr/bin/env python3
"""BEV diagnostic: GT (white outlines) vs predictions (colored outlines) overlaid."""
import math, sys, numpy as np, cv2, torch
from pathlib import Path
from pyquaternion import Quaternion

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))

from model import BEVFormerTiny
from data  import NuScenesMiniLoader
from eval  import _build_remap
from visualizer import (_global_to_pixel, _box_corners_global,
                        _draw_drivable_area, GROUP_COLORS, CLASS_GROUP, _BG_COLOR)

PC = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
NUSC_CAT = {
    'vehicle':                    'vehicle',
    'human.pedestrian':           'pedestrian',
    'movable_object.barrier':     'barrier',
    'movable_object.trafficcone': 'traffic_cone',
}

def main():
    model = BEVFormerTiny(pretrained_backbone=False); model.eval()
    ckpt  = torch.load('model/checkpoints/bevformer_tiny_fp16_epoch_24.pth', map_location='cpu')
    model.load_state_dict(_build_remap(ckpt.get('state_dict', ckpt)), strict=False)

    loader = NuScenesMiniLoader('/Users/trish/Downloads/nuScenes_miniV1.0')
    nusc   = loader.nusc

    prev_bev = None
    for fi, sample in enumerate(loader.iter_scene(0)):
        if fi >= 2: break
        with torch.no_grad():
            out = model(sample['imgs'], sample['img_metas'], prev_bev=prev_bev)
        prev_bev = out['bev_feat'].detach()
        if fi == 1:   # frame 1 has car detections
            cls_out      = out['cls_logits'][0].cpu()
            reg_out      = out['reg_preds'][0].cpu()
            ref_out      = out['ref_pts'][0].cpu()
            sample_token = sample['img_metas'][0]['sample_token']

    nusc_s   = nusc.get('sample', sample_token)
    lidar_sd = nusc.get('sample_data', nusc_s['data']['LIDAR_TOP'])
    lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

    ego_tx, ego_ty = ego_pose['translation'][:2]
    ego_yaw = Quaternion(ego_pose['rotation']).yaw_pitch_roll[0]
    yaw_l2e   = Quaternion(lidar_cs['rotation']).yaw_pitch_roll[0]
    yaw_total = ego_yaw + yaw_l2e
    cos_t, sin_t = math.cos(yaw_total), math.sin(yaw_total)

    print(f"ego_yaw={math.degrees(ego_yaw):.1f}°  yaw_l2e={math.degrees(yaw_l2e):.4f}°"
          f"  yaw_total={math.degrees(yaw_total):.1f}°")

    CSIZE = 700; CRANGE = 140.0
    scale = CSIZE / CRANGE; chalf = CSIZE // 2
    patch_ox, patch_oy = ego_tx, ego_ty

    try:
        from nuscenes.map_expansion.map_api import NuScenesMap
        scene    = nusc.get('scene', nusc_s['scene_token'])
        loc      = nusc.get('log', scene['log_token'])['location']
        nmap     = NuScenesMap('/Users/trish/Downloads/nuScenes_miniV1.0', loc)
    except Exception as e:
        print(f"[WARN] map: {e}"); nmap = None

    canvas = np.full((CSIZE, CSIZE, 3), _BG_COLOR, dtype=np.uint8)
    _draw_drivable_area(canvas, nmap, patch_ox, patch_oy, CRANGE, CSIZE)

    def draw_box(col_c, row_c, l_px, w_px, yaw, color, thick=2):
        if not (-30 <= col_c < CSIZE+30 and -30 <= row_c < CSIZE+30):
            return
        corn = _box_corners_global(col_c, row_c, l_px, w_px, yaw)
        cv2.polylines(canvas, [corn], True, color, thick, cv2.LINE_AA)

    # ── GT boxes: white outlines ──────────────────────────────────────────────
    for ann_tok in nusc_s['anns']:
        ann = nusc.get('sample_annotation', ann_tok)
        cat = ann['category_name']
        grp = next((g for p, g in NUSC_CAT.items() if cat.startswith(p)), None)
        if grp is None: continue
        tx, ty   = ann['translation'][:2]
        w_a, l_a = ann['size'][0], ann['size'][1]  # nuScenes [width, length, h]
        yaw_a    = Quaternion(ann['rotation']).yaw_pitch_roll[0]
        col_c, row_c = _global_to_pixel(tx, ty, patch_ox, patch_oy, chalf, scale)
        draw_box(col_c, row_c, max(l_a*scale, 4), max(w_a*scale, 3), yaw_a,
                 (255, 255, 255), 2)
        label = cat.split('.')[-1][:5]
        cv2.putText(canvas, label, (col_c, row_c - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)

    # ── Predicted boxes: group colors ─────────────────────────────────────────
    scores, labels = cls_out.float().sigmoid().max(-1)
    order = scores.argsort(descending=True)[:200]
    keep  = [int(i) for i in order if float(scores[i]) > 0.25]
    print(f"Predictions above 0.25: {len(keep)}")
    for idx in keep:
        r   = reg_out[idx].float(); p = ref_out[idx].float()
        lbl = int(labels[idx])
        grp = CLASS_GROUP[lbl % len(CLASS_GROUP)]
        fill = GROUP_COLORS[grp]
        x_lid = float(p[0]) * (PC[3]-PC[0]) + PC[0]
        y_lid = float(p[1]) * (PC[4]-PC[1]) + PC[1]
        gx = cos_t * x_lid - sin_t * y_lid + ego_tx
        gy = sin_t * x_lid + cos_t * y_lid + ego_ty
        w_m = float(r[2].exp().clamp(0.2, 20))
        l_m = float(r[3].exp().clamp(0.4, 20))
        yaw_lid = math.atan2(float(r[6]), float(r[7]))
        yaw_glo = yaw_total - yaw_lid - math.pi / 2
        col_c, row_c = _global_to_pixel(gx, gy, patch_ox, patch_oy, chalf, scale)
        draw_box(col_c, row_c, max(l_m*scale, 4), max(w_m*scale, 3), yaw_glo, fill, 1)

    # ── Ego ───────────────────────────────────────────────────────────────────
    col_e, row_e = _global_to_pixel(ego_tx, ego_ty, patch_ox, patch_oy, chalf, scale)
    cv2.circle(canvas, (col_e, row_e), 10, (255, 255, 255), -1)
    tip_c = int(col_e + math.cos(ego_yaw) * 25)
    tip_r = int(row_e - math.sin(ego_yaw) * 25)
    cv2.arrowedLine(canvas, (col_e, row_e), (tip_c, tip_r),
                    (255, 255, 255), 2, tipLength=0.3, line_type=cv2.LINE_AA)

    out_path = 'bev_outputs/diag_bev.png'
    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"Saved {out_path}   WHITE=GT  COLORED=pred")

if __name__ == '__main__':
    main()
