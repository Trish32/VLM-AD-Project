"""
Hyper-parameters for BEVFusion PointPillars config `bevf_pp_2x8_1x_nusc.py`,
mirrored from the official ADLab-AutoDrive repo so the pure-PyTorch port stays
in lock-step with the released checkpoint.

Pure PyTorch / MPS only — no mmcv / mmdet3d. See CLAUDE.md.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
POINT_CLOUD_RANGE = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
VOXEL_SIZE = [0.25, 0.25, 8.0]
# grid sizes derived from range / voxel_size
#   nx = (50-(-50))/0.25 = 400, ny = 400, nz = (3-(-5))/8 = 1
GRID_SIZE = [400, 400, 1]          # x, y, z  (PointPillars: single z-pillar)
SPARSE_SHAPE = [1, 400, 400]       # z, y, x for scatter output_shape=[400,400]

# camera lift-splat grid (grid=0.5)
LSS_GRID = 0.5
FINAL_DIM = (900, 1600)            # original camera resolution (frustum coords)
DOWNSAMPLE = 8                     # FPNC output stride => feat (112, 200)
IMC = 256                          # camera BEV channel dim
LIC = 384                          # lidar BEV channel dim
CAMERA_DEPTH_RANGE = [4.0, 45.0, 1.0]   # D = 41 depth bins

# ---------------------------------------------------------------------------
# Classes — order MUST match the checkpoint meta (NuScenesDataset default)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
]
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Voxelization
# ---------------------------------------------------------------------------
MAX_NUM_POINTS = 64
MAX_VOXELS_TEST = 40000
MAX_VOXELS_TRAIN = 30000
POINT_DIM = 4                      # [x, y, z, dt]  (intensity dropped, dt added)
SWEEPS_NUM = 10

# ---------------------------------------------------------------------------
# Image preprocessing  (MyResize keep_ratio -> MyNormalize -> MyPad /32)
# ---------------------------------------------------------------------------
IMG_SCALE = (800, 448)            # (W, H) target bound, keep aspect ratio
SIZE_DIVISOR = 32
IMG_MEAN = [123.675, 116.28, 103.53]   # RGB
IMG_STD = [58.395, 57.12, 57.375]
NUM_VIEWS = 6
CAM_NAMES = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]

# ---------------------------------------------------------------------------
# Anchor3DHead
# ---------------------------------------------------------------------------
ANCHOR_RANGES = [
    [-49.6, -49.6, -1.80032795, 49.6, 49.6, -1.80032795],
    [-49.6, -49.6, -1.74440365, 49.6, 49.6, -1.74440365],
    [-49.6, -49.6, -1.68526504, 49.6, 49.6, -1.68526504],
    [-49.6, -49.6, -1.67339111, 49.6, 49.6, -1.67339111],
    [-49.6, -49.6, -1.61785072, 49.6, 49.6, -1.61785072],
    [-49.6, -49.6, -1.80984986, 49.6, 49.6, -1.80984986],
    [-49.6, -49.6, -1.763965, 49.6, 49.6, -1.763965],
]
ANCHOR_SIZES = [
    [1.95017717, 4.60718145, 1.72270761],   # car
    [2.4560939, 6.73778078, 2.73004906],     # truck
    [2.87427237, 12.01320693, 3.81509561],   # trailer
    [0.60058911, 1.68452161, 1.27192197],    # bicycle
    [0.66344886, 0.7256437, 1.75748069],     # pedestrian
    [0.39694519, 0.40359262, 1.06232151],    # traffic_cone
    [2.49008838, 0.48578221, 0.98297065],    # barrier
]
ANCHOR_ROTATIONS = [0.0, 1.57]
ANCHOR_CUSTOM_VALUES = [0.0, 0.0]            # vx, vy
CODE_SIZE = 9
DIR_OFFSET = 0.7854                          # pi/4
DIR_LIMIT_OFFSET = 0.0

# ---------------------------------------------------------------------------
# Test / NMS config
# ---------------------------------------------------------------------------
TEST_CFG = dict(
    use_rotate_nms=True,
    nms_across_levels=False,
    nms_pre=1000,
    nms_thr=0.2,
    score_thr=0.05,
    min_bbox_size=0,
    max_num=500,
)

# ---------------------------------------------------------------------------
# Train assigner / loss config (for fine-tune)
# ---------------------------------------------------------------------------
TRAIN_CFG = dict(
    pos_iou_thr=0.6,
    neg_iou_thr=0.3,
    min_pos_iou=0.3,
    code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
)

# ---------------------------------------------------------------------------
# Data root
# ---------------------------------------------------------------------------
DATAROOT = '/Users/trish/Downloads/nuScenes_miniV1.0'
VERSION = 'v1.0-mini'
CHECKPOINT = (
    '/Users/trish/VLMProjects/bevfusion_vldrive/'
    'BEVFusion_vl/model/checkpoints/bevf_pp_2x8_1x_nusc.pth'
)
