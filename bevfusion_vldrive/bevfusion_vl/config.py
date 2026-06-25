"""
Config for the pure-PyTorch MPS port of MIT-HAN-LAB BEVFusion (det + seg).
Mirrors the resolved YAML hierarchy. No mmcv/mmdet3d. See CLAUDE.md.
"""
from __future__ import annotations

DATAROOT = '/Users/trish/Downloads/nuScenes_miniV1.0'
VERSION = 'v1.0-mini'
CKPT_DIR = '/Users/trish/VLMProjects/bevfusion_vldrive/bevfusion_vl/model/checkpoints'
CKPT_DET = f'{CKPT_DIR}/bevfusion-det.pth'
CKPT_SEG = f'{CKPT_DIR}/bevfusion-seg.pth'

# Classes
OBJECT_CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]
MAP_CLASSES = [
    'drivable_area', 'ped_crossing', 'walkway', 'stop_line',
    'carpark_area', 'divider',
]

# Image
IMAGE_SIZE = (256, 704)          # H, W
NUM_VIEWS = 6
IMG_MEAN = [0.485, 0.456, 0.406]  # ImageNet, applied to [0,1] RGB
IMG_STD = [0.229, 0.224, 0.225]
TEST_RESIZE_LIM = (0.48, 0.48)   # ImageAug3D test resize
CAM_NAMES = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]

# Points
LOAD_DIM = 5
USE_DIM = 5                       # [x, y, z, intensity, dt]
SWEEPS_NUM = 9

# Shared SparseEncoder architecture
SE_IN_CHANNELS = 5
SE_BASE_CHANNELS = 16
SE_OUTPUT_CHANNELS = 128
SE_ENCODER_CHANNELS = ((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128))
SE_ENCODER_PADDINGS = ((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0))

# Shared decoder
SECOND_IN = 256
SECOND_OUT = (128, 256)
SECOND_LAYER_NUMS = (5, 5)
SECOND_STRIDES = (1, 2)
SECONDFPN_IN = (128, 256)
SECONDFPN_OUT = (256, 256)
SECONDFPN_UP = (1, 2)

# Swin (camera backbone)
SWIN = dict(embed_dims=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
            window_size=7, mlp_ratio=4, out_indices=(1, 2, 3))


class DetCfg:
    TASK = 'det'
    IMAGE_SIZE = IMAGE_SIZE
    NUM_VIEWS = NUM_VIEWS
    VOXEL_SIZE = [0.075, 0.075, 0.2]
    POINT_CLOUD_RANGE = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
    SPARSE_SHAPE = [1440, 1440, 41]          # x, y, z
    MAX_NUM_POINTS = 10
    MAX_VOXELS = 160000                       # test
    # camera vtransform (DepthLSSTransform)
    VT_TYPE = 'DepthLSSTransform'
    VT_IN = 256
    VT_OUT = 80
    XBOUND = [-54.0, 54.0, 0.3]
    YBOUND = [-54.0, 54.0, 0.3]
    ZBOUND = [-10.0, 10.0, 20.0]
    DBOUND = [1.0, 60.0, 0.5]
    VT_DOWNSAMPLE = 2
    GRID_SIZE = [1440, 1440, 41]
    CLASSES = OBJECT_CLASSES


class SegCfg:
    TASK = 'seg'
    IMAGE_SIZE = IMAGE_SIZE
    NUM_VIEWS = NUM_VIEWS
    VOXEL_SIZE = [0.1, 0.1, 0.2]
    POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    SPARSE_SHAPE = [1024, 1024, 41]
    MAX_NUM_POINTS = 10
    MAX_VOXELS = 120000
    VT_TYPE = 'LSSTransform'
    VT_IN = 256
    VT_OUT = 80
    XBOUND = [-51.2, 51.2, 0.4]
    YBOUND = [-51.2, 51.2, 0.4]
    ZBOUND = [-10.0, 10.0, 20.0]
    DBOUND = [1.0, 60.0, 0.5]
    VT_DOWNSAMPLE = 2
    # seg head grid transform
    SEG_INPUT_SCOPE = [[-51.2, 51.2, 0.8], [-51.2, 51.2, 0.8]]
    SEG_OUTPUT_SCOPE = [[-50, 50, 0.5], [-50, 50, 0.5]]
    CLASSES = MAP_CLASSES
