"""Resolve & load Occ3D-nuScenes occupancy ground truth (labels.npz).

The GT is NOT part of raw nuScenes-mini.  Download the `gts` folder from the
CVPR2023-3D-Occupancy-Prediction benchmark (or the gdrive link in FlashOcc's
doc/install.md) and point ``--occ-root`` at it.  Layout::

    <occ_root>/<scene_name>/<sample_token>/labels.npz
        semantics   (200, 200, 16)  uint8   class id 0..17 (17 = free)
        mask_camera (200, 200, 16)  bool
        mask_lidar  (200, 200, 16)  bool
"""
import os
import numpy as np

DOWNLOAD_HINT = (
    "Occ3D-nuScenes occupancy GT (labels.npz) not found.\n"
    "  Download the 'gts' folder from\n"
    "    https://github.com/CVPR2023-3D-Occupancy-Prediction/"
    "CVPR2023-3D-Occupancy-Prediction\n"
    "  or the gdrive in FlashOCC/doc/install.md:\n"
    "    https://drive.google.com/file/d/1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY\n"
    "  then pass --occ-root <dir containing scene-*/<token>/labels.npz>")


def find_label_path(occ_root, scene_name, sample_token):
    """Return path to labels.npz for a sample, or None if absent."""
    # canonical layout
    p = os.path.join(occ_root, scene_name, sample_token, 'labels.npz')
    if os.path.exists(p):
        return p
    # some releases drop the scene level
    p = os.path.join(occ_root, sample_token, 'labels.npz')
    if os.path.exists(p):
        return p
    return None


def load_label(path):
    z = np.load(path)
    return (z['semantics'],
            z['mask_lidar'].astype(bool),
            z['mask_camera'].astype(bool))


def coverage(loader, occ_root, indices):
    """Return dict index -> labels.npz path for samples that have GT."""
    found = {}
    for i in indices:
        s = loader.samples[i]
        scene = loader.nusc.get('scene', s['scene_token'])['name']
        path = find_label_path(occ_root, scene, s['token'])
        if path is not None:
            found[i] = path
    return found
