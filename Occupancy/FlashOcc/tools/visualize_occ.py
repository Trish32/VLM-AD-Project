"""FlashOcc occupancy visualisation → GIF (pure PyTorch / MPS).

Runs BEVDetOCC over one nuScenes-mini scene and, per frame, renders a composite:
  left  — the 6 surround cameras (raw input)
  right — a VoxFormer-style forward 3-D voxel render of the predicted occupancy
          (solid, lit cubes; low camera looking along +x so the road recedes to a
          vanishing point). The occupancy is the full 360° grid fused from all 6
          cameras; this view shows its forward region. Ego-centric, so the ego is
          fixed and the world streams toward the camera as it drives.

Example:
    PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive \
        python tools/visualize_occ.py --scene scene-0103 --max-frames 20 --device mps
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import BEVDetOCC, load_flashocc_checkpoint           # noqa: E402
from data.loader import NuScenesOccLoader, CAMS                 # noqa: E402

GRID_CONFIG = {'x': [-40, 40, 0.4], 'y': [-40, 40, 0.4],
               'z': [-1, 5.4, 6.4], 'depth': [1.0, 45.0, 0.5]}

OCC_CLASSES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free']

# Occ3D-nuScenes palette (matches the official FlashOCC vis_occ.py colors_map);
# 17 = free → white, as in the official visualisation.
PALETTE = np.array([
    [  0,   0,   0], [255, 120,  50], [255, 192, 203], [255, 255,   0],
    [  0, 150, 245], [  0, 255, 255], [200, 180,   0], [255,   0,   0],
    [255, 240, 150], [135,  60,   0], [160,  32, 240], [255,   0, 255],
    [139, 137, 137], [ 75,   0,  75], [150, 240,  80], [230, 230, 250],
    [  0, 175,   0], [255, 255, 255],
], dtype=np.uint8)

# A compact, fixed legend (the classes that actually carry a scene).
LEGEND_IDS = [11, 13, 14, 15, 16, 4, 10, 7, 1]


def bev_occ_rgb(occ):
    """(200,200,16) class volume → (200,200,3) BEV image, forward-up / left-left.

    Matches the official FlashOCC ``vis_occ.py``: each BEV cell takes the class of
    its **topmost non-free voxel** (the surface seen looking straight down). On an
    open road the topmost voxel of a car's column is the car itself, so vehicles,
    buildings and vegetation read correctly without any special-casing; ``free``
    columns stay white. Nearest-neighbour upscaling keeps the blocky voxel grid.
    """
    free = 17
    occupied = occ != free                       # (X, Y, Z)
    has = occupied.any(2)
    # highest occupied z = first non-free scanning from the top
    z_top = (occ.shape[2] - 1) - occupied[:, :, ::-1].argmax(2)
    cmap = np.take_along_axis(occ, z_top[..., None], axis=2)[..., 0]
    cmap[~has] = free
    rgb = PALETTE[cmap]                           # (X, Y, 3) indexed [x, y]
    # x = forward, y = left → flip both so row0 = front (up), col0 = left.
    return rgb[::-1, ::-1]


def camera_grid(nusc, sample, size=(240, 135)):
    """Raw 6-camera 2×3 grid (FL,F,FR / BL,B,BR)."""
    imgs = []
    for cam in CAMS:
        sd = nusc.get('sample_data', sample['data'][cam])
        path = os.path.join(nusc.dataroot, sd['filename'])
        imgs.append(np.asarray(Image.open(path).convert('RGB').resize(size)))
    return np.vstack([np.hstack(imgs[:3]), np.hstack(imgs[3:])])


def _downsample2(occ):
    """(200,200,16) → (100,100,16), keeping a non-free class when the 2×2 xy block
    has one (so thin objects survive the coarsening)."""
    free = 17
    o = occ.reshape(100, 2, 100, 2, 16).transpose(0, 2, 4, 1, 3).reshape(100, 100, 16, 4)
    nonfree = o != free
    has = nonfree.any(-1)
    chosen = np.take_along_axis(o, nonfree.argmax(-1)[..., None], -1)[..., 0]
    chosen[~has] = free
    return chosen


# Cube-face templates: (axis, dir, 4 corner offsets, shade) — top brightest,
# bottom darkest, sides mid → the lit-cube look of the official renderer.
_FACES = [
    (2,  1, [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], 1.00),   # +z top
    (2, -1, [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)], 0.55),   # -z bottom
    (0,  1, [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)], 0.85),   # +x
    (0, -1, [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)], 0.85),   # -x
    (1,  1, [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)], 0.70),   # +y
    (1, -1, [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], 0.70),   # -y
]


def voxel_faces(occ):
    """Exposed cube faces of the volume → (quads (M,4,3), facecolors (M,3)).

    Only faces between an occupied voxel and a *free* neighbour are emitted (the
    visible shell, as real cubes). Out-of-grid neighbours are treated as occupied
    so the 80 m grid-boundary walls are never drawn. Each face is shaded by its
    orientation to give solid, lit cubes rather than a flat point cloud.
    """
    free = 17
    F = occ != free
    quads, cols = [], []
    for axis, d, corners, shade in _FACES:
        nbr = np.ones_like(F)                      # out-of-grid = occupied
        dst = [slice(None)] * 3; src = [slice(None)] * 3
        n = F.shape[axis]
        if d > 0:
            dst[axis] = slice(0, n - 1); src[axis] = slice(1, n)
        else:
            dst[axis] = slice(1, n); src[axis] = slice(0, n - 1)
        nbr[tuple(dst)] = F[tuple(src)]
        exp = F & ~nbr                              # face shows where neighbour free
        xs, ys, zs = np.nonzero(exp)
        if len(xs) == 0:
            continue
        base = np.stack([xs, ys, zs], 1)[:, None, :]            # (N,1,3)
        quads.append(base + np.array(corners)[None])           # (N,4,3)
        cols.append(PALETTE[occ[xs, ys, zs]] / 255.0 * shade)
    return np.concatenate(quads), np.concatenate(cols)


def render(out_path, cam_grid, occ, frame_idx, scene, occ_pct):
    fig = plt.figure(figsize=(12, 5), dpi=120)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.02)

    axc = fig.add_subplot(gs[0]); axc.imshow(cam_grid); axc.axis("off")
    axc.set_title("6 surround cameras (input)", color="black", fontsize=10)

    # ---- VoxFormer-style forward-driving voxel-cube render ---------------
    # Ego-centric: the ego is fixed, the world streams toward the camera as it
    # drives (motion shows as optical flow of the cubes across the clip).
    axv = fig.add_subplot(gs[1], projection="3d")
    axv.set_facecolor("white")
    quads, cols = voxel_faces(_downsample2(occ))             # (100,100,16)
    axv.add_collection3d(Poly3DCollection(quads, facecolors=cols,
                                          edgecolors=(0, 0, 0, 0.12), linewidths=0.1))
    try:
        axv.set_proj_type("persp", focal_length=0.55)
    except TypeError:
        axv.set_proj_type("persp")
    # low camera looking along +x (forward); road recedes to a vanishing point
    axv.view_init(elev=14, azim=180)
    axv.set_box_aspect((50, 84, 22))
    axv.set_xlim(50, 100); axv.set_ylim(8, 92); axv.set_zlim(0, 14)
    axv.set_axis_off()
    axv.set_title("predicted 3-D occupancy — voxels (forward)", color="black",
                  fontsize=10, y=0.92)

    handles = [Patch(facecolor=PALETTE[i] / 255.0, edgecolor="none",
                     label=OCC_CLASSES[i]) for i in LEGEND_IDS]
    axv.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=6.5, facecolor="white", edgecolor="#cccccc",
               labelcolor="black", framealpha=0.95)

    fig.text(0.012, 0.96, f"{scene}  frame {frame_idx:02d}   occupied {occ_pct:.0f}%",
             color="black", fontsize=9, family="monospace", va="top")
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def make_gif(paths, gif_path, duration_ms=400, max_height=360):
    frames = [Image.open(p).convert("RGB") for p in paths]
    if max_height:
        frames = [f.resize((round(f.width * max_height / f.height), max_height))
                  if f.height > max_height else f for f in frames]
    pframes = [f.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.NONE)
               for f in frames]
    pframes[0].save(gif_path, save_all=True, append_images=pframes[1:],
                    duration=duration_ms, loop=0, disposal=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', default='/Users/trish/Downloads/nuScenes_miniV1.0')
    ap.add_argument('--ckpt', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'model/checkpoints/flashocc-r50-256x704.pth'))
    ap.add_argument('--device', default='mps')
    ap.add_argument('--scene', default='scene-0103')
    ap.add_argument('--max-frames', type=int, default=20)
    ap.add_argument('--out', default='occ_outputs')
    args = ap.parse_args()

    device = torch.device(args.device if (
        args.device != 'mps' or torch.backends.mps.is_available()) else 'cpu')
    model = BEVDetOCC(grid_config=GRID_CONFIG, input_size=(256, 704),
                      numC_Trans=64, num_classes=18, Dz=16)
    load_flashocc_checkpoint(model, args.ckpt)
    model.eval().to(device)

    loader = NuScenesOccLoader(args.dataroot)
    idxs = loader.scene_indices(args.scene)[:args.max_frames]
    print(f'[occ-viz] {args.scene}: {len(idxs)} frames  device {device}')

    outdir = args.out
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(outdir):
        if f.startswith('occ_') and f.endswith('.png'):
            os.remove(os.path.join(outdir, f))
    paths = []
    for j, idx in enumerate(idxs):
        inputs = tuple(t.to(device) for t in loader.get_batched(idx))
        occ = model.predict_occ(inputs)[0].numpy()             # (200,200,16)
        occ_pct = 100.0 * (occ != 17).mean()
        cam_grid = camera_grid(loader.nusc, loader.samples[idx])
        p = os.path.join(outdir, f'occ_{j:03d}.png')
        render(p, cam_grid, occ, j, args.scene, occ_pct)
        paths.append(p)
        print(f'  frame {j:02d}: occupied {occ_pct:4.1f}%')

    gif = os.path.join(outdir, f'{args.scene}_occ.gif')
    make_gif(paths, gif)
    print(f'[occ-viz] wrote {gif}  ({len(paths)} frames)')


if __name__ == '__main__':
    main()
