"""
Visualize all six nuScenes cameras for a chosen sample in a 2x3 grid.

Layout:
  [FRONT_LEFT]  [FRONT]  [FRONT_RIGHT]
  [BACK_LEFT]   [BACK]   [BACK_RIGHT]
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DATAROOT = Path("/Users/trish/Downloads/nuScenes_miniV1.0")
META = DATAROOT / "v1.0-mini"  # pathlib.Path object (represents the directory path)

# Display order: (row, col) → camera channel name
GRID_LAYOUT = [
    ("CAM_FRONT_LEFT",  0, 0),
    ("CAM_FRONT",       0, 1),
    ("CAM_FRONT_RIGHT", 0, 2),
    ("CAM_BACK_LEFT",   1, 0),
    ("CAM_BACK",        1, 1),
    ("CAM_BACK_RIGHT",  1, 2),
]

# Json data loading
def load_json(name: str) -> list[dict]:
    return json.loads((META / name).read_text())


def build_lookup(records: list[dict], key: str = "token") -> dict:
    return {r[key]: r for r in records}


def get_camera_image_paths(sample_token: str) -> dict[str, Path]:
    """Return {channel: absolute image path} for the six cameras of one sample."""
    sample_data = load_json("sample_data.json")

    paths: dict[str, Path] = {}
    for sd in sample_data:
        if sd["sample_token"] != sample_token:
            continue
        fname = sd["filename"]          # e.g. "samples/CAM_FRONT/..."
        if not fname.startswith("samples/CAM"):
            continue
        channel = fname.split("/")[1]   # e.g. "CAM_FRONT"
        paths[channel] = DATAROOT / fname

    return paths


def get_calibration(sample_token: str) -> dict[str, dict]:
    """Return {channel: {intrinsic, rotation, translation}} for the six cameras."""
    sample_data   = load_json("sample_data.json")
    cal_sensors   = build_lookup(load_json("calibrated_sensor.json"))
    sensors       = build_lookup(load_json("sensor.json"))

    # sample -> sample data -> calibrated_sensor -> sensor
    calib: dict[str, dict] = {}
    for sd in sample_data:
        if sd["sample_token"] != sample_token:
            continue
        cs = cal_sensors[sd["calibrated_sensor_token"]]
        s  = sensors[cs["sensor_token"]]
        if s["modality"] != "camera":
            continue
        channel = s["channel"]
        calib[channel] = {
            "intrinsic":    np.array(cs["camera_intrinsic"]),   # 3×3
            "rotation":     cs["rotation"],                     # quaternion [w,x,y,z]
            "translation":  np.array(cs["translation"]),        # [x,y,z] in ego frame
        }
    return calib


def annotate_axis(ax: plt.Axes, channel: str, calib: dict) -> None:
    """Overlay camera name and focal lengths on the image axis."""
    K = calib["intrinsic"]
    fx, fy = K[0, 0], K[1, 1]  # Focal lengths fx/fy in pixels
    label = channel.replace("CAM_", "").replace("_", " ")
    ax.set_title(label, fontsize=10, fontweight="bold", pad=3)
    ax.text(
        0.01, 0.01,
        f"fx={fx:.0f}  fy={fy:.0f}",
        transform=ax.transAxes,
        fontsize=7, color="white",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
    )


def visualize(sample_index: int = 0, show_calib: bool = True) -> None:
    samples = load_json("sample.json")
    if sample_index >= len(samples):
        raise ValueError(f"sample_index {sample_index} out of range (0–{len(samples)-1})")

    sample  = samples[sample_index]
    tok     = sample["token"]
    ts_sec  = sample["timestamp"] / 1e6
    scene_tok = sample["scene_token"]

    scenes  = build_lookup(load_json("scene.json"), key="token")
    scene_name = scenes[scene_tok]["name"]

    img_paths = get_camera_image_paths(tok)
    calib_map = get_calibration(tok) if show_calib else {}

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle(
        f"nuScenes — scene: {scene_name}  |  sample {sample_index}  |  t={ts_sec:.3f}s",
        fontsize=12, fontweight="bold", y=1.0,
    )

    for channel, row, col in GRID_LAYOUT:
        ax = axes[row][col]
        if channel not in img_paths:
            ax.set_visible(False)
            continue

        img = mpimg.imread(img_paths[channel])
        ax.imshow(img)
        ax.axis("off")

        if show_calib and channel in calib_map:
            annotate_axis(ax, channel, calib_map[channel])
        else:
            label = channel.replace("CAM_", "").replace("_", " ")
            ax.set_title(label, fontsize=10, fontweight="bold", pad=3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize nuScenes 6-camera rig")
    parser.add_argument("--sample", type=int, default=0,
                        help="Sample index within v1.0-mini (0-403)")  # --sample selects which sample to visualize (default 0)
    parser.add_argument("--no-calib", action="store_true",
                        help="Skip intrinsic overlay")  # --no-calib hides the intrinsic text 
    args = parser.parse_args()

    
    visualize(sample_index=args.sample, show_calib=not args.no_calib)
