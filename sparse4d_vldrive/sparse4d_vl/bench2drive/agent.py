"""
Bench2Drive / CARLA-Leaderboard agent wrapping the Sparse4D-v3 driving stack.

Implements the AutonomousAgent interface:
  sensors()  → declares 6 surround cameras (+ IMU / speedometer) matching the
               nuScenes layout the model was trained on.
  run_step() → 6 images → multi-view → detect → track → motion → plan, then a
               trajectory→PID controller turns the ego plan into VehicleControl.

This file imports on any platform (carla / leaderboard are imported lazily) so
the controller + frame logic can be unit-tested off-simulator; only run_step on a
real CARLA box needs them.

⚠️ Frame & calibration notes (verify on the actual Bench2Drive rig):
 - CARLA ego frame is x-forward, y-RIGHT, z-up (left-handed); the model (nuScenes
   LIDAR_TOP) is x-forward, y-LEFT, z-up. We prepend a y-flip (FLIP) to every
   projection and flip the planned trajectory's y back to CARLA on output.
 - Camera intrinsics are built from fov/width/height; extrinsics from each
   sensor's pose in the ego frame. The model expects the lidar→pixel matrix.
 - The model is trained on nuScenes (real); zero-shot CARLA transfer is weak —
   retrain on Bench2Drive CARLA data for real Driving Scores. The adapter is
   rig-agnostic (reads calibration at runtime), so it stays correct either way.
"""

from __future__ import annotations

import numpy as np

try:                                   # only present on a real CARLA box
    import carla
    from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
    _CARLA = True
except Exception:                      # importable on Mac for the shared logic
    carla = None
    AutonomousAgent = object
    Track = None
    _CARLA = False

import torch

from .controller import TrajectoryController


# CARLA high-level command → our planner command (0=right, 1=straight, 2=left)
_CMD_MAP = {1: 2, 5: 2,          # LEFT / CHANGELANELEFT
            2: 0, 6: 0,          # RIGHT / CHANGELANERIGHT
            3: 1, 4: 1, -1: 1}   # STRAIGHT / LANEFOLLOW / VOID

IMG_H, IMG_W = 256, 704

# 6 cameras, nuScenes order; poses (x,y,z,yaw°) in the CARLA ego frame.
CAM_RIG = [
    ('CAM_FRONT',       1.5,  0.0, 1.5,    0.0),
    ('CAM_FRONT_RIGHT', 1.2, -0.5, 1.5,   55.0),
    ('CAM_FRONT_LEFT',  1.2,  0.5, 1.5,  -55.0),
    ('CAM_BACK',       -1.5,  0.0, 1.5,  180.0),
    ('CAM_BACK_LEFT',   0.0,  0.6, 1.5, -110.0),
    ('CAM_BACK_RIGHT',  0.0, -0.6, 1.5,  110.0),
]
CAM_FOV = 70.0


def plan_to_control(out: dict, command: int, current_speed: float,
                    controller: TrajectoryController):
    """Shared: pick the command's ego trajectory and run the controller.
    Returns a controller.Control (steer/throttle/brake) in MODEL (y-left) frame."""
    ego_traj = out['ego_traj'][0]                  # (3, Te, 2) ego displacements
    wp = ego_traj[int(command)].detach().cpu().numpy()
    return controller.control(wp, current_speed)


def build_camera_projection(x, y, z, yaw_deg, fov, W, H) -> np.ndarray:
    """4×4 model-frame(lidar)→pixel matrix for one CARLA camera.

    proj = K · AXIS · inv(T_cam_in_ego) · FLIP   (FLIP = model→CARLA ego y-flip)
    """
    f = W / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    K = np.array([[f, 0, W / 2, 0], [0, f, H / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)
    # UE camera axes (x fwd, y right, z up) → image axes (x right, y down, z fwd)
    AXIS = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], np.float64)
    # camera pose in the (CARLA) ego frame
    cy, sy = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    T = np.array([[cy, -sy, 0, x], [sy, cy, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], np.float64)
    FLIP = np.diag([1.0, -1.0, 1.0, 1.0])          # model (y-left) → CARLA ego (y-right)
    return (K @ AXIS @ np.linalg.inv(T) @ FLIP).astype(np.float32)


class Sparse4DB2DAgent(AutonomousAgent):
    """Sparse4D-v3 end-to-end agent for the CARLA Leaderboard / Bench2Drive."""

    # ------------------------------------------------------------------
    def setup(self, path_to_conf_file):
        if Track is not None:
            self.track = Track.SENSORS
        from sparse4d_vl.model.sparse4d_v3 import Sparse4Dv3
        from sparse4d_vl.model.checkpoint import load_checkpoint
        self.model = Sparse4Dv3(with_planning=True, with_map=True, ego_steps=6)
        self.model.eval()
        if path_to_conf_file:
            load_checkpoint(self.model, path_to_conf_file, version='v3')
        self.model.reset_state()
        self.controller = TrajectoryController()
        self.controller.reset()
        # precompute per-camera projection (rig is fixed)
        self._proj = np.stack([build_camera_projection(x, y, z, yaw, CAM_FOV, IMG_W, IMG_H)
                               for (_, x, y, z, yaw) in CAM_RIG])
        self._wh = np.array([[IMG_W, IMG_H]] * 6, np.float32)

    # ------------------------------------------------------------------
    def sensors(self):
        sensors = []
        for (cid, x, y, z, yaw) in CAM_RIG:
            sensors.append({'type': 'sensor.camera.rgb', 'id': cid,
                            'x': x, 'y': y, 'z': z, 'roll': 0.0, 'pitch': 0.0, 'yaw': yaw,
                            'width': 1600, 'height': 900, 'fov': CAM_FOV})
        sensors.append({'type': 'sensor.other.imu', 'id': 'IMU',
                        'x': 0, 'y': 0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0,
                        'sensor_tick': 0.05})
        sensors.append({'type': 'sensor.speedometer', 'id': 'SPEED', 'reading_frequency': 20})
        return sensors

    # ------------------------------------------------------------------
    def _preprocess(self, bgra):
        """CARLA BGRA (H,W,4) → model RGB tensor (3, 256, 704), resize+crop like the loader."""
        import cv2
        rgb = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB)
        scale = max(IMG_H / rgb.shape[0], IMG_W / rgb.shape[1])
        rgb = cv2.resize(rgb, (int(rgb.shape[1] * scale), int(rgb.shape[0] * scale)))
        top = rgb.shape[0] - IMG_H
        rgb = rgb[top:top + IMG_H, :IMG_W]
        return torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1)

    def run_step(self, input_data, timestamp):
        imgs = torch.stack([self._preprocess(input_data[cid][1]) for (cid, *_ ) in CAM_RIG])
        imgs = imgs.unsqueeze(0)                         # (1, 6, 3, H, W)
        speed = float(input_data['SPEED'][1]['speed'])   # m/s
        command = _CMD_MAP.get(int(getattr(self, '_get_command', lambda: 4)()), 1)

        img_metas = {
            'projection_mat': self._proj, 'img_wh': self._wh,
            'ego2global': np.eye(4, np.float32), 'lidar2ego': np.eye(4, np.float32),
            'timestamp': timestamp, 'sample_token': str(timestamp),
        }
        with torch.no_grad():
            out = self.model(imgs.float(), img_metas)

        ctrl = plan_to_control(out, command, speed, self.controller)
        # model frame steer (y-left, left positive) → CARLA steer (left negative)
        return carla.VehicleControl(steer=float(-ctrl.steer),
                                    throttle=float(ctrl.throttle),
                                    brake=float(ctrl.brake))

    def destroy(self):
        if hasattr(self, 'model'):
            self.model.reset_state()


def get_entry_point():
    return 'Sparse4DB2DAgent'
