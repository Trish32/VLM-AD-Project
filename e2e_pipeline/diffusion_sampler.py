"""Candidate generators with real lateral diversity, and what the denoiser adds.

WHY THE OBVIOUS FRAMING IS WRONG. "The denoiser samples from noise rather than
fixed anchors, so it produces diverse trajectories" is true of a vanilla
diffusion planner and false of DiffusionDrive specifically. Its headline is
TRUNCATED diffusion: inference seeds `add_noise(anchor, noise, t=8)` and takes
two steps. Measured on the real schedule:

    t=8    noise coefficient 0.0316  ->  0.095 m lateral, 0.128 m longitudinal
    t=40   noise coefficient 0.0814  ->  0.244 m lateral, 0.329 m longitudinal

t=8 is the inference seed and t=40 is the training maximum, so 0.095 m is the
diversity the sampling actually injects -- about 0.23 m of endpoint spread over
six steps. Upstream says so directly: "magic number 40 means that we add little
noise for each anchor". The noise refines; it does not diversify.

WHERE THE DIVERSITY ACTUALLY IS. The vocabulary is (3 commands x 6 anchors), and
`diffusiondrive_anchor_planner` hardcodes command='straight' -- index 2:

    command      lateral endpoint spread
    right (0)     5.7 m
    left  (1)     8.7 m
    straight (2)  0.5 m     <- the only one the pipeline ever proposes
    all 18       19.9 m

So the planner has been choosing among six trajectories that differ by half a
metre laterally, in scenes whose occlusion is lateral. That is a 40x gap against
the vocabulary it already has on disk, and it costs nothing to close.

This module provides both levers separately so their contributions can be told
apart: `commands` widens the candidate set, `n_diffusion_samples` adds the real
truncated-diffusion perturbation on top.
"""
from __future__ import annotations

import numpy as np

from .scene import SceneRepresentation


def multi_command_planner(anchor_npy: str, dt: float = 0.5,
                          commands: tuple = (2,),
                          n_diffusion_samples: int = 0,
                          seed: int = 0,
                          speed_condition: bool = True,
                          max_accel: float = 3.0):
    """Candidates from several command modes, optionally diffusion-perturbed.

    `commands=(2,)` reproduces `diffusiondrive_anchor_planner` exactly, so the
    baseline arm is the same code path rather than a re-implementation of it.
    """
    from .vlm_planner import INDEX_COMMAND, DrivingIntent, intent_conditioned_planner

    inner = intent_conditioned_planner(anchor_npy, dt=dt, max_accel=max_accel)
    rng = np.random.default_rng(seed)

    _ddim = None
    if n_diffusion_samples > 0:
        from diffusiondrive_planner.truncated_diffusion import TruncatedDDIM
        _ddim = TruncatedDDIM()

    def _perturb(cands: np.ndarray) -> np.ndarray:
        """One truncated-diffusion sample around each candidate.

        Runs in upstream's own normalized delta space with its own schedule, so
        the perturbation size is the real one rather than a chosen epsilon.
        Frames differ: this package is (forward, left), upstream is
        (lateral, forward) -- the same conversion `intent_conditioned_planner`
        does at load, inverted.
        """
        import torch

        from diffusiondrive_planner.truncated_diffusion import (
            anchor_to_deltas, denormalize_traj)
        up = np.stack([-cands[..., 1], cands[..., 0]], axis=-1)
        t = torch.as_tensor(up, dtype=torch.float32)
        deltas = anchor_to_deltas(t)
        noise = torch.as_tensor(rng.standard_normal(tuple(deltas.shape)),
                                dtype=torch.float32)
        seeded = _ddim.seed_from_anchor(deltas, noise)
        samp = denormalize_traj(seeded).cumsum(dim=-2).numpy()
        return np.stack([samp[..., 1], -samp[..., 0]], axis=-1)

    def plan(scene: SceneRepresentation, command: int):
        v = max(scene.ego.speed, 2.0) if speed_condition else scene.ego.speed
        out, sc = [], []
        for ci in commands:
            base, scores = inner(
                scene, DrivingIntent(command=INDEX_COMMAND[ci],
                                     target_speed_mps=v))
            base = np.asarray(base, dtype=np.float64)
            scores = (np.asarray(scores, dtype=np.float64)
                      if scores is not None else np.ones(len(base)))
            out.append(base)
            sc.append(scores)
            for _ in range(n_diffusion_samples):
                out.append(_perturb(base))
                sc.append(scores)
        return np.concatenate(out, axis=0), np.concatenate(sc, axis=0)

    return plan


def lateral_spread(planner, scene, command: int = 2) -> dict:
    """Diagnostic: how far apart are this planner's candidates, actually."""
    cands, _ = planner(scene, command)
    c = np.asarray(cands, dtype=np.float64)
    end = c[:, -1, :]
    return {
        'n': int(len(c)),
        'lateral_spread_m': float(end[:, 1].max() - end[:, 1].min()),
        'forward_spread_m': float(end[:, 0].max() - end[:, 0].min()),
        'mean_pairwise_m': float(np.mean([
            np.linalg.norm(end[i] - end[j])
            for i in range(len(end)) for j in range(i + 1, len(end))])
            if len(end) > 1 else 0.0),
    }
