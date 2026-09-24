"""Extract (features, residual) pairs from logged agent motion and fit.

Target = actual agent velocity at t+1 minus what the IDM prior predicts. Purely
supervised from recordings: the question is "what did this agent do", not "what
would it have done", so no counterfactuals are required.

Usage:
    PYTHONPATH=. python e2e_pipeline/fit_residual.py
"""
import os

import numpy as np
from nuscenes import NuScenes

from e2e_pipeline.closed_loop import GTWorldModel
from e2e_pipeline.residual import ResidualDynamics, agent_features
from e2e_pipeline.world_model import (IDM_A_MAX, IDM_B, IDM_S0, IDM_T,
                                      LANE_HALF_WIDTH)

DT = 0.5


def idm_next_velocity(agent, ego_speed):
    """What the prior predicts for this agent, and the decel it applied."""
    v = np.asarray(agent.vxy, dtype=np.float64)
    speed = float(np.linalg.norm(v))
    if speed < 0.5:
        return v, 0.0
    fwd = v / speed
    lat = np.array([-fwd[1], fwd[0]])
    rel = -np.asarray(agent.xy, dtype=np.float64)     # ego from the agent
    gap, offset = float(rel @ fwd), abs(float(rel @ lat))
    decel = 0.0
    if gap > 0.0 and offset <= LANE_HALF_WIDTH:
        s_star = IDM_S0 + max(0.0, speed * IDM_T)
        decel = min(IDM_A_MAX * (s_star / max(gap, 0.5)) ** 2, IDM_B * 3.0)
    return fwd * max(0.0, speed - decel * DT), decel


def collect(nusc, scenes):
    X, Y = [], []
    for si in scenes:
        w = GTWorldModel(nusc, scene_idx=si)
        for k in range(len(w.samples) - 1):
            fr, nx = w.samples[k], w.samples[k + 1]
            ego_v = float(np.linalg.norm(nx['xy'] - fr['xy']) / DT)
            now = {a.track_id: a for a in w.agents_at(k * DT, fr['xy'], fr['yaw'])}
            later = {a.track_id: a for a in w.agents_at((k + 1) * DT, fr['xy'], fr['yaw'])}
            for tid, a in now.items():
                b = later.get(tid)
                if b is None:
                    continue
                pred_v, decel = idm_next_velocity(a, ego_v)
                X.append(agent_features(a, ego_v, decel))
                Y.append(np.asarray(b.vxy, dtype=np.float64) - pred_v)
    return np.array(X), np.array(Y)


def main():
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'),
                    verbose=False)
    for tr, te, tag in (([0, 1, 2, 3, 4], [5, 6, 7, 8, 9], 'train 0-4 / test 5-9'),
                        ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4], 'train 5-9 / test 0-4')):
        Xtr, Ytr = collect(nusc, tr)
        Xte, Yte = collect(nusc, te)
        m = ResidualDynamics().fit(Xtr, Ytr, epochs=400)
        print(f'\n  {tag}   train {len(Xtr)}  test {len(Xte)}')
        print('   train:', m.evaluate(Xtr, Ytr).describe())
        print('   TEST :', m.evaluate(Xte, Yte).describe())


if __name__ == '__main__':
    main()
