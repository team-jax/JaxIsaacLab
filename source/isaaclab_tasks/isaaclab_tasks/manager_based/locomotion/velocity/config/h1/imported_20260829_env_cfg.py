# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward weights reproducing the downloaded submission `env_20260829183920.yaml` /
`report.html` (model_best.pt / policy.onnx in ~/다운로드), so the checkpoint can be
retrained or replayed against the exact reward config it was produced with.

Diff vs. this repo's `H1Rewards` defaults (rough_env_cfg.py):
    termination_penalty  -200.0 -> -20.0
    track_lin_vel_xy_exp    1.0 ->  0.5
    feet_air_time          0.25 ->  0.2
    flat_orientation_l2    -1.0 -> -1.5
    action_rate_l2       -0.005 -> -0.0005
track_ang_vel_z_exp, feet_slide, dof_pos_limits, joint_deviation_* and
dof_torques_l2/dof_acc_l2 already matched the repo defaults.
"""

from isaaclab.utils import configclass

from .rough_env_cfg import H1RoughEnvCfg, H1RoughEnvCfg_PLAY


@configclass
class H1RoughEnvCfg_Imported20260829(H1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.termination_penalty.weight = -20.0
        self.rewards.track_lin_vel_xy_exp.weight = 0.5
        self.rewards.feet_air_time.weight = 0.2
        self.rewards.flat_orientation_l2.weight = -1.5
        self.rewards.action_rate_l2.weight = -0.0005


@configclass
class H1RoughEnvCfg_Imported20260829_PLAY(H1RoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.termination_penalty.weight = -20.0
        self.rewards.track_lin_vel_xy_exp.weight = 0.5
        self.rewards.feet_air_time.weight = 0.2
        self.rewards.flat_orientation_l2.weight = -1.5
        self.rewards.action_rate_l2.weight = -0.0005
