# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Full environment reproducing the downloaded submission `env_20260901190711.yaml` /
`report.html` (round 2, model_best.pt / play_video.mp4 from
drive-download-20260901T124100Z-1-001.zip), so the checkpoint can be retrained or
replayed against the exact config it was produced with.

Verified by diffing the submitted `env.yaml` against this repo's own auto-dumped
`params/env.yaml` from an actual run (not just spot-checking the rewards section, which
is how the terrain/command gaps below were originally missed). Remaining diffs vs.
this repo's `H1RoughEnvCfg`/`H1Rewards` defaults:

Rewards (rough_env_cfg.py):
    track_lin_vel_xy_exp    1.0 ->  1.5
    ang_vel_xy_l2         -0.05 -> -0.1
    action_rate_l2       -0.005 -> -0.002
    feet_air_time          0.25 -> 0.35
    flat_orientation_l2    -1.0 -> -1.5
    termination_penalty  -200.0 -> -75.0  (submitted value was -150.0, lowered further by hand)
    joint_deviation_hip    -0.2 -> -0.3
    joint_deviation_torso  -0.1 -> -0.15
track_ang_vel_z_exp, dof_torques_l2, dof_acc_l2, dof_pos_limits, feet_slide and
joint_deviation_arms already matched the repo defaults.

Terrain generator sub_terrains (ROUGH_TERRAINS_CFG defaults in parens):
    pyramid_stairs     proportion 0.2  -> 0.15
    pyramid_stairs_inv proportion 0.2  -> 0.15
    boxes              proportion 0.2  -> 0.3, grid_height_range (0.05,0.2) -> (0.15,0.5)
    random_rough       proportion 0.2  -> 0.1
hf_pyramid_slope(_inv) proportions already matched (0.1 each).

Commands (velocity_env_cfg.py / rough_env_cfg.py defaults in parens):
    lin_vel_x range (0.0, 1.0)  -> (-1.0, 1.5)   (submission allows walking backward, faster)
    lin_vel_y range (0.0, 0.0)  -> (-0.5, 0.5)   (submission allows sideways strafing)

Report's own diagnostics flag feet_air_time=0.35 (>0.3) as a hopping/bounding-gait
risk, and fall rate stayed at 26.1% with exploration std=1.537 (still not converged).
"""

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from .rough_env_cfg import H1RoughEnvCfg, H1RoughEnvCfg_PLAY


def _apply_submission_overrides(cfg, match_command_ranges: bool = True):
    # rewards
    cfg.rewards.track_lin_vel_xy_exp.weight = 1.5
    cfg.rewards.ang_vel_xy_l2.weight = -0.1
    cfg.rewards.action_rate_l2.weight = -0.002
    cfg.rewards.feet_air_time.weight = 0.35
    cfg.rewards.flat_orientation_l2.weight = -1.5
    cfg.rewards.termination_penalty.weight = -75.0
    cfg.rewards.joint_deviation_hip.weight = -0.3
    cfg.rewards.joint_deviation_torso.weight = -0.15

    # terrain generator sub-terrain mix
    sub_terrains = cfg.scene.terrain.terrain_generator.sub_terrains
    sub_terrains["pyramid_stairs"].proportion = 0.15
    sub_terrains["pyramid_stairs_inv"].proportion = 0.15
    sub_terrains["boxes"].proportion = 0.3
    sub_terrains["boxes"].grid_height_range = (0.15, 0.5)
    sub_terrains["random_rough"].proportion = 0.1

    # commanded velocity ranges (skipped for the PLAY variant, which deliberately pins
    # a single fixed forward command for a clean, interpretable video — same convention
    # this repo's own H1RoughEnvCfg_PLAY already uses)
    if match_command_ranges:
        cfg.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.5)
        cfg.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)


@configclass
class H1RoughEnvCfg_Imported20260901(H1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_submission_overrides(self)


@configclass
class H1RoughEnvCfg_Imported20260901_PLAY(H1RoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_submission_overrides(self, match_command_ranges=False)


@configclass
class H1RoughEnvCfg_Imported20260901_PLAY_Stairs(H1RoughEnvCfg_Imported20260901_PLAY):
    """Same as `H1RoughEnvCfg_Imported20260901_PLAY`, but every terrain cell is an
    ascending pyramid-stairs patch (instead of the usual 20%-of-cells mix), so a
    recorded video is guaranteed to show the robot climbing stairs."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.sub_terrains = {
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=1.0,
                step_height_range=(0.05, 0.23),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
        }
