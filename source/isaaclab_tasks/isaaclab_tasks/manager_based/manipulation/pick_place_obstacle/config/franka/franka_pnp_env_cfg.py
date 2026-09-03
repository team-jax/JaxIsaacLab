# SPDX-License-Identifier: BSD-3-Clause
"""Franka Panda placeholder config for pick-and-place + obstacle avoidance.

!! PLACEHOLDER !!
This uses Franka Panda (7 DOF, 2-finger) as a stand-in for the actual robot.
When the actual 5-DOF + 3-finger URDF is ready, replace the items marked
with  # [SWAP]  below:

  1. Import your robot's ArticulationCfg instead of FRANKA_PANDA_CFG.
  2. Update arm_action joint_names regex to match your arm joints.
  3. Update gripper_action joint_names / open/close commands.
  4. Update ee_frame prim_path to the correct wrist link name.
  5. Update OffsetCfg pos to the correct palm TCP offset.
  6. Update object_dims / obstacle_dims in ObservationsCfg to match.
  7. Verify workspace ranges in EventCfg match the new robot's reach.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pick_place_obstacle import mdp
from isaaclab_tasks.manager_based.manipulation.pick_place_obstacle.pick_place_obstacle_env_cfg import (
    PnPObstacleEnvCfg,
)

# [SWAP] Replace with your robot's ArticulationCfg when URDF is ready
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaPnPObstacleEnvCfg(PnPObstacleEnvCfg):
    """Franka Panda instantiation of pick-and-place + obstacle avoidance."""

    def __post_init__(self):
        super().__post_init__()

        # -- Robot --  [SWAP] Replace FRANKA_PANDA_CFG with your robot cfg
        # Use .copy() (deep copy) so we don't mutate the shared FRANKA_PANDA_CFG object.
        # activate_contact_sensors is required for ee_contact ContactSensor to work.
        _robot_cfg = FRANKA_PANDA_CFG.copy()
        _robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        _robot_cfg.spawn.activate_contact_sensors = True
        self.scene.robot = _robot_cfg

        # -- Arm action: joint position control --
        # [SWAP] Update joint_names regex when robot changes (e.g., "joint_[1-5]")
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],  # [SWAP]
            scale=0.5,
            use_default_offset=True,
        )

        # -- Gripper action: binary open / close --
        # [SWAP] For 3-finger gripper: update joint_names and command expressions
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],  # [SWAP]
            open_command_expr={"panda_finger_.*": 0.04},   # [SWAP] open width per joint
            close_command_expr={"panda_finger_.*": 0.0},   # [SWAP] close command
        )

        # -- End-effector frame tracker --
        # [SWAP] Change prim_path base link and target frame to your robot's links
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/EEFrame"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",  # [SWAP] base link
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",  # [SWAP] EE link
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),  # [SWAP] TCP offset
                ),
            ],
        )

        # -- Reduce env count for easier first run --
        self.scene.num_envs = 1024


@configclass
class FrankaPnPObstacleEnvCfg_PLAY(FrankaPnPObstacleEnvCfg):
    """Play/eval mode: fewer envs, no corruption, start with obstacles active."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        # Force obstacles on for eval viewing
        self.curriculum.obstacle_stage.params["stage_1_threshold"] = 0
        self.curriculum.obstacle_stage.params["stage_2_threshold"] = 0
