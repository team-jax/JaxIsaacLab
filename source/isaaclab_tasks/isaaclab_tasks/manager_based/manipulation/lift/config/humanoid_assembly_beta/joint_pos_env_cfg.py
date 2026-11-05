# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.humanoid_assembly_beta.lift_humanoid_assembly_beta_env_cfg import (
    LiftEnvCfg,
)

from isaaclab_assets.robots.humanoid_assembly_beta import (
    HUMANOID_ASSEMBLY_BETA_CFG,
    HUMANOID_ASSEMBLY_BETA_RIGHT_ARM_JOINT_NAMES,
)

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class HumanoidAssemblyBetaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Humanoid Assembly Beta as robot
        self.scene.robot = HUMANOID_ASSEMBLY_BETA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type: drive the right arm's 7 joints, keep the left
        # arm/gripper passively held at their default (init) pose since this is a uni-manual task.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=HUMANOID_ASSEMBLY_BETA_RIGHT_ARM_JOINT_NAMES,
            scale=0.5,
            use_default_offset=True,
        )

        # Open/close targets for the 3-finger right gripper. The source URDF has no defined "open"/
        # "close" hand pose, so these are a heuristic first pass (base joints curl within their
        # [-pi/2, pi/2] / [0, pi] limits, distal joints curl within their [0, ~120deg] limits) and
        # should be checked visually and re-tuned in the simulator.
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "R_wrist_yaw_to_R_finger_1_link1",
                "R_wrist_yaw_to_R_finger_2_link1",
                "R_rotation_base_to_R_finger_3_link1",
                "R_finger_1_link1_to_R_finger_1_link2",
                "R_finger_2_link1_to_R_finger_2_link2",
                "R_finger_3_link1_to_R_finger_3_link2",
            ],
            open_command_expr={
                "R_wrist_yaw_to_R_finger_1_link1": 0.3,
                "R_wrist_yaw_to_R_finger_2_link1": 0.3,
                "R_rotation_base_to_R_finger_3_link1": 0.3,
                "R_finger_1_link1_to_R_finger_1_link2": 0.1,
                "R_finger_2_link1_to_R_finger_2_link2": 0.1,
                "R_finger_3_link1_to_R_finger_3_link2": 0.1,
            },
            close_command_expr={
                "R_wrist_yaw_to_R_finger_1_link1": 0.9,
                "R_wrist_yaw_to_R_finger_2_link1": 0.9,
                "R_rotation_base_to_R_finger_3_link1": 0.9,
                "R_finger_1_link1_to_R_finger_1_link2": 1.7,
                "R_finger_2_link1_to_R_finger_2_link2": 1.7,
                "R_finger_3_link1_to_R_finger_3_link2": 1.7,
            },
        )

        # Set the body name for the end effector: the palm link the three fingers attach to
        self.commands.object_pose.body_name = "R_rotation_base_1"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/R_rotation_base_1",
                    name="end_effector",
                ),
            ],
        )


@configclass
class HumanoidAssemblyBetaCubeLiftEnvCfg_PLAY(HumanoidAssemblyBetaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
