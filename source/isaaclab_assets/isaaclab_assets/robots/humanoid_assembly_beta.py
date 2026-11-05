# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Humanoid Assembly Beta dual-arm robot.

This is a fixed-base torso with two 7-DOF arms (shoulder pitch/roll/yaw, elbow, wrist pitch/yaw/roll),
each ending in a 3-finger gripper (2 joints per finger). The asset is spawned directly from the source
URDF (converted to USD on first load and cached under the same data directory) rather than a
pre-converted USD file, since it was imported from a user-provided CAD export.

The following configurations are available:

* :obj:`HUMANOID_ASSEMBLY_BETA_CFG`: Humanoid Assembly Beta robot with both arms and grippers.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

HUMANOID_ASSEMBLY_BETA_DATA_DIR = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Robots", "HumanoidAssemblyBeta")
"""Path to the directory holding the Humanoid Assembly Beta source URDF, meshes, and cached USD."""

# joint names of the right 7-DOF arm (shoulder pitch/roll/yaw, elbow, wrist pitch/yaw/roll)
HUMANOID_ASSEMBLY_BETA_RIGHT_ARM_JOINT_NAMES = [
    "R_shoulder_pitch_to_R_shoulder_roll",
    "R_shoulder_roll_to_R_shoulder_yaw",
    "R_shoulder_yaw_to_R_upper_arm",
    "R_upper_arm_to_R_under_arm",
    "R_under_arm_to_R_wrist_pitch",
    "R_wrist_pitch_to_R_wrist_yaw",
    "R_wrist_yaw_to_R_rotation_base",
]
HUMANOID_ASSEMBLY_BETA_LEFT_ARM_JOINT_NAMES = [
    name.replace("R_", "L_") for name in HUMANOID_ASSEMBLY_BETA_RIGHT_ARM_JOINT_NAMES
]
HUMANOID_ASSEMBLY_BETA_ARM_JOINT_NAMES = (
    HUMANOID_ASSEMBLY_BETA_RIGHT_ARM_JOINT_NAMES + HUMANOID_ASSEMBLY_BETA_LEFT_ARM_JOINT_NAMES
)

# joint names of the right 3-finger gripper (2 joints per finger)
HUMANOID_ASSEMBLY_BETA_RIGHT_GRIPPER_JOINT_NAMES = [
    "R_wrist_yaw_to_R_finger_1_link1",
    "R_wrist_yaw_to_R_finger_2_link1",
    "R_rotation_base_to_R_finger_3_link1",
    "R_finger_1_link1_to_R_finger_1_link2",
    "R_finger_2_link1_to_R_finger_2_link2",
    "R_finger_3_link1_to_R_finger_3_link2",
]
HUMANOID_ASSEMBLY_BETA_LEFT_GRIPPER_JOINT_NAMES = [
    name.replace("R_", "L_") for name in HUMANOID_ASSEMBLY_BETA_RIGHT_GRIPPER_JOINT_NAMES
]
HUMANOID_ASSEMBLY_BETA_GRIPPER_JOINT_NAMES = (
    HUMANOID_ASSEMBLY_BETA_RIGHT_GRIPPER_JOINT_NAMES + HUMANOID_ASSEMBLY_BETA_LEFT_GRIPPER_JOINT_NAMES
)

HUMANOID_ASSEMBLY_BETA_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=os.path.join(HUMANOID_ASSEMBLY_BETA_DATA_DIR, "urdf", "Humanoid_Assembly_Beta.urdf"),
        usd_dir=os.path.join(HUMANOID_ASSEMBLY_BETA_DATA_DIR, "usd"),
        usd_file_name="Humanoid_Assembly_Beta.usd",
        # the source URDF has no leg/base joint: fix the torso to the world so the arms have something
        # to push and pull against instead of floating freely under gravity
        fix_base=True,
        merge_fixed_joints=True,
        self_collision=False,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            # placeholder gains: the source URDF carries no real motor spec (uniform effort=100,
            # velocity=100 on every joint), so these are only a starting point for tuning. The
            # ImplicitActuatorCfg groups below set the gains actually used during simulation.
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=1.0),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={name: 0.0 for name in HUMANOID_ASSEMBLY_BETA_ARM_JOINT_NAMES}
        | {name: 0.0 for name in HUMANOID_ASSEMBLY_BETA_GRIPPER_JOINT_NAMES},
    ),
    actuators={
        "arms": ImplicitActuatorCfg(
            joint_names_expr=HUMANOID_ASSEMBLY_BETA_ARM_JOINT_NAMES,
            velocity_limit_sim=100.0,
            effort_limit_sim=100.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=HUMANOID_ASSEMBLY_BETA_GRIPPER_JOINT_NAMES,
            velocity_limit_sim=100.0,
            effort_limit_sim=100.0,
            stiffness=200.0,
            damping=10.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of the Humanoid Assembly Beta dual-arm robot."""

HUMANOID_ASSEMBLY_BETA_HIGH_PD_CFG = HUMANOID_ASSEMBLY_BETA_CFG.copy()
HUMANOID_ASSEMBLY_BETA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
HUMANOID_ASSEMBLY_BETA_HIGH_PD_CFG.actuators["arms"].stiffness = 400.0
HUMANOID_ASSEMBLY_BETA_HIGH_PD_CFG.actuators["arms"].damping = 80.0
"""Configuration of the Humanoid Assembly Beta robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
