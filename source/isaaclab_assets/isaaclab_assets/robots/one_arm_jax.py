"""Original JAX single-arm URDF with position controlled joints."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

JAX_ARM_JOINTS = [
    "shoulder_pitch_to_shoulder_roll", "shoulder_roll_to_shoulder_yaw",
    "shoulder_yaw_to_upper_arm", "upper_arm_to_under_arm",
    "under_arm_to_arm_yaw", "arm_yaw_to_wrist_pitch",
    "wrist_pitch_to_wrist_yaw", "wrist_yaw_to_rotation_base",
]
JAX_FINGER_JOINTS = [
    "wrist_yaw_to_finger_2_link1", "wrist_yaw_to_finger_3_link1",
    "rotation_base_to_finger_1_link1", "wrist_yaw_to_finger_2_link2",
    "wrist_yaw_to_finger_3_link2", "finger_1_link1_to_finger_1_link2",
]
_data = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Robots", "OneArmJaxUrdf")

ONE_ARM_JAX_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=os.path.join(_data, "urdf", "One_Arm_JAX_URDF.urdf"),
        usd_dir=os.path.join(_data, "usd"),
        usd_file_name="One_Arm_JAX_URDF.usd",
        fix_base=True,
        merge_fixed_joints=True,
        self_collision=False,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            drive_type="force", target_type="position",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=4.0),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=3.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={name: 0.0 for name in JAX_ARM_JOINTS + JAX_FINGER_JOINTS},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=JAX_ARM_JOINTS, effort_limit_sim=100.0,
            velocity_limit_sim=1.0, stiffness=100.0, damping=8.0,
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=JAX_FINGER_JOINTS, effort_limit_sim=100.0,
            velocity_limit_sim=1.0, stiffness=180.0, damping=12.0,
        ),
    },
)
