"""November 8 JAX V3 assembly (URDF_1108-2), prepared from the exporter package: link names without the "_1" suffix,
shoulder roll and wrist pitch axes flipped so joint angles keep the V2 sign convention (see the URDF header), base_link
collision removed. Same joint names, order and drives as V2."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

JAX_V3_ARM_JOINTS = [
    "shoulder_pitch_to_shoulder_roll",
    "shoulder_roll_to_shoulder_yaw",
    "shoulder_yaw_to_upper_arm",
    "upper_arm_to_under_arm",
    "under_arm_to_wrist_pitch",
    "wrist_pitch_to_wrist_yaw",
    "wrist_yaw_to_hand_base",
    "hand_base_to_rotation_base",
]
JAX_V3_FINGER_JOINTS = [
    "hand_base_to_finger_1_1",
    "hand_base_to_finger_2_1",
    "rotation_base_to_finger_3_1",
    "finger_1_1_to_finger_1_2",
    "finger_1_2_to_finger_2_2",
    "finger_3_1_to_finger_3_2",
]
_data = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Robots", "JaxV3Assembly1108")

JAX_V3_ASSEMBLY_1108_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=os.path.join(_data, "urdf", "JAXV3_Assembly.urdf"),
        usd_dir=os.path.join(_data, "usd"),
        usd_file_name="JAXV3_Assembly.usd",
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
        joint_pos={name: 0.0 for name in JAX_V3_ARM_JOINTS + JAX_V3_FINGER_JOINTS},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=JAX_V3_ARM_JOINTS, effort_limit_sim=100.0,
            velocity_limit_sim=1.0, stiffness=100.0, damping=8.0,
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=JAX_V3_FINGER_JOINTS, effort_limit_sim=100.0,
            velocity_limit_sim=1.0, stiffness=180.0, damping=12.0,
        ),
    },
)

# diagnostic variant: V3 kinematics and collision, V2 inertials (isolates the mass change); GL_ROBOT=v3m
JAX_V3M_ASSEMBLY_1108_CFG = JAX_V3_ASSEMBLY_1108_CFG.replace(spawn=JAX_V3_ASSEMBLY_1108_CFG.spawn.replace(
    asset_path=os.path.join(_data, "urdf", "JAXV3_Assembly_v2mass.urdf"), usd_dir=os.path.join(_data, "usd_v2mass"),
    usd_file_name="JAXV3_Assembly_v2mass.usd"))
