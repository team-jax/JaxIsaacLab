"""Separate Stage 0 task for the October 6 JAX V2 URDF.

The elbow sign and link names differ from the original robot. Finger closure
geometry has not yet been calibrated for a grasp task.
"""

import torch

from isaaclab.utils import configclass

from isaaclab_assets.robots.jax_v2_assembly_1006 import (
    JAX_V2_ARM_JOINTS,
    JAX_V2_ASSEMBLY_1006_CFG,
    JAX_V2_FINGER_JOINTS,
)

from .pregrasp import OneArmJaxPregraspDownEnvCfg, OneArmJaxPregraspEnv


@configclass
class JaxV2PregraspDownEnvCfg(OneArmJaxPregraspDownEnvCfg):
    robot_cfg = JAX_V2_ASSEMBLY_1006_CFG.replace(
        init_state=JAX_V2_ASSEMBLY_1006_CFG.init_state.replace(
            pos=(0.0, 0.0, 1.20 - 0.895 + float(__import__("os").environ.get("GL_ROBOT_DZ", "0"))),  # GL_ROBOT_DZ: equivalence check
            rot=(0.7071067811865476, 0.0, 0.0, -0.7071067811865476),
        )
    )
    arm_joint_names = JAX_V2_ARM_JOINTS
    finger_joint_names = JAX_V2_FINGER_JOINTS
    fingertip_body_names = ("Finger_1_2", "Finger_2_2", "Finger_3_2")
    wrist_body_name = "MG996R"
    # Open pregrasp fitted to the fixed cube: fingers 1/2 on world +Y and
    # finger 3 on world -Y, with their link origins outside the cube.
    # The revised URDF bends toward the table for a positive elbow command.
    initial_arm_q = (0.0035, 0.0300, -0.8992, 1.5500,
                     0.9310, -0.8527, -0.6241, -1.2487)
    finger_open = (-0.8000, -0.4751, -0.2817, 0.0, 0.0, 0.0)
    # Closure targets are provisional; only the open pose was verified.
    finger_close = (0.3, 0.3, 0.3, 1.0, 1.0, 1.0)
    reset_finger_closure_fraction = 0.0
    table_center = (-0.74, 0.0, 0.72)
    object_x = (-0.40, -0.40)
    object_y = (0.08, 0.08)


class JaxV2PregraspEnv(OneArmJaxPregraspEnv):
    """Begin with neutral fingers; the policy chooses when to spread them."""

    cfg: JaxV2PregraspDownEnvCfg

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        q = self.targets[env_ids].clone()
        q[:, self.finger_ids] = 0.0
        self.targets[env_ids] = q
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)
