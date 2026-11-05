"""Continue the measured lift/pregrasp policy into a three-finger cube grasp."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from .v2_pregrasp_pose import JaxV2PregraspPoseEnv, JaxV2PregraspPoseEnvCfg


@configclass
class JaxV2GraspAfterLiftEnvCfg(JaxV2PregraspPoseEnvCfg):
    # The existing second-phalanx sensors supply binary valid contacts. Force
    # magnitude is used for thresholds and safety, not for increasing reward.
    inner_contact_scale = 0.03
    stable_grasp_scale = 0.02
    grasp_success_scale = 10.0
    early_close_scale = 0.002
    excessive_force_scale = 0.01
    # Pregrasp is a preparation signal, not a terminal success in this task.
    pregrasp_success_scale = 0.0
    # The opposed_near policy parked wrist_yaw_to_hand_base at its +120 deg
    # URDF limit with the open hand facing sideways. The fitted cube pregrasp
    # (initial_arm_q) uses about -36 deg, so keep the target within +/-60 deg.
    wrist_yaw_limit_rad = (-1.05, 1.05)
    # v6 saturated the approach reward at 0.13 m and the policy parked there;
    # keep the gradient until the tip center reaches the cube faces.
    cube_approach_full_m = 0.03
    # v6 penalized any tip below 8 cm over the table (up to 0.10/step), which
    # outweighs the 0.01/step proximity reward; a 5 cm cube is gripped at
    # about 2.5 cm, so the policy hovered at ~9 cm. Penalize only near-contact.
    table_clearance_target_m = 0.015
    # Initial grasp uses only each finger's joint nearest the palm; the second
    # joints stay straight (open = close = 0). Closing range is up to the +90 deg
    # URDF limit (the thumb is meant to close fully; the scan put +q on the
    # fingers 1/2 side), and the policy chooses how far fingers 1/2 close.
    # Order: finger_1_1, finger_2_1, finger_3_1, finger_1_2, finger_2_2, finger_3_2.
    finger_close = (1.5708, 1.5708, 1.5708, 0.0, 0.0, 0.0)


class JaxV2GraspAfterLiftEnv(JaxV2PregraspPoseEnv):
    cfg: JaxV2GraspAfterLiftEnvCfg

    def _finger_open_stage(self, arm: torch.Tensor, tips_w: torch.Tensor) -> torch.Tensor:
        # v6 also required 4 cm tip clearance over the table, which resets the
        # finger targets to 0 rad at cube height. Keep it only before the lift
        # transition (phase 2).
        local_tips = tips_w - self.scene.env_origins[:, None, :]
        forward_workspace = (arm[:, 2] > -1.20) & (arm[:, 1].abs() < 0.35)
        forward_reach = local_tips[..., 0].mean(dim=1) < -0.20
        lifted = (self.reward_phase == 2) & (arm[:, 3] > 1.35) & forward_workspace & forward_reach
        return super()._finger_open_stage(arm, tips_w) | lifted

    def _pre_physics_step(self, actions: torch.Tensor):
        super()._pre_physics_step(actions)
        wrist_yaw_id = self.arm_ids[6]
        self.targets[:, wrist_yaw_id] = self.targets[:, wrist_yaw_id].clamp(*self.cfg.wrist_yaw_limit_rad)

    def _setup_scene(self):
        super()._setup_scene()
        # PhysX contact-point buffer unpacking costs ~18 s/iteration at 7056
        # envs. Retain actual filtered second-link contact forces; classify
        # side contacts with the link pose and force direction instead.
        for i in (1, 2, 3):
            self.scene.sensors[f"finger_{i}"].cfg.track_contact_points = False

    def _contacts(self):
        sensors = [self.scene.sensors[f"finger_{i}"].data for i in (1, 2, 3)]
        forces_w = torch.stack([sensor.force_matrix_w[:, 0, 0] for sensor in sensors], dim=1)
        force_n = torch.linalg.vector_norm(forces_w, dim=-1)
        cube_pos = self.object.data.root_pos_w
        cube_q = self.object.data.root_quat_w[:, None, :].expand(-1, 3, -1)
        finger_pos = self.robot.data.body_pos_w[:, self.fingertip_ids]
        local_pos = quat_apply_inverse(
            cube_q.reshape(-1, 4), (finger_pos - cube_pos[:, None]).reshape(-1, 3)
        ).reshape(-1, 3, 3)
        local_force = quat_apply_inverse(
            cube_q.reshape(-1, 4), forces_w.reshape(-1, 3)
        ).reshape(-1, 3, 3)
        half_x, half_y, half_z = (size / 2 for size in self.cfg.object_size)
        expected_y = local_pos.new_tensor((half_y + 0.025, half_y + 0.025, -half_y - 0.025))
        on_opposing_sides = (local_pos[..., 1] - expected_y).abs() < 0.035
        beside_cube = (local_pos[..., 0].abs() < half_x + 0.025) & (local_pos[..., 2].abs() < half_z + 0.012)
        side_force = local_force[..., 1].abs() / force_n.clamp_min(1e-6)
        valid = (
            on_opposing_sides & beside_cube & (side_force > 0.4)
            & (force_n >= self.cfg.min_contact_force_n)
            & (force_n <= self.cfg.max_safe_contact_force_n)
        )
        quality = (valid.float().sum(dim=1) / 3.0).square()
        return valid, force_n, valid.all(dim=1), quality

    def _get_rewards(self):
        reward = super()._get_rewards()
        valid, force_n, all_valid, contact_quality = self._contacts()
        tips = self.robot.data.body_pos_w[:, self.fingertip_ids]
        cube = self.object.data.root_pos_w
        tip_cube_distance = torch.linalg.vector_norm(tips.mean(dim=1) - cube, dim=1)
        hand_velocity = self.robot.data.body_lin_vel_w[:, self.fingertip_ids].mean(dim=1)
        relative_speed = torch.linalg.vector_norm(
            self.object.data.root_lin_vel_w - hand_velocity, dim=1
        )
        stable = all_valid & (relative_speed < self.cfg.max_relative_speed_mps)
        self.hold_steps = torch.where(stable, self.hold_steps + 1, 0)
        self.success = self.hold_steps >= self.required_hold_steps

        close_q = self.robot.data.joint_pos.new_tensor(self.cfg.finger_close)
        finger_q = self.robot.data.joint_pos[:, self.finger_ids]
        # Measure closure from the 0 rad neutral pose, not finger_open: the
        # fingers rest at 0 during the lift and pass through it while opening,
        # which is already ~30% of the open->close range.
        closing_joints = close_q > 1e-3
        closure = (finger_q / close_q.clamp_min(1e-6)).clamp(0.0, 1.0)[:, closing_joints]
        early_close = (closure.mean(dim=1) > 0.25) & (tip_cube_distance > 0.13)
        excessive_force = ((force_n.amax(dim=1) - self.cfg.max_safe_contact_force_n) / 30.0).clamp(0.0, 2.0)
        unsafe = force_n.amax(dim=1) > self.cfg.unsafe_contact_force_n
        self.unsafe_steps = torch.where(unsafe, self.unsafe_steps + 1, 0)
        unsafe_failure = self.unsafe_steps >= self.cfg.unsafe_consecutive_steps
        contact_reward = self.cfg.inner_contact_scale * contact_quality
        stable_reward = self.cfg.stable_grasp_scale * stable.float()
        reward = (
            reward + contact_reward + stable_reward + self.cfg.grasp_success_scale * self.success.float()
            - self.cfg.early_close_scale * early_close.float()
            - self.cfg.excessive_force_scale * excessive_force
            - self.cfg.failure_termination_penalty * unsafe_failure.float()
        )
        self.extras["log"].update({
            "inner_contact_reward": contact_reward.mean(),
            "stable_grasp_reward": stable_reward.mean(),
            "grasp_success": self.success.float().mean(),
            "contact_3": all_valid.float().mean(),
            "early_close_penalty": (self.cfg.early_close_scale * early_close.float()).mean(),
            "excessive_force_penalty": (self.cfg.excessive_force_scale * excessive_force).mean(),
            "max_contact_force_n": force_n.amax(dim=1).mean(),
        })
        self.extras["metrics"].update({
            "valid_contact": valid,
            "contact_3": all_valid,
            "stable": stable,
            "hold": self.hold_steps > 0,
            "success": self.success,
            "unsafe_contact": unsafe,
            "force_n": force_n,
            "early_close": early_close,
        })
        return reward
