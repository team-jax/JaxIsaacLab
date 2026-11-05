"""V2 arm-down to three-finger grasp, using geometric rewards, not joint replay."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from .v2_pregrasp import JaxV2PregraspDownEnvCfg, JaxV2PregraspEnv


@configclass
class JaxV2GraspEnvCfg(JaxV2PregraspDownEnvCfg):
    episode_length_s = 12.0
    # Per control step. The final success is paid once when the episode ends.
    # Dense rewards remain small relative to the one-time +10 success across
    # the full 12-second (720-step) episode; loitering should not dominate.
    posture_scale = 0.003
    reach_scale = 0.004
    pregrasp_scale = 0.004
    inner_contact_scale = 0.01
    stable_grasp_scale = 0.015
    success_scale = 10.0
    early_close_scale = 0.005
    table_proximity_scale = 0.03
    action_rate_scale = 0.001
    failure_termination_penalty = 2.0
    # Disabled for the original v1 experiment. The controlled-lift task below
    # turns these on without changing the saved v1 reward configuration.
    lift_path_scale = 0.0
    early_rotation_scale = 0.0


@configclass
class JaxV2ControlledLiftEnvCfg(JaxV2GraspEnvCfg):
    """Reward the measured shoulder/elbow clearance arc before wrist motion."""

    lift_path_scale = 0.012
    early_rotation_scale = 0.020


class JaxV2GraspEnv(JaxV2PregraspEnv):
    cfg: JaxV2GraspEnvCfg

    def _pregrasp_quality(self):
        """Two fingers on cube local +Y, one on local -Y; all outside the box."""
        tips_w = self.robot.data.body_pos_w[:, self.fingertip_ids]
        cube_w = self.object.data.root_pos_w
        cube_q = self.object.data.root_quat_w[:, None, :].expand(-1, 3, -1)
        tips_o = quat_apply_inverse(
            cube_q.reshape(-1, 4), (tips_w - cube_w[:, None, :]).reshape(-1, 3)
        ).reshape(-1, 3, 3)
        half_y = self.cfg.object_size[1] / 2
        anchors = tips_o.new_tensor((
            (0.0, half_y + 0.025, 0.0),
            (0.0, half_y + 0.025, 0.0),
            (0.0, -half_y - 0.025, 0.0),
        ))
        placement = torch.linalg.vector_norm(tips_o - anchors[None], dim=-1).mean(dim=1)
        wrist_w = self.robot.data.body_pos_w[:, self.wrist_id]
        axis = tips_w.mean(dim=1) - wrist_w
        toward_cube = cube_w - wrist_w
        alignment = 0.5 + 0.5 * (axis * toward_cube).sum(dim=1) / (
            torch.linalg.vector_norm(axis, dim=1)
            * torch.linalg.vector_norm(toward_cube, dim=1)
        ).clamp_min(1e-6)
        quality = torch.exp(-10.0 * placement) * alignment.clamp(0.0, 1.0)
        distance = torch.linalg.vector_norm(cube_w - tips_w.mean(dim=1), dim=1)
        return quality, distance

    def _contacts(self):
        """Only second-phalanx inner contacts on opposing cube Y side faces count."""
        sensors = [self.scene.sensors[f"finger_{i}"].data for i in (1, 2, 3)]
        forces_w = torch.stack([sensor.force_matrix_w[:, 0, 0] for sensor in sensors], dim=1)
        points_w = torch.stack([sensor.contact_pos_w[:, 0, 0] for sensor in sensors], dim=1)
        force_n = torch.linalg.vector_norm(forces_w, dim=-1)
        cube_w = self.object.data.root_pos_w
        cube_q = self.object.data.root_quat_w[:, None, :].expand(-1, 3, -1)
        relative = quat_apply_inverse(
            cube_q.reshape(-1, 4), (points_w - cube_w[:, None, :]).reshape(-1, 3)
        ).reshape(-1, 3, 3)
        local_force = quat_apply_inverse(
            cube_q.reshape(-1, 4), forces_w.reshape(-1, 3)
        ).reshape(-1, 3, 3)
        finger_w = self.robot.data.body_pos_w[:, self.fingertip_ids]
        finger_q = self.robot.data.body_quat_w[:, self.fingertip_ids]
        opposite_w = torch.stack((
            finger_w[:, 2], finger_w[:, 2], 0.5 * (finger_w[:, 0] + finger_w[:, 1])
        ), dim=1)
        local_contact = quat_apply_inverse(
            finger_q.reshape(-1, 4), (points_w - finger_w).reshape(-1, 3)
        ).reshape(-1, 3, 3)[..., (0, 2)]
        local_inside = quat_apply_inverse(
            finger_q.reshape(-1, 4), (opposite_w - finger_w).reshape(-1, 3)
        ).reshape(-1, 3, 3)[..., (0, 2)]
        inside_cos = (local_contact * local_inside).sum(dim=-1) / (
            torch.linalg.vector_norm(local_contact, dim=-1)
            * torch.linalg.vector_norm(local_inside, dim=-1)
        ).clamp_min(1e-6)
        half_x, half_y, half_z = (size / 2 for size in self.cfg.object_size)
        expected_y = relative.new_tensor((half_y, half_y, -half_y))
        on_side = (relative[..., 1] - expected_y).abs() < 0.012
        within_face = (relative[..., 0].abs() < half_x + 0.005) & (relative[..., 2].abs() < half_z - 0.003)
        side_force = local_force[..., 1].abs() / force_n.clamp_min(1e-6)
        valid = (
            torch.isfinite(points_w).all(dim=-1)
            & (inside_cos > 0.2)
            & (torch.linalg.vector_norm(local_contact, dim=-1) > 0.001)
            & on_side & within_face & (side_force > 0.25)
            & (force_n >= self.cfg.min_contact_force_n)
            & (force_n <= self.cfg.max_safe_contact_force_n)
        )
        quality = (valid.float().sum(dim=1) / 3.0).square()
        return valid, force_n, valid.all(dim=1), quality

    def _get_rewards(self):
        tips = self.robot.data.body_pos_w[:, self.fingertip_ids]
        cube = self.object.data.root_pos_w
        distance = torch.linalg.vector_norm(tips.mean(dim=1) - cube, dim=1)
        pregrasp_quality, _ = self._pregrasp_quality()
        q = self.robot.data.joint_pos
        elbow_error = (q[:, self.arm_ids[3]] - 1.55).abs()
        elbow_quality = torch.exp(-2.0 * elbow_error)
        open_q = torch.tensor(self.cfg.finger_open, device=self.device)
        close_q = torch.tensor(self.cfg.finger_close, device=self.device)
        closure = ((q[:, self.finger_ids] - open_q) / (close_q - open_q).clamp_min(1e-5)).clamp(0.0, 1.0)
        open_quality = 1.0 - closure.mean(dim=1)
        # Use each clone's local coordinates; world X includes the env origin.
        # The arm may hang below table height while it is behind the near edge.
        local_tips = tips - self.scene.env_origins[:, None, :]
        table_x, table_y, _ = self.cfg.table_center
        table_size_x, table_size_y, _ = self.cfg.table_cfg.size
        over_table = (
            (local_tips[..., 0] > table_x - table_size_x / 2)
            & (local_tips[..., 0] < table_x + table_size_x / 2)
            & ((local_tips[..., 1] - table_y).abs() < table_size_y / 2)
        )
        low_over_table = (over_table & (tips[..., 2] < self.cfg.table_top_z + 0.02)).any(dim=1)
        ready_to_close = (distance < 0.12) & ~low_over_table
        posture_reward = self.cfg.posture_scale * (elbow_quality + open_quality) / 2.0
        # In the scripted collision-free lift, shoulder pitch and yaw change
        # about -0.32 and -0.35 rad per radian of elbow flexion. Wrist and
        # shoulder roll stay near zero until the elbow reaches ~90 degrees.
        # These are soft rewards; the policy still commands every joint.
        arm = q[:, self.arm_ids]
        elbow = arm[:, 3].clamp_min(0.0)
        path_error = (
            (arm[:, 0] + 0.32 * elbow).abs()
            + (arm[:, 2] + 0.35 * elbow).abs()
        ) / 2.0
        early_gate = ((1.55 - elbow) / 0.35).clamp(0.0, 1.0)
        other_rotation = torch.cat((arm[:, 1:2], arm[:, 4:8]), dim=1)
        unwanted_rotation = other_rotation.square().mean(dim=1).clamp(0.0, 1.0)
        lift_path_reward = (
            self.cfg.lift_path_scale * (elbow / 1.45).clamp(0.0, 1.0)
            * torch.exp(-2.0 * (path_error + unwanted_rotation))
        )
        early_rotation_penalty = self.cfg.early_rotation_scale * early_gate * (
            path_error.square().clamp(0.0, 1.0) + unwanted_rotation
        )
        # Nonzero gradient even from the arm-down reset, without a progress
        # difference that can punish slow approach.
        reach_reward = self.cfg.reach_scale / (1.0 + 4.0 * distance)
        pregrasp_reward = self.cfg.pregrasp_scale * pregrasp_quality
        valid, force_n, all_valid, contact_quality = self._contacts()
        contact_reward = self.cfg.inner_contact_scale * contact_quality
        hand_vel = self.robot.data.body_lin_vel_w[:, self.fingertip_ids].mean(dim=1)
        relative_speed = torch.linalg.vector_norm(self.object.data.root_lin_vel_w - hand_vel, dim=1)
        stable = all_valid & (relative_speed < self.cfg.max_relative_speed_mps)
        self.hold_steps = torch.where(stable, self.hold_steps + 1, 0)
        self.success = self.hold_steps >= self.required_hold_steps
        stable_reward = self.cfg.stable_grasp_scale * stable.float()
        neutral_closure = ((-open_q) / (close_q - open_q)).clamp(0.0, 1.0).mean()
        early_close = (closure.mean(dim=1) > neutral_closure + 0.05) & ~ready_to_close
        unsafe = force_n.amax(dim=1) > self.cfg.unsafe_contact_force_n
        self.unsafe_steps = torch.where(unsafe, self.unsafe_steps + 1, 0)
        fallen = cube[:, 2] < self.cfg.table_top_z - 0.05
        failed = fallen | (self.unsafe_steps >= self.cfg.unsafe_consecutive_steps)
        action_rate = ((self.actions - self.previous_actions) ** 2).sum(dim=1)
        reward = (
            posture_reward + lift_path_reward + reach_reward + pregrasp_reward + contact_reward + stable_reward
            + self.cfg.success_scale * self.success.float()
            - self.cfg.early_close_scale * early_close.float()
            - early_rotation_penalty
            - self.cfg.table_proximity_scale * low_over_table.float()
            - self.cfg.failure_termination_penalty * failed.float()
            - self.cfg.action_rate_scale * action_rate
        )
        self.extras["log"] = {
            "posture_reward": posture_reward.mean(),
            "lift_path_reward": lift_path_reward.mean(),
            "early_rotation_penalty": early_rotation_penalty.mean(),
            "reach_reward": reach_reward.mean(),
            "pregrasp_reward": pregrasp_reward.mean(),
            "inner_contact_reward": contact_reward.mean(),
            "stable_grasp_reward": stable_reward.mean(),
            "early_close_penalty": (self.cfg.early_close_scale * early_close.float()).mean(),
            "table_proximity_penalty": (self.cfg.table_proximity_scale * low_over_table.float()).mean(),
            "failure_penalty": (self.cfg.failure_termination_penalty * failed.float()).mean(),
            "action_rate_penalty": (self.cfg.action_rate_scale * action_rate).mean(),
            "pregrasp_quality": pregrasp_quality.mean(),
            "three_finger_contact": all_valid.float().mean(),
            "hold_steps": self.hold_steps.float().mean(),
            "success": self.success.float().mean(),
            "max_contact_force_n": force_n.amax(dim=1).mean(),
        }
        self.extras["metrics"] = {
            "reach": distance < self.cfg.reach_distance_m,
            "pregrasp_quality": pregrasp_quality,
            "valid_contact": valid,
            "contact_3": all_valid,
            "stable": stable,
            "hold": self.hold_steps > 0,
            "success": self.success,
            "unsafe_contact": unsafe,
            "table_proximity": low_over_table,
            "early_close": early_close,
            "force_n": force_n,
        }
        return reward
