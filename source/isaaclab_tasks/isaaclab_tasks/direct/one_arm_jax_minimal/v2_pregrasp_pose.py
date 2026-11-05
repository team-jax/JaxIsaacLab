"""Lift first, then learn an open three-finger pose around the cube.

The policy controls the arm, RotationBody, and (after lifting) the fingers.
No joint-angle demonstration or pressure reward is used for the pregrasp.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from .v2_lift_only import JaxV2LiftOnlyEnv, JaxV2LiftOnlyEnvCfg


@configclass
class JaxV2PregraspPoseEnvCfg(JaxV2LiftOnlyEnvCfg):
    episode_length_s = 25.0
    # Leave the phase-0 arc reward once the arm has cleared the table. The yaw
    # target is capped below, so this corridor cannot admit the old -1.56 rad
    # folded-back posture.
    use_clearance_corridor = True
    table_edge_margin_m = 0.18
    table_clearance_target_m = 0.08
    table_clearance_penalty_scale = 0.10
    # RotationBody stays neutral while the arm clears the table. A measured
    # sweep places finger 3 opposite fingers 1/2 near -pi/2, not -pi.
    lift_rotation_target_rad = 0.0
    pregrasp_rotation_target_rad = -1.57
    pregrasp_rotation_progress_scale = 2.0
    pregrasp_rotation_tolerance_rad = 0.30
    raised_wrist_pitch_min_rad = 0.65
    # The actuator tracks a 1.45 rad target at roughly 1.43 rad at rest.
    raised_elbow_min_rad = 1.40
    raised_hold_duration_s = 0.5
    pregrasp_hold_duration_s = 0.5
    pregrasp_face_margin_m = 0.025
    pregrasp_position_tolerance_m = 0.035
    pregrasp_open_tolerance_rad = 0.15
    open_progress_scale = 2.0
    forward_progress_scale = 2.0
    cube_approach_progress_scale = 2.0
    cube_proximity_scale = 0.01
    # Tip-center/cube distance at which the approach quality saturates.
    cube_approach_full_m = 0.13
    open_pose_cube_distance_m = 0.13
    face_pose_progress_scale = 0.0
    tip_height_progress_scale = 0.0
    lift_transition_bonus = 2.0
    pregrasp_success_scale = 10.0


class JaxV2PregraspPoseEnv(JaxV2LiftOnlyEnv):
    cfg: JaxV2PregraspPoseEnvCfg

    def __init__(self, cfg: JaxV2PregraspPoseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._raised_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._pregrasp_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._previous_open_quality = torch.zeros(self.num_envs, device=self.device)
        self._previous_face_quality = torch.zeros(self.num_envs, device=self.device)
        self._previous_tip_height_quality = torch.zeros(self.num_envs, device=self.device)
        self._previous_forward_quality = torch.zeros(self.num_envs, device=self.device)
        self._previous_cube_approach_quality = torch.zeros(self.num_envs, device=self.device)
        self._previous_rotation_quality = torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if hasattr(self, "_raised_steps"):
            ids = self.robot._ALL_INDICES if env_ids is None else env_ids
            self._raised_steps[ids] = 0
            self._pregrasp_steps[ids] = 0
            self._previous_open_quality[ids] = 0.0
            self._previous_face_quality[ids] = 0.0
            self._previous_tip_height_quality[ids] = 0.0
            self._previous_forward_quality[ids] = 0.0
            self._previous_cube_approach_quality[ids] = 0.0
            self._previous_rotation_quality[ids] = 0.0

    def _finger_open_stage(self, arm: torch.Tensor, tips_w: torch.Tensor) -> torch.Tensor:
        local_tips = tips_w - self.scene.env_origins[:, None, :]
        table_x, table_y, _ = self.cfg.table_center
        size_x, size_y, _ = self.cfg.table_cfg.size
        margin = 0.04
        over_table = (
            (local_tips[..., 0] > table_x - size_x / 2 - margin)
            & (local_tips[..., 0] < table_x + size_x / 2 + margin)
            & ((local_tips[..., 1] - table_y).abs() < size_y / 2 + margin)
        )
        tip_clearance = tips_w[..., 2].amin(dim=1) - self.cfg.table_top_z
        safe_to_open = (~over_table.any(dim=1)) | (tip_clearance > self.cfg.min_lift_tip_clearance_m)
        # The prior checkpoint has yaw near -0.8 rad; the regressed policy
        # folded behind the base near -1.5 rad with a large shoulder roll.
        forward_workspace = (arm[:, 2] > -1.20) & (arm[:, 1].abs() < 0.35)
        # The cube is at X=-0.40 m. Opening at X=-0.11 m, as in the last
        # video, is behind the actual workspace and must not count as success.
        forward_reach = local_tips[..., 0].mean(dim=1) < -0.20
        return (arm[:, 3] > 1.35) & safe_to_open & forward_workspace & forward_reach

    def _pre_physics_step(self, actions: torch.Tensor):
        # The first two phases are the existing learned lift. Fingers remain
        # neutral, and RotationBody is held neutral until the arm is raised.
        previous_finger_targets = self.targets[:, self.finger_ids].clone()
        super()._pre_physics_step(actions)
        # The failed policy drove shoulder yaw to -1.56 rad and left the hand
        # behind the robot. Keep this preparation task in the verified forward
        # workspace (the earlier arm path used about -0.8 rad).
        yaw_id = self.arm_ids[2]
        self.targets[:, yaw_id] = self.targets[:, yaw_id].clamp_min(-1.0)
        preparing = self.reward_phase < 2
        self.targets[:, self.arm_ids[7]] = torch.where(
            preparing, torch.zeros_like(self.targets[:, self.arm_ids[7]]), self.targets[:, self.arm_ids[7]]
        )
        arm = self.robot.data.joint_pos[:, self.arm_ids]
        tips = self.robot.data.body_pos_w[:, self.fingertip_ids]
        preshaping = self._finger_open_stage(arm, tips)
        finger_actions = actions[:, 8:11].clamp(-1.0, 1.0)
        self.actions[:, 8:11] = torch.where(preshaping[:, None], finger_actions, 0.0)
        open_q = self.targets.new_tensor(self.cfg.finger_open)
        close_q = self.targets.new_tensor(self.cfg.finger_close)
        six_actions = torch.cat((finger_actions, finger_actions), dim=1)
        finger_target = previous_finger_targets + (
            self.cfg.gripper_delta_scale * six_actions * (close_q - open_q)
        )
        finger_target = torch.minimum(torch.maximum(finger_target, torch.minimum(open_q, close_q)),
                                      torch.maximum(open_q, close_q))
        self.targets[:, self.finger_ids] = torch.where(
            preshaping[:, None], finger_target, self.targets[:, self.finger_ids]
        )

    def _get_rewards(self):
        # Reuse the verified arm path and geometric table-clearance penalty.
        # The lift-only success is a transition here, not episode termination.
        reward = super()._get_rewards() - self.cfg.success_scale * self.success.float()
        arm = self.robot.data.joint_pos[:, self.arm_ids]
        finger = self.robot.data.joint_pos[:, self.finger_ids]
        tips_w = self.robot.data.body_pos_w[:, self.fingertip_ids]
        tip_clearance = tips_w[..., 2].amin(dim=1) - self.cfg.table_top_z
        tip_center_x = (tips_w[..., 0] - self.scene.env_origins[:, None, 0]).mean(dim=1)
        forward_quality = ((0.02 - tip_center_x) / 0.22).clamp(0.0, 1.0)
        forward_reward = self.cfg.forward_progress_scale * (
            forward_quality - self._previous_forward_quality
        ) * (arm[:, 3] > 1.0)
        self._previous_forward_quality = forward_quality.detach()
        tip_cube_distance = torch.linalg.vector_norm(
            tips_w.mean(dim=1) - self.object.data.root_pos_w, dim=1
        )
        cube_approach_quality = (
            (0.30 - tip_cube_distance) / (0.30 - self.cfg.cube_approach_full_m)
        ).clamp(0.0, 1.0)
        cube_approach_reward = self.cfg.cube_approach_progress_scale * (
            cube_approach_quality - self._previous_cube_approach_quality
        ) * (arm[:, 3] > 1.0)
        self._previous_cube_approach_quality = cube_approach_quality.detach()
        rotation_error = (arm[:, 7] - self.cfg.pregrasp_rotation_target_rad).abs()
        rotation_quality = (1.0 - rotation_error / 1.57).clamp(0.0, 1.0)
        rotation_reward = self.cfg.pregrasp_rotation_progress_scale * (
            rotation_quality - self._previous_rotation_quality
        ) * (self.reward_phase == 2) * cube_approach_quality
        cube_proximity_reward = self.cfg.cube_proximity_scale * cube_approach_quality * (self.reward_phase == 2)
        self._previous_rotation_quality = rotation_quality.detach()
        tip_height_quality = (
            (tip_clearance + 0.25) / (0.25 + self.cfg.min_lift_tip_clearance_m)
        ).clamp(0.0, 1.0)
        height_reward = self.cfg.tip_height_progress_scale * (
            tip_height_quality - self._previous_tip_height_quality
        ) * (self.reward_phase < 2)
        self._previous_tip_height_quality = tip_height_quality.detach()
        arm_speed = self.robot.data.joint_vel[:, self.arm_ids].abs().amax(dim=1)
        raised = (
            (arm[:, 0] < self.cfg.safe_raised_shoulder_pitch_rad + 0.03)
            & (arm[:, 3] > self.cfg.raised_elbow_min_rad)
            & (arm[:, 4] > self.cfg.raised_wrist_pitch_min_rad)
            & (tip_clearance > self.cfg.min_lift_tip_clearance_m)
            & (arm_speed < self.cfg.max_hold_arm_speed_rad_s)
        )
        self._raised_steps = torch.where((self.reward_phase == 1) & raised, self._raised_steps + 1, 0)
        lift_steps = round(self.cfg.raised_hold_duration_s / (self.cfg.sim.dt * self.cfg.decimation))
        lift_transition = (self.reward_phase == 1) & (self._raised_steps >= lift_steps)
        self.reward_phase = torch.where(lift_transition, 2, self.reward_phase)

        # Link2 origins in cube-local coordinates: fingers 1/2 oppose finger 3
        # across the cube's Y faces. Distance from each target remains dense
        # while the final pose requires all three fingers in place.
        cube_w = self.object.data.root_pos_w
        cube_q = self.object.data.root_quat_w[:, None, :].expand(-1, 3, -1)
        tips_o = quat_apply_inverse(
            cube_q.reshape(-1, 4), (tips_w - cube_w[:, None, :]).reshape(-1, 3)
        ).reshape(-1, 3, 3)
        side = self.cfg.object_size[1] / 2 + self.cfg.pregrasp_face_margin_m
        anchors = tips_o.new_tensor(((0.0, side, 0.0), (0.0, side, 0.0), (0.0, -side, 0.0)))
        face_error = torch.linalg.vector_norm(tips_o - anchors[None], dim=-1)
        face_quality = torch.exp(-8.0 * face_error.mean(dim=1))
        open_error = (finger - finger.new_tensor(self.cfg.finger_open)).abs()
        open_quality = torch.exp(-2.0 * open_error.amax(dim=1))
        preshaping = self._finger_open_stage(arm, tips_w)
        open_reward = self.cfg.open_progress_scale * (
            open_quality - self._previous_open_quality
        ) * preshaping
        face_reward = self.cfg.face_pose_progress_scale * (
            face_quality - self._previous_face_quality
        ) * preshaping
        self._previous_open_quality = open_quality.detach()
        self._previous_face_quality = face_quality.detach()

        # A three-sided pose must be held with open fingers, without touching
        # the table. Actual grasp/contact belongs to the next stage.
        pose_ready = (
            preshaping
            & (tip_cube_distance < self.cfg.open_pose_cube_distance_m)
            & (rotation_error < self.cfg.pregrasp_rotation_tolerance_rad)
            & (face_error.mean(dim=1) < 0.075)
            & (open_error.amax(dim=1) < self.cfg.pregrasp_open_tolerance_rad)
            & (arm_speed < self.cfg.max_hold_arm_speed_rad_s)
        )
        self._pregrasp_steps = torch.where(pose_ready, self._pregrasp_steps + 1, 0)
        required_steps = round(self.cfg.pregrasp_hold_duration_s / (self.cfg.sim.dt * self.cfg.decimation))
        self.success = self._pregrasp_steps >= required_steps
        finger_action_rate = ((self.actions[:, 8:11] - self.previous_actions[:, 8:11]) ** 2).sum(dim=1)
        reward = (
            reward + self.cfg.lift_transition_bonus * lift_transition.float()
            + height_reward + forward_reward + cube_approach_reward + cube_proximity_reward
            + rotation_reward + open_reward + face_reward
            + self.cfg.pregrasp_success_scale * self.success.float()
            - self.cfg.action_rate_scale * finger_action_rate
        )
        self.extras["log"].update({
            "lift_transition": lift_transition.float().mean(),
            "rotation_body_rad": arm[:, 7].mean(),
            "rotation_progress_reward": rotation_reward.mean(),
            "rotation_error_rad": rotation_error.mean(),
            "open_progress_reward": open_reward.mean(),
            "forward_progress_reward": forward_reward.mean(),
            "cube_approach_progress_reward": cube_approach_reward.mean(),
            "cube_proximity_reward": cube_proximity_reward.mean(),
            "tip_cube_distance_m": tip_cube_distance.mean(),
            "tip_height_progress_reward": height_reward.mean(),
            "face_pose_progress_reward": face_reward.mean(),
            "face_pose_error_m": face_error.mean(),
            "pregrasp_ready": pose_ready.float().mean(),
            "finger_open_stage": preshaping.float().mean(),
            "success": self.success.float().mean(),
        })
        self.extras["metrics"].update({
            "phase": self.reward_phase,
            "lift_transition": lift_transition,
            "face_pose_error_m": face_error,
            "pregrasp_ready": pose_ready,
            "tip_cube_distance_m": tip_cube_distance,
            "finger_open_stage": preshaping,
            "success": self.success,
            "rotation_body_rad": arm[:, 7],
            "rotation_error_rad": rotation_error,
        })
        return reward
