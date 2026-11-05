"""Stage 0: learn a collision-aware arm lift with all finger motors locked neutral."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .v2_learn_sequence import JaxV2LearnSequenceEnv, JaxV2LearnSequenceEnvCfg


@configclass
class JaxV2LiftOnlyEnvCfg(JaxV2LearnSequenceEnvCfg):
    episode_length_s = 20.0
    safe_raised_shoulder_pitch_rad = -0.20
    # Keep the original 11-action policy interface for later grasp training.
    # The final three finger actions are ignored during this lift-only stage.
    lift_hold_duration_s = 0.5
    max_hold_arm_speed_rad_s = 0.2
    min_lift_tip_clearance_m = 0.04
    # This stage uses geometry only; finger pressure calibration belongs to
    # the later grasp task.
    diagnostic_table_contacts = False
    table_edge_margin_m = 0.06
    table_clearance_target_m = 0.04
    table_clearance_penalty_scale = 0.05
    # Reaching the clearance pose already pays a one-time bonus. Do not pay
    # indefinitely for waiting in phase 1 instead of finishing the lift.
    phase_progress_scale = 0.0
    raised_pose_scale = 2.0
    wrist_raise_progress_scale = 2.0
    lift_rotation_target_rad = None
    use_clearance_corridor = False
    # Arm indices (into arm_joint_names) whose error shapes the phase-1 raised
    # pose. None keeps all eight joints.
    raised_shaping_arm_indices: tuple[int, ...] | None = None
    # Table clearance target once reward_phase == 2. None keeps
    # table_clearance_target_m in every phase.
    grasp_table_clearance_target_m: float | None = None


class JaxV2LiftOnlyEnv(JaxV2LearnSequenceEnv):
    cfg: JaxV2LiftOnlyEnvCfg

    def __init__(self, cfg: JaxV2LiftOnlyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._previous_raised_quality = torch.zeros(self.num_envs, device=self.device)
        self._previous_wrist_quality = torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if hasattr(self, "_previous_raised_quality"):
            self._previous_raised_quality[self.robot._ALL_INDICES if env_ids is None else env_ids] = 0.0
            self._previous_wrist_quality[self.robot._ALL_INDICES if env_ids is None else env_ids] = 0.0

    def _pre_physics_step(self, actions: torch.Tensor):
        arm_only_actions = actions.clamp(-1.0, 1.0).clone()
        arm_only_actions[:, 8:] = 0.0
        super()._pre_physics_step(arm_only_actions)
        pitch_id = self.arm_ids[0]
        aligning = self.reward_phase == 1
        self.targets[:, pitch_id] = torch.where(
            aligning,
            self.targets[:, pitch_id].clamp_max(self.cfg.safe_raised_shoulder_pitch_rad),
            self.targets[:, pitch_id],
        )
        self.targets[:, self.finger_ids] = 0.0

    def _get_rewards(self):
        arm = self.robot.data.joint_pos[:, self.arm_ids]
        arm_speed = self.robot.data.joint_vel[:, self.arm_ids].abs().amax(dim=1)
        finger = self.robot.data.joint_pos[:, self.finger_ids]
        clearance = arm.new_tensor((-0.45, 0.0, -0.50, 1.45, 0.0, 0.0, 0.0, 0.0))
        clearance_error = (arm - clearance).abs()
        clearance_ready = clearance_error.amax(dim=1) < 0.18
        if self.cfg.use_clearance_corridor:
            # The waypoint is a safety corridor, not an exact joint pose.
            # A strict yaw equality can strand an otherwise lifted arm in
            # phase 0 and leave all later rewards/actions disabled.
            clearance_ready = (
                (arm[:, 0] < -0.35) & (arm[:, 0] > -0.70)
                & (arm[:, 2] < -0.45) & (arm[:, 2] > -1.10)
                & (arm[:, 3] > 1.35)
                & (arm[:, 1].abs() < 0.25)
                & (arm[:, 4:8].abs().amax(dim=1) < 0.25)
            )
        raised = arm.new_tensor(self.cfg.initial_arm_q)
        raised[0] = self.cfg.safe_raised_shoulder_pitch_rad
        raised[3] = self.cfg.raised_elbow_rad
        if self.cfg.lift_rotation_target_rad is not None:
            raised[7] = self.cfg.lift_rotation_target_rad
        raised_error = (arm - raised).abs()
        # The success gate uses the worst joint error. Use the same bottleneck
        # for shaping so seven aligned joints cannot hide an unraised elbow.
        shaping_error = raised_error
        if self.cfg.raised_shaping_arm_indices is not None:
            shaping_error = raised_error[:, list(self.cfg.raised_shaping_arm_indices)]
        raised_quality = torch.exp(-2.0 * shaping_error.amax(dim=1))
        wrist_quality = torch.exp(-2.0 * raised_error[:, 4])
        raised_ready = raised_error.amax(dim=1) < self.cfg.raised_arm_tolerance_rad

        elbow = arm[:, 3].clamp(0.0, 1.45)
        progress = elbow / 1.45
        arc_error = ((arm[:, 0] + 0.45 * progress).abs() + (arm[:, 2] + 0.50 * progress).abs()) / 2.0
        wrist_error = torch.cat((arm[:, 1:2], arm[:, 4:8]), dim=1).square().mean(dim=1)
        arc_quality = progress * torch.exp(-2.0 * (arc_error + wrist_error))

        old_phase = self.reward_phase.clone()
        arm_transition = (old_phase == 0) & clearance_ready
        self.reward_phase = torch.where(arm_transition, 1, old_phase)
        tips = self.robot.data.body_pos_w[:, self.fingertip_ids]
        tip_clearance = tips[..., 2].amin(dim=1) - self.cfg.table_top_z
        clear = tip_clearance > self.cfg.min_lift_tip_clearance_m
        held = (self.reward_phase == 1) & raised_ready & clear & (arm_speed < self.cfg.max_hold_arm_speed_rad_s)
        self.hold_steps = torch.where(held, self.hold_steps + 1, 0)
        required_steps = round(self.cfg.lift_hold_duration_s / (self.cfg.sim.dt * self.cfg.decimation))
        self.success = self.hold_steps >= required_steps

        local_tips = tips - self.scene.env_origins[:, None, :]
        table_x, table_y, _ = self.cfg.table_center
        size_x, size_y, _ = self.cfg.table_cfg.size
        edge = self.cfg.table_edge_margin_m
        near_table = (
            (local_tips[..., 0] > table_x - size_x / 2 - edge)
            & (local_tips[..., 0] < table_x + size_x / 2 + edge)
            & ((local_tips[..., 1] - table_y).abs() < size_y / 2 + edge)
        )
        clearance_target = torch.full_like(tip_clearance, self.cfg.table_clearance_target_m)
        if self.cfg.grasp_table_clearance_target_m is not None:
            clearance_target = torch.where(
                self.reward_phase == 2, self.cfg.grasp_table_clearance_target_m, clearance_target
            )
        clearance_deficit = (
            (clearance_target[:, None] - (local_tips[..., 2] - self.cfg.table_top_z))
            / clearance_target[:, None]
        ).clamp(0.0, 1.0)
        clearance_risk = (clearance_deficit * near_table).amax(dim=1)
        table_proximity = clearance_risk > 0.0
        fallen = self.object.data.root_pos_w[:, 2] < self.cfg.table_top_z - 0.05
        action_rate = ((self.actions[:, :8] - self.previous_actions[:, :8]) ** 2).sum(dim=1)
        lift_reward = self.cfg.lift_arc_scale * arc_quality * (old_phase == 0)
        raised_reward = self.cfg.raised_pose_scale * (
            raised_quality - self._previous_raised_quality
        ) * (old_phase == 1)
        wrist_raise_reward = self.cfg.wrist_raise_progress_scale * (
            wrist_quality - self._previous_wrist_quality
        ) * (old_phase == 1)
        self._previous_raised_quality = raised_quality.detach()
        self._previous_wrist_quality = wrist_quality.detach()
        reward = (
            lift_reward + raised_reward + wrist_raise_reward
            + self.cfg.phase_progress_scale * (old_phase == 1).float()
            + self.cfg.arm_ready_bonus * arm_transition.float()
            + self.cfg.success_scale * self.success.float()
            - self.cfg.table_clearance_penalty_scale * clearance_risk
            - self.cfg.failure_termination_penalty * fallen.float()
            - self.cfg.action_rate_scale * action_rate
        )
        self.extras["log"] = {
            "lift_arc_reward": lift_reward.mean(),
            "raised_pose_reward": raised_reward.mean(),
            "wrist_raise_reward": wrist_raise_reward.mean(),
            "wrist_pitch_rad": arm[:, 4].mean(),
            "clearance_ready": clearance_ready.float().mean(),
            "raised_ready": raised_ready.float().mean(),
            "lift_hold": held.float().mean(),
            "finger_max_abs_rad": finger.abs().amax(dim=1).mean(),
            "tip_clearance_m": tip_clearance.mean(),
            "table_clearance_penalty": (self.cfg.table_clearance_penalty_scale * clearance_risk).mean(),
            "action_rate_penalty": (self.cfg.action_rate_scale * action_rate).mean(),
            "success": self.success.float().mean(),
        }
        self.extras["metrics"] = {
            "phase": self.reward_phase,
            "clearance_ready": clearance_ready,
            "raised_ready": raised_ready,
            "lift_hold": held,
            "success": self.success,
            "table_proximity": table_proximity,
            "finger_max_abs_rad": finger.abs().amax(dim=1),
            "tip_clearance_m": tip_clearance,
        }
        return reward
