"""One PPO policy learns the verified lift-then-open preparation and grasp.

Phases gate rewards only. Every arm and finger action is always applied by the
original delta-position controller in JaxV2GraspEnv.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .v2_grasp import JaxV2GraspEnv, JaxV2GraspEnvCfg


@configclass
class JaxV2LearnSequenceEnvCfg(JaxV2GraspEnvCfg):
    episode_length_s = 25.0
    raised_elbow_rad = 1.70
    raised_arm_tolerance_rad = 0.12
    open_finger_tolerance_rad = 0.15
    lift_arc_scale = 0.020
    phase_progress_scale = 0.002
    raised_pose_scale = 0.020
    open_after_raise_scale = 0.020
    lower_open_hand_scale = 0.020
    arm_ready_bonus = 1.0
    fingers_open_bonus = 1.0
    lowered_open_hand_bonus = 1.0
    premature_open_scale = 0.040
    premature_rotation_scale = 0.010


class JaxV2LearnSequenceEnv(JaxV2GraspEnv):
    cfg: JaxV2LearnSequenceEnvCfg

    def __init__(self, cfg: JaxV2LearnSequenceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.reward_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if hasattr(self, "reward_phase"):
            self.reward_phase[self.robot._ALL_INDICES if env_ids is None else env_ids] = 0

    def _pre_physics_step(self, actions: torch.Tensor):
        super()._pre_physics_step(actions)
        # Project only arm motor targets onto the clearance path verified for
        # the V2 URDF. PPO still chooses the elbow motion and every finger
        # action; the projection prevents an elbow lift with the shoulder
        # pointed toward the tabletop.
        preparing = self.reward_phase == 0
        elbow_id = self.arm_ids[3]
        elbow = self.targets[:, elbow_id].clamp(0.0, 1.45)
        progress = elbow / 1.45
        self.targets[:, elbow_id] = torch.where(preparing, elbow, self.targets[:, elbow_id])
        pitch_id, yaw_id = self.arm_ids[0], self.arm_ids[2]
        self.targets[:, pitch_id] = torch.where(
            preparing, torch.minimum(self.targets[:, pitch_id], -0.15 - 0.30 * progress), self.targets[:, pitch_id]
        )
        self.targets[:, yaw_id] = torch.where(
            preparing, torch.minimum(self.targets[:, yaw_id], -0.50 * progress), self.targets[:, yaw_id]
        )
        # The verified wrist stays neutral until the clearance pose is reached.
        for arm_index in (1, 4, 5, 6, 7):
            joint_id = self.arm_ids[arm_index]
            self.targets[:, joint_id] = torch.where(
                preparing, torch.zeros_like(self.targets[:, joint_id]), self.targets[:, joint_id]
            )
        # Once clear of the table edge, keep the elbow above the measured
        # clearance angle during arm orientation and finger opening.
        raised_preparation = (self.reward_phase > 0) & (self.reward_phase < 4)
        self.targets[:, elbow_id] = torch.where(
            raised_preparation, self.targets[:, elbow_id].clamp_min(1.45), self.targets[:, elbow_id]
        )

    def _get_rewards(self):
        arm = self.robot.data.joint_pos[:, self.arm_ids]
        finger = self.robot.data.joint_pos[:, self.finger_ids]
        tips = self.robot.data.body_pos_w[:, self.fingertip_ids]
        cube = self.object.data.root_pos_w
        distance = torch.linalg.vector_norm(tips.mean(dim=1) - cube, dim=1)
        pregrasp_quality, _ = self._pregrasp_quality()

        final = torch.tensor(self.cfg.initial_arm_q, device=self.device)
        raised = final.clone()
        raised[3] = self.cfg.raised_elbow_rad
        arm_error = (arm - raised).abs()
        raised_quality = torch.exp(-2.0 * arm_error.mean(dim=1))
        arm_ready = arm_error.amax(dim=1) < self.cfg.raised_arm_tolerance_rad

        open_q = torch.tensor(self.cfg.finger_open, device=self.device)
        close_q = torch.tensor(self.cfg.finger_close, device=self.device)
        open_error = (finger - open_q).abs()
        open_quality = torch.exp(-2.0 * open_error.mean(dim=1))
        fingers_open = open_error.amax(dim=1) < self.cfg.open_finger_tolerance_rad

        # The verified motion first reaches a clearance pose with neutral
        # fingers, then rotates to the raised pose before finger opening.
        clearance = torch.zeros_like(raised)
        clearance[0], clearance[2], clearance[3] = -0.45, -0.50, 1.45
        clearance_error = (arm - clearance).abs()
        clearance_ready = clearance_error.amax(dim=1) < 0.18
        elbow = arm[:, 3].clamp_min(0.0)
        arc_error = ((arm[:, 0] + 0.45 * (elbow / 1.45).clamp(max=1.0)).abs()
                     + (arm[:, 2] + 0.50 * (elbow / 1.45).clamp(max=1.0)).abs()) / 2.0
        wrist_rotation = torch.cat((arm[:, 1:2], arm[:, 4:8]), dim=1).square().mean(dim=1)
        arc_quality = (elbow / 1.45).clamp(0.0, 1.0) * torch.exp(
            -2.0 * (arc_error + wrist_rotation)
        )
        arc_ready = clearance_ready

        old_phase = self.reward_phase.clone()
        neutral_fingers = finger.abs().amax(dim=1) < 0.12
        self.reward_phase = torch.where((old_phase == 0) & arc_ready & neutral_fingers, 1, old_phase)
        # Reaching the arm goal with fingers already open does not skip the
        # required "arm first, then open" order.
        arm_transition = (self.reward_phase == 1) & arm_ready & neutral_fingers
        self.reward_phase = torch.where(arm_transition, 2, self.reward_phase)
        finger_transition = (old_phase == 2) & fingers_open & arm_ready
        self.reward_phase = torch.where(finger_transition, 3, self.reward_phase)
        final_error = (arm - final).abs()
        final_quality = torch.exp(-2.0 * final_error.mean(dim=1))
        lowered_open_hand = (final_error.amax(dim=1) < self.cfg.raised_arm_tolerance_rad) & fingers_open
        lower_transition = (old_phase == 3) & lowered_open_hand
        self.reward_phase = torch.where(lower_transition, 4, self.reward_phase)

        lift_reward = self.cfg.lift_arc_scale * arc_quality * (old_phase == 0)
        phase_progress_reward = self.cfg.phase_progress_scale * old_phase.float()
        raised_reward = self.cfg.raised_pose_scale * raised_quality * (old_phase == 1)
        open_reward = (
            self.cfg.open_after_raise_scale * open_quality * raised_quality * (old_phase == 2)
        )
        lower_reward = self.cfg.lower_open_hand_scale * final_quality * open_quality * (old_phase == 3)
        # The policy can move every joint at every step. Opening before the
        # raised arm is verified gives no opening reward and incurs a penalty.
        # Neutral fingers are required throughout the lift and arm alignment.
        # Penalize the largest deviating finger so one early-spread finger
        # cannot be hidden by the other five near-zero finger joints.
        premature_open = finger.abs().amax(dim=1).clamp(0.0, 1.0) * (old_phase < 2)
        premature_rotation = wrist_rotation.clamp(0.0, 1.0) * (old_phase == 0)

        reach_reward = self.cfg.reach_scale / (1.0 + 4.0 * distance) * (old_phase == 4)
        pregrasp_reward = self.cfg.pregrasp_scale * pregrasp_quality * (old_phase == 4)
        valid, force_n, all_valid, contact_quality = self._contacts()
        contact_reward = self.cfg.inner_contact_scale * contact_quality * (old_phase == 4)
        hand_vel = self.robot.data.body_lin_vel_w[:, self.fingertip_ids].mean(dim=1)
        relative_speed = torch.linalg.vector_norm(self.object.data.root_lin_vel_w - hand_vel, dim=1)
        stable = all_valid & (relative_speed < self.cfg.max_relative_speed_mps)
        self.hold_steps = torch.where(stable & (old_phase == 4), self.hold_steps + 1, 0)
        self.success = self.hold_steps >= self.required_hold_steps
        stable_reward = self.cfg.stable_grasp_scale * stable.float() * (old_phase == 4)

        local_tips = tips - self.scene.env_origins[:, None, :]
        table_x, table_y, _ = self.cfg.table_center
        size_x, size_y, _ = self.cfg.table_cfg.size
        over_table = (
            (local_tips[..., 0] > table_x - size_x / 2)
            & (local_tips[..., 0] < table_x + size_x / 2)
            & ((local_tips[..., 1] - table_y).abs() < size_y / 2)
        )
        low_over_table = (over_table & (local_tips[..., 2] < self.cfg.table_top_z + 0.02)).any(dim=1)
        unsafe = force_n.amax(dim=1) > self.cfg.unsafe_contact_force_n
        self.unsafe_steps = torch.where(unsafe, self.unsafe_steps + 1, 0)
        fallen = cube[:, 2] < self.cfg.table_top_z - 0.05
        failed = fallen | (self.unsafe_steps >= self.cfg.unsafe_consecutive_steps)
        action_rate = ((self.actions - self.previous_actions) ** 2).sum(dim=1)
        reward = (
            lift_reward + phase_progress_reward + raised_reward + open_reward + lower_reward + reach_reward + pregrasp_reward
            + contact_reward + stable_reward
            + self.cfg.arm_ready_bonus * arm_transition.float()
            + self.cfg.fingers_open_bonus * finger_transition.float()
            + self.cfg.lowered_open_hand_bonus * lower_transition.float()
            + self.cfg.success_scale * self.success.float()
            - self.cfg.premature_open_scale * premature_open
            - self.cfg.premature_rotation_scale * premature_rotation
            - self.cfg.table_proximity_scale * low_over_table.float()
            - self.cfg.failure_termination_penalty * failed.float()
            - self.cfg.action_rate_scale * action_rate
        )
        self.extras["log"] = {
            "lift_arc_reward": lift_reward.mean(),
            "phase_progress_reward": phase_progress_reward.mean(),
            "raised_pose_reward": raised_reward.mean(),
            "open_after_raise_reward": open_reward.mean(),
            "lower_open_hand_reward": lower_reward.mean(),
            "reach_reward": reach_reward.mean(),
            "pregrasp_reward": pregrasp_reward.mean(),
            "inner_contact_reward": contact_reward.mean(),
            "stable_grasp_reward": stable_reward.mean(),
            "arm_ready": arm_ready.float().mean(),
            "fingers_open_after_raise": finger_transition.float().mean(),
            "phase_lift": (self.reward_phase == 0).float().mean(),
            "phase_align": (self.reward_phase == 1).float().mean(),
            "phase_open": (self.reward_phase == 2).float().mean(),
            "phase_lower": (self.reward_phase == 3).float().mean(),
            "phase_grasp": (self.reward_phase == 4).float().mean(),
            "premature_open_penalty": (self.cfg.premature_open_scale * premature_open).mean(),
            "premature_rotation_penalty": (self.cfg.premature_rotation_scale * premature_rotation).mean(),
            "table_proximity_penalty": (self.cfg.table_proximity_scale * low_over_table.float()).mean(),
            "failure_penalty": (self.cfg.failure_termination_penalty * failed.float()).mean(),
            "action_rate_penalty": (self.cfg.action_rate_scale * action_rate).mean(),
            "success": self.success.float().mean(),
        }
        self.extras["metrics"] = {
            "phase": self.reward_phase,
            "clearance_ready": clearance_ready,
            "lowered_open_hand": lowered_open_hand,
            "arc_ready": arc_ready,
            "arm_ready": arm_ready,
            "fingers_open": fingers_open,
            "reach": distance < self.cfg.reach_distance_m,
            "pregrasp_quality": pregrasp_quality,
            "valid_contact": valid,
            "contact_3": all_valid,
            "stable": stable,
            "hold": self.hold_steps > 0,
            "success": self.success,
            "unsafe_contact": unsafe,
            "table_proximity": low_over_table,
            "early_close": premature_open > 0.01,
            "force_n": force_n,
        }
        return reward
