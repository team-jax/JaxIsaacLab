"""Preparatory posture task: bend the elbow, open the fingers, and face the cube.

This task shares the Stage 1 robot, scene, observations, and delta actions. It
does not reward contact or closing the fingers and leaves the grasp task intact.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .env import OneArmJaxMinimalEnv, OneArmJaxMinimalEnvCfg


@configclass
class OneArmJaxPregraspEnvCfg(OneArmJaxMinimalEnvCfg):
    episode_length_s = 4.0
    # The reset is deliberately short of the desired posture so opening and
    # elbow flexion require policy actions. The nominal grasp reset is unchanged.
    reset_elbow_offset_rad = 0.25
    reset_shoulder_pitch_offset_rad = 0.20
    reset_shoulder_roll_offset_rad = -0.20
    reset_finger_closure_fraction = 0.30
    reset_from_arm_down = False
    posture_hold_duration_s = 0.5
    posture_reward_scale = 0.02
    posture_success_scale = 5.0
    pregrasp_contact_limit_n = 2.0


class OneArmJaxPregraspEnv(OneArmJaxMinimalEnv):
    cfg: OneArmJaxPregraspEnvCfg

    def __init__(self, cfg: OneArmJaxPregraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.posture_steps = torch.zeros_like(self.hold_steps)
        self.required_posture_steps = round(cfg.posture_hold_duration_s / (cfg.sim.dt * cfg.decimation))

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        q = self.targets[env_ids].clone()
        if self.cfg.reset_from_arm_down:
            q[:, self.arm_ids] = 0.0
        else:
            q[:, self.arm_ids[0]] += self.cfg.reset_shoulder_pitch_offset_rad
            q[:, self.arm_ids[1]] += self.cfg.reset_shoulder_roll_offset_rad
            q[:, self.arm_ids[3]] += self.cfg.reset_elbow_offset_rad
        open_q = torch.tensor(self.cfg.finger_open, device=self.device)
        close_q = torch.tensor(self.cfg.finger_close, device=self.device)
        q[:, self.finger_ids] = open_q + self.cfg.reset_finger_closure_fraction * (close_q - open_q)
        q = torch.minimum(torch.maximum(q, self.lower), self.upper)
        self.targets[env_ids] = q
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)
        if hasattr(self, "posture_steps"):
            self.posture_steps[env_ids] = 0

    def _get_rewards(self):
        q = self.robot.data.joint_pos
        elbow_error = (q[:, self.arm_ids[3]] - self.cfg.initial_arm_q[3]).abs()
        elbow_quality = torch.exp(-5.0 * elbow_error)
        arm_target = torch.tensor(self.cfg.initial_arm_q, device=self.device)
        arm_pose_error = (q[:, self.arm_ids] - arm_target).abs().mean(dim=1)
        arm_pose_quality = 1.0 / (1.0 + arm_pose_error)
        open_q = torch.tensor(self.cfg.finger_open, device=self.device)
        close_q = torch.tensor(self.cfg.finger_close, device=self.device)
        closure = ((q[:, self.finger_ids] - open_q) / (close_q - open_q)).clamp(0.0, 1.0)
        open_quality = 1.0 - closure.mean(dim=1)
        # Stage 0 asks the open hand to face the cube. Exact positions on the
        # three cube faces belong to the later grasp task, not this posture task.
        tips = self.robot.data.body_pos_w[:, self.fingertip_ids]
        wrist = self.robot.data.body_pos_w[:, self.wrist_id]
        cube = self.object.data.root_pos_w
        hand_axis = tips.mean(dim=1) - wrist
        to_cube = cube - wrist
        facing = (hand_axis * to_cube).sum(dim=1) / (
            torch.linalg.vector_norm(hand_axis, dim=1)
            * torch.linalg.vector_norm(to_cube, dim=1)
        ).clamp_min(1e-6)
        tip_distance = torch.linalg.vector_norm(cube - tips.mean(dim=1), dim=1)
        facing_quality = (0.5 * (facing + 1.0)).clamp(0.0, 1.0)
        if self.cfg.reset_from_arm_down:
            # At the distant arm-down reset, a product of direction and
            # proximity was virtually zero. Give each a usable gradient.
            reach_quality = 1.0 / (1.0 + 4.0 * tip_distance)
            alignment_quality = 0.5 * (facing_quality + reach_quality)
            aligned = (facing_quality > 0.90) & (tip_distance < 0.12)
        else:
            alignment_quality = facing_quality * torch.exp(
                -6.0 * (tip_distance - 0.08).clamp_min(0.0)
            )
            aligned = alignment_quality > 0.90
        _, force_n, _, _ = self._contacts()
        min_tip_z = self.robot.data.body_pos_w[:, self.fingertip_ids, 2].amin(dim=1)
        clear = min_tip_z > self.cfg.table_top_z + 0.02
        clearance_quality = ((min_tip_z - (self.cfg.table_top_z - 0.05)) / 0.09).clamp(0.0, 1.0)
        no_contact = force_n.amax(dim=1) < self.cfg.pregrasp_contact_limit_n
        posture_ready = (
            (elbow_error < 0.10) & (open_quality > 0.85)
            & aligned & clear & no_contact
        )
        self.posture_steps = torch.where(posture_ready, self.posture_steps + 1, 0)
        self.success = self.posture_steps >= self.required_posture_steps
        action_rate = ((self.actions - self.previous_actions) ** 2).sum(dim=1)
        unsafe = force_n.amax(dim=1) > self.cfg.unsafe_contact_force_n
        self.unsafe_steps = torch.where(unsafe, self.unsafe_steps + 1, 0)
        fallen = self.object.data.root_pos_w[:, 2] < self.cfg.table_top_z - 0.05
        failed = fallen | (self.unsafe_steps >= self.cfg.unsafe_consecutive_steps)
        posture_reward = (
            0.04 * arm_pose_quality + 0.03 * alignment_quality
            + 0.01 * open_quality + 0.03 * clearance_quality
            if self.cfg.reset_from_arm_down else
            self.cfg.posture_reward_scale * (elbow_quality + open_quality + alignment_quality) / 3.0
        )
        reward = (
            posture_reward + self.cfg.posture_success_scale * self.success.float()
            - 0.005 * (~no_contact).float()
            - self.cfg.failure_termination_penalty * failed.float()
            - self.cfg.action_rate_scale * action_rate
        )
        self.extras["log"] = {
            "elbow_quality": elbow_quality.mean(),
            "arm_pose_quality": arm_pose_quality.mean(),
            "open_quality": open_quality.mean(),
            "alignment_quality": alignment_quality.mean(),
            "tip_distance_m": tip_distance.mean(),
            "clearance_quality": clearance_quality.mean(),
            "facing_quality": facing_quality.mean(),
            "posture_ready": posture_ready.float().mean(),
            "posture_hold_steps": self.posture_steps.float().mean(),
            "success": self.success.float().mean(),
            "contact_force_n": force_n.amax(dim=1).mean(),
            "contact_penalty": (0.005 * (~no_contact).float()).mean(),
            "action_rate_penalty": (self.cfg.action_rate_scale * action_rate).mean(),
        }
        self.extras["metrics"] = {
            "elbow_ready": elbow_error < 0.10,
            "open_ready": open_quality > 0.85,
            "aligned": aligned,
            "clear": clear,
            "no_contact": no_contact,
            "posture_ready": posture_ready,
            "success": self.success,
            "elbow_error_rad": elbow_error,
            "open_quality": open_quality,
            "alignment_quality": alignment_quality,
            "tip_distance_m": tip_distance,
            "facing_quality": facing_quality,
            "force_n": force_n,
        }
        return reward


@configclass
class OneArmJaxPregraspDownEnvCfg(OneArmJaxPregraspEnvCfg):
    """The same posture goal, starting from the URDF arm-down zero pose."""

    reset_from_arm_down = True
