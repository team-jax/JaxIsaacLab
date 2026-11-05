"""v10: raise the arm from hanging into an L pose with the hand lined up on the cube's y.

L pose (measured on the V2 URDF from the all-zero hanging start): shoulder
pitch and roll 0 keep the upper arm vertical, elbow +90 deg makes the forearm
horizontal toward the table (-x), and the three wrist joints at 0 keep the hand
along the forearm. Shoulder yaw then turns the forearm about the vertical upper
arm; at about -30 deg the finger second links sit at the cube's y (0.077 m vs
0.080 m), 15 cm above it. Yaw is left free and the y error is rewarded directly.

The env keeps the v9 observation (43) and action (11) interface so later grasp
stages can resume from this checkpoint. Fingers and RotationBody are held at
their reset values; the cube only has to stay untouched.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass

from .v2_pad_sensor_grasp import JaxV2PadSensorGraspEnv, JaxV2PadSensorGraspEnvCfg


@configclass
class JaxV2LRaiseEnvCfg(JaxV2PadSensorGraspEnvCfg):
    # The raise takes ~2 s; 10 s episodes spent most samples holding. 4 s
    # gives 2.5x more raises per iteration.
    episode_length_s = 4.0
    sensor_randomize = False
    auto_lift = False
    # Arm joints 0..6: shoulder pitch, roll, yaw, elbow, forearm roll, wrist
    # pitch, hand yaw. None leaves the joint free (yaw aligns y).
    # Yaw free (None) left no signal to turn the forearm inward before the
    # elbow rises, and at yaw 0 the fingertips meet the table edge; -30 deg is
    # the measured yaw that puts the finger second links at the cube's y.
    l_pose_q = (0.0, 0.0, math.radians(-30.0), math.pi / 2, 0.0, 0.0, 0.0)
    # From hanging the pose error is ~90 deg and the y error ~0.21 m, where
    # exp(-3 err) * exp(-y / 0.02) is ~1e-7 (no signal). Linear terms give a
    # gradient from the start; the sharp product pays for the exact pose.
    l_pose_linear_rad = 1.6
    l_y_linear_m = 0.25
    l_pose_sharpness = 5.0
    # y_only model_999 met the y target with the hand behind the body
    # (shoulder pitch -33 deg) and every shaped joint parked at the same 33 deg
    # error, where improving one joint does not lower the max. Pay y only with
    # the finger second links in front of the robot, and shape with the mean
    # joint error (success still uses the max).
    l_front_x_m = -0.25
    l_front_sharpness_m = 0.03
    l_y_scale_m = 0.01
    l_progress_scale = 2.0
    # yaw30 model_299 barely moved (elbow 14 deg): at 0.05 a full elbow raise
    # was worth ~0.002/step, below the exploration noise.
    l_hold_scale = 0.5
    l_success_pose_rad = 0.08
    l_success_y_m = 0.01
    l_success_speed_rad_s = 0.2
    l_success_scale = 0.1
    l_table_clearance_m = 0.03
    l_table_scale = 0.05
    # y_only swung the arm backward (pitch -33 deg) and turned it past the
    # pole with zero measured intrusion: the keep-out samples segment centre
    # lines only, so allow for the limb thickness and forbid backward pitch.
    body_margin_m = 0.04
    l_body_scale = 5.0
    l_min_shoulder_pitch_rad = -0.05
    # hold05 model_149 found the mirrored L: shoulder yaw +165 deg with the
    # elbow bent backward (-93 deg). Keep the elbow forward and the yaw near
    # the measured -30 deg.
    l_elbow_range_rad = (0.0, 1.75)
    l_shoulder_yaw_range_rad = (-1.57, 0.52)
    l_cube_move_scale = 1.0
    l_action_rate_scale = 0.002


class JaxV2LRaiseEnv(JaxV2PadSensorGraspEnv):
    cfg: JaxV2LRaiseEnvCfg

    def __init__(self, cfg: JaxV2LRaiseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        shaped = [i for i, q in enumerate(cfg.l_pose_q) if q is not None]
        self._l_ids = [self.arm_ids[i] for i in shaped]
        self._l_q = torch.tensor([cfg.l_pose_q[i] for i in shaped], device=self.device)
        self._l_prev_quality = torch.zeros(self.num_envs, device=self.device)
        self._l_hand_ids = list(self.fingertip_ids) + self.proximal_ids + [self._hand_id]

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if hasattr(self, "_l_prev_quality"):
            self._l_prev_quality[self.robot._ALL_INDICES if env_ids is None else env_ids] = 0.0

    def _pre_physics_step(self, actions: torch.Tensor):
        # Delta-position control of the seven arm joints through the v8 action
        # filter; fingers and RotationBody keep their reset targets.
        alpha = self.cfg.action_filter_alpha
        self._filtered_actions = alpha * actions.clamp(-1.0, 1.0) + (1.0 - alpha) * self._filtered_actions
        self.previous_actions.copy_(self.actions)
        self.actions = self._filtered_actions.clone()
        arm = self.arm_ids[:7]
        self.targets[:, arm] += self.cfg.arm_delta_scale * self.actions[:, :7]
        self.targets[:, arm] = torch.minimum(torch.maximum(self.targets[:, arm], self.lower[arm]), self.upper[arm])
        pitch = self.arm_ids[0]
        self.targets[:, pitch] = self.targets[:, pitch].clamp_min(self.cfg.l_min_shoulder_pitch_rad)
        yaw, elbow = self.arm_ids[2], self.arm_ids[3]
        self.targets[:, yaw] = self.targets[:, yaw].clamp(*self.cfg.l_shoulder_yaw_range_rad)
        self.targets[:, elbow] = self.targets[:, elbow].clamp(*self.cfg.l_elbow_range_rad)
        self.targets[:, self.finger_ids] = 0.0

    def _l_errors(self):
        pose_err = (self.robot.data.joint_pos[:, self._l_ids] - self._l_q).abs().amax(dim=1)
        grasp_y = self.robot.data.body_pos_w[:, self.fingertip_ids, 1].mean(dim=1)
        y_err = (grasp_y - self.object.data.root_pos_w[:, 1]).abs()
        return pose_err, y_err

    def _l_front(self):
        grasp_x = (self.robot.data.body_pos_w[:, self.fingertip_ids, 0] - self.scene.env_origins[:, None, 0]).mean(dim=1)
        return torch.sigmoid((self.cfg.l_front_x_m - grasp_x) / self.cfg.l_front_sharpness_m), grasp_x

    def _get_dones(self):
        self._body_intrusion = self._body_intrusion_depth()
        self._cube_shift, self._cube_rotation_deg = self._cube_pose_change()
        fallen = self.object.data.root_pos_w[:, 2] < self.cfg.table_top_z - 0.05
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return fallen, timeout

    def _get_rewards(self):
        cube_pos = self.object.data.root_pos_w
        self._cube_start = torch.where(self._cube_start_valid[:, None], self._cube_start, cube_pos)
        self._cube_start_quat = torch.where(
            self._cube_start_valid[:, None], self._cube_start_quat, self.object.data.root_quat_w
        )
        self._cube_start_valid[:] = True

        pose_err, y_err = self._l_errors()
        front, grasp_x = self._l_front()
        mean_err = (self.robot.data.joint_pos[:, self._l_ids] - self._l_q).abs().mean(dim=1)
        pose_lin = (1.0 - mean_err / self.cfg.l_pose_linear_rad).clamp(0.0, 1.0)
        y_lin = (1.0 - y_err / self.cfg.l_y_linear_m).clamp(0.0, 1.0) * front
        sharp = torch.exp(-self.cfg.l_pose_sharpness * pose_err) * torch.exp(-y_err / self.cfg.l_y_scale_m) * front
        quality = 0.25 * pose_lin + 0.25 * y_lin + 0.5 * sharp
        fresh = self.episode_length_buf <= 1
        progress = torch.where(fresh, 0.0, quality - self._l_prev_quality)
        self._l_prev_quality = quality.detach()
        arm_speed = self.robot.data.joint_vel[:, self.arm_ids[:7]].abs().amax(dim=1)
        success = (
            (pose_err < self.cfg.l_success_pose_rad) & (y_err < self.cfg.l_success_y_m) & (front > 0.5)
            & (arm_speed < self.cfg.l_success_speed_rad_s)
        )

        local = self.robot.data.body_pos_w[:, self._l_hand_ids] - self.scene.env_origins[:, None, :]
        table_x, table_y, _ = self.cfg.table_center
        size_x, size_y, _ = self.cfg.table_cfg.size
        over_table = ((local[..., 0] - table_x).abs() < size_x / 2 + 0.03) & (
            (local[..., 1] - table_y).abs() < size_y / 2 + 0.03
        )
        target = self.cfg.l_table_clearance_m
        deficit = ((target - (local[..., 2] - self.cfg.table_top_z)) / target).clamp(0.0, 1.0) * over_table
        table_penalty = self.cfg.l_table_scale * deficit.amax(dim=1)
        body_penalty = self.cfg.l_body_scale * self._body_intrusion
        cube_penalty = self.cfg.l_cube_move_scale * (self._cube_shift + (self._cube_rise.abs() > 0.005).float() * 0.01)
        action_rate = ((self.actions[:, :7] - self.previous_actions[:, :7]) ** 2).sum(dim=1)
        rate_penalty = self.cfg.l_action_rate_scale * action_rate

        reward = (
            self.cfg.l_progress_scale * progress + self.cfg.l_hold_scale * quality
            + self.cfg.l_success_scale * success.float()
            - table_penalty - body_penalty - cube_penalty - rate_penalty
        )
        self.extras["log"] = {
            "l_quality": quality.mean(),
            "l_success": success.float().mean(),
            "l_pose_err_deg": torch.rad2deg(pose_err).mean(),
            "l_y_err_mm": (1000.0 * y_err).mean(),
            "l_grasp_x_m": grasp_x.mean(),
            "l_shoulder_yaw_deg": torch.rad2deg(self.robot.data.joint_pos[:, self.arm_ids[2]]).mean(),
            "l_table_penalty": table_penalty.mean(),
            "l_body_intrusion_m": self._body_intrusion.mean(),
            "l_cube_shift_m": self._cube_shift.mean(),
        }
        return reward
