"""v14: v13 pre-grasp and close, then lift the held cube.

Same as v13 up to ``lift_after_s``. After that the arm is paid for raising the
cube (up to ``lift_target_m``) while the thumb and finger 1 or 2 press opposite
faces; dropping it back toward the table after it rose is penalized. The
v12/v13 placement terms follow the cube, so they keep the hand on it.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass

from .v2_pad_sensor_grasp import JaxV2PadSensorGraspEnv
from .v2_front_grasp_close import JaxV2FrontGraspCloseEnv, JaxV2FrontGraspCloseEnvCfg


@configclass
class JaxV2FrontLiftEnvCfg(JaxV2FrontGraspCloseEnvCfg):
    lift_after_s = 8.0
    lift_target_m = 0.08
    lift_scale = 2.0
    lift_hold_scale = 0.5
    lift_drop_scale = 5.0
    # v14_lift model_2098 (policy arm, hold + 0.5 rad x action): average cube
    # rise 3-5 mm, lift success ~0; random arm offsets slipped the grasp. As
    # in v8 (auto_lift_rot30, 15/16 lifts), the environment raises the hand
    # straight up by differential IK from lift_after_s at lift_speed_mps up to
    # lift_target_m; the policy adds a small arm residual and keeps its finger
    # residual.
    lift_speed_mps = 0.025
    lift_arm_range_rad = 0.05
    # lift_probe (model_1849, scripted lift, no residual): held 97% while the
    # hand rose 17 mm but the cube only 2.5 mm, so the pads slid along the
    # faces. During the lift close the fingers further past the guide shape
    # (deg, finger_ids order) to squeeze harder.
    lift_squeeze_deg = (0.0, 0.0, 10.0, 20.0, 20.0, 20.0)


class JaxV2FrontLiftEnv(JaxV2FrontGraspCloseEnv):
    cfg: JaxV2FrontLiftEnvCfg

    def _pre_physics_step(self, actions: torch.Tensor):
        super()._pre_physics_step(actions)
        arm = self.arm_ids[:7]
        lifting = self.episode_length_buf * self.step_dt >= self.cfg.lift_after_s
        hand_z = self.robot.data.body_pos_w[:, self._hand_id, 2]
        if not hasattr(self, "_ik_arm"):
            self._ik_arm = self.targets[:, arm].clone()
            self._ik_z0 = hand_z.clone()
        # Until the lift starts, track the held arm target and hand height.
        self._ik_arm = torch.where(lifting[:, None], self._ik_arm, self.targets[:, arm])
        self._ik_z0 = torch.where(lifting, self._ik_z0, hand_z)
        rising = lifting & (hand_z - self._ik_z0 < self.cfg.lift_target_m)
        twist = torch.zeros((self.num_envs, 6, 1), device=self.device)
        twist[:, 2, 0] = self.cfg.lift_speed_mps * self.step_dt * rising
        jac = self.robot.root_physx_view.get_jacobians()[:, self._hand_id - 1][:, :, arm]
        dq = torch.linalg.lstsq(jac, twist).solution[:, :, 0]
        self._ik_arm = self._ik_arm + dq * lifting[:, None]
        target = self._ik_arm + self.cfg.lift_arm_range_rad * self.actions[:, :7]
        target = torch.minimum(torch.maximum(target, self.lower[arm]), self.upper[arm])
        self.targets[:, arm] = torch.where(lifting[:, None], target, self.targets[:, arm])
        squeeze = self.targets.new_tensor([math.radians(v) for v in self.cfg.lift_squeeze_deg])
        fingers = self.targets[:, self.finger_ids] + squeeze * lifting[:, None]
        fingers = torch.minimum(torch.maximum(fingers, self.lower[self.finger_ids]), self.upper[self.finger_ids])
        self.targets[:, self.finger_ids] = fingers

    def _get_rewards(self):
        reward = super()._get_rewards()
        log = self.extras["log"]
        if not hasattr(self, "_max_rise"):
            self._max_rise = torch.zeros(self.num_envs, device=self.device)
        lifting = self.episode_length_buf * self.step_dt >= self.cfg.lift_after_s
        fresh = self.episode_length_buf <= 1
        rise = self._cube_rise
        self._max_rise = torch.where(fresh, 0.0, torch.maximum(self._max_rise, rise))
        # Held = v13 distal grasp: thumb and finger 1 or 2 second-link pads.
        distal = JaxV2PadSensorGraspEnv.pad_forces(self)[:, 3:] >= self.cfg.grasp_distal_force_n
        both = distal[:, 2] & distal[:, :2].any(dim=1)
        progress = (rise / self.cfg.lift_target_m).clamp(0.0, 1.0)
        drop = (self._max_rise - rise).clamp_min(0.0)
        lift = (
            both.float() * (self.cfg.lift_scale * progress + self.cfg.lift_hold_scale)
            - self.cfg.lift_drop_scale * drop
        )
        reward = reward + lifting.float() * lift
        n = lifting.float().sum().clamp_min(1.0)
        log["lift_rise_m"] = (rise * lifting).sum() / n
        log["lift_held"] = (both & lifting).float().sum() / n
        log["lift_success"] = ((rise > 0.5 * self.cfg.lift_target_m) & both & lifting).float().sum() / n
        return reward
