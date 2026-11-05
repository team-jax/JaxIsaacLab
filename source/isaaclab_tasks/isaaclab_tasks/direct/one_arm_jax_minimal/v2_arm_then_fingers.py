"""Scripted collision-free preparation, followed by policy-controlled grasp.

The preparation is not learned. PPO receives observations throughout, but its
actions take effect only after the arm is raised, fingers open, and arm lowered
to the verified ready pose. This isolates the grasp-learning task.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass

from .v2_grasp import JaxV2GraspEnv, JaxV2GraspEnvCfg


@configclass
class JaxV2ArmThenFingersEnvCfg(JaxV2GraspEnvCfg):
    episode_length_s = 31.0
    # The 19-second setup leaves about 12 seconds for PPO grasp control.
    prepared_elbow_rad = 1.70
    prepared_arm_tolerance_rad = 0.12


class JaxV2ArmThenFingersEnv(JaxV2GraspEnv):
    cfg: JaxV2ArmThenFingersEnvCfg

    def __init__(self, cfg: JaxV2ArmThenFingersEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.prep_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.preparing = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if hasattr(self, "prep_step"):
            ids = self.robot._ALL_INDICES if env_ids is None else env_ids
            self.prep_step[ids] = 0
            self.preparing[ids] = True

    @staticmethod
    def _smooth(t: torch.Tensor, start: float, duration: float) -> torch.Tensor:
        fraction = ((t - start) / duration).clamp(0.0, 1.0)
        return 0.5 - 0.5 * torch.cos(math.pi * fraction)

    def _pre_physics_step(self, actions: torch.Tensor):
        super()._pre_physics_step(actions)
        t = self.prep_step.float() * (self.cfg.sim.dt * self.cfg.decimation)
        self.preparing = t < 19.0
        if not self.preparing.any():
            return

        start = torch.zeros((self.num_envs, 8), device=self.device)
        clearance = start.clone()
        clearance[:, 0] = -0.15
        clearance[:, 2] = -0.50
        clearance[:, 3] = 1.45
        raised = torch.tensor(self.cfg.initial_arm_q, device=self.device).expand_as(start).clone()
        raised[:, 3] = self.cfg.prepared_elbow_rad
        final = torch.tensor(self.cfg.initial_arm_q, device=self.device).expand_as(start)

        first = self._smooth(t, 0.5, 5.0)[:, None]
        second = self._smooth(t, 5.5, 2.5)[:, None]
        lift = self._smooth(t, 7.0, 1.5)[:, None]
        orient = self._smooth(t, 8.5, 3.0)[:, None]
        lower = self._smooth(t, 15.5, 3.0)[:, None]
        arm_target = first * clearance
        arm_target[:, 0] += first[:, 0] * (-0.45 - clearance[:, 0]) + second[:, 0] * 0.30
        lifted_clearance = clearance.clone()
        lifted_clearance[:, 3] = self.cfg.prepared_elbow_rad
        arm_target += lift * (lifted_clearance - clearance)
        arm_target += orient * (raised - lifted_clearance)
        arm_target += lower * (final - raised)

        # At the opening boundary, wait for the *actual* arm joints to reach
        # the raised target. The finger command remains neutral until then.
        arm_error = (self.robot.data.joint_pos[:, self.arm_ids] - raised).abs().amax(dim=1)
        hold_for_arm = (t >= 12.5) & (t < 15.5) & (
            arm_error > self.cfg.prepared_arm_tolerance_rad
        )
        open_fraction = self._smooth(t, 12.5, 3.0)[:, None]
        finger_open = torch.tensor(self.cfg.finger_open, device=self.device)
        finger_target = open_fraction * finger_open
        finger_target = torch.where(hold_for_arm[:, None], torch.zeros_like(finger_target), finger_target)
        self.prep_step += (~hold_for_arm).long()

        self.targets[:, self.arm_ids] = torch.where(
            self.preparing[:, None], arm_target, self.targets[:, self.arm_ids]
        )
        self.targets[:, self.finger_ids] = torch.where(
            self.preparing[:, None], finger_target, self.targets[:, self.finger_ids]
        )
        # Ignored policy actions must not incur an action-rate penalty.
        self.actions[self.preparing] = 0.0
        self.previous_actions[self.preparing] = 0.0

    def _get_rewards(self):
        reward = super()._get_rewards()
        self.extras["log"]["scripted_preparation"] = self.preparing.float().mean()
        self.extras["metrics"]["scripted_preparation"] = self.preparing
        return torch.where(self.preparing, torch.zeros_like(reward), reward)
