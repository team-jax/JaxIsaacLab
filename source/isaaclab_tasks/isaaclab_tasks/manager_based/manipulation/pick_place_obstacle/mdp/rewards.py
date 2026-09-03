# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Floor / table penalty
# ---------------------------------------------------------------------------

def ee_floor_penalty(
    env: "ManagerBasedRLEnv",
    min_height: float = 0.05,
    std: float = 0.03,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalty when EE descends below min_height (table surface guard).

    Returns 0 when EE is above min_height, rises smoothly toward +1 as EE
    drops below it. Pair with a negative weight to get a penalty.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_z = ee_frame.data.target_pos_w[..., 0, 2]
    below = torch.clamp(min_height - ee_z, min=0.0)
    return torch.tanh(below / std)


# ---------------------------------------------------------------------------
# Approach rewards
# ---------------------------------------------------------------------------

def ee_object_distance(
    env: "ManagerBasedRLEnv",
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Dense reward: EE approaching the target object (tanh kernel)."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    dist = torch.norm(obj.data.root_pos_w - ee_frame.data.target_pos_w[..., 0, :], dim=-1)
    return 1.0 - torch.tanh(dist / std)


def object_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward when object z exceeds minimal_height."""
    return (env.scene[object_cfg.name].data.root_pos_w[:, 2] > minimal_height).float()


# ---------------------------------------------------------------------------
# Goal rewards
# ---------------------------------------------------------------------------

def object_goal_distance(
    env: "ManagerBasedRLEnv",
    std: float,
    minimal_height: float,
    goal_pos_w: list,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense reward: lifted object approaching fixed goal (tanh kernel)."""
    obj: RigidObject = env.scene[object_cfg.name]
    goal = torch.tensor(goal_pos_w, device=env.device, dtype=torch.float32)
    dist = torch.norm(obj.data.root_pos_w - goal.unsqueeze(0), dim=-1)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    return lifted.float() * (1.0 - torch.tanh(dist / std))


def place_success(
    env: "ManagerBasedRLEnv",
    threshold: float,
    goal_pos_w: list,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Sparse reward: object placed within threshold of goal position."""
    obj: RigidObject = env.scene[object_cfg.name]
    goal = torch.tensor(goal_pos_w, device=env.device, dtype=torch.float32)
    dist = torch.norm(obj.data.root_pos_w - goal.unsqueeze(0), dim=-1)
    return (dist < threshold).float()


# ---------------------------------------------------------------------------
# Grasp intermediate reward
# ---------------------------------------------------------------------------

def grasp_effort(
    env: "ManagerBasedRLEnv",
    std: float,
    close_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Intermediate reward: bonus when EE is close to object AND gripper is closing.

    Bridges the learning gap between "approach" and "lift":
      near  = tanh-gated EE-object distance
      close = gripper close signal from last action (last action dim, + = close)
      reward = near × clamp(close, 0, 1) × (dist < close_threshold)
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    dist = torch.norm(obj.data.root_pos_w - ee_pos, dim=-1)
    near = 1.0 - torch.tanh(dist / std)
    gripper_signal = torch.clamp(env.action_manager.action[:, -1], min=0.0, max=1.0)
    within_reach = (dist < close_threshold).float()
    return near * gripper_signal * within_reach


# ---------------------------------------------------------------------------
# Obstacle proximity penalty
# ---------------------------------------------------------------------------

def obstacle_proximity_penalty(
    env: "ManagerBasedRLEnv",
    safe_dist: float,
    std: float,
    obstacle_0_a_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_0_a"),
    obstacle_0_b_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_0_b"),
    obstacle_0_c_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_0_c"),
    obstacle_0_d_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_0_d"),
    obstacle_0_e_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_0_e"),
    obstacle_1_a_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_1_a"),
    obstacle_1_b_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_1_b"),
    obstacle_1_c_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_1_c"),
    obstacle_1_d_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_1_d"),
    obstacle_1_e_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_1_e"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Potential-field penalty: negative reward when EE approaches active obstacles.

    Parked variants (x ≈ 200 m) contribute ≈ 0 penalty automatically.
    Penalty is normalised over 2 slots, not 10 variants.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]

    all_cfgs = [
        obstacle_0_a_cfg, obstacle_0_b_cfg, obstacle_0_c_cfg, obstacle_0_d_cfg, obstacle_0_e_cfg,
        obstacle_1_a_cfg, obstacle_1_b_cfg, obstacle_1_c_cfg, obstacle_1_d_cfg, obstacle_1_e_cfg,
    ]

    penalty = torch.zeros(env.num_envs, device=env.device)
    for cfg in all_cfgs:
        obs: RigidObject = env.scene[cfg.name]
        dist = torch.norm(ee_pos - obs.data.root_pos_w, dim=-1)
        delta = torch.clamp(safe_dist - dist, min=0.0)
        penalty += 1.0 - torch.tanh(delta / std)

    return -(penalty / 2.0)  # normalise by number of slots


# ---------------------------------------------------------------------------
# Obstacle contact penalty (paired with obstacle_contact termination)
# ---------------------------------------------------------------------------

def obstacle_contact_penalty(
    env: "ManagerBasedRLEnv",
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_contact"),
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """One-shot penalty applied on the termination step when obstacle contact occurs.

    Returns -1.0 for envs where panda_hand contact force > threshold AND the EE
    is geometrically near an active obstacle (same two-gate logic as the termination
    function). Weight is set large in RewardsCfg to dwarf the proximity penalty.
    """
    from .terminations import obstacle_contact as _contact_check
    triggered = _contact_check(env, contact_sensor_cfg, force_threshold)
    return -triggered.float()
