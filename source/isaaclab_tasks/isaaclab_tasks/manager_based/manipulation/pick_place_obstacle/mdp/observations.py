# SPDX-License-Identifier: BSD-3-Clause
"""Custom observation functions — vision-pipeline-compatible format.

Obstacle system: 2 slots, each with 5 box size variants.
  Slot 0 → obstacle_0_a / _b / _c / _d / _e
  Slot 1 → obstacle_1_a / _b / _c / _d / _e

Each episode one variant per slot is randomly activated (placed in workspace);
the other four are parked at x=200 m. Observations return zeros for inactive slots.
env._obstacle_active_variants: (num_envs, 2) long tensor, -1 = inactive.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ---------------------------------------------------------------------------
# Variant metadata (must match scene config sizes and init_state z values)
# [slot_idx][variant_idx] → (scene_name, dims [l, w, h])
# ---------------------------------------------------------------------------

OBSTACLE_VARIANTS: list[list[tuple[str, list[float]]]] = [
    [
        ("obstacle_0_a", [0.05, 0.05, 0.10]),  # small pillar
        ("obstacle_0_b", [0.08, 0.08, 0.15]),  # medium pillar
        ("obstacle_0_c", [0.06, 0.06, 0.22]),  # tall pillar
        ("obstacle_0_d", [0.10, 0.10, 0.08]),  # wide squat block
        ("obstacle_0_e", [0.07, 0.07, 0.18]),  # medium-tall pillar
    ],
    [
        ("obstacle_1_a", [0.05, 0.05, 0.10]),
        ("obstacle_1_b", [0.08, 0.08, 0.15]),
        ("obstacle_1_c", [0.06, 0.06, 0.22]),
        ("obstacle_1_d", [0.10, 0.10, 0.08]),
        ("obstacle_1_e", [0.07, 0.07, 0.18]),
    ],
]

_INACTIVE_THRESHOLD = 100.0  # x > this → obstacle is parked / inactive


# ---------------------------------------------------------------------------
# Internal helpers (also used by terminations.py)
# ---------------------------------------------------------------------------

def _active_obstacle_pos(env: "ManagerBasedRLEnv", slot: int) -> torch.Tensor:
    """Return (N, 3) world-frame position of the active variant; zeros when inactive."""
    active = getattr(env, "_obstacle_active_variants", None)
    variants = OBSTACLE_VARIANTS[slot]
    pos_out = torch.zeros(env.num_envs, 3, device=env.device)

    for v_idx, (name, _) in enumerate(variants):
        asset: RigidObject = env.scene[name]
        pos = asset.data.root_pos_w
        mask = (active[:, slot] == v_idx) if active is not None else (pos[:, 0].abs() < _INACTIVE_THRESHOLD)
        pos_out[mask] = pos[mask]

    return pos_out


def _active_obstacle_quat(env: "ManagerBasedRLEnv", slot: int) -> torch.Tensor:
    """Return (N, 4) quaternion of active variant; identity [1,0,0,0] when inactive."""
    active = getattr(env, "_obstacle_active_variants", None)
    variants = OBSTACLE_VARIANTS[slot]
    quat_out = torch.zeros(env.num_envs, 4, device=env.device)
    quat_out[:, 0] = 1.0

    for v_idx, (name, _) in enumerate(variants):
        asset: RigidObject = env.scene[name]
        quat = asset.data.root_quat_w
        mask = (active[:, slot] == v_idx) if active is not None else (asset.data.root_pos_w[:, 0].abs() < _INACTIVE_THRESHOLD)
        quat_out[mask] = quat[mask]

    return quat_out


def _active_obstacle_dims(env: "ManagerBasedRLEnv", slot: int) -> torch.Tensor:
    """Return (N, 3) bounding-box dims of active variant; zeros when inactive."""
    active = getattr(env, "_obstacle_active_variants", None)
    variants = OBSTACLE_VARIANTS[slot]
    dims_out = torch.zeros(env.num_envs, 3, device=env.device)

    for v_idx, (name, dims) in enumerate(variants):
        d = torch.tensor(dims, device=env.device, dtype=torch.float32)
        if active is not None:
            mask = active[:, slot] == v_idx
        else:
            asset: RigidObject = env.scene[name]
            mask = asset.data.root_pos_w[:, 0].abs() < _INACTIVE_THRESHOLD
        dims_out[mask] = d

    return dims_out


# ---------------------------------------------------------------------------
# End-effector observations
# ---------------------------------------------------------------------------

def ee_position(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[..., 0, :]


def ee_orientation(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w[..., 0, :]


# ---------------------------------------------------------------------------
# Object observations
# ---------------------------------------------------------------------------

def object_position(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    return env.scene[object_cfg.name].data.root_pos_w


def object_orientation(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    return env.scene[object_cfg.name].data.root_quat_w


def object_dims_obs(
    env: "ManagerBasedRLEnv",
    object_dims: list,
) -> torch.Tensor:
    dims = torch.tensor(object_dims, device=env.device, dtype=torch.float32)
    return dims.unsqueeze(0).expand(env.num_envs, -1)


# ---------------------------------------------------------------------------
# Obstacle slot 0 observations
# ---------------------------------------------------------------------------

def obstacle_0_position(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _active_obstacle_pos(env, slot=0)


def obstacle_0_orientation(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _active_obstacle_quat(env, slot=0)


def obstacle_0_dims(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _active_obstacle_dims(env, slot=0)


# ---------------------------------------------------------------------------
# Obstacle slot 1 observations
# ---------------------------------------------------------------------------

def obstacle_1_position(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _active_obstacle_pos(env, slot=1)


def obstacle_1_orientation(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _active_obstacle_quat(env, slot=1)


def obstacle_1_dims(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _active_obstacle_dims(env, slot=1)


# ---------------------------------------------------------------------------
# Goal observation
# ---------------------------------------------------------------------------

def goal_position(
    env: "ManagerBasedRLEnv",
    goal_pos_w: list,
) -> torch.Tensor:
    goal = torch.tensor(goal_pos_w, device=env.device, dtype=torch.float32)
    return goal.unsqueeze(0).expand(env.num_envs, -1)
