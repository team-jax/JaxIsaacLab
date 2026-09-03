# SPDX-License-Identifier: BSD-3-Clause
"""Custom termination functions for pick-and-place with obstacle avoidance."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .observations import OBSTACLE_VARIANTS


def _near_active_obstacle(
    env: "ManagerBasedRLEnv",
    margin: float = 0.06,
) -> torch.Tensor:
    """(N,) bool: True when EE is geometrically inside any active obstacle's bbox + margin."""
    from isaaclab.sensors import FrameTransformer

    ee_frame: FrameTransformer = env.scene["ee_frame"]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]  # (N, 3)
    active = getattr(env, "_obstacle_active_variants", None)
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for slot in range(2):
        for v_idx, (name, dims) in enumerate(OBSTACLE_VARIANTS[slot]):
            if active is not None:
                slot_mask = active[:, slot] == v_idx
            else:
                asset = env.scene[name]
                slot_mask = asset.data.root_pos_w[:, 0].abs() < 100.0
            if not slot_mask.any():
                continue
            obs_pos = env.scene[name].data.root_pos_w
            half = torch.tensor(dims, device=env.device) / 2.0 + margin
            diff = (ee_pos - obs_pos).abs()
            inside = (diff < half).all(dim=-1) & slot_mask
            result = result | inside

    return result


def obstacle_contact(
    env: "ManagerBasedRLEnv",
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_contact"),
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate when panda_hand contact force exceeds threshold near an active obstacle.

    Two-gate check prevents false positives from table or grasped-object contacts:
      1. Physics gate: net contact force on panda_hand > force_threshold (N)
      2. Geometry gate: EE center is within obstacle bounding box + margin

    The physics gate fires only on significant impacts (kinematic obstacles generate
    large constraint forces). The geometry gate ensures we're near an obstacle, not
    touching the table or the grasped object (whose contacts are primarily on the
    fingers, not panda_hand).
    """
    sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    # net_forces_w: (N, history_length, 3) — use most-recent step (index 0)
    net_force = sensor.data.net_forces_w[:, 0, :].norm(dim=-1)  # (N,)
    has_contact = net_force > force_threshold

    if not has_contact.any():
        return has_contact

    return has_contact & _near_active_obstacle(env)
