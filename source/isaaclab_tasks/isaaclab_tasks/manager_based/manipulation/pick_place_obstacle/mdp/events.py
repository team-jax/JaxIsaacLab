# SPDX-License-Identifier: BSD-3-Clause
"""Custom event functions for pick-and-place with obstacle avoidance."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs.mdp.events import reset_root_state_uniform as _reset_root_state_uniform
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Local-frame position offset used when parking an inactive obstacle.
# After adding env_origin this places the obstacle ~200 m away from every workspace.
_PARKED_LOCAL = torch.tensor([200.0, 200.0, 0.0])

# Number of size variants per obstacle slot (must match OBSTACLE_VARIANTS in observations.py)
_NUM_VARIANTS = 5


def reset_obstacle_random_variant(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    obstacle_idx: int,
    pose_range: dict[str, tuple[float, float]],
    variant_0_cfg: SceneEntityCfg,
    variant_1_cfg: SceneEntityCfg,
    variant_2_cfg: SceneEntityCfg,
    variant_3_cfg: SceneEntityCfg,
    variant_4_cfg: SceneEntityCfg,
) -> None:
    """Reset one obstacle slot by randomly selecting one of five size variants.

    Per episode, for each env:
      - Inactive slot (obstacle_idx >= current stage): park ALL five variants far away.
      - Active slot: randomly choose one variant to place in the workspace;
        park the other four at x≈200 m.

    Tracks the chosen variant index in env._obstacle_active_variants[:, obstacle_idx].
      -1  → inactive slot
      0-4 → variant index

    Position convention:
      pose_range values are LOCAL-frame offsets from the env origin.
      write_root_state_to_sim() expects WORLD-frame positions.
      Conversion: root_state[:, :3] += env.scene.env_origins[env_ids]
    """
    stage = getattr(env, "_obstacle_stage", 0)
    num = len(env_ids)
    env_origins = env.scene.env_origins[env_ids]
    parked = _PARKED_LOCAL.to(env.device)

    # Initialise tracker tensor on first call
    if not hasattr(env, "_obstacle_active_variants"):
        env._obstacle_active_variants = torch.full(
            (env.num_envs, 2), -1, dtype=torch.long, device=env.device
        )

    all_variants = [
        variant_0_cfg, variant_1_cfg, variant_2_cfg, variant_3_cfg, variant_4_cfg,
    ]

    if obstacle_idx >= stage:
        # Slot inactive: park all variants
        env._obstacle_active_variants[env_ids, obstacle_idx] = -1
        for cfg in all_variants:
            asset: RigidObject = env.scene[cfg.name]
            root_state = asset.data.default_root_state[env_ids].clone()
            root_state[:, 7:] = 0.0
            root_state[:, 0] = parked[0]
            root_state[:, 1] = parked[1]
            root_state[:, 2] = parked[2]
            root_state[:, :3] += env_origins
            asset.write_root_state_to_sim(root_state, env_ids=env_ids)
    else:
        # Slot active: one random variant per env, park the rest
        chosen = torch.randint(0, _NUM_VARIANTS, (num,), device=env.device)
        env._obstacle_active_variants[env_ids, obstacle_idx] = chosen

        for v_idx, cfg in enumerate(all_variants):
            asset: RigidObject = env.scene[cfg.name]
            root_state = asset.data.default_root_state[env_ids].clone()
            root_state[:, 7:] = 0.0

            active_mask = chosen == v_idx

            if active_mask.any():
                rs_act = root_state[active_mask].clone()
                n_act = int(active_mask.sum().item())
                for col, key in enumerate(("x", "y", "z")):
                    if key in pose_range:
                        lo, hi = pose_range[key]
                        rs_act[:, col] = torch.rand(n_act, device=env.device) * (hi - lo) + lo
                rs_act[:, :3] += env_origins[active_mask]
                root_state[active_mask] = rs_act

            park_mask = ~active_mask
            if park_mask.any():
                rs_park = root_state[park_mask].clone()
                rs_park[:, 0] = parked[0]
                rs_park[:, 1] = parked[1]
                rs_park[:, 2] = parked[2]
                rs_park[:, :3] += env_origins[park_mask]
                root_state[park_mask] = rs_park

            asset.write_root_state_to_sim(root_state, env_ids=env_ids)


def reset_object_position_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    narrow_range: dict,
    full_range: dict,
    expand_start_steps: int,
    expand_end_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Reset object with curriculum: narrow spawn range → full range over training.

    Phase 1 (0 → expand_start_steps): object spawns in narrow_range (close to default EE).
    Phase 2 (expand_start_steps → expand_end_steps): linearly interpolates to full_range.
    Phase 3 (expand_end_steps → ∞): full_range.
    """
    step = env.common_step_counter
    t = float(min(1.0, max(0.0, (step - expand_start_steps) / max(1, expand_end_steps - expand_start_steps))))

    pose_range = {}
    for axis in set(narrow_range.keys()) | set(full_range.keys()):
        n_lo, n_hi = narrow_range.get(axis, full_range[axis])
        f_lo, f_hi = full_range.get(axis, narrow_range[axis])
        pose_range[axis] = (n_lo + (f_lo - n_lo) * t, n_hi + (f_hi - n_hi) * t)

    _reset_root_state_uniform(env, env_ids, pose_range=pose_range, velocity_range={}, asset_cfg=asset_cfg)
