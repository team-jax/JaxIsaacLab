# SPDX-License-Identifier: BSD-3-Clause
"""Curriculum functions for pick-and-place with obstacle avoidance.

The obstacle curriculum advances in stages based on total environment steps:
  Stage 0 (steps < stage_1_threshold): no active obstacles
  Stage 1 (steps < stage_2_threshold): 1 obstacle in workspace
  Stage 2 (steps >= stage_2_threshold): 2 obstacles in workspace

Inactive obstacles are placed at a far-away "parked" position by the reset events.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def set_obstacle_stage(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    stage_1_threshold: int,
    stage_2_threshold: int,
) -> torch.Tensor | None:
    """Set env._obstacle_stage based on current training step count.

    Called every step by the CurriculumManager. Stores the current stage as
    an attribute on the env so that reset events can read it.

    Returns None (curriculum functions that return None are treated as no-op).
    """
    step = env.common_step_counter
    if step < stage_1_threshold:
        stage = 0
    elif step < stage_2_threshold:
        stage = 1
    else:
        stage = 2

    # Store on env so obstacle reset events can read it without extra params
    env._obstacle_stage = stage
    return None
