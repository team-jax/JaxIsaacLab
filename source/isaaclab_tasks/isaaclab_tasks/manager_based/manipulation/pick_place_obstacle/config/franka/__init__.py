# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-PickPlace-Obstacle-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_pnp_env_cfg:FrankaPnPObstacleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PnPObstaclePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-Obstacle-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_pnp_env_cfg:FrankaPnPObstacleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PnPObstaclePPORunnerCfg",
    },
    disable_env_checker=True,
)
