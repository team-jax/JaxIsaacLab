"""Object-conditioned reach, grasp, lift, and hold with the JAX one-arm robot."""

import gymnasium as gym

gym.register(
    id="Isaac-OneArmJax-PickPlace-Direct-v1",
    entry_point=f"{__name__}.env:OneArmJaxPickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:OneArmJaxPickPlaceEnvCfg",
        "skrl_sac_cfg_entry_point": f"{__name__}.agents:skrl_sac_cfg.yaml",
        "skrl_cfg_entry_point": f"{__name__}.agents:skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:OneArmJaxLiftHoldPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-OneArmJax-LiftHold-Direct-v1",
    entry_point=f"{__name__}.env:OneArmJaxPickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:OneArmJaxPickPlaceEnvCfg",
        "skrl_cfg_entry_point": f"{__name__}.agents:skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:OneArmJaxLiftHoldPPORunnerCfg",
    },
)
