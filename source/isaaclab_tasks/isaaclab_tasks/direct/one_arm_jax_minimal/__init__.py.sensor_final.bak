"""Minimal Stage 1 grasp task; the original JAX task remains independent."""

import gymnasium as gym

gym.register(
    id="Isaac-OneArmJax-Minimal-Grasp-Direct-v1",
    entry_point=f"{__name__}.env:OneArmJaxMinimalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:OneArmJaxMinimalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:OneArmJaxMinimalPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-OneArmJax-Pregrasp-Direct-v1",
    entry_point=f"{__name__}.pregrasp:OneArmJaxPregraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pregrasp:OneArmJaxPregraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:OneArmJaxMinimalPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-OneArmJax-Pregrasp-Down-Direct-v1",
    entry_point=f"{__name__}.pregrasp:OneArmJaxPregraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pregrasp:OneArmJaxPregraspDownEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:OneArmJaxMinimalPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-Pregrasp-Down-Direct-v1",
    entry_point=f"{__name__}.v2_pregrasp:JaxV2PregraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_pregrasp:JaxV2PregraspDownEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:OneArmJaxMinimalPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-ArmDown-Grasp-Direct-v1",
    entry_point=f"{__name__}.v2_grasp:JaxV2GraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_grasp:JaxV2GraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2GraspPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-ArmDown-ControlledLift-Direct-v2",
    entry_point=f"{__name__}.v2_grasp:JaxV2GraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_grasp:JaxV2ControlledLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2ControlledLiftPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-ArmThenFingers-Grasp-Direct-v3",
    entry_point=f"{__name__}.v2_arm_then_fingers:JaxV2ArmThenFingersEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_arm_then_fingers:JaxV2ArmThenFingersEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2ArmThenFingersPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-Learn-LiftThenOpen-Direct-v4",
    entry_point=f"{__name__}.v2_learn_sequence:JaxV2LearnSequenceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_learn_sequence:JaxV2LearnSequenceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2LearnSequencePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-ArmLift-LockedFingers-Direct-v5",
    entry_point=f"{__name__}.v2_lift_only:JaxV2LiftOnlyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_lift_only:JaxV2LiftOnlyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2LiftOnlyPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-LiftThenPregrasp-Direct-v6",
    entry_point=f"{__name__}.v2_pregrasp_pose:JaxV2PregraspPoseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_pregrasp_pose:JaxV2PregraspPoseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2PregraspPosePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-LiftThenGrasp-Direct-v7",
    entry_point=f"{__name__}.v2_grasp_after_lift:JaxV2GraspAfterLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_grasp_after_lift:JaxV2GraspAfterLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2GraspAfterLiftPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-SmoothLiftGrasp-Direct-v8",
    entry_point=f"{__name__}.v2_smooth_grasp:JaxV2SmoothGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_smooth_grasp:JaxV2SmoothGraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2SmoothGraspPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-PadSensorGrasp-Direct-v9",
    entry_point=f"{__name__}.v2_pad_sensor_grasp:JaxV2PadSensorGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_pad_sensor_grasp:JaxV2PadSensorGraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2PadSensorGraspPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-PadSensorGrasp-FirstPhalanx-Direct-v9",
    entry_point=f"{__name__}.v2_pad_sensor_grasp:JaxV2PadSensorGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_pad_sensor_grasp:JaxV2PadSensorFirstPhalanxEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2PadSensorGraspPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-LRaise-Direct-v10",
    entry_point=f"{__name__}.v2_l_raise:JaxV2LRaiseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_l_raise:JaxV2LRaiseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2LRaisePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-FrontGrasp-Direct-v11",
    entry_point=f"{__name__}.v2_front_grasp:JaxV2FrontGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_front_grasp:JaxV2FrontGraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2FrontGraspPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-FrontPregrasp-Direct-v12",
    entry_point=f"{__name__}.v2_front_pregrasp:JaxV2FrontPregraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_front_pregrasp:JaxV2FrontPregraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2FrontPregraspPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-FrontGraspClose-Direct-v13",
    entry_point=f"{__name__}.v2_front_grasp_close:JaxV2FrontGraspCloseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_front_grasp_close:JaxV2FrontGraspCloseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2FrontGraspClosePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-FrontLift-Direct-v14",
    entry_point=f"{__name__}.v2_front_lift:JaxV2FrontLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_front_lift:JaxV2FrontLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2FrontLiftPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-FrontLiftRL-Direct-v15",
    entry_point=f"{__name__}.v2_front_lift_rl:JaxV2FrontLiftRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_front_lift_rl:JaxV2FrontLiftRLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2FrontLiftRLPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-FrontGraspLift-Direct-v16",
    entry_point=f"{__name__}.v2_front_grasp_lift:JaxV2FrontGraspLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_front_grasp_lift:JaxV2FrontGraspLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2FrontGraspLiftPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-ArmRaise-Direct-v17",
    entry_point=f"{__name__}.v2_arm_raise:JaxV2ArmRaiseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_arm_raise:JaxV2ArmRaiseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2ArmRaisePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-RaiseToPregrasp-Direct-v18",
    entry_point=f"{__name__}.v2_arm_raise:JaxV2ArmRaiseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_arm_raise:JaxV2RaiseToPregraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2RaiseToPregraspPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-RaiseLearned-Direct-v19",
    entry_point=f"{__name__}.v2_arm_raise:JaxV2ArmRaiseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_arm_raise:JaxV2RaiseLearnedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2RaiseLearnedPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-RaiseToL-Direct-v20",
    entry_point=f"{__name__}.v2_arm_raise:JaxV2ArmRaiseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_arm_raise:JaxV2RaiseToLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2RaiseToLPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-RaiseResidual-Direct-v21",
    entry_point=f"{__name__}.v2_arm_raise:JaxV2ArmRaiseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_arm_raise:JaxV2RaiseResidualEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2RaiseResidualPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-RaiseTable-Direct-v22",
    entry_point=f"{__name__}.v2_arm_raise:JaxV2RaiseTableEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_arm_raise:JaxV2RaiseTableEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2RaiseTablePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-RaiseLTable-Direct-v24",
    entry_point=f"{__name__}.v2_arm_raise:JaxV2RaiseTableEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_arm_raise:JaxV2RaiseLTableEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2RaiseLTablePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-RaiseCube-Direct-v25",
    entry_point=f"{__name__}.v2_arm_raise:JaxV2RaiseTableEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_arm_raise:JaxV2RaiseCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2RaiseCubePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-JaxV2-RaiseCubeWide-Direct-v27",
    entry_point=f"{__name__}.v2_arm_raise:JaxV2RaiseTableEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v2_arm_raise:JaxV2RaiseCubeWideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:JaxV2RaiseCubePPORunnerCfg",
    },
)
