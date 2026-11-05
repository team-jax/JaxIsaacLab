"""Unmodified RSL-RL PPO runner with a separate experiment name."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class OneArmJaxMinimalPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 25
    experiment_name = "one_arm_jax_minimal_stage1_cube5cm_v3_1k"
    clip_actions = 1.0
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.36787944, std_type="log"),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 256], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=2.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=4,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )


@configclass
class JaxV2GraspPPORunnerCfg(OneArmJaxMinimalPPORunnerCfg):
    experiment_name = "jax_v2_1006_arm_down_grasp_v1"


@configclass
class JaxV2ControlledLiftPPORunnerCfg(OneArmJaxMinimalPPORunnerCfg):
    experiment_name = "jax_v2_1006_controlled_lift_v2"


@configclass
class JaxV2ArmThenFingersPPORunnerCfg(OneArmJaxMinimalPPORunnerCfg):
    experiment_name = "jax_v2_1006_arm_then_fingers_v3"


@configclass
class JaxV2LearnSequencePPORunnerCfg(OneArmJaxMinimalPPORunnerCfg):
    experiment_name = "jax_v2_1006_learn_lift_then_open_v4"


@configclass
class JaxV2LiftOnlyPPORunnerCfg(OneArmJaxMinimalPPORunnerCfg):
    experiment_name = "jax_v2_1006_arm_lift_locked_fingers_v5"


@configclass
class JaxV2PregraspPosePPORunnerCfg(OneArmJaxMinimalPPORunnerCfg):
    experiment_name = "jax_v2_1006_pregrasp_pose_v6"


@configclass
class JaxV2GraspAfterLiftPPORunnerCfg(OneArmJaxMinimalPPORunnerCfg):
    # Keep the v6 log root so the verified experiment-2 checkpoint can be
    # loaded by the unmodified Isaac Lab / RSL-RL resume path.
    experiment_name = "jax_v2_1006_pregrasp_pose_v6"


@configclass
class JaxV2SmoothGraspPPORunnerCfg(OneArmJaxMinimalPPORunnerCfg):
    experiment_name = "jax_v2_1006_smooth_grasp_v8"


@configclass
class JaxV2PadSensorGraspPPORunnerCfg(OneArmJaxMinimalPPORunnerCfg):
    experiment_name = "jax_v2_1006_pad_sensor_v9"


@configclass
class JaxV2LRaisePPORunnerCfg(OneArmJaxMinimalPPORunnerCfg):
    experiment_name = "jax_v2_1006_l_raise_v10"


@configclass
class JaxV2FrontGraspPPORunnerCfg(OneArmJaxMinimalPPORunnerCfg):
    experiment_name = "jax_v2_1006_front_grasp_v11"


@configclass
class JaxV2FrontPregraspPPORunnerCfg(OneArmJaxMinimalPPORunnerCfg):
    experiment_name = "jax_v2_1006_front_pregrasp_v12"


@configclass
class JaxV2FrontGraspClosePPORunnerCfg(OneArmJaxMinimalPPORunnerCfg):
    experiment_name = "jax_v2_1006_front_grasp_close_v13"


@configclass
class JaxV2FrontLiftPPORunnerCfg(OneArmJaxMinimalPPORunnerCfg):
    experiment_name = "jax_v2_1006_front_lift_v14"


@configclass
class JaxV2FrontLiftRLPPORunnerCfg(OneArmJaxMinimalPPORunnerCfg):
    experiment_name = "jax_v2_1006_front_lift_rl_v15"


@configclass
class JaxV2FrontGraspLiftPPORunnerCfg(OneArmJaxMinimalPPORunnerCfg):
    experiment_name = "jax_v2_1006_front_grasp_lift_v16"

    def __post_init__(self):
        super().__post_init__()
        import os
        # resuming with a rescaled reward: the clipped value loss would let the
        # critic move only 0.2 per update toward the new value scale
        if os.environ.get("GL_NO_VCLIP"):
            self.algorithm.use_clipped_value_loss = False
        if os.environ.get("GL_LR"):
            self.algorithm.learning_rate = float(os.environ["GL_LR"])


@configclass
class JaxV2ArmRaisePPORunnerCfg(OneArmJaxMinimalPPORunnerCfg):
    experiment_name = "jax_v2_1006_arm_raise_v17"


@configclass
class JaxV2RaiseToPregraspPPORunnerCfg(JaxV2ArmRaisePPORunnerCfg):
    experiment_name = "jax_v2_1006_raise_pregrasp_v18"


@configclass
class JaxV2RaiseLearnedPPORunnerCfg(JaxV2ArmRaisePPORunnerCfg):
    experiment_name = "jax_v2_1006_raise_learned_v19"


@configclass
class JaxV2RaiseToLPPORunnerCfg(JaxV2ArmRaisePPORunnerCfg):
    experiment_name = "jax_v2_1006_raise_to_l_v20"


@configclass
class JaxV2RaiseResidualPPORunnerCfg(JaxV2ArmRaisePPORunnerCfg):
    experiment_name = "jax_v2_1006_raise_residual_v21"


@configclass
class JaxV2RaiseTablePPORunnerCfg(JaxV2ArmRaisePPORunnerCfg):
    experiment_name = "jax_v2_1006_raise_table_v22"

    def __post_init__(self):
        super().__post_init__()
        # v21 degraded after ~50 iterations at 1e-4 in every reward variant
        self.algorithm.learning_rate = 3.0e-5


@configclass
class JaxV2RaiseLTablePPORunnerCfg(JaxV2RaiseTablePPORunnerCfg):
    experiment_name = "jax_v2_1006_raise_ltable_v24"


@configclass
class JaxV2RaiseCubePPORunnerCfg(JaxV2RaiseTablePPORunnerCfg):
    experiment_name = "jax_v2_1006_raise_cube_v25"
