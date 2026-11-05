"""RSL-RL PPO configuration for the JAX arm Stage 1 grasp task."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class OneArmJaxLiftHoldPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "one_arm_jax_grasp_closure_v4"
    clip_actions = 1.0
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.36787944, std_type="log"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256],
        activation="elu",
        obs_normalization=True,
    )
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
