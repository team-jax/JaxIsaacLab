"""v3: the policy's observation and action at the first steps of the grasp-lift episode on the V2 / V3 robot (one cell, fixed
turn, no start noise). argv: checkpoint out.pt"""
import sys
CK, OUT = sys.argv[1], sys.argv[2]; sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import os, importlib.metadata, torch, gymnasium as gym
os.environ["GL_CUBE_BOX"] = "1"
import isaaclab_tasks  # noqa
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2FrontGraspLiftPPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_lift import JaxV2FrontGraspLiftEnvCfg
cfg = JaxV2FrontGraspLiftEnvCfg(); cfg.episode_length_s = 12.0; cfg.scene.num_envs = 2
raw = gym.make("Isaac-JaxV2-FrontGraspLift-Direct-v16", cfg=cfg); u = raw.unwrapped
u._gl_cube_off_fixed = torch.tensor([[-0.08, 0.0]], device=u.device).expand(2, -1)
ac = handle_deprecated_rsl_rl_cfg(JaxV2FrontGraspLiftPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(raw, clip_actions=1.0)
runner = OnPolicyRunner(env, ac.to_dict(), log_dir=None, device=ac.device); runner.load(CK); pol = runner.get_inference_policy(device=u.device)
u._reset_idx(u.robot._ALL_INDICES); obs = env.get_observations()
O, A, P = [], [], []
with torch.inference_mode():
    for k in range(40):
        a = pol(obs); O.append((obs["policy"] if isinstance(obs, dict) else obs)[0].clone().cpu()); A.append(a[0].clone().cpu())
        obs, _, _, _ = env.step(a)
torch.save({"obs": torch.stack(O), "act": torch.stack(A), "body_names": u.robot.body_names, "joint_names": u.robot.joint_names}, OUT)
print("OBS_PROBE", OUT, flush=True)
os._exit(0)
