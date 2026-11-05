import argparse, importlib.metadata, json
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); app = AppLauncher(args).app
import gymnasium as gym, torch
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2FrontGraspClosePPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_close import JaxV2FrontGraspCloseEnvCfg
cfg = JaxV2FrontGraspCloseEnvCfg(); cfg.scene.num_envs = 256
raw = gym.make("Isaac-JaxV2-FrontGraspClose-Direct-v13", cfg=cfg)
ac = handle_deprecated_rsl_rl_cfg(JaxV2FrontGraspClosePPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(raw, clip_actions=ac.clip_actions)
r = OnPolicyRunner(env, ac.to_dict(), log_dir=None, device=ac.device); r.load(args.checkpoint)
pol = r.get_inference_policy(device=env.unwrapped.device); u = raw.unwrapped
for mode in ("open", "half", "closed", "policy"):
    tot = torch.zeros(256, device=u.device); logs = {}
    with torch.inference_mode():
        u.episode_length_buf[:] = 0; u._reset_idx(None); obs = env.get_observations()
        for k in range(710):
            a = pol(obs)
            if mode != "policy":
                a[:, 8:11] = {"open": -1.0, "half": 0.0, "closed": 1.0}[mode]
            obs, rew, done, ext = env.step(a)
            if k >= 300:
                tot += rew
                for key in ("grasp_both", "pre_cube_shift_m", "grasp_side_n", "pre_touching", "pre_success"):
                    logs[key] = logs.get(key, 0.0) + float(u.extras["log"][key]) / 410
    print("PROBE", mode, json.dumps({"close_phase_reward": round(float(tot.mean()), 2), **{k: round(v, 3) for k, v in logs.items()}}), flush=True)
app.close()
