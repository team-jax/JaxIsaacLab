import argparse, importlib.metadata, json
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True)
AppLauncher.add_app_launcher_args(p); args = p.parse_args(); app = AppLauncher(args).app
import gymnasium as gym, torch
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2FrontPregraspPPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_pregrasp import JaxV2FrontPregraspEnvCfg
cfg = JaxV2FrontPregraspEnvCfg(); cfg.scene.num_envs = 64
raw = gym.make("Isaac-JaxV2-FrontPregrasp-Direct-v12", cfg=cfg); u = raw.unwrapped
ac = handle_deprecated_rsl_rl_cfg(JaxV2FrontPregraspPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(raw, clip_actions=ac.clip_actions)
r = OnPolicyRunner(env, ac.to_dict(), log_dir=None, device=ac.device); r.load(args.checkpoint)
pol = r.get_inference_policy(device=u.device); obs = env.get_observations()
with torch.inference_mode():
    for k in range(360):
        obs, *_ = env.step(pol(obs))
        if k == 359:
            cube = u.object.data.root_pos_w
            tips = u.robot.data.body_pos_w[:, u.fingertip_ids]; prox = u.robot.data.body_pos_w[:, u.proximal_ids]
            ax = tips - prox; ax = ax / ax.norm(dim=-1, keepdim=True)
            dc = tips + 0.03 * ax
            rel = (dc - cube[:, None]) * 1000
            out = {"distal_centre_rel_cube_mm(f1,f2,thumb)": [[round(float(v), 1) for v in rel[:, i].mean(0)] for i in range(3)],
                   "link2_origin_rel_cube_mm": [[round(float(v), 1) for v in ((tips[:, i] - cube) * 1000).mean(0)] for i in range(3)],
                   "finger_axis(f1,f2)": [[round(float(v), 2) for v in ax[:, i].mean(0)] for i in range(2)],
                   "lowest_hand_link_above_table_mm": round(float(((u.robot.data.body_pos_w[:, [u._hand_id] + u.proximal_ids + list(u.fingertip_ids), 2].amin(1)) - cfg.table_top_z).mean() * 1000), 1),
                   "cube_z_above_table_mm": round(float((cube[:, 2] - cfg.table_top_z).mean() * 1000), 1)}
            print("DP", json.dumps(out), flush=True)
import os; os._exit(0)
