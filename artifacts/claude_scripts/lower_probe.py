import argparse, importlib.metadata, json
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--joint", type=int, default=0)
AppLauncher.add_app_launcher_args(p); args = p.parse_args(); app = AppLauncher(args).app
import gymnasium as gym, torch
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2FrontPregraspPPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_pregrasp import JaxV2FrontPregraspEnvCfg
cfg = JaxV2FrontPregraspEnvCfg(); cfg.scene.num_envs = 13; cfg.episode_length_s = 30.0
raw = gym.make("Isaac-JaxV2-FrontPregrasp-Direct-v12", cfg=cfg); u = raw.unwrapped
ac = handle_deprecated_rsl_rl_cfg(JaxV2FrontPregraspPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(raw, clip_actions=ac.clip_actions)
r = OnPolicyRunner(env, ac.to_dict(), log_dir=None, device=ac.device); r.load(args.checkpoint)
pol = r.get_inference_policy(device=u.device); obs = env.get_observations()
deltas = torch.linspace(-0.3, 0.3, 13, device=u.device)
with torch.inference_mode():
    for k in range(360): obs, *_ = env.step(pol(obs))
    frozen = u.targets.clone(); j = u.arm_ids[args.joint]
    cube0 = u.object.data.root_pos_w.clone()
    def pre(actions):
        u.targets[:] = frozen; u.targets[:, j] = frozen[:, j] + deltas
    u._pre_physics_step = pre
    for k in range(180): obs, *_ = env.step(torch.zeros_like(pol(obs)))
    tips = u.robot.data.body_pos_w[:, u.fingertip_ids]
    links = [u._hand_id] + u.proximal_ids + list(u.fingertip_ids)
    zmin = u.robot.data.body_pos_w[:, links, 2].amin(1) - cfg.table_top_z
    argm = u.robot.data.body_pos_w[:, links, 2].argmin(1)
    names = [u.robot.body_names[l] for l in links]
    for i in range(13):
        print("LP", json.dumps({"joint": args.joint, "delta_deg": round(float(deltas[i]) * 57.3, 1),
            "track_err_deg": round(float((u.targets[i, j] - u.robot.data.joint_pos[i, j]) * 57.3), 1),
            "f1_z_rel_cube_mm": round(float((tips[i, 0, 2] - cube0[i, 2]) * 1000), 1), "f2_z_rel_cube_mm": round(float((tips[i, 1, 2] - cube0[i, 2]) * 1000), 1),
            "lowest_link": names[int(argm[i])], "lowest_mm": round(float(zmin[i] * 1000), 1),
            "cube_move_mm": round(float((u.object.data.root_pos_w[i] - cube0[i]).norm() * 1000), 1)}), flush=True)
import os; os._exit(0)
