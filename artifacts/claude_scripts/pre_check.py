import argparse, importlib.metadata, json, math
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--nostop", action="store_true"); p.add_argument("--thumb", type=float, default=None)
AppLauncher.add_app_launcher_args(p); args = p.parse_args(); app = AppLauncher(args).app
import gymnasium as gym, torch
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2FrontPregraspPPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_pregrasp import JaxV2FrontPregraspEnvCfg
cfg = JaxV2FrontPregraspEnvCfg(); cfg.scene.num_envs = 64
if args.nostop: cfg.table_stop_force_n = None
if args.thumb is not None: cfg.hand_shape_deg = (0.0, 0.0, args.thumb, 0.0, 0.0, 0.0)
raw = gym.make("Isaac-JaxV2-FrontPregrasp-Direct-v12", cfg=cfg); u = raw.unwrapped
ac = handle_deprecated_rsl_rl_cfg(JaxV2FrontPregraspPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(raw, clip_actions=ac.clip_actions)
r = OnPolicyRunner(env, ac.to_dict(), log_dir=None, device=ac.device); r.load(args.checkpoint)
pol = r.get_inference_policy(device=u.device); obs = env.get_observations()
with torch.inference_mode():
    for k in range(240):
        obs, *_ = env.step(pol(obs))
        if k in (29, 119, 239):
            q = [round(math.degrees(float(v)), 1) for v in u.robot.data.joint_pos[0, list(u.arm_ids[:7])]]
            print("PC", json.dumps({"t": round((k + 1) / 60, 1), "q": q, "stop_frac": round(float(getattr(u, "_table_stop", torch.zeros(1)).float().mean()), 2),
                  "err_m": round(float(u.extras["log"]["pre_err_m"]), 3),
                  "fmat_shape": list(u.scene.sensors["table_contact_MG996R"].data.force_matrix_w.shape),
                  "table_f_env0": {n: round(float(torch.linalg.vector_norm(u.scene.sensors[f"table_contact_{n}"].data.force_matrix_w[0, 0, 0])), 2) for n in cfg.table_contact_body_names},
                  "zmin_mm": round(1000 * (float(u.robot.data.body_pos_w[:, [u.robot.body_names.index(n) for n in cfg.table_contact_body_names], 2].min()) - cfg.table_top_z), 1)}), flush=True)
import os; os._exit(0)
