import argparse, importlib.metadata, json, math
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); app = AppLauncher(args).app
import gymnasium as gym, torch
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2FrontPregraspPPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_pregrasp import JaxV2FrontPregraspEnvCfg
cfg = JaxV2FrontPregraspEnvCfg(); cfg.scene.num_envs = 1
raw = gym.make("Isaac-JaxV2-FrontPregrasp-Direct-v12", cfg=cfg)
ac = handle_deprecated_rsl_rl_cfg(JaxV2FrontPregraspPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(raw, clip_actions=ac.clip_actions)
r = OnPolicyRunner(env, ac.to_dict(), log_dir=None, device=ac.device); r.load(args.checkpoint)
pol = r.get_inference_policy(device=env.unwrapped.device); u = raw.unwrapped; R = u.robot; B = R.body_names
arm = list(u.arm_ids[:7])
obs = env.get_observations()
with torch.inference_mode():
    for k in range(700):
        obs, *_ = env.step(pol(obs))
        if k % 30 == 29:
            P = R.data.body_pos_w[0]
            low = {n: round(1000 * (float(P[B.index(n), 2]) - cfg.table_top_z), 1) for n in ("MG996R", "Finger_1_1", "Finger_1_2", "Finger_2_1", "Finger_2_2", "Finger_3_1", "Finger_3_2", "RotationBody")}
            err = [round(math.degrees(float(u.targets[0, i] - R.data.joint_pos[0, i])), 1) for i in arm]
            tq = R.data.applied_torque[0, arm] if hasattr(R.data, "applied_torque") else None
            print("PRESS " + json.dumps({"t": round((k + 1) / 60, 1), "target_minus_q_deg": err, "height_over_table_mm": low,
                  "torque": [round(float(v), 2) for v in tq] if tq is not None else None}), flush=True)
app.close()
