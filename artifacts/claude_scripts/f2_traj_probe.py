"""Thumb tip height and RotationBody angle over the pre-grasp (model, 4 envs)."""
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
cfg = JaxV2FrontPregraspEnvCfg(); cfg.scene.num_envs = 4
raw = gym.make("Isaac-JaxV2-FrontPregrasp-Direct-v12", cfg=cfg); u = raw.unwrapped
ac = handle_deprecated_rsl_rl_cfg(JaxV2FrontPregraspPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(raw, clip_actions=ac.clip_actions)
r = OnPolicyRunner(env, ac.to_dict(), log_dir=None, device=ac.device); r.load(args.checkpoint)
pol = r.get_inference_policy(device=u.device); obs = env.get_observations()
rb = u.arm_ids[7]; th = u.fingertip_ids[2]; f2 = u.fingertip_ids[1]
with torch.inference_mode():
    for k in range(int(6.0 / u.step_dt)):
        obs, *_ = env.step(pol(obs))
        if k % 12 == 11:
            cube = u.object.data.root_pos_w[0]
            print("TT", json.dumps({"t": round((k + 1) * u.step_dt, 2), "rb_deg": round(float(u.robot.data.joint_pos[0, rb].rad2deg()), 1),
                  "rb_target_deg": round(float(u.targets[0, rb].rad2deg()), 1), "thumb_tip_z_rel_cube_mm": round(float((u.robot.data.body_pos_w[0, th, 2] - cube[2]) * 1000), 1),
                  "thumb_q1_deg": round(float(u.robot.data.joint_pos[0, u.finger_ids[2]].rad2deg()), 1),
                  "hand_z_rel_cube_mm": round(float((u.robot.data.body_pos_w[0, u._hand_id, 2] - cube[2]) * 1000), 1),
                  "f2_2_deg": round(float(u.robot.data.joint_pos[0, u.finger_ids[4]].rad2deg()), 1), "f1_2_deg": round(float(u.robot.data.joint_pos[0, u.finger_ids[3]].rad2deg()), 1), "f2_net_N": round(float(torch.linalg.vector_norm(u.scene.sensors["finger_2"].data.net_forces_w[0, 0])), 2), "plate_min_mm": round(float((u._finger_plate_min_z()[0] - cfg.table_top_z) * 1000), 1), "f2_tip_z_rel_cube_mm": round(float((u.robot.data.body_pos_w[0, f2, 2] - cube[2]) * 1000), 1)}), flush=True)
import os; os._exit(0)
