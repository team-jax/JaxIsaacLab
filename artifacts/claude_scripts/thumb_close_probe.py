"""v13 (frozen pre-grasp + balanced close): thumb motion over time, env 0..3."""
import argparse, json
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); args = p.parse_args(); app = AppLauncher(args).app
import gymnasium as gym, torch
import isaaclab_tasks  # noqa
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_close import JaxV2FrontGraspCloseEnvCfg
cfg = JaxV2FrontGraspCloseEnvCfg(); cfg.scene.num_envs = 4; cfg.sensor_randomize = False
raw = gym.make("Isaac-JaxV2-FrontGraspClose-Direct-v13", cfg=cfg); u = raw.unwrapped; raw.reset()
act = torch.zeros(4, 11, device=u.device); th = u.fingertip_ids[2]; rb = u.arm_ids[7]
with torch.inference_mode():
    for k in range(int(9.0 / u.step_dt)):
        raw.step(act)
        t = round((k + 1) * u.step_dt, 2)
        if k % 15 == 14 and t >= 2.9:
            cube = u.object.data.root_pos_w[0]
            print("TC", json.dumps({"t": t, "rb_deg": round(float(u.robot.data.joint_pos[0, rb].rad2deg()), 1), "rb_target": round(float(u.targets[0, rb].rad2deg()), 1),
                "thumb_q_deg": [round(float(u.robot.data.joint_pos[0, u.finger_ids[i]].rad2deg()), 1) for i in (2, 5)],
                "thumb_tip_rel_cube_mm": [round(float(v), 1) for v in (u.robot.data.body_pos_w[0, th] - cube) * 1000],
                "elbow": round(float(u.robot.data.joint_pos[0, u.arm_ids[3]].rad2deg()), 1)}), flush=True)
import os; os._exit(0)
