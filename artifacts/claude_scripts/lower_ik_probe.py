"""Diagnostic only: model_3038 pre-grasp, at close onset move the hand down dz by
differential IK (close_shift_m), close the flat shape; report contacts/heights."""
import argparse, json, math
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--dz", type=float, required=True); p.add_argument("--rb", type=float, default=0.0); p.add_argument("--pre", default="")
AppLauncher.add_app_launcher_args(p); args = p.parse_args(); app = AppLauncher(args).app
import gymnasium as gym, torch
import isaaclab_tasks  # noqa
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_close import JaxV2FrontGraspCloseEnvCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_pad_sensor_grasp import JaxV2PadSensorGraspEnv
cfg = JaxV2FrontGraspCloseEnvCfg(); cfg.scene.num_envs = 64; cfg.sensor_randomize = False
if args.pre: cfg.pre_policy_path = args.pre
pass
cfg.close_shift_m = (0.0, 0.0, -args.dz); cfg.close_rotation_delta_deg = args.rb
cfg.grasp_guide_deg = (10.0, 10.0, -30.0, 0.0, 0.0, 0.0)
raw = gym.make("Isaac-JaxV2-FrontGraspClose-Direct-v13", cfg=cfg); u = raw.unwrapped; raw.reset()
act = torch.zeros(64, 11, device=u.device)
with torch.inference_mode():
    for k in range(int(8.0 / u.step_dt)):
        if k == int(4.9 / u.step_dt): cube0 = u.object.data.root_pos_w.clone()
        raw.step(act)
    pad = JaxV2PadSensorGraspEnv.pad_forces(u); cube = u.object.data.root_pos_w
    tips = u.robot.data.body_pos_w[:, u.fingertip_ids]
    links = [u._hand_id] + u.proximal_ids + list(u.fingertip_ids)
    zmin = (u.robot.data.body_pos_w[:, links, 2].amin(1) - cfg.table_top_z) * 1000
    arm = u.arm_ids[:7]; trk = (u.targets[:, arm] - u.robot.data.joint_pos[:, arm]).abs().amax(1).rad2deg()
    on = (pad[:, 3:] >= 0.3)
    print("LZ", json.dumps({"dz_mm": args.dz * 1000, "all3_distal": round(float(on.all(1).float().mean()), 2), "f1_distal_on": round(float(on[:, 0].float().mean()), 2),
        "f2_distal_on": round(float(on[:, 1].float().mean()), 2), "thumb_distal_on": round(float(on[:, 2].float().mean()), 2),
        "distal_N": [round(float(v), 2) for v in pad[:, 3:].mean(0)], "prox_N": [round(float(v), 2) for v in pad[:, :3].mean(0)],
        "tip_z_rel_cube_mm(f1,f2,th)": [round(float(v), 1) for v in ((tips[..., 2] - cube[:, None, 2]) * 1000).mean(0)],
        "lowest_link_mm": round(float(zmin.mean()), 1), "arm_track_err_deg": round(float(trk.mean()), 1),
        "f2_q1_deg": round(float(u.robot.data.joint_pos[:, u.finger_ids[1]].mean().rad2deg()), 1), "f1_q1_deg": round(float(u.robot.data.joint_pos[:, u.finger_ids[0]].mean().rad2deg()), 1),
        "f2_table_N": round(float((torch.linalg.vector_norm(u.scene.sensors["finger_2"].data.net_forces_w[:, 0], dim=-1) - torch.linalg.vector_norm(u.scene.sensors["finger_2"].data.force_matrix_w[:, 0, 0], dim=-1)).clamp_min(0).mean()), 2),
        "cube_move_mm": round(float((cube - cube0).norm(dim=1).mean() * 1000), 1)}), flush=True)
import os; os._exit(0)
