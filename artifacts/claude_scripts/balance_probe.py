"""Close-phase comparison: fixed guide angles vs balanced close. 64 envs, zero
actions, v13 frozen pre-grasp. Reports cube x/y/rot motion 5-9 s and forces."""
import argparse, json, math
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--balance", type=int, default=1); p.add_argument("--squeeze", type=float, default=2.0); p.add_argument("--pre", default=""); p.add_argument("--iters", type=int, default=0)
AppLauncher.add_app_launcher_args(p); args = p.parse_args(); app = AppLauncher(args).app
import gymnasium as gym, torch
import isaaclab_tasks  # noqa
from isaaclab.utils.math import quat_mul, quat_conjugate
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_close import JaxV2FrontGraspCloseEnvCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_pad_sensor_grasp import JaxV2PadSensorGraspEnv
cfg = JaxV2FrontGraspCloseEnvCfg(); cfg.scene.num_envs = 64; cfg.sensor_randomize = False
cfg.close_force_balance = bool(args.balance); cfg.close_squeeze_n = args.squeeze
if args.pre: cfg.pre_policy_path = args.pre
import os as _os
if _os.environ.get("PRE_ROT_START"): cfg.pre_rotation_start_s = float(_os.environ["PRE_ROT_START"]); cfg.pre_rotation_ramp_s = float(_os.environ["PRE_ROT_RAMP"])
if args.iters:
    cfg.robot_cfg = cfg.robot_cfg.replace(spawn=cfg.robot_cfg.spawn.replace(articulation_props=cfg.robot_cfg.spawn.articulation_props.replace(solver_position_iteration_count=args.iters, solver_velocity_iteration_count=4)))
raw = gym.make("Isaac-JaxV2-FrontGraspClose-Direct-v13", cfg=cfg); u = raw.unwrapped; raw.reset()
act = torch.zeros(64, 11, device=u.device)
xy_max = torch.zeros(64, device=u.device); rot_max = torch.zeros(64, device=u.device)
with torch.inference_mode():
    for k in range(int(9.0 / u.step_dt)):
        raw.step(act)
        if k == 0: cstart = u.object.data.root_pos_w.clone(); pre_touch = torch.zeros(64, device=u.device)
        if (k + 1) * u.step_dt < 4.95:
            pad0 = JaxV2PadSensorGraspEnv.pad_forces(u); pre_touch = torch.maximum(pre_touch, pad0.amax(1))
        t = (k + 1) * u.step_dt
        if abs(t - 4.95) < u.step_dt / 2:
            c0 = u.object.data.root_pos_w.clone(); q0 = u.object.data.root_quat_w.clone()
        if t > 4.95:
            d = u.object.data.root_pos_w - c0
            xy_max = torch.maximum(xy_max, d[:, :2].norm(dim=1))
            dq = quat_mul(u.object.data.root_quat_w, quat_conjugate(q0))
            rot_max = torch.maximum(rot_max, 2 * torch.acos(dq[:, 0].abs().clamp(max=1.0)))
        if any(abs(t - x) < u.step_dt / 2 for x in (5.5, 6.0, 7.0, 9.0)):
            pad = JaxV2PadSensorGraspEnv.pad_forces(u); f = pad[:, :3] + pad[:, 3:]
            on = pad[:, 3:] >= 0.3
            print("BP", json.dumps({"balance": args.balance, "t": round(t, 2), "all3_distal": round(float(on.all(1).float().mean()), 2),
                "digit_N(f1,f2,th)": [round(float(v), 2) for v in f.mean(0)], "distal_N": [round(float(v), 2) for v in pad[:, 3:].mean(0)],
                "prox_N": [round(float(v), 2) for v in pad[:, :3].mean(0)],
                "thumb_minus_fingers_N": round(float((f[:, 2] - f[:, 0] - f[:, 1]).abs().mean()), 2),
                "approach_touch_frac": round(float((pre_touch > 0.3).float().mean()), 2), "cube_from_start_mm": round(float((u.object.data.root_pos_w - cstart).norm(dim=1).mean() * 1000), 1), "cube_xy_max_mm": round(float(xy_max.mean() * 1000), 1), "cube_rot_max_deg": round(float(rot_max.mean().rad2deg()), 1),
                "q1_deg": [round(float(v), 1) for v in u.robot.data.joint_pos[:, u.finger_ids[:3]].mean(0).rad2deg()]}), flush=True)
import os; os._exit(0)
