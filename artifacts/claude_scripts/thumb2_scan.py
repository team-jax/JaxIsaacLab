"""Level-lift hold vs thumb second joint. v16 env, no policy: frozen pre-grasp,
from 5 s a balanced close (first joints force-controlled, thumb first joint
allowed -70..+30 deg) with the thumb second joint fixed per env group; from
6 s elbow up with the wrist coupling (level hand). 8 groups x 8 envs."""
import argparse, json, math
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--squeeze", type=float, default=2.0); p.add_argument("--couple", type=int, default=1)
AppLauncher.add_app_launcher_args(p); args = p.parse_args(); app = AppLauncher(args).app
import gymnasium as gym, torch
import isaaclab_tasks  # noqa
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_lift import JaxV2FrontGraspLiftEnvCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_pad_sensor_grasp import JaxV2PadSensorGraspEnv
TH2 = [0, 15, 30, 45, 60, 75, 90, 105]; G = 8; N = G * len(TH2)
cfg = JaxV2FrontGraspLiftEnvCfg(); cfg.scene.num_envs = N; cfg.sensor_randomize = False
cfg.lift_keep_hand_pitch = bool(args.couple); cfg.close_squeeze_n = args.squeeze; cfg.close_squeeze_range_n = 0.0
cfg.close_thumb_max_range_deg = 0.0; cfg.lift_active_joints = (0, 3, 5)
raw = gym.make("Isaac-JaxV2-FrontGraspLift-Direct-v16", cfg=cfg); u = raw.unwrapped; raw.reset()
dev = u.device
th2 = torch.tensor([math.radians(v) for v in TH2], device=dev).repeat_interleave(G)
lo = u._hand_shape[:3].clone(); lo[2] = math.radians(-70.0)
hi = torch.tensor([math.radians(v) for v in (30.0, 30.0, 30.0)], device=dev)

def bal():
    closing = u._closing()
    if not hasattr(u, "_sq"):
        u._sq = u._hand_shape[:3].expand(N, -1).clone()
    u._sq = torch.where(closing[:, None], u._sq, u._hand_shape[:3].expand(N, -1))
    pad = JaxV2PadSensorGraspEnv.pad_forces(u); f = pad[:, :3] + pad[:, 3:]
    on = f >= cfg.close_touch_n; all_on = on.all(1, keepdim=True); s = cfg.close_squeeze_n
    want = torch.where(all_on, s * f.new_tensor([0.5, 0.5, 1.0]).expand_as(f), torch.full_like(f, cfg.close_touch_n))
    dmax = cfg.close_speed_rad_s * u.step_dt
    step = (cfg.close_gain_rad_per_n * (want - f)).clamp(-dmax, dmax)
    step = torch.where(on, step, torch.full_like(step, dmax))
    u._sq = torch.minimum(torch.maximum(u._sq + step, lo), hi)
    q2 = u._hand_shape[3:].expand(N, -1).clone(); q2[:, 2] = th2
    return torch.cat((u._sq, q2), 1)
u._balanced_close = bal

act = torch.zeros(N, u.single_action_space.shape[0], device=dev)
names = [f"proximal_{i}" for i in (1, 2, 3)] + [f"finger_{i}" for i in (1, 2, 3)]
with torch.inference_mode():
    for k in range(int(10.0 / u.step_dt)):
        t = (k + 1) * u.step_dt
        act[:, 3] = 0.5 if t > 6.0 else 0.0
        raw.step(act)
        if abs(t - 5.95) < u.step_dt / 2:
            c0 = u.object.data.root_pos_w.clone(); h0 = u.robot.data.body_pos_w[:, u._hand_id].clone(); q0 = u.object.data.root_quat_w.clone()
        if abs(t - 5.95) < u.step_dt / 2 or abs(t - 9.5) < u.step_dt / 2:
            pad = JaxV2PadSensorGraspEnv.pad_forces(u)
            cr = (u.object.data.root_pos_w[:, 2] - c0[:, 2]) * 1000; hr = (u.robot.data.body_pos_w[:, u._hand_id, 2] - h0[:, 2]) * 1000
            xy = (u.object.data.root_pos_w[:, :2] - u._cube_start[:, :2]).norm(dim=1) * 1000
            rot = torch.rad2deg(2 * torch.acos((u.object.data.root_quat_w * q0).sum(1).abs().clamp(max=1.0)))
            for i, v in enumerate(TH2):
                s = slice(i * G, (i + 1) * G)
                print("T2", json.dumps({"t": round(t, 2), "th2": v, "cube_rise_mm": round(float(cr[s].mean()), 1),
                    "hand_rise_mm": round(float(hr[s].mean()), 1), "held_frac": round(float(((cr[s] > 0.7 * hr[s]) & (hr[s] > 20)).float().mean()), 2),
                    "xy_mm": round(float(xy[s].mean()), 1), "rot_deg": round(float(rot[s].mean()), 1),
                    "pad_N": [round(float(x), 2) for x in pad[s].mean(0)],
                    "th1_deg": round(float(torch.rad2deg(u.robot.data.joint_pos[s, u.finger_ids[2]]).mean()), 1),
                    "th2_deg": round(float(torch.rad2deg(u.robot.data.joint_pos[s, u.finger_ids[5]]).mean()), 1)}), flush=True)
import os; os._exit(0)
