"""v16 slip diagnosis without a policy: frozen pre-grasp, balanced close at
--squeeze N, then from 6 s a constant elbow-up command (action 3) with the
wrist coupling on/off. Reports cube vs hand rise, per-pad force on the cube
(world vector) and the summed vertical force the pads put on the cube."""
import argparse, json
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--couple", type=int, default=1); p.add_argument("--squeeze", type=float, default=2.0)
p.add_argument("--elbow", type=float, default=0.5); p.add_argument("--n", type=int, default=64); p.add_argument("--mu", type=float, default=0.0)
AppLauncher.add_app_launcher_args(p); args = p.parse_args(); app = AppLauncher(args).app
import gymnasium as gym, torch
import isaaclab_tasks  # noqa
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_lift import JaxV2FrontGraspLiftEnvCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_pad_sensor_grasp import JaxV2PadSensorGraspEnv
cfg = JaxV2FrontGraspLiftEnvCfg(); cfg.scene.num_envs = args.n; cfg.sensor_randomize = False
cfg.lift_keep_hand_pitch = bool(args.couple); cfg.close_squeeze_n = args.squeeze; cfg.close_squeeze_range_n = 0.0
cfg.lift_active_joints = (0, 3, 5)
if args.mu:
    sp = cfg.object_cfg.spawn
    cfg.object_cfg = cfg.object_cfg.replace(spawn=sp.replace(physics_material=sp.physics_material.replace(static_friction=args.mu, dynamic_friction=args.mu)))
    print("SP_MU", cfg.object_cfg.spawn.physics_material, flush=True)
raw = gym.make("Isaac-JaxV2-FrontGraspLift-Direct-v16", cfg=cfg); u = raw.unwrapped; raw.reset()
act = torch.zeros(args.n, raw.unwrapped.single_action_space.shape[0], device=u.device)
names = [f"proximal_{i}" for i in (1, 2, 3)] + [f"finger_{i}" for i in (1, 2, 3)]
with torch.inference_mode():
    for k in range(int(10.0 / u.step_dt)):
        t = (k + 1) * u.step_dt
        act[:, 3] = args.elbow if t > 6.0 else 0.0
        raw.step(act)
        if abs(t - 5.9) < u.step_dt / 2:
            c0 = u.object.data.root_pos_w.clone(); h0 = u.robot.data.body_pos_w[:, u._hand_id].clone()
        if any(abs(t - x) < u.step_dt / 2 for x in (5.9, 7.0, 8.0, 9.5)):
            fv = torch.stack([u.scene.sensors[n].data.force_matrix_w[:, 0, 0] for n in names], 1)  # force on the pad from the cube
            pad = JaxV2PadSensorGraspEnv.pad_forces(u)
            d = {"couple": args.couple, "sq": args.squeeze, "t": round(t, 2),
                 "cube_rise_mm": round(float((u.object.data.root_pos_w[:, 2] - c0[:, 2]).mean() * 1000), 1),
                 "hand_rise_mm": round(float((u.robot.data.body_pos_w[:, u._hand_id, 2] - h0[:, 2]).mean() * 1000), 1),
                 "pad_N": [round(float(v), 2) for v in pad.mean(0)],
                 "force_on_cube_z_N": round(float(-fv[..., 2].sum(1).mean()), 3),
                 "per_link_fz_on_cube": [round(float(v), 2) for v in (-fv[..., 2]).mean(0)],
                 "per_link_fxy_on_cube": [[round(float(a), 2) for a in v] for v in (-fv[..., :2]).mean(0)],
                 "table_force_n": round(float(u._table_force.mean()), 2) if hasattr(u, "_table_force") else None,
                 "q_arm": [round(float(v), 1) for v in u.robot.data.joint_pos[0, u.arm_ids[:7]].rad2deg()],
                 "q_fing": [round(float(v), 1) for v in u.robot.data.joint_pos[0, u.finger_ids].rad2deg()]}
            print("SP", json.dumps(d), flush=True)
import os; os._exit(0)
