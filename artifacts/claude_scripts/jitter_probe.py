"""Per-step jitter of a v16 policy: per joint and phase, the RMS of the
step-to-step change of joint velocity (high-frequency shake), the RMS of the
joint target change, and the policy action change per action dim."""
import argparse, json, importlib.metadata
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--n", type=int, default=64); p.add_argument("--effort", type=float, default=0.0); p.add_argument("--ema", type=float, default=0.0); p.add_argument("--kp", type=float, default=0.0); p.add_argument("--kd", type=float, default=0.0); p.add_argument("--viters", type=int, default=0); p.add_argument("--arm", type=float, default=0.0)
AppLauncher.add_app_launcher_args(p); args = p.parse_args(); app = AppLauncher(args).app
import gymnasium as gym, torch
import isaaclab_tasks  # noqa
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2FrontGraspLiftPPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_lift import JaxV2FrontGraspLiftEnvCfg
cfg = JaxV2FrontGraspLiftEnvCfg(); cfg.scene.num_envs = args.n; cfg.sensor_randomize = False
cfg.close_force_ema = args.ema
if args.effort:
    acts = dict(cfg.robot_cfg.actuators); acts["fingers"] = acts["fingers"].replace(effort_limit_sim=args.effort)
    cfg.robot_cfg = cfg.robot_cfg.replace(actuators=acts)
if args.kp:
    acts = dict(cfg.robot_cfg.actuators); acts["fingers"] = acts["fingers"].replace(stiffness=args.kp, damping=args.kd, armature=args.arm or None)
    cfg.robot_cfg = cfg.robot_cfg.replace(actuators=acts)
if args.viters:
    ap = cfg.robot_cfg.spawn.articulation_props
    cfg.robot_cfg = cfg.robot_cfg.replace(spawn=cfg.robot_cfg.spawn.replace(articulation_props=ap.replace(solver_velocity_iteration_count=args.viters)))
print("CFG_FINGERS", cfg.robot_cfg.actuators["fingers"].stiffness, cfg.robot_cfg.actuators["fingers"].damping, cfg.robot_cfg.actuators["fingers"].effort_limit_sim, cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count, flush=True)
raw = gym.make("Isaac-JaxV2-FrontGraspLift-Direct-v16", cfg=cfg)
acfg = handle_deprecated_rsl_rl_cfg(JaxV2FrontGraspLiftPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(raw, clip_actions=acfg.clip_actions)
runner = OnPolicyRunner(env, acfg.to_dict(), log_dir=None, device=acfg.device); runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = raw.unwrapped
ids = list(u.arm_ids[:8]) + list(u.finger_ids)
names = [u.robot.joint_names[i][:14] for i in ids]
dt = u.step_dt
T = int(12.0 / dt) - 2
acc = {k: torch.zeros(len(ids), device=u.device) for k in ("pre", "close", "hold")}
tgt = {k: torch.zeros(len(ids), device=u.device) for k in acc}
act = {k: torch.zeros(u.single_action_space.shape[0], device=u.device) for k in acc}
cnt = {k: 0 for k in acc}
obs = env.get_observations(); pv = None; pt = None; pa = None
with torch.inference_mode():
    for k in range(T):
        a = policy(obs); obs, _, _, _ = env.step(a)
        t = (k + 1) * dt
        ph = "pre" if t < 5.0 else ("close" if t < 8.0 else "hold")
        v = u.robot.data.joint_vel[:, ids]; tg = u.targets[:, ids]; aa = u.actions.clone()
        if abs(t - 4.95) < dt / 2:
            hq0 = u.robot.data.body_quat_w[:, u._hand_id].clone(); cq0 = u.object.data.root_quat_w.clone(); q_arm0 = u.robot.data.joint_pos[:, u.arm_ids[:8]].clone()
        if any(abs(t - x) < dt / 2 for x in (6.0, 7.0, 8.0, 10.0, 11.9)):
            from isaaclab.utils.math import quat_mul, quat_conjugate, axis_angle_from_quat
            ha = axis_angle_from_quat(quat_mul(u.robot.data.body_quat_w[:, u._hand_id], quat_conjugate(hq0)))
            ca = axis_angle_from_quat(quat_mul(u.object.data.root_quat_w, quat_conjugate(cq0)))
            dq = torch.rad2deg(u.robot.data.joint_pos[:, u.arm_ids[:8]] - q_arm0)
            print("ROT", json.dumps({"t": round(t, 2), "hand_rotvec_deg_xyz": [round(float(x), 1) for x in torch.rad2deg(ha).mean(0)],
                "cube_rotvec_deg_xyz": [round(float(x), 1) for x in torch.rad2deg(ca).mean(0)],
                "arm_dq_deg_mean": [round(float(x), 1) for x in dq.mean(0)], "arm_dq_deg_absmean": [round(float(x), 1) for x in dq.abs().mean(0)]}), flush=True)
        if abs(t - 10.0) < dt / 2:
            lift = {"rise_mm": round(float(u._cube_rise.mean() * 1000), 1), "lifted": round(float((u._cube_rise > 0.04).float().mean()), 2),
                    "xy_mm": round(float(u._cube_shift.mean() * 1000), 1), "rot_deg": round(float(u._cube_rotation_deg.mean()), 1)}
        if pv is not None and k > 1:
            acc[ph] += (v - pv).square().mean(0); tgt[ph] += (tg - pt).square().mean(0); act[ph] += (aa - pa).square().mean(0); cnt[ph] += 1
        pv, pt, pa = v.clone(), tg.clone(), aa
print("LIFT", json.dumps({"effort": args.effort, "ema": args.ema, "kp": args.kp, "kd": args.kd, "vit": args.viters, "armature": args.arm, **lift}), flush=True)
for ph in acc:
    n = max(cnt[ph], 1)
    print("JIT", json.dumps({"phase": ph, "dvel_rms_deg_s": {nm: round(float(x), 2) for nm, x in zip(names, torch.rad2deg((acc[ph] / n).sqrt()))},
        "dtarget_rms_deg": {nm: round(float(x), 3) for nm, x in zip(names, torch.rad2deg((tgt[ph] / n).sqrt()))},
        "daction_rms": [round(float(x), 3) for x in (act[ph] / n).sqrt()]}), flush=True)
import os; os._exit(0)
