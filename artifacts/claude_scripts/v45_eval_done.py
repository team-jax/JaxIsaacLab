"""v27: v26_eval over the wide cube box (x -8/-3/+2 cm, y -5/0/+6 cm), exact-IK pre-grasp (R unused):
start pose check, then deterministic policy: lifted (>4 cm at the end), table force, cube xy drift."""
import sys
CK = sys.argv[1]; R = float(sys.argv[2]); sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import os, json, math, importlib.metadata, torch, gymnasium as gym
os.environ["GL_CUBE_BOX"] = "1"
import isaaclab_tasks  # noqa
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2FrontGraspLiftPPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_lift import JaxV2FrontGraspLiftEnvCfg
cfg = JaxV2FrontGraspLiftEnvCfg(); cfg.episode_length_s = 5.0 + float(os.environ.get("EVAL_S", "6.5")) + 1.0; NE = int(os.environ.get("NE", "36")); cfg.scene.num_envs = NE; cfg.sensor_randomize = False
raw = gym.make("Isaac-JaxV2-FrontGraspLift-Direct-v16", cfg=cfg); u = raw.unwrapped
grid = torch.tensor([[dx, dy] for dx in (-0.08, -0.03, 0.02) for dy in (-0.05, 0.0, 0.06)], device=u.device)
u._gl_cube_off_fixed = grid[torch.arange(NE, device=u.device) % 9]
ac = handle_deprecated_rsl_rl_cfg(JaxV2FrontGraspLiftPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(raw, clip_actions=1.0)
runner = OnPolicyRunner(env, ac.to_dict(), log_dir=None, device=ac.device); runner.load(CK); pol = runner.get_inference_policy(device=u.device)
u._reset_idx(u.robot._ALL_INDICES)
obs = env.get_observations()
names = ["proximal_1", "proximal_2", "finger_1", "finger_2"]
with torch.inference_mode():
    q0 = torch.rad2deg(u.robot.data.joint_pos[0, list(u.arm_ids)])
    print("START", json.dumps({"t": round(float(u.episode_length_buf[0]) * u.step_dt, 2), "q_env0": [round(float(v), 1) for v in q0]}), flush=True)
    from isaaclab.utils.math import quat_mul, quat_conjugate
    def ang(q1, q0):
        return torch.rad2deg(2 * torch.acos((q1 * q0).sum(-1).abs().clamp(max=1.0)))
    cq0 = u.object.data.root_quat_w.clone(); hq0 = u.robot.data.body_quat_w[:, u._hand_id].clone()
    crot_lift = torch.zeros(NE, device=u.device); hrot_lift = torch.zeros(NE, device=u.device); crot_ret = torch.zeros(NE, device=u.device)
    rel0 = None; relmax = torch.zeros(NE, device=u.device)
    cube0 = u.object.data.root_pos_w.clone(); tmax = torch.zeros(NE, device=u.device)
    print("Z cube0_z", round(float(cube0[:, 2].mean() - u.scene.env_origins[:, 2].mean()), 4), "table_z", round(float(u.table.data.root_pos_w[:, 2].mean()), 4) if hasattr(u, "table") else None, flush=True)
    for k in range(round(float(os.environ.get("EVAL_S", "6.5")) * 60)):
        obs, _, _, _ = env.step(pol(obs))
        tmax = torch.maximum(tmax, u._table_force)
        cq = u.object.data.root_quat_w; hq = u.robot.data.body_quat_w[:, u._hand_id]
        up = (u.object.data.root_pos_w[:, 2] - cube0[:, 2]) > 0.01
        ret = (u._ret_t >= 0) if hasattr(u, "_ret_t") else torch.zeros_like(up)
        rel = quat_mul(quat_conjugate(hq), cq)  # cube orientation in the hand frame
        if rel0 is None: rel0 = rel.clone()
        rel0 = torch.where((~up)[:, None], rel, rel0)  # latch at lift-off
        lifting_now = up & ~ret
        crot_lift = torch.where(lifting_now, torch.maximum(crot_lift, ang(cq, cq0)), crot_lift)
        hrot_lift = torch.where(lifting_now, torch.maximum(hrot_lift, ang(hq, hq0)), hrot_lift)
        crot_ret = torch.where(up & ret, torch.maximum(crot_ret, ang(cq, cq0)), crot_ret)
        relmax = torch.where(up, torch.maximum(relmax, ang(rel, rel0)), relmax)
        # tilt (cube z axis vs world z) and yaw (about world z), from the start orientation
        from isaaclab.utils.math import quat_apply
        ez = torch.zeros_like(u.object.data.root_pos_w); ez[:, 2] = 1.0
        ex = torch.zeros_like(ez); ex[:, 0] = 1.0
        zc = quat_apply(cq, ez); tilt = torch.rad2deg(torch.acos(zc[:, 2].clamp(-1, 1)))
        xc = quat_apply(cq, ex); x0 = quat_apply(cq0, ex)
        yaw = torch.rad2deg(torch.atan2(xc[:, 1], xc[:, 0]) - torch.atan2(x0[:, 1], x0[:, 0]))
        yaw = (yaw + 180) % 360 - 180
        if k == 0:
            tilt_close = torch.zeros(NE, device=u.device); yaw_close = torch.zeros(NE, device=u.device)
            tilt_lift = torch.zeros(NE, device=u.device); yaw_lift = torch.zeros(NE, device=u.device)
        tilt_close = torch.where(~up, torch.maximum(tilt_close, tilt), tilt_close); yaw_close = torch.where(~up, torch.maximum(yaw_close, yaw.abs()), yaw_close)
        tilt_lift = torch.where(lifting_now, torch.maximum(tilt_lift, tilt), tilt_lift); yaw_lift = torch.where(lifting_now, torch.maximum(yaw_lift, yaw.abs()), yaw_lift)
    rise = u.object.data.root_pos_w[:, 2] - cube0[:, 2]
    xy = (u.object.data.root_pos_w[:, :2] - cube0[:, :2]).norm(dim=1)
for g in range(9):
    m = (torch.arange(NE, device=u.device) % 9) == g
    print("RES", json.dumps({"dx_cm": round(float(grid[g, 0]) * 100, 1), "dy_cm": round(float(grid[g, 1]) * 100, 1),
          "lifted": round(float((rise[m] > 0.04).float().mean()), 2), "rise_mm": round(float(rise[m].mean()) * 1000, 1),
          "table_N_max": round(float(tmax[m].max()), 2), "xy_drift_mm": round(float(xy[m].mean()) * 1000, 1)}), flush=True)
ok = rise > 0.04
print("ROT cube_world_max_during_lift_deg %.1f hand_max_during_lift_deg %.1f cube_world_max_during_return_deg %.1f cube_in_hand_max_deg %.1f (lifted envs, mean)" % tuple(float(v[ok].mean()) if ok.any() else -1 for v in (crot_lift, hrot_lift, crot_ret, relmax)), flush=True)
print("SPLIT on_table(before 1cm): tilt %.1f yaw %.1f | lifting: tilt %.1f yaw %.1f (deg, max, lifted envs mean)" % tuple(float(v[ok].mean()) if ok.any() else -1 for v in (tilt_close, yaw_close, tilt_lift, yaw_lift)), flush=True)
c = u.cfg; pf = u.pad_forces(); arm = list(u.arm_ids)
rise_l = u.object.data.root_pos_w[:, 2] - u._lift_p0[:, 2]
arrived = u._ret_t >= c.lift_return_s
grip = (pf[:, 5] >= 0.3) & ((pf[:, 3] >= 0.3) | (pf[:, 4] >= 0.3))
print("DIAG arrived %.3f rise4 %.3f grip_distal %.3f grip_any %.3f | pad mean N %s | ret_t mean %.2f | done_cnt mean %.1f" % (
    float(arrived.float().mean()), float((rise_l >= 0.04).float().mean()), float(grip.float().mean()),
    float(((pf[:, [2, 5]].amax(1) >= 0.3) & (pf[:, [0, 1, 3, 4]].amax(1) >= 0.3)).float().mean()),
    [round(float(v), 2) for v in pf.mean(0)], float(u._ret_t.mean()), float(u._done_cnt.mean())), flush=True)
for nm, m in (("arr&rise", arrived & (rise_l >= 0.04)), ("arr&rise&grip", arrived & (rise_l >= 0.04) & grip)):
    print("DIAG", nm, round(float(m.float().mean()), 3), flush=True)
dt_ = u.lift_done_t[u.lift_done] - c.close_after_s
print("DONE_T mean %.2f p90 %.2f max %.2f (s after close start)" % ((float(dt_.mean()), float(dt_.quantile(0.9)), float(dt_.max())) if len(dt_) else (-1, -1, -1)), flush=True)
print("DONE lift_done_frac %.3f" % (float(u.lift_done.float().mean()) if hasattr(u, "lift_done") else -1), flush=True)
print("ALL lifted", round(float((rise > 0.04).float().mean()), 3), flush=True)
sys.stdout.flush()
os._exit(0)
