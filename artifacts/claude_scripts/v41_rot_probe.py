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
cfg = JaxV2FrontGraspLiftEnvCfg(); NE = int(os.environ.get("NE", "36")); cfg.scene.num_envs = NE; cfg.sensor_randomize = False
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
    for k in range(round(6.5 * 60)):
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
    rise = u.object.data.root_pos_w[:, 2] - cube0[:, 2]
    xy = (u.object.data.root_pos_w[:, :2] - cube0[:, :2]).norm(dim=1)
for g in range(9):
    m = (torch.arange(NE, device=u.device) % 9) == g
    print("RES", json.dumps({"dx_cm": round(float(grid[g, 0]) * 100, 1), "dy_cm": round(float(grid[g, 1]) * 100, 1),
          "lifted": round(float((rise[m] > 0.04).float().mean()), 2), "rise_mm": round(float(rise[m].mean()) * 1000, 1),
          "table_N_max": round(float(tmax[m].max()), 2), "xy_drift_mm": round(float(xy[m].mean()) * 1000, 1)}), flush=True)
ok = rise > 0.04
print("ROT cube_world_max_during_lift_deg %.1f hand_max_during_lift_deg %.1f cube_world_max_during_return_deg %.1f cube_in_hand_max_deg %.1f (lifted envs, mean)" % tuple(float(v[ok].mean()) if ok.any() else -1 for v in (crot_lift, hrot_lift, crot_ret, relmax)), flush=True)
print("ALL lifted", round(float((rise > 0.04).float().mean()), 3), flush=True)
sys.stdout.flush()
os._exit(0)
