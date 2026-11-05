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
cfg = JaxV2FrontGraspLiftEnvCfg(); cfg.scene.num_envs = 36; cfg.sensor_randomize = False
raw = gym.make("Isaac-JaxV2-FrontGraspLift-Direct-v16", cfg=cfg); u = raw.unwrapped
grid = torch.tensor([[dx, dy] for dx in (-0.08, -0.03, 0.02) for dy in (-0.05, 0.0, 0.06)], device=u.device)
u._gl_cube_off_fixed = grid[torch.arange(36, device=u.device) % 9]
ac = handle_deprecated_rsl_rl_cfg(JaxV2FrontGraspLiftPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(raw, clip_actions=1.0)
runner = OnPolicyRunner(env, ac.to_dict(), log_dir=None, device=ac.device); runner.load(CK); pol = runner.get_inference_policy(device=u.device)
u._reset_idx(u.robot._ALL_INDICES)
obs = env.get_observations()
names = ["proximal_1", "proximal_2", "finger_1", "finger_2"]
with torch.inference_mode():
    q0 = torch.rad2deg(u.robot.data.joint_pos[0, list(u.arm_ids)])
    print("START", json.dumps({"t": round(float(u.episode_length_buf[0]) * u.step_dt, 2), "q_env0": [round(float(v), 1) for v in q0]}), flush=True)
    cube0 = u.object.data.root_pos_w.clone(); tmax = torch.zeros(36, device=u.device)
    print("Z cube0_z", round(float(cube0[:, 2].mean() - u.scene.env_origins[:, 2].mean()), 4), "table_z", round(float(u.table.data.root_pos_w[:, 2].mean()), 4) if hasattr(u, "table") else None, flush=True)
    for k in range(round(6.5 * 60)):
        obs, _, _, _ = env.step(pol(obs))
        tmax = torch.maximum(tmax, u._table_force)
        if k in (60, 120):
            from isaaclab.utils.math import quat_apply_inverse
            cp = u.object.data.root_pos_w; cq = u.object.data.root_quat_w
            tips = u.robot.data.body_pos_w[:, u.fingertip_ids]
            rel = quat_apply_inverse(cq[:, None].expand(-1, 3, -1).reshape(-1, 4), (tips - cp[:, None]).reshape(-1, 3)).reshape(-1, 3, 3) * 1000
            tilt = torch.rad2deg(2 * torch.acos((cq[:, 0]).abs().clamp(max=1)))
            pf = u.pad_forces()
            G = torch.arange(36, device=u.device) % 9
            for g in range(9):
                m = G == g
                print("GR k%d (%+.0f,%+.0f)" % (k, grid[g, 0] * 100, grid[g, 1] * 100), "tip_rel_cube_mm[f1,f2,th](x,y,z)", [[round(float(v)) for v in rel[m, i].mean(0)] for i in range(3)], "pads", [round(float(v), 2) for v in pf[m, 3:].mean(0)], "cube_tilt_deg %.1f" % float(tilt[m].mean()), "rise %.0f" % float(((cp[:, 2] - cube0[:, 2])[m]).mean() * 1000), flush=True)
    rise = u.object.data.root_pos_w[:, 2] - cube0[:, 2]
    xy = (u.object.data.root_pos_w[:, :2] - cube0[:, :2]).norm(dim=1)
for g in range(9):
    m = (torch.arange(36, device=u.device) % 9) == g
    print("RES", json.dumps({"dx_cm": round(float(grid[g, 0]) * 100, 1), "dy_cm": round(float(grid[g, 1]) * 100, 1),
          "lifted": round(float((rise[m] > 0.04).float().mean()), 2), "rise_mm": round(float(rise[m].mean()) * 1000, 1),
          "table_N_max": round(float(tmax[m].max()), 2), "xy_drift_mm": round(float(xy[m].mean()) * 1000, 1)}), flush=True)
print("ALL lifted", round(float((rise > 0.04).float().mean()), 3), flush=True)
sys.stdout.flush()
os._exit(0)
