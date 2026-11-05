"""v25 base path only (zero residual) at a 3x3 grid of cube offsets on the nominal table:
non-cube contact, keep-out, cube motion, final error to the shifted pre-grasp pose."""
import sys; sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import json, math, torch, gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_arm_raise import JaxV2RaiseCubeEnvCfg
cfg = JaxV2RaiseCubeEnvCfg(); cfg.scene.num_envs = 9; cfg.raise_rsi_prob = 0.0; cfg.raise_gain_rand = 0.0
u = gym.make("Isaac-JaxV2-RaiseCube-Direct-v25", cfg=cfg).unwrapped
offs = torch.tensor([[dx, dy] for dx in (-0.02, 0.0, 0.02) for dy in (-0.02, 0.0, 0.02)], device=u.device)
u.reset()
u._table_off_fixed = torch.zeros(9, 2, device=u.device)
u._cube_off_fixed = offs
u._reset_idx(u.robot._ALL_INDICES)
a = torch.zeros(9, u.single_action_space.shape[0], device=u.device)
names = [f"proximal_{i}" for i in (1, 2, 3)] + [f"finger_{i}" for i in (1, 2, 3)]
cmax = torch.zeros(9, device=u.device); kmax = torch.zeros(9, device=u.device)
with torch.inference_mode():
    cp = u.object.data.root_pos_w - u.scene.env_origins
    print("CUBE", [[round(float(v), 3) for v in r] for r in cp[:, :2]], flush=True)
    for k in range(int(9.5 * 60)):
        u.step(a)
        cn = torch.stack([(torch.linalg.vector_norm(u.scene.sensors[n].data.net_forces_w[:, 0], dim=-1)
                           - torch.linalg.vector_norm(u.scene.sensors[n].data.force_matrix_w[:, 0, 0], dim=-1)).clamp_min(0) for n in names], 1).sum(1)
        cmax = torch.maximum(cmax, cn); kmax = torch.maximum(kmax, u.raise_keepout())
    goal = u._raise_goal() + u._table_shift_q(torch.full((9,), 9.0, device=u.device))
    err = torch.rad2deg((u.robot.data.joint_pos[:, list(u.arm_ids)] - goal).abs()).amax(1)
    cube = torch.linalg.vector_norm(u.object.data.root_pos_w - u.scene.env_origins - cp, dim=1)
    tips = u.robot.data.body_pos_w[:, list(u.fingertip_ids)].mean(1) - u.object.data.root_pos_w
    for i in range(9):
        print("RES", json.dumps({"cube_dx_cm": round(float(offs[i, 0]) * 100, 1), "cube_dy_cm": round(float(offs[i, 1]) * 100, 1), "table_contact_N": round(float(cmax[i]), 2),
              "keepout_mm": round(float(kmax[i]) * 1000, 1), "err_deg": round(float(err[i]), 2), "cube_moved_mm": round(float(cube[i]) * 1000, 2),
              "tips_minus_cube_mm": [round(float(v) * 1000, 1) for v in tips[i]]}), flush=True)
sys.stdout.flush()
import os; os._exit(0)
