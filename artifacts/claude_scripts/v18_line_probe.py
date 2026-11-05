"""Offline path design for v18: joint poses that move the hand straight back
(+x, toward the robot) from the pre-grasp pose at the same height and hand
pitch (shoulder pitch, elbow, wrist pitch; all about world y). Prints the
deepest hand-mesh point over the table and its clearance above the top."""
import sys; DZ = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0; sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import json, math, torch, gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_arm_raise import JaxV2RaiseToPregraspEnvCfg
cfg = JaxV2RaiseToPregraspEnvCfg(); cfg.scene.num_envs = 1
u = gym.make("Isaac-JaxV2-RaiseToPregrasp-Direct-v18", cfg=cfg).unwrapped; u.reset()
arm = list(u.arm_ids); cols = [arm[0], arm[3], arm[5]]
goal = torch.tensor(cfg.raise_goal_q, device=u.device)
fing = torch.zeros(len(u.finger_ids), device=u.device); fing[2] = math.radians(-30)
def put(q8):
    q = u.robot.data.joint_pos.clone(); q[0, arm] = q8; q[0, u.finger_ids] = fing
    u.robot.write_joint_state_to_sim(q, torch.zeros_like(q)); u.sim.forward(); u.robot.update(0.0)
def hand():
    p = u.robot.data.body_pos_w[0, u._hand_id] - u.scene.env_origins[0]
    return p
with torch.inference_mode():
    put(goal); p0 = hand().clone()
    q = goal.clone(); out = []
    for k in range(0, 11, 2):
        dx = 0.01 * k
        for it in range(30):
            put(q); err = torch.zeros(3, device=u.device)
            p = hand(); err[0] = p0[0] + dx - p[0]; err[1] = p0[2] + DZ - p[2]
            J = u.robot.root_physx_view.get_jacobians()[0, u._hand_id - 1]
            # pitch: keep the sum of the three pitch joints (all about world y)
            Js = torch.stack([J[0, cols], J[2, cols], torch.ones(3, device=u.device)])
            err[2] = (goal[0] + goal[3] + goal[5]) - (q[0] + q[3] + q[5])
            dq = torch.linalg.lstsq(Js, err[:, None]).solution[:, 0]
            q[[0, 3, 5]] += dq
        put(q); m = u._hand_mesh_points()[0] if hasattr(u, "_hand_mesh_points") else None
        over = m[:, 0] < -0.34
        zmin_over = float(m[over, 2].min()) if over.any() else None
        out.append({"dx": round(dx, 2), "q_deg": [round(float(v), 1) for v in torch.rad2deg(q)], "hand_xz": [round(float(hand()[0]), 3), round(float(hand()[2]), 3)],
                    "mesh_x_min": round(float(m[:, 0].min()), 3), "mesh_z_min": round(float(m[:, 2].min()), 3),
                    "zmin_over_table": None if zmin_over is None else round(zmin_over, 3)})
        print("LINE " + json.dumps(out[-1]), flush=True)
import os; os._exit(0)
