"""v34 reach: how far forward (-x) the pre-grasp fingertip midpoint can go with the hand orientation kept.\nv27: exact per-offset IK of the v23 path for a cube moved (dx, dy) on the table (wider range than
the linear v25 table). Per waypoint and grid point: move the fingertip midpoint by (dx, dy, 0) with hand
roll/pitch kept, yaw free (as v25), warm-started along the grid. Reports residual and limit hits;
saves data/v27_shift_grid.pt {"ts", "gx", "gy", "dq": (n_wp, nx, ny, 8) rad} when argv[1] == save."""
import sys; SAVE = len(sys.argv) > 1 and sys.argv[1] == "save"; R = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08
NR = int(sys.argv[3]) if len(sys.argv) > 3 else int(__import__("os").environ.get("REACH_NR", "2"))  # 3: hand yaw fixed too (v27b)
OUT = sys.argv[4] if len(sys.argv) > 4 else "v27_shift_grid.pt"
DZ = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0  # v34: table (and cube) height offset
sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import math, torch, gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab.utils.math import quat_mul, quat_conjugate, axis_angle_from_quat
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_arm_raise import JaxV2RaiseToPregraspEnvCfg, V23_WAYPOINTS
cfg = JaxV2RaiseToPregraspEnvCfg(); cfg.scene.num_envs = 1
u = gym.make("Isaac-JaxV2-RaiseToPregrasp-Direct-v18", cfg=cfg).unwrapped; u.reset()
arm = list(u.arm_ids); lo = u.lower[arm]; hi = u.upper[arm]
use = [0, 2, 3, 4, 5, 6]
def put(q8):
    q = u.robot.data.joint_pos.clone(); q[0, arm] = q8; q[0, u.finger_ids[2]] = math.radians(-30)
    u.robot.write_joint_state_to_sim(q, torch.zeros_like(q)); u.sim.forward(); u.robot.update(0.0)
def pose():
    tip = u.robot.data.body_pos_w[0, list(u.fingertip_ids)].mean(0) - u.scene.env_origins[0]
    return tip, u.robot.data.body_quat_w[0, u._hand_id]
def solve(q0, p0, r0, d, qinit):
    q = qinit.clone()
    for it in range(60):
        put(q); p, r = pose()
        err = torch.cat((p0 + d - p, axis_angle_from_quat(quat_mul(r0[None], quat_conjugate(r[None])))[0][:NR]))
        Jh = u.robot.root_physx_view.get_jacobians()[0, u._hand_id - 1][:, [arm[j] for j in use]]
        r_ = p - (u.robot.data.body_pos_w[0, u._hand_id] - u.scene.env_origins[0])
        skew = torch.tensor([[0, -r_[2], r_[1]], [r_[2], 0, -r_[0]], [-r_[1], r_[0], 0]], device=u.device)
        J = torch.cat((Jh[:3] - skew @ Jh[3:], Jh[3:3 + NR]))
        dq = torch.linalg.lstsq(J.T @ J + 1e-4 * torch.eye(len(use), device=u.device), (J.T @ err[:, None])).solution[:, 0]
        q[use] += dq.clamp(-0.1, 0.1); q = torch.minimum(torch.maximum(q, lo), hi)
    put(q); p, r = pose()
    return q, float((p - p0 - d).norm() * 1000)
with torch.inference_mode():
    w = V23_WAYPOINTS[-1]
    q0 = torch.tensor([math.radians(v) for v in w[1]], device=u.device)
    put(q0); p0, r0 = [x.clone() for x in pose()]
    print("BASE tip", [round(float(v), 3) for v in p0], flush=True)
    q = q0.clone()
    import os as _os
    DZ = float(_os.environ.get("REACH_DZ", "0"))
    for dx in [-0.02 * k for k in range(0, 10)]:
        d = torch.tensor([dx, 0.0, DZ], device=u.device)
        q, res = solve(q0, p0, r0, d, q)
        at = ((q - lo).abs() < 1e-3) | ((hi - q).abs() < 1e-3); at[1] = False
        _hq = u.robot.data.body_quat_w[0, u._hand_id]; _tip = u.robot.data.body_pos_w[0, list(u.fingertip_ids)].mean(0); _d = _tip - u.robot.data.body_pos_w[0, u._hand_id]; _d = _d / _d.norm()
        print("HAND_DIR pitch_deg %.1f" % float(torch.rad2deg(torch.asin(_d[2]))), end="  ")
        print("REACH dz %.3f dx %.2f mid_x %.3f resid_mm %.1f" % (DZ, dx, float(p0[0] + dx), res), "q", [round(math.degrees(float(v)), 1) for v in q[:7]], "atlim", [k for k in range(8) if at[k]], flush=True)

if SAVE:
    torch.save({"ts": torch.tensor([w[0] for w in V23_WAYPOINTS]), "gx": gx, "gy": gy, "dq": out},
               "source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/data/" + OUT)
    print("SAVED", flush=True)
sys.stdout.flush()
import os; os._exit(0)
