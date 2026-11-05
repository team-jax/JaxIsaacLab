"""v25: per-waypoint joint shift for a hand shift in x, y, z at constant hand orientation, all 7
arm joints (least squares on the hand Jacobian: 3 position + 3 orientation rows). For cube (x, y)
and table (z) offsets. Saves data/v25_shift_xyz.pt (n_wp, 8, 3) rad per m [dx, dy, dz]."""
import sys; sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import math, torch, gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab.utils.math import quat_error_magnitude, quat_mul, quat_conjugate, axis_angle_from_quat
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_arm_raise import JaxV2RaiseToPregraspEnvCfg, V23_WAYPOINTS
cfg = JaxV2RaiseToPregraspEnvCfg(); cfg.scene.num_envs = 1
u = gym.make("Isaac-JaxV2-RaiseToPregrasp-Direct-v18", cfg=cfg).unwrapped; u.reset()
arm = list(u.arm_ids)
def put(q8):
    q = u.robot.data.joint_pos.clone(); q[0, arm] = q8; q[0, u.finger_ids[2]] = math.radians(-30)
    u.robot.write_joint_state_to_sim(q, torch.zeros_like(q)); u.sim.forward(); u.robot.update(0.0)
def pose():
    # the grasp point (fingertip midpoint) follows the cube, not the hand origin: with the hand
    # yaw free, the hand origin alone let the fingertips swing ~1 cm sideways into the cube
    tip = u.robot.data.body_pos_w[0, list(u.fingertip_ids)].mean(0) - u.scene.env_origins[0]
    return tip, u.robot.data.body_quat_w[0, u._hand_id]
out = []
with torch.inference_mode():
    for w in V23_WAYPOINTS:
        q0 = torch.tensor([math.radians(v) for v in w[1]], device=u.device)
        S = torch.zeros(8, 3, device=u.device)
        for k in range(3):
            d = torch.zeros(3, device=u.device); d[k] = 0.01
            put(q0); p0, r0 = pose(); p0 = p0.clone(); r0 = r0.clone(); q = q0.clone()
            for it in range(40):
                put(q); p, r = pose()
                # keep hand roll and pitch (world x, y rotation); yaw (world z) free: the hand
                # may turn toward a cube moved sideways instead of counter-twisting forearm and wrist
                err = torch.cat((p0 + d - p, axis_angle_from_quat(quat_mul(r0[None], quat_conjugate(r[None])))[0][:2]))
                use = [0, 2, 3, 4, 5, 6]  # shoulder roll sits at its 0 deg limit in these poses
                Jh = u.robot.root_physx_view.get_jacobians()[0, u._hand_id - 1][:, [arm[j] for j in use]]
                r_ = p - (u.robot.data.body_pos_w[0, u._hand_id] - u.scene.env_origins[0])
                skew = torch.tensor([[0, -r_[2], r_[1]], [r_[2], 0, -r_[0]], [-r_[1], r_[0], 0]], device=u.device)
                J = torch.cat((Jh[:3] - skew @ Jh[3:], Jh[3:5]))  # grasp-point velocity, hand roll/pitch
                q[use] += torch.linalg.lstsq(J, err[:, None]).solution[:, 0]
            put(q); p, r = pose()
            S[:, k] = (q - q0) / 0.01
            res = float((p - p0 - d).norm() * 1000)
        out.append(S.cpu())
        print("SHIFT", w[0], "deg/cm dx:", [round(math.degrees(float(v)) / 100, 2) for v in S[:7, 0]], "dy:", [round(math.degrees(float(v)) / 100, 2) for v in S[:7, 1]], "dz:", [round(math.degrees(float(v)) / 100, 2) for v in S[:7, 2]], "resid_mm", round(res, 2), flush=True)
torch.save(torch.stack(out), "source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/data/v25_shift_xyz.pt")
print("SAVED", flush=True)
sys.stdout.flush()
import os; os._exit(0)
