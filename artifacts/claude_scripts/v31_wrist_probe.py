"""v31: wrist pitch/yaw at the L (0,0,0,90 deg) that make the hand point straight ahead (-x): direction
fingertip(1,2) midpoint - hand base, fingers at the hand-over grip shape. Newton on (j5, j6), also j4 roll scan."""
import sys; sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import math, torch, gymnasium as gym, isaaclab_tasks  # noqa
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_arm_raise import JaxV2RaiseCubeWideEnvCfg
c = JaxV2RaiseCubeWideEnvCfg(); c.scene.num_envs = 1
u = gym.make("Isaac-JaxV2-RaiseCubeWide-Direct-v27", cfg=c).unwrapped; u.reset()
arm = list(u.arm_ids); fids = list(u.finger_ids)
fing = torch.tensor([math.radians(v) for v in (13.63, 10.58, -30.43, 0, 0, 0)], device=u.device)
def dirn(q8):
    q = u.robot.data.joint_pos.clone(); q[0, arm] = q8; q[0, fids] = fing
    u.robot.write_joint_state_to_sim(q, torch.zeros_like(q)); u.sim.forward(); u.robot.update(0.0)
    d = u.robot.data.body_pos_w[0, list(u.fingertip_ids)].mean(0) - u.robot.data.body_pos_w[0, u._hand_id]  # all three tips
    return d / d.norm()
JS = (4, 5)
with torch.inference_mode():
    q8 = torch.tensor([math.radians(v) for v in (-0.15, 0.0, 0.0, 89.82, 0.15, -0.02, -0.16, -169.59)], device=u.device)
    print("NOW dir", [round(float(v), 3) for v in dirn(q8)], flush=True)
    for it in range(80):
        d0 = dirn(q8); err = torch.stack((d0[1], d0[2]))
        J = torch.zeros(2, 2, device=u.device)
        for k, j in enumerate(JS):
            qq = q8.clone(); qq[j] += 1e-3; d1 = dirn(qq); J[:, k] = (torch.stack((d1[1], d1[2])) - err) / 1e-3
        dq = torch.linalg.solve(J.T @ J + 1e-4 * torch.eye(2, device=u.device), -(J.T @ err[:, None]))[:, 0].clamp(-0.03, 0.03); q8[JS[0]] += dq[0]; q8[JS[1]] += dq[1]
    print("SOLVED", JS, [round(math.degrees(float(q8[j])), 2) for j in range(7)], "dir", [round(float(v), 4) for v in dirn(q8)], flush=True)
    from isaaclab.utils.math import quat_mul, quat_conjugate, axis_angle_from_quat, euler_xyz_from_quat
    def hq(q8):
        dirn(q8); return u.robot.data.body_quat_w[0, u._hand_id].clone()
    pre = torch.tensor([math.radians(v) for v in (-7.3, 0.0, 2.5, 73.7, -5.6, 14.7, -6.6, -169.59)], device=u.device)
    qp = hq(pre); qL = hq(torch.tensor([math.radians(v) for v in (-0.15, 0.0, 0.0, 89.82, 0.15, -0.02, -0.16, -169.59)], device=u.device))
    rel = axis_angle_from_quat(quat_mul(qp[None], quat_conjugate(qL[None])))[0]
    print("PRE vs L hand rotation (world axis-angle deg)", [round(math.degrees(float(v)), 2) for v in rel], "pre dir", [round(float(v), 3) for v in dirn(pre)], flush=True)
    q8 = torch.tensor([math.radians(v) for v in (-0.15, 0.0, 0.0, 89.82, 0.15, -0.02, -0.16, -169.59)], device=u.device)
    for it in range(40):
        e = axis_angle_from_quat(quat_mul(qp[None], quat_conjugate(hq(q8)[None])))[0]
        J = torch.zeros(3, 3, device=u.device)
        for k, j in enumerate((4, 5, 6)):
            qq = q8.clone(); qq[j] += 1e-3
            e1 = axis_angle_from_quat(quat_mul(qp[None], quat_conjugate(hq(qq)[None])))[0]; J[:, k] = (e1 - e) / 1e-3
        step = torch.linalg.solve(J.T @ J + 1e-3 * torch.eye(3, device=u.device), J.T @ e[:, None])[:, 0]
        q8[[4, 5, 6]] -= step.clamp(-0.05, 0.05)
    rem = axis_angle_from_quat(quat_mul(qp[None], quat_conjugate(hq(q8)[None])))[0]
    print("residual deg", round(math.degrees(float(rem.norm())), 3))
    print("MATCH_PRE j4 %.2f j5 %.2f j6 %.2f" % tuple(math.degrees(float(q8[j])) for j in (4, 5, 6)), "dir", [round(float(v), 3) for v in dirn(q8)], flush=True)
import os; os._exit(0)
