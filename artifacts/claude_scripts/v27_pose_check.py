"""v27: at the exact-IK pre-grasp pose for each cube offset, hand mesh clearance to the table top
and penetration into the cube box (static, no physics)."""
import sys; sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import math, torch, gymnasium as gym, isaaclab_tasks  # noqa
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_arm_raise import JaxV2RaiseCubeWideEnvCfg, grid_weights
c = JaxV2RaiseCubeWideEnvCfg(); c.scene.num_envs = 1
u = gym.make("Isaac-JaxV2-RaiseCubeWide-Direct-v27", cfg=c).unwrapped; u.reset()
d = torch.load("source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/data/v27_shift_grid.pt")
gx, gy, G = d["gx"].to(u.device), d["gy"].to(u.device), d["dq"][-1].reshape(-1, 8).to(u.device)
q0 = torch.tensor([math.radians(v) for v in (-7.3, 0.0, 2.5, 73.7, -5.6, 14.7, -6.6, -170.1)], device=u.device)
arm = list(u.arm_ids); cube0 = torch.tensor([-0.40, 0.26, 0.7675], device=u.device); top = 0.74
with torch.inference_mode():
    for dx in (-0.08, -0.06, -0.04, -0.03, -0.02, 0.0, 0.02):
        line = []
        for dy in (-0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06):
            off = torch.tensor([[dx, dy]], device=u.device)
            q = u.robot.data.joint_pos.clone(); q[0, arm] = q0 + grid_weights(gx, gy, off) @ G
            q[0, u.finger_ids] = u._hand_shape[0] if u._hand_shape.dim() > 1 else u._hand_shape
            u.robot.write_joint_state_to_sim(q, torch.zeros_like(q)); u.sim.forward(); u.robot.update(0.0)
            p = u._hand_mesh_points()[0]
            clear = float((p[:, 2] - top).min()) * 1000
            rel = p - (cube0 + torch.tensor([dx, dy, 0.0], device=u.device))
            inside = (rel.abs() < 0.0275).all(dim=1)
            pen = float((0.0275 - rel.abs()).min(dim=1).values[inside].max()) * 1000 if inside.any() else 0.0
            line.append(f"({dy*100:+.0f}){clear:5.1f}/{pen:4.1f}")
        print("DX", f"{dx*100:+.0f}", " ".join(line), flush=True)
import os; os._exit(0)
