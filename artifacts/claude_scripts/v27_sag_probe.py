"""v27: gravity sag at the exact-IK pre-grasp pose of every cube grid point (81 envs in parallel).
Targets = geometric pose + comp; settle 2 s; comp += geometric - actual (3 rounds). Saves the sag
offset relative to the nominal comp (gl_sag_comp_deg) as data/v27_sag_grid.pt (nx, ny, 8) rad."""
import sys; sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import math, torch, gymnasium as gym, isaaclab_tasks  # noqa
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_arm_raise import JaxV2RaiseCubeWideEnvCfg
d = torch.load("source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/data/v27_shift_grid.pt")
nx, ny = len(d["gx"]), len(d["gy"])
c = JaxV2RaiseCubeWideEnvCfg(); c.scene.num_envs = nx * ny
u = gym.make("Isaac-JaxV2-RaiseCubeWide-Direct-v27", cfg=c).unwrapped; u.reset()
arm = list(u.arm_ids); fids = list(u.finger_ids); N = u.num_envs
q0 = torch.tensor([math.radians(v) for v in (-7.3, 0.0, 2.5, 73.7, -5.6, 14.7, -6.6, -170.1)], device=u.device)
sag0 = torch.tensor([math.radians(v) for v in (1.9, 0.0, 0.0, 2.1, 0.0, 0.7, -0.2, 0.0)], device=u.device)
geo = q0 + d["dq"][-1].reshape(-1, 8).to(u.device)
comp = sag0.expand(N, -1).clone()
with torch.inference_mode():
    away = u.object.data.root_state_w.clone(); away[:, 0] += 3.0; away[:, 7:] = 0.0
    for rnd in range(4):
        q = u.robot.data.joint_pos.clone(); q[:, arm] = geo; q[:, fids] = u.targets[:, fids]
        u.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        u.object.write_root_pose_to_sim(away[:, :7]); u.object.write_root_velocity_to_sim(away[:, 7:])
        tg = u.targets.clone(); tg[:, arm] = geo + comp
        for k in range(240):
            u.robot.set_joint_position_target(tg); u.scene.write_data_to_sim(); u.sim.step(render=False); u.scene.update(u.physics_dt)
        e = geo - u.robot.data.joint_pos[:, arm]
        print("ROUND", rnd, "max_err_deg", round(math.degrees(float(e.abs().max())), 2), "comp_deg range sp/el", [round(math.degrees(float(comp[:, j].min())), 2) for j in (0, 3)], [round(math.degrees(float(comp[:, j].max())), 2) for j in (0, 3)], flush=True)
        if rnd < 3:
            comp = comp + e
    for i in range(nx):
        print("SAG dx", round(float(d["gx"][i]) * 100), [round(math.degrees(float(v)), 2) for v in (comp - sag0)[i * ny:(i + 1) * ny, 3]], flush=True)
torch.save((comp - sag0).reshape(nx, ny, 8).cpu(), "source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/data/v27_sag_grid.pt")
print("SAVED", flush=True)
import os; os._exit(0)
