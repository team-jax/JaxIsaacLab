"""v48j smoke test: GL_MU_RAND per-episode friction - reset, read back the cube/robot materials, step a little."""
import sys; sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import os, torch, gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_lift import JaxV2FrontGraspLiftEnvCfg
cfg = JaxV2FrontGraspLiftEnvCfg(); cfg.scene.num_envs = 8
print("CFG gl_mu_rand", cfg.gl_mu_rand, flush=True)
u = gym.make("Isaac-JaxV2-FrontGraspLift-Direct-v16", cfg=cfg).unwrapped
u.reset()
for name, a in (("cube", u.object), ("robot", u.robot)):
    m = a.root_physx_view.get_material_properties()
    print("MAT", name, tuple(m.shape), "static per env", [round(float(v), 3) for v in m[:, 0, 0]], "dyn", [round(float(v), 3) for v in m[:, 0, 1]], flush=True)
print("GL_MU env", [round(float(v), 3) for v in u._gl_mu] if hasattr(u, "_gl_mu") else None, flush=True)
a = torch.zeros((8, u.cfg.action_space), device=u.device)
for k in range(30):
    u.step(a)
u._reset_idx(torch.tensor([0, 1, 2], device=u.device))
m = u.object.root_physx_view.get_material_properties()
print("AFTER_RESET cube static", [round(float(v), 3) for v in m[:, 0, 0]], flush=True)
print("SMOKE_OK", flush=True)
import os as _o; _o._exit(0)
