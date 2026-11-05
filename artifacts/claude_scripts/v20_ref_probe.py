"""v20: keep-out depth of the hand mesh along the reference path (kinematic, states written)."""
import sys; sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import json, math, torch, gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_arm_raise import JaxV2RaiseToLEnvCfg, raise_nominal
cfg = JaxV2RaiseToLEnvCfg(); cfg.scene.num_envs = 1
u = gym.make("Isaac-JaxV2-RaiseToL-Direct-v20", cfg=cfg).unwrapped; u.reset()
arm = list(u.arm_ids); goal = u._raise_goal(); worst = 0.0
with torch.inference_mode():
    for k in range(0, 7 * 60, 6):
        q = u.robot.data.joint_pos.clone(); q[0, arm] = raise_nominal(cfg, torch.tensor([k / 60.0], device=u.device), goal)[0]
        u.robot.write_joint_state_to_sim(q, torch.zeros_like(q)); u.sim.forward(); u.robot.update(0.0)
        ko = float(u.raise_keepout()[0]); worst = max(worst, ko)
        m = u._hand_mesh_points()[0]
        if k % 60 == 0 or ko > 0:
            print("REFP", json.dumps({"t": round(k / 60, 2), "keepout_mm": round(ko * 1000, 1), "mesh_x_min": round(float(m[:, 0].min()), 3), "mesh_z_min": round(float(m[:, 2].min()), 3)}), flush=True)
print("REFP_WORST_MM", round(worst * 1000, 2))
import os; os._exit(0)
