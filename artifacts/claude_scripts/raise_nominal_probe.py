"""v17 with zero actions (nominal path only): keep-out depth, hand-table contact, final pose error."""
import sys; sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import json, torch, gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_arm_raise import JaxV2ArmRaiseEnvCfg
cfg = JaxV2ArmRaiseEnvCfg(); cfg.scene.num_envs = 8
env = gym.make("Isaac-JaxV2-ArmRaise-Direct-v17", cfg=cfg).unwrapped; env.reset()
a = torch.zeros(8, env.single_action_space.shape[0], device=env.device)
sens = [f"proximal_{i}" for i in (1, 2, 3)] + [f"finger_{i}" for i in (1, 2, 3)]
with torch.inference_mode():
    for k in range(int(4.9 / env.step_dt)):
        env.step(a)
        if k % 30 == 29:
            err, spd = env.raise_errors()
            cn = sum(torch.linalg.vector_norm(env.scene.sensors[n].data.net_forces_w[:, 0], dim=-1) for n in sens)
            p = env.robot.data.body_pos_w[0, list(env.fingertip_ids)] - env.scene.env_origins[0]
            print("NP", json.dumps({"t": round((k + 1) * env.step_dt, 2), "keepout_mm": round(float(env.raise_keepout().max() * 1000), 1),
                  "contact_n": round(float(cn.max()), 2), "err_max_deg": round(float(torch.rad2deg(err.amax(1)).max()), 1),
                  "tips_xz": [[round(float(v[0]), 3), round(float(v[2]), 3)] for v in p]}), flush=True)
import os; os._exit(0)
