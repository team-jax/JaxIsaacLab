import argparse, math
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--off", action="store_true"); AppLauncher.add_app_launcher_args(p); args = p.parse_args(); app = AppLauncher(args).app
import gymnasium as gym, torch
import isaaclab_tasks  # noqa
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_pregrasp import JaxV2FrontPregraspEnvCfg
cfg = JaxV2FrontPregraspEnvCfg(); cfg.scene.num_envs = 4; cfg.episode_length_s = 100.0
if args.off: cfg.table_stop_force_n = None
raw = gym.make("Isaac-JaxV2-FrontPregrasp-Direct-v12", cfg=cfg); u = raw.unwrapped; raw.reset()
zd = torch.zeros(4, dtype=torch.bool, device=u.device); u._get_dones = lambda: (zd, zd)
parked = u.object.data.root_state_w[:, :7].clone(); parked[:, 0] -= 0.30; parked[:, 1] -= 0.15
arm = list(u.arm_ids[:7]); push = torch.zeros((4, 7), device=u.device); orig = u._pre_physics_step
def pre(actions):
    orig(actions)
    tw = torch.zeros((4, 6, 1), device=u.device); tw[:, 2, 0] = -0.03 * u.step_dt
    jac = u.robot.root_physx_view.get_jacobians()[:, u._hand_id - 1][:, :, arm]
    push.add_(torch.linalg.lstsq(jac, tw).solution[:, :, 0]); u.targets[:, arm] += push
u._pre_physics_step = pre
a = torch.zeros_like(u.actions)
with torch.inference_mode():
    for k in range(360):
        u.object.write_root_pose_to_sim(parked); raw.step(a)
        if k % 40 == 39:
            links = [u.robot.body_names.index(n) for n in cfg.table_contact_body_names]
            z = (u.robot.data.body_pos_w[:, links, 2].min(dim=1).values - cfg.table_top_z) * 1000
            f = max(float(torch.linalg.vector_norm(u.scene.sensors[f"table_contact_{n}"].data.force_matrix_w[:, 0, 0], dim=-1).max()) for n in cfg.table_contact_body_names)
            print("TS", k + 1, "min_link_mm", round(float(z.min()), 1), "table_force_N", round(f, 2), "stop", float(getattr(u, "_table_stop", torch.zeros(1)).float().mean()), flush=True)
import os; os._exit(0)
