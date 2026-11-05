"""v34: pre-grasp start geometry per table dz (-3/0/+3 cm) at 9 cube spots: fingertip-mid minus cube (mm, env
frame) and three-fingertip direction from the hand, right after reset (no policy)."""
import sys
sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import os, math, torch, gymnasium as gym
os.environ["GL_CUBE_BOX"] = "1"; os.environ["GL_TABLE_DZ_FIXED"] = "0"
import isaaclab_tasks  # noqa
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_lift import JaxV2FrontGraspLiftEnvCfg
cfg = JaxV2FrontGraspLiftEnvCfg(); cfg.scene.num_envs = 9; cfg.sensor_randomize = False; cfg.gl_start_noise_deg = 0.0
u = gym.make("Isaac-JaxV2-FrontGraspLift-Direct-v16", cfg=cfg).unwrapped
grid = torch.tensor([[dx, dy] for dx in (-0.08, -0.03, 0.02) for dy in (-0.05, 0.0, 0.06)], device=u.device)
u._gl_cube_off_fixed = grid
for dz in (-0.03, 0.0, 0.03):
    u.cfg.gl_table_dz_fixed = dz
    with torch.inference_mode():
        u._reset_idx(u.robot._ALL_INDICES)
        for _ in range(30):  # settle 0.5 s under the held targets
            u.robot.set_joint_position_target(u.targets); u.scene.write_data_to_sim(); u.sim.step(render=False); u.scene.update(u.cfg.sim.dt)
        tips = u.robot.data.body_pos_w[:, u.fingertip_ids]; mid = tips.mean(1)
        rel = (mid - u.object.data.root_pos_w) * 1000
        d = mid - u.robot.data.body_pos_w[:, u._hand_id]; d = d / d.norm(dim=1, keepdim=True)
        pitch = torch.rad2deg(torch.asin(d[:, 2])); yaw = torch.rad2deg(torch.atan2(d[:, 1], -d[:, 0]))
        tz = (tips[..., 2] - u.object.data.root_pos_w[:, 2:3]) * 1000
        print("QSTART dz%+.2f" % dz, "q(-8,0)", [round(float(v), 2) for v in torch.rad2deg(u.robot.data.joint_pos[1, list(u.arm_ids)])], "tg", [round(float(v), 2) for v in torch.rad2deg(u.targets[1, list(u.arm_ids)])], flush=True)
    for i in range(9):
        print("S dz%+.2f (%+.0f,%+.0f) mid-cube %s tip_z %s pitch %.1f yaw %.1f" % (dz, grid[i, 0] * 100, grid[i, 1] * 100,
              [round(float(v), 1) for v in rel[i]], [round(float(v), 1) for v in tz[i]], float(pitch[i]), float(yaw[i])), flush=True)
os._exit(0)
