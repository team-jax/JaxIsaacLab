"""Print the v16 scene layout and link positions at the hanging pose and at the L start pose."""
import sys; sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import torch, gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_lift import JaxV2FrontGraspLiftEnvCfg
cfg = JaxV2FrontGraspLiftEnvCfg(); cfg.scene.num_envs = 1
print("TABLE", cfg.table_center, cfg.table_cfg.size, cfg.table_top_z, "CUBE", cfg.object_cfg.init_state.pos, "ROBOT", cfg.robot_cfg.init_state.pos, flush=True)
env = gym.make("Isaac-JaxV2-FrontGraspLift-Direct-v16", cfg=cfg).unwrapped; env.reset()
o = env.scene.env_origins[0]
names = env.robot.body_names
def dump(tag):
    p = env.robot.data.body_pos_w[0] - o
    print(tag, {n: [round(float(v), 3) for v in p[i]] for i, n in enumerate(names)}, flush=True)
dump("L_START")
print("CUBE_POS", [round(float(v), 3) for v in env.object.data.root_pos_w[0] - o], flush=True)
q = env.robot.data.joint_pos.clone(); q[:] = 0.0
env.robot.write_joint_state_to_sim(q, torch.zeros_like(q)); env.robot.set_joint_position_target(q)
for _ in range(5):
    env.scene.write_data_to_sim(); env.sim.step(); env.scene.update(env.sim.get_physics_dt())
dump("HANGING")
print("JOINTS", env.robot.joint_names, flush=True)
import os; os._exit(0)
