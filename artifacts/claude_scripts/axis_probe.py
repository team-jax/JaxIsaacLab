"""World rotation axes of the arm joints at the v16 close onset (5 s)."""
import sys; sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import torch, gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_lift import JaxV2FrontGraspLiftEnvCfg
cfg = JaxV2FrontGraspLiftEnvCfg(); cfg.scene.num_envs = 4
env = gym.make("Isaac-JaxV2-FrontGraspLift-Direct-v16", cfg=cfg).unwrapped
env.reset()
a = torch.zeros((4, env.single_action_space.shape[0]), device=env.device)
for _ in range(int(4.9 / env.step_dt)):
    env.step(a)
arm = list(env.arm_ids[:7])
J = env.robot.root_physx_view.get_jacobians()[:, env._hand_id - 1][:, 3:, arm]
torch.set_printoptions(precision=3, sci_mode=False)
print("AXES (rows wx,wy,wz; cols arm joints 0..6)", J[0], flush=True)
print("Q_DEG", torch.rad2deg(env.robot.data.joint_pos[0, arm]), flush=True)
import os; os._exit(0)
