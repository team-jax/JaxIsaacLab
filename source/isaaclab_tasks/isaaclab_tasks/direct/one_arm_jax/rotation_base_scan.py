"""One batched physical sweep of the third-finger rotation-base joint; no RL training."""

import csv
from isaaclab.app import AppLauncher

import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)
simulation_app = app.app

import gymnasium as gym
import numpy as np
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.one_arm_jax.env import OneArmJaxPickPlaceEnvCfg

angles = np.linspace(0.0, np.pi, 33)
cfg = OneArmJaxPickPlaceEnvCfg()
cfg.scene.num_envs = len(angles)
cfg.episode_length_s = 12
cfg.finger_open = (1.4, 1.4, 1.4, 0, 0, 0)
cfg.robot_cfg.actuators["fingers"].effort_limit_sim = 20
cfg.robot_cfg.actuators["fingers"].stiffness = 80
env = gym.make("Isaac-OneArmJax-LiftHold-Direct-v1", cfg=cfg)
env.reset(seed=123)
core = env.unwrapped
device = core.device
n = len(angles)
state = core.object.data.default_root_state.clone()
state[:, :3] = core.scene.env_origins + torch.tensor((-0.205, 0.085, 0.764), device=device)
state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device)
state[:, 7:] = 0
core.object.write_root_pose_to_sim(state[:, :7])
core.object.write_root_velocity_to_sim(state[:, 7:])
core.initial_object_z[:] = state[:, 2]
q = core.robot.data.joint_pos.clone()
arm = torch.tensor([0.51693858, -0.043632, -0.97900159, -1.23282265,
                    0.46861325, 1.32154267, -0.55498539, 0.69469648], device=device)
q[:, core.arm_ids] = arm
q[:, core.arm_ids[-1]] = torch.tensor(angles, device=device)
q[:, core.finger_ids] = torch.tensor(cfg.finger_open, device=device)
core.targets[:] = q
core.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
core.robot.set_joint_position_target(q)
zero = torch.zeros((n, 9), device=device)
for _ in range(20):
    env.step(zero)
curl = torch.zeros((n, 3), device=device)
open_finger = torch.tensor(cfg.finger_open, device=device)
close_finger = torch.tensor(cfg.finger_close, device=device)
max_force = torch.zeros((n, 3), device=device)
max_run = torch.zeros(n, device=device, dtype=torch.long)
run = torch.zeros_like(max_run)
contact_steps = torch.zeros_like(max_run)
for _ in range(200):
    force = core._pressure_forces()
    curl += torch.where(force < 3, 0.005, torch.where(force > 10, -0.01, 0.0))
    curl.clamp_(0, 1)
    core.targets[:, core.finger_ids] = open_finger + torch.cat((curl, curl), dim=1) * (close_finger - open_finger)
    env.step(zero)
    force = core._pressure_forces()
    max_force = torch.maximum(max_force, force)
    contact = (force >= 2).all(dim=1)
    run = torch.where(contact, run + 1, 0)
    max_run = torch.maximum(max_run, run)
    contact_steps += contact.long()
obj = core.object.data.root_pos_w - core.scene.env_origins
with open("artifacts/rotation_base_scan.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(("q7_rad", "contact_steps", "max_run", "p1_max_n", "p2_max_n", "p3_max_n", "final_x", "final_y"))
    for i, angle in enumerate(angles):
        writer.writerow((angle, contact_steps[i].item(), max_run[i].item(), *max_force[i].tolist(),
                         obj[i, 0].item(), obj[i, 1].item()))
print("BEST", sorted(zip(max_run.tolist(), angles.tolist()), reverse=True)[:10], flush=True)
env.close()
simulation_app.close()
