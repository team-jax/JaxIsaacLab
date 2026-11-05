"""One short batched physical scan of fixed cube locations; no RL training."""

import argparse
import csv
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="artifacts/grasp_contact_scan.csv")
parser.add_argument("--optimized-wide", action="store_true")
parser.add_argument("--staged", action="store_true")
parser.add_argument("--low-palm", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)
simulation_app = app.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.one_arm_jax.env import OneArmJaxPickPlaceEnvCfg

xs = np.linspace(-0.255, -0.195, 13)
ys = np.linspace(0.025, 0.095, 15)
positions = [(float(x), float(y)) for x in xs for y in ys]
cfg = OneArmJaxPickPlaceEnvCfg()
cfg.scene.num_envs = len(positions)
cfg.episode_length_s = 16.0 if args.staged else 12.0
if args.optimized_wide:
    cfg.finger_open = (1.40, 1.40, 1.40, 0, 0, 0)
    cfg.robot_cfg.actuators["fingers"].effort_limit_sim = 20.0
    cfg.robot_cfg.actuators["fingers"].stiffness = 80.0
env = gym.make("Isaac-OneArmJax-LiftHold-Direct-v1", cfg=cfg)
env.reset(seed=123)
core = env.unwrapped
device = core.device
ids = torch.arange(len(positions), device=device)
state = core.object.data.default_root_state.clone()
state[:, :3] = core.scene.env_origins + torch.tensor(
    [[x, y, 0.764] for x, y in positions], device=device)
state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device)
state[:, 7:] = 0
core.object.write_root_pose_to_sim(state[:, :7])
core.object.write_root_velocity_to_sim(state[:, 7:])
core.initial_object_z[:] = state[:, 2]
q = core.robot.data.joint_pos.clone()
grasp_arm_q = torch.tensor(
    [0.66526774, -0.0436329, -1.0471979, -1.38850439,
     0.57322713, 1.54101851, -0.65936561, 0.95230107], device=device)
if args.optimized_wide:
    grasp_arm_q = torch.tensor(
        [0.61936741, -0.03479314, -0.98698333, -1.40757144,
         0.55121264, 1.50558886, -0.69994159, 0.93852028], device=device)
approach_arm_q = torch.tensor(
    [0.785397, -0.00269022, -1.04719699, -1.51999425,
     0.72478146, 1.570795, -0.93259199, 1.31012272], device=device)
if args.low_palm:
    grasp_arm_q = torch.tensor(
        [0.51693858, -0.043632, -0.97900159, -1.23282265,
         0.46861325, 1.32154267, -0.55498539, 0.69469648], device=device)
    approach_arm_q = torch.tensor(
        [0.57597683, -0.02022471, -1.04549025, -1.49861287,
         0.51903925, 1.5401905, 0.32026581, 0.00000646], device=device)
q[:, core.arm_ids] = approach_arm_q if args.staged else grasp_arm_q
q[:, core.finger_ids] = torch.tensor(cfg.finger_open, device=device)
core.targets[:] = q
core.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
core.robot.set_joint_position_target(q)
zero_action = torch.zeros((len(positions), 9), device=device)
best_min_force = torch.zeros(len(positions), device=device)
best_contact_count = torch.zeros(len(positions), device=device, dtype=torch.long)
three_contact_steps = torch.zeros(len(positions), device=device, dtype=torch.long)
three_contact_run = torch.zeros_like(three_contact_steps)
longest_three_contact_run = torch.zeros_like(three_contact_steps)
max_lift = torch.zeros(len(positions), device=device)
max_force = torch.zeros((len(positions), 3), device=device)
for _ in range(12 if args.staged else 20):
    env.step(zero_action)
if args.staged:
    for j in range(150):
        alpha = (j + 1) / 150
        core.targets[:, core.arm_ids] = approach_arm_q * (1 - alpha) + grasp_arm_q * alpha
        env.step(zero_action)
    for _ in range(60):
        env.step(zero_action)
open_finger = torch.tensor(cfg.finger_open, device=device)
close_finger = torch.tensor(cfg.finger_close, device=device)
curl = torch.zeros((len(positions), 3), device=device)
for _ in range(200 if args.optimized_wide else 120):
    force = core._pressure_forces()
    curl += torch.where(force < 3.0, 0.005 if args.optimized_wide else 0.015,
                        torch.where(force > 10.0, -0.01, 0.0))
    curl.clamp_(0, 1)
    core.targets[:, core.finger_ids] = open_finger + torch.cat((curl, curl), dim=1) * (close_finger - open_finger)
    env.step(zero_action)
    force = core._pressure_forces()
    max_force = torch.maximum(max_force, force)
    best_min_force = torch.maximum(best_min_force, force.min(dim=1).values)
    best_contact_count = torch.maximum(best_contact_count, (force >= 2.0).sum(dim=1))
    three = (force >= 2.0).all(dim=1)
    three_contact_steps += three.long()
    three_contact_run = torch.where(three, three_contact_run + 1, 0)
    longest_three_contact_run = torch.maximum(longest_three_contact_run, three_contact_run)
    max_lift = torch.maximum(max_lift, core.object.data.root_pos_w[:, 2] - core.initial_object_z)
if args.staged:
    for j in range(90):
        alpha = (j + 1) / 90
        core.targets[:, core.arm_ids] = grasp_arm_q + (approach_arm_q - grasp_arm_q) * (0.8 * alpha)
        env.step(zero_action)
        force = core._pressure_forces()
        three = (force >= 2.0).all(dim=1)
        three_contact_steps += three.long()
        three_contact_run = torch.where(three, three_contact_run + 1, 0)
        longest_three_contact_run = torch.maximum(longest_three_contact_run, three_contact_run)
        max_lift = torch.maximum(max_lift, core.object.data.root_pos_w[:, 2] - core.initial_object_z)
obj = core.object.data.root_pos_w - core.scene.env_origins
with open(args.output, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(("x", "y", "best_min_force_n", "max_contact_count", "three_contact_steps", "longest_three_contact_run", "max_lift_m", "p1_max_n", "p2_max_n", "p3_max_n", "final_x", "final_y"))
    for i, (x, y) in enumerate(positions):
        writer.writerow((x, y, best_min_force[i].item(), best_contact_count[i].item(),
                         three_contact_steps[i].item(), longest_three_contact_run[i].item(), max_lift[i].item(),
                         *max_force[i].tolist(), obj[i, 0].item(), obj[i, 1].item()))
print("SCAN_OUTPUT=" + os.path.abspath(args.output), flush=True)
top = torch.argsort(best_min_force, descending=True)[:10].tolist()
for i in top:
    print("SCAN_TOP", positions[i], best_min_force[i].item(), best_contact_count[i].item(), max_force[i].tolist(), flush=True)
env.close()
simulation_app.close()
