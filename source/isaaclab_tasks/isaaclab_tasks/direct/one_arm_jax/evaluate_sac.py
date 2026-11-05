"""Fixed-seed 16-environment evaluation with a mandatory rendered video."""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--algorithm", choices=("SAC", "PPO"), default="SAC")
parser.add_argument("--tag", default="")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--steps", type=int, default=480)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args)
simulation_app = app.app

import gymnasium as gym
import torch
import yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner

from isaaclab_tasks.direct.one_arm_jax.env import OneArmJaxPickPlaceEnvCfg

run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
cfg = OneArmJaxPickPlaceEnvCfg()
cfg.scene.num_envs = 16
cfg.seed = args.seed
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.eye = (1.4, 1.5, 1.7)
cfg.viewer.lookat = (-0.25, 0.0, 0.95)
cfg.viewer.resolution = (960, 540)
cfg.log_dir = run_dir
raw_env = gym.make("Isaac-OneArmJax-PickPlace-Direct-v1", cfg=cfg, render_mode="rgb_array")
suffix = args.algorithm.lower() + ("_" + args.tag if args.tag else "")
video_dir = os.path.join(run_dir, "videos", "evaluation_" + suffix)
os.makedirs(video_dir, exist_ok=True)
video_env = gym.wrappers.RecordVideo(
    raw_env, video_folder=video_dir, step_trigger=lambda step: step == 0,
    video_length=args.steps, disable_logger=True,
)
env = SkrlVecEnvWrapper(video_env, ml_framework="torch")
agent_path = os.path.join(os.path.dirname(__file__), "agents", "skrl_" + args.algorithm.lower() + "_cfg.yaml")
with open(agent_path) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["agent"]["experiment"].update(
    directory=os.path.join(run_dir, "eval_logs"), experiment_name="evaluation",
    write_interval=0, checkpoint_interval=0,
)
runner = Runner(env, agent_cfg)
runner.agent.load(args.checkpoint)
runner.agent.enable_training_mode(False, apply_to_models=True)

obs, _ = env.reset()
states = env.state()
active = torch.ones(16, dtype=torch.bool, device=env.device)
keys = ("contact", "grasp", "lift", "lift_held", "transport", "target_reach", "drop", "place", "success")
ever = {key: torch.zeros(16, dtype=torch.bool, device=env.device) for key in keys}
reward_sum = torch.zeros(16, device=env.device)
max_action = 0.0
joint_limit_steps = 0
finger_sum = 0.0
finger_joint_sum = torch.zeros(6, device=env.device)
finger_target_sum = torch.zeros(6, device=env.device)
gripper_action_sum = 0.0
gripper_open_steps = 0
closest_ee_distance = torch.full((16,), float("inf"), device=env.device)
contact_steps = torch.zeros(16, device=env.device)
finger_contact_steps = torch.zeros(3, device=env.device)
opposing_contact_steps = 0
for step in range(args.steps):
    with torch.inference_mode():
        actions, outputs = runner.agent.act(obs, states, timestep=100000, timesteps=100000)
        actions = outputs.get("mean_actions", actions).clamp(-1.0, 1.0)
        max_action = max(max_action, float(actions.abs().max()))
        obs, rewards, terminated, truncated, _ = env.step(actions)
        metrics = raw_env.unwrapped.extras["metrics"]
        core = raw_env.unwrapped
        obj, ee, _ = core._base_states()
        closest_ee_distance = torch.minimum(
            closest_ee_distance, torch.linalg.vector_norm(obj - ee, dim=-1)
        )
        contact_steps += metrics["contact"].float() * active
        finger_contacts = core._contacts()
        finger_contact_steps += (finger_contacts & active[:, None]).float().sum(dim=0)
        opposing_contact_steps += int((finger_contacts[:, 2] & (finger_contacts[:, 0] | finger_contacts[:, 1]) & active).sum())
        finger_sum += float(core.robot.data.joint_pos[:, core.finger_ids[:3]].mean())
        finger_joint_sum += core.robot.data.joint_pos[:, core.finger_ids].mean(dim=0)
        finger_target_sum += core.targets[:, core.finger_ids].mean(dim=0)
        gripper_action_sum += float(actions[:, 8].mean())
        gripper_open_steps += int((actions[:, 8] < -0.5).sum())
        for key in keys:
            ever[key] |= metrics[key] & active
        reward_sum += rewards.flatten() * active
        robot = raw_env.unwrapped.robot
        q = robot.data.joint_pos
        lower, upper = raw_env.unwrapped.lower, raw_env.unwrapped.upper
        joint_limit_steps += int((((q < lower + 0.02) | (q > upper - 0.02)).any(dim=1) & active).sum())
        active &= ~(terminated.flatten() | truncated.flatten())
        states = env.state()
counts = {key: int(ever[key].sum()) for key in keys}
summary = {
    "checkpoint": args.checkpoint,
    "seed": args.seed,
    "num_envs": 16,
    "steps": args.steps,
    "counts": counts,
    "reward_mean": float(reward_sum.mean()),
    "max_abs_action": max_action,
    "joint_limit_env_steps": joint_limit_steps,
    "finger_proximal_mean_rad": finger_sum / args.steps,
    "finger_joint_mean_rad": (finger_joint_sum / args.steps).cpu().tolist(),
    "finger_target_mean_rad": (finger_target_sum / args.steps).cpu().tolist(),
    "gripper_action_mean": gripper_action_sum / args.steps,
    "gripper_open_fraction": gripper_open_steps / (args.steps * 16),
    "closest_ee_object_mean_m": float(closest_ee_distance.mean()),
    "contact_steps_mean": float(contact_steps.mean()),
    "finger_contact_steps": finger_contact_steps.cpu().tolist(),
    "opposing_contact_steps": opposing_contact_steps,
}
output_path = os.path.join(run_dir, "evaluation_" + suffix + ".json")
with open(output_path, "w") as f:
    json.dump(summary, f, indent=2)
print("EVALUATION_JSON=" + output_path, flush=True)
print(json.dumps(summary), flush=True)
env.close()
simulation_app.close()
