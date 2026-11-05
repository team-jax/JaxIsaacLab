"""Fixed-seed evaluation of the reach/grasp/lift/hold policy, including video."""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--steps", type=int, default=480)
parser.add_argument("--camera-eye", type=float, nargs=3, default=(-1.2, 1.1, 1.4))
parser.add_argument("--camera-lookat", type=float, nargs=3, default=(-0.25, 0.04, 0.82))
parser.add_argument("--video-tag", type=str, default="")
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
cfg.viewer.eye = tuple(args.camera_eye)
cfg.viewer.lookat = tuple(args.camera_lookat)
cfg.viewer.resolution = (960, 540)
cfg.log_dir = run_dir
raw_env = gym.make("Isaac-OneArmJax-LiftHold-Direct-v1", cfg=cfg, render_mode="rgb_array")
video_dir = os.path.join(run_dir, "videos", "evaluation_lift_hold" + ("_" + args.video_tag if args.video_tag else ""))
os.makedirs(video_dir, exist_ok=True)
video_env = gym.wrappers.RecordVideo(
    raw_env, video_folder=video_dir, step_trigger=lambda step: step == 0,
    video_length=args.steps, disable_logger=True,
)
env = SkrlVecEnvWrapper(video_env, ml_framework="torch")
agent_path = os.path.join(os.path.dirname(__file__), "agents", "skrl_ppo_cfg.yaml")
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
keys = ("reach", "raw_contact_3", "contact_2", "contact_3", "grasp_quality", "grasp",
        "lift_1cm", "lift_2cm", "lift_4cm", "grasped_lift_4cm", "holding", "hold_4cm", "drop", "success")
ever = {key: torch.zeros(16, dtype=torch.bool, device=env.device) for key in keys}
ever_finger = torch.zeros((16, 3), dtype=torch.bool, device=env.device)
ever_side_finger = torch.zeros((16, 3), dtype=torch.bool, device=env.device)
pressure_sum = torch.zeros(3, device=env.device)
pressure_max = torch.zeros(3, device=env.device)
pressure_saturation = 0
active_env_steps = 0
reward_sum = torch.zeros(16, device=env.device)
joint_limit_steps = 0
finger_relative_samples = {}
observations_finite = True
rewards_finite = True
reward_terms_finite = True
observation_shape = None
reward_shape = None
contact_position_shape = None

for step in range(args.steps):
    with torch.inference_mode():
        actions, outputs = runner.agent.act(obs, states, timestep=100000, timesteps=100000)
        actions = outputs.get("mean_actions", actions).clamp(-1.0, 1.0)
        obs, rewards, terminated, truncated, _ = env.step(actions)
        core = raw_env.unwrapped
        metrics = core.extras["metrics"]
        observation_shape = tuple(obs.shape)
        reward_shape = tuple(rewards.shape)
        contact_position_shape = tuple(core.scene.sensors["finger_1"].data.contact_pos_w.shape)
        observations_finite &= bool(torch.isfinite(obs).all())
        rewards_finite &= bool(torch.isfinite(rewards).all())
        reward_terms_finite &= all(bool(torch.isfinite(value).all()) for value in core.extras["log"].values())
        if step in (100, 200):
            relative = core.robot.data.body_pos_w[:, core.fingertip_ids] - core.object.data.root_pos_w[:, None]
            finger_relative_samples[str(step)] = relative[active].mean(dim=0).cpu().tolist()
        pressure = core._pressures()
        active_env_steps += int(active.sum())
        pressure_sum += (pressure * active[:, None]).sum(dim=0)
        pressure_max = torch.maximum(pressure_max, torch.where(active[:, None], pressure, 0.0).max(dim=0).values)
        pressure_saturation += int(((pressure >= 1.0).any(dim=1) & active).sum())
        ever_finger |= metrics["pressure_contact"] & active[:, None]
        ever_side_finger |= metrics["side_contact"] & active[:, None]
        for key in keys:
            ever[key] |= metrics[key] & active
        reward_sum += rewards.flatten() * active
        q = core.robot.data.joint_pos
        joint_limit_steps += int((((q < core.lower + 0.02) | (q > core.upper - 0.02)).any(dim=1) & active).sum())
        active &= ~(terminated.flatten() | truncated.flatten())
        states = env.state()

summary = {
    "checkpoint": os.path.abspath(args.checkpoint), "seed": args.seed,
    "num_envs": 16, "steps": args.steps,
    "counts": {key: int(value.sum()) for key, value in ever.items()},
    "finger_contact_counts": ever_finger.sum(dim=0).cpu().tolist(),
    "meaningful_contact_counts": ever_side_finger.sum(dim=0).cpu().tolist(),
    "pressure_mean": (pressure_sum / max(active_env_steps, 1)).cpu().tolist(),
    "pressure_max": pressure_max.cpu().tolist(),
    "pressure_saturation_env_steps": pressure_saturation,
    "reward_mean": float(reward_sum.mean()),
    "joint_limit_env_steps": joint_limit_steps,
    "observation_shape": observation_shape, "reward_shape": reward_shape,
    "contact_position_shape": contact_position_shape,
    "observations_finite": observations_finite, "rewards_finite": rewards_finite,
    "reward_terms_finite": reward_terms_finite,
    "finger_relative_samples": finger_relative_samples,
    "video_dir": video_dir,
}
output_path = os.path.join(run_dir, "evaluation_lift_hold" + ("_" + args.video_tag if args.video_tag else "") + ".json")
with open(output_path, "w") as f:
    json.dump(summary, f, indent=2)
print("EVALUATION_JSON=" + output_path, flush=True)
print(json.dumps(summary), flush=True)
env.close()
simulation_app.close()
