"""Fixed-seed video and quantitative evaluation for an RSL-RL JAX arm checkpoint."""

import argparse
import importlib.metadata
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--steps", type=int, default=480)
parser.add_argument("--camera-eye", type=float, nargs=3, default=(-0.7, 0.55, 1.12))
parser.add_argument("--camera-lookat", type=float, nargs=3, default=(-0.22, 0.08, 0.78))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args)
simulation_app = app.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.direct.one_arm_jax.agents.rsl_rl_ppo_cfg import OneArmJaxLiftHoldPPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax.env import OneArmJaxPickPlaceEnvCfg

run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
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
video_dir = os.path.join(run_dir, "videos", "evaluation_grasp_stage1_rsl")
os.makedirs(video_dir, exist_ok=True)
video_env = gym.wrappers.RecordVideo(
    raw_env, video_folder=video_dir, step_trigger=lambda step: step == 0,
    video_length=args.steps, disable_logger=True,
)
agent_cfg = handle_deprecated_rsl_rl_cfg(
    OneArmJaxLiftHoldPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib")
)
env = RslRlVecEnvWrapper(video_env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

obs = env.get_observations()
active = torch.ones(16, dtype=torch.bool, device=env.device)
keys = ("reach", "raw_contact_3", "contact_2", "contact_3", "grasp_quality", "grasp",
        "grasp_hold", "pressure_excess", "success")
ever = {key: torch.zeros(16, dtype=torch.bool, device=env.device) for key in keys}
ever_finger = torch.zeros((16, 3), dtype=torch.bool, device=env.device)
ever_meaningful = torch.zeros_like(ever_finger)
pressure_sum = torch.zeros(3, device=env.device)
pressure_max = torch.zeros(3, device=env.device)
reward_sum = torch.zeros(16, device=env.device)
active_env_steps = 0
joint_limit_steps = 0
finite = True

for _ in range(args.steps):
    with torch.inference_mode():
        actions = policy(obs)
        obs, rewards, dones, _ = env.step(actions)
        core = raw_env.unwrapped
        metrics = core.extras["metrics"]
        pressure = core._pressures()
        active_env_steps += int(active.sum())
        pressure_sum += (pressure * active[:, None]).sum(dim=0)
        pressure_max = torch.maximum(pressure_max, torch.where(active[:, None], pressure, 0.0).max(dim=0).values)
        ever_finger |= metrics["pressure_contact"] & active[:, None]
        ever_meaningful |= metrics["side_contact"] & active[:, None]
        for key in keys:
            ever[key] |= metrics[key] & active
        reward_sum += rewards.flatten() * active
        q = core.robot.data.joint_pos
        joint_limit_steps += int((((q < core.lower + 0.02) | (q > core.upper - 0.02)).any(dim=1) & active).sum())
        finite &= bool(torch.isfinite(rewards).all())
        active &= ~dones.flatten().bool()

summary = {
    "checkpoint": os.path.abspath(args.checkpoint), "seed": args.seed,
    "num_envs": 16, "steps": args.steps,
    "counts": {key: int(value.sum()) for key, value in ever.items()},
    "finger_contact_counts": ever_finger.sum(dim=0).cpu().tolist(),
    "meaningful_contact_counts": ever_meaningful.sum(dim=0).cpu().tolist(),
    "pressure_mean": (pressure_sum / max(active_env_steps, 1)).cpu().tolist(),
    "pressure_max": pressure_max.cpu().tolist(),
    "reward_mean": float(reward_sum.mean()),
    "joint_limit_env_steps": joint_limit_steps,
    "rewards_finite": finite,
    "video_dir": video_dir,
}
output_path = os.path.join(run_dir, "evaluation_grasp_stage1_rsl.json")
with open(output_path, "w") as output:
    json.dump(summary, output, indent=2)
print("EVALUATION_JSON=" + output_path, flush=True)
env.close()
simulation_app.close()
