"""Fixed-seed Stage 1 evaluation with video for a trained RSL-RL checkpoint."""

import argparse
import importlib.metadata
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=480)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import OneArmJaxMinimalPPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.env import OneArmJaxMinimalEnvCfg

checkpoint = os.path.abspath(args.checkpoint)
run_dir = os.path.dirname(checkpoint)
cfg = OneArmJaxMinimalEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.eye = (-0.75, 0.62, 1.12)
cfg.viewer.lookat = (-0.22, 0.08, 0.78)
cfg.viewer.resolution = (960, 540)
raw_env = gym.make("Isaac-OneArmJax-Minimal-Grasp-Direct-v1", cfg=cfg, render_mode="rgb_array")
video_dir = os.path.join(run_dir, "videos", f"evaluation_seed_{args.seed}")
os.makedirs(video_dir, exist_ok=True)
video_env = gym.wrappers.RecordVideo(
    raw_env, video_folder=video_dir,
    step_trigger=lambda step: step == 0,
    video_length=args.steps, disable_logger=True,
)
agent_cfg = handle_deprecated_rsl_rl_cfg(
    OneArmJaxMinimalPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib")
)
env = RslRlVecEnvWrapper(video_env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

observation = env.get_observations()
active = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
keys = ("reach", "grasp_region", "contact_3", "grasp_candidate", "hold", "success", "unsafe_contact", "failure_termination")
ever = {key: torch.zeros_like(active) for key in keys}
max_pregrasp_quality = torch.zeros(args.num_envs, device=env.device)
ever_finger = torch.zeros((args.num_envs, 3), dtype=torch.bool, device=env.device)
max_force = torch.zeros(args.num_envs, device=env.device)
initial_finger_q = raw_env.unwrapped.robot.data.joint_pos[:, raw_env.unwrapped.finger_ids].clone()
max_closure = torch.zeros(args.num_envs, device=env.device)
finite = True
for _ in range(args.steps):
    with torch.inference_mode():
        action = policy(observation)
        observation, reward, done, _ = env.step(action)
        metrics = raw_env.unwrapped.extras["metrics"]
        for key in keys:
            ever[key] |= metrics[key] & active
        max_pregrasp_quality = torch.maximum(
            max_pregrasp_quality, torch.where(active, metrics["pregrasp_quality"], 0.0)
        )
        ever_finger |= metrics["valid_contact"] & active[:, None]
        max_force = torch.maximum(max_force, torch.where(active, metrics["force_n"].amax(dim=1), 0.0))
        finger_q = raw_env.unwrapped.robot.data.joint_pos[:, raw_env.unwrapped.finger_ids]
        open_q = torch.tensor(raw_env.unwrapped.cfg.finger_open, device=env.device)
        close_q = torch.tensor(raw_env.unwrapped.cfg.finger_close, device=env.device)
        closure = ((finger_q - open_q) / (close_q - open_q)).clamp(0.0, 1.0).mean(dim=1)
        max_closure = torch.maximum(max_closure, torch.where(active, closure, 0.0))
        finite &= bool(torch.isfinite(observation["policy"]).all() and torch.isfinite(reward).all())
        active &= ~done.flatten().bool()

summary = {
    "checkpoint": checkpoint,
    "seed": args.seed,
    "num_envs": args.num_envs,
    "steps": args.steps,
    "counts": {key: int(value.sum()) for key, value in ever.items()},
    "valid_finger_contact_counts": ever_finger.sum(dim=0).cpu().tolist(),
    "max_pregrasp_quality": max_pregrasp_quality.cpu().tolist(),
    "max_contact_force_n": max_force.cpu().tolist(),
    "initial_finger_q": initial_finger_q.cpu().tolist(),
    "max_closure_fraction": max_closure.cpu().tolist(),
    "finite": finite,
    "video_dir": video_dir,
}
output_path = os.path.join(run_dir, f"evaluation_minimal_seed_{args.seed}.json")
with open(output_path, "w") as output:
    json.dump(summary, output, indent=2)
print("EVALUATION_JSON=" + output_path, flush=True)
env.close()
simulation_app.close()
