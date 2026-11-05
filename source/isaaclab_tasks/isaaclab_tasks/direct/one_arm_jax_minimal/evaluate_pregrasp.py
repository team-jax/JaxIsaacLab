"""Fixed-seed video and metrics for the independent pre-grasp posture task."""

import argparse
import importlib.metadata
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--wide", action="store_true", help="Show the elbow and whole arm in the video")
parser.add_argument("--arm-down-reset", action="store_true", help="Start from the URDF zero arm pose")
parser.add_argument("--from-down-task", action="store_true", help="Evaluate the arm-down training task")
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
from isaaclab_tasks.direct.one_arm_jax_minimal.pregrasp import OneArmJaxPregraspEnvCfg, OneArmJaxPregraspDownEnvCfg

checkpoint = os.path.abspath(args.checkpoint)
run_dir = os.path.dirname(checkpoint)
cfg = OneArmJaxPregraspDownEnvCfg() if args.from_down_task else OneArmJaxPregraspEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.eye = (-1.15, 0.90, 1.35) if args.wide else (-0.75, 0.62, 1.12)
cfg.viewer.lookat = (-0.15, 0.04, 0.83) if args.wide else (-0.22, 0.08, 0.78)
cfg.viewer.resolution = (960, 540)
task_name = "Isaac-OneArmJax-Pregrasp-Down-Direct-v1" if args.from_down_task else "Isaac-OneArmJax-Pregrasp-Direct-v1"
raw_env = gym.make(task_name, cfg=cfg, render_mode="rgb_array")
video_dir = os.path.join(
    run_dir, "videos", f"evaluation_seed_{args.seed}"
    + ("_wide" if args.wide else "") + ("_arm_down" if args.arm_down_reset or args.from_down_task else ""),
)
os.makedirs(video_dir, exist_ok=True)
video_env = gym.wrappers.RecordVideo(
    raw_env, video_folder=video_dir, step_trigger=lambda step: step == 0,
    video_length=args.steps, disable_logger=True,
)
agent_cfg = handle_deprecated_rsl_rl_cfg(
    OneArmJaxMinimalPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib")
)
env = RslRlVecEnvWrapper(video_env, clip_actions=agent_cfg.clip_actions)
if args.arm_down_reset:
    unwrapped = raw_env.unwrapped
    q = unwrapped.robot.data.joint_pos.clone()
    q[:, unwrapped.arm_ids] = 0.0
    q[:, unwrapped.finger_ids] = torch.tensor(unwrapped.cfg.finger_open, device=env.device)
    q = torch.minimum(torch.maximum(q, unwrapped.lower), unwrapped.upper)
    unwrapped.targets[:] = q
    unwrapped.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    unwrapped.robot.set_joint_position_target(q)
    unwrapped.posture_steps[:] = 0
    unwrapped.success[:] = False
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

observation = env.get_observations()
active = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
keys = ("elbow_ready", "open_ready", "aligned", "clear", "no_contact", "posture_ready", "success")
ever = {key: torch.zeros_like(active) for key in keys}
max_force = torch.zeros(args.num_envs, device=env.device)
max_alignment = torch.zeros(args.num_envs, device=env.device)
min_elbow_error = torch.full((args.num_envs,), float("inf"), device=env.device)
max_open_quality = torch.zeros(args.num_envs, device=env.device)
finite = True
for _ in range(args.steps):
    with torch.inference_mode():
        action = policy(observation)
        observation, reward, done, _ = env.step(action)
        metrics = raw_env.unwrapped.extras["metrics"]
        for key in keys:
            ever[key] |= metrics[key] & active
        max_force = torch.maximum(max_force, torch.where(active, metrics["force_n"].amax(dim=1), 0.0))
        max_alignment = torch.maximum(max_alignment, torch.where(active, metrics["alignment_quality"], 0.0))
        min_elbow_error = torch.minimum(min_elbow_error, torch.where(active, metrics["elbow_error_rad"], float("inf")))
        max_open_quality = torch.maximum(max_open_quality, torch.where(active, metrics["open_quality"], 0.0))
        finite &= bool(torch.isfinite(observation["policy"]).all() and torch.isfinite(reward).all())
        active &= ~done.flatten().bool()

summary = {
    "checkpoint": checkpoint, "seed": args.seed, "num_envs": args.num_envs,
    "steps": args.steps,
    "counts": {key: int(value.sum()) for key, value in ever.items()},
    "max_contact_force_n": max_force.cpu().tolist(),
    "max_alignment_quality": max_alignment.cpu().tolist(),
    "min_elbow_error_rad": min_elbow_error.cpu().tolist(),
    "max_open_quality": max_open_quality.cpu().tolist(),
    "finite": finite, "video_dir": video_dir,
}
output_path = os.path.join(
    run_dir, f"evaluation_pregrasp_seed_{args.seed}"
    + ("_wide" if args.wide else "") + ("_arm_down" if args.arm_down_reset or args.from_down_task else "") + ".json",
)
with open(output_path, "w") as output:
    json.dump(summary, output, indent=2)
print("EVALUATION_JSON=" + output_path, flush=True)
env.close()
simulation_app.close()
