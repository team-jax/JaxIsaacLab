"""Record and score a fixed-seed V2 grasp checkpoint with a close camera."""

import argparse
import importlib.metadata
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=720)
parser.add_argument("--camera", choices=("close", "front"), default="close")
parser.add_argument("--controlled-lift-v2", action="store_true")
parser.add_argument("--arm-then-fingers-v3", action="store_true")
parser.add_argument("--learn-sequence-v4", action="store_true")
parser.add_argument("--tag", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import (
    JaxV2GraspPPORunnerCfg, JaxV2ControlledLiftPPORunnerCfg,
    JaxV2ArmThenFingersPPORunnerCfg, JaxV2LearnSequencePPORunnerCfg,
)
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_grasp import JaxV2GraspEnvCfg, JaxV2ControlledLiftEnvCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_arm_then_fingers import JaxV2ArmThenFingersEnvCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_learn_sequence import JaxV2LearnSequenceEnvCfg

checkpoint = os.path.abspath(args.checkpoint)
run_dir = os.path.dirname(checkpoint)
cfg = (
    JaxV2LearnSequenceEnvCfg() if args.learn_sequence_v4 else
    JaxV2ArmThenFingersEnvCfg() if args.arm_then_fingers_v3 else
    JaxV2ControlledLiftEnvCfg() if args.controlled_lift_v2 else JaxV2GraspEnvCfg()
)
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
if args.camera == "front":
    cfg.viewer.eye = (-0.78, 0.08, 1.04)
    cfg.viewer.lookat = (-0.29, 0.08, 0.80)
else:
    cfg.viewer.eye = (-0.53, 0.39, 1.02)
    cfg.viewer.lookat = (-0.35, 0.08, 0.79)
cfg.viewer.resolution = (960, 540)
task_name = (
    "Isaac-JaxV2-Learn-LiftThenOpen-Direct-v4" if args.learn_sequence_v4 else
    "Isaac-JaxV2-ArmThenFingers-Grasp-Direct-v3" if args.arm_then_fingers_v3 else
    "Isaac-JaxV2-ArmDown-ControlledLift-Direct-v2" if args.controlled_lift_v2
    else "Isaac-JaxV2-ArmDown-Grasp-Direct-v1"
)
raw_env = gym.make(task_name, cfg=cfg, render_mode="rgb_array")
version = "learn_sequence_v4" if args.learn_sequence_v4 else "arm_then_fingers_v3" if args.arm_then_fingers_v3 else (
    "controlled_lift_v2" if args.controlled_lift_v2 else "v2"
)
video_dir = os.path.join(
    run_dir, "videos", f"evaluation_{version}_{args.camera}_seed_{args.seed}_{os.path.splitext(os.path.basename(checkpoint))[0]}{('_' + args.tag) if args.tag else ''}"
)
os.makedirs(video_dir, exist_ok=True)
video_env = gym.wrappers.RecordVideo(
    raw_env, video_folder=video_dir,
    step_trigger=lambda step: step == 0,
    video_length=args.steps, disable_logger=True,
)
agent_cfg = handle_deprecated_rsl_rl_cfg(
    JaxV2LearnSequencePPORunnerCfg() if args.learn_sequence_v4 else
    JaxV2ArmThenFingersPPORunnerCfg() if args.arm_then_fingers_v3 else
    JaxV2ControlledLiftPPORunnerCfg() if args.controlled_lift_v2 else JaxV2GraspPPORunnerCfg(),
    importlib.metadata.version("rsl-rl-lib")
)
env = RslRlVecEnvWrapper(video_env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

observation = env.get_observations()
active = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
keys = ("reach", "contact_3", "stable", "hold", "success", "unsafe_contact", "table_proximity", "early_close")
ever = {key: torch.zeros_like(active) for key in keys}
max_pregrasp = torch.zeros(args.num_envs, device=env.device)
max_force = torch.zeros(args.num_envs, device=env.device)
ever_phase = torch.zeros((args.num_envs, 4), dtype=torch.bool, device=env.device)
for _ in range(args.steps):
    with torch.inference_mode():
        action = policy(observation)
        observation, reward, done, _ = env.step(action)
        metrics = raw_env.unwrapped.extras["metrics"]
        if "phase" in metrics:
            ever_phase |= torch.nn.functional.one_hot(metrics["phase"].long(), num_classes=4).bool() & active[:, None]
        for key in keys:
            ever[key] |= metrics[key] & active
        max_pregrasp = torch.maximum(max_pregrasp, torch.where(active, metrics["pregrasp_quality"], 0.0))
        max_force = torch.maximum(max_force, torch.where(active, metrics["force_n"].amax(dim=1), 0.0))
        active &= ~done.flatten().bool()

summary = {
    "checkpoint": checkpoint,
    "seed": args.seed,
    "num_envs": args.num_envs,
    "steps": args.steps,
    "counts": {key: int(value.sum()) for key, value in ever.items()},
    "max_pregrasp_quality": max_pregrasp.cpu().tolist(),
    "max_contact_force_n": max_force.cpu().tolist(),
    "phase_reached_counts": ever_phase.sum(dim=0).cpu().tolist(),
    "video_dir": video_dir,
}
output_path = os.path.join(video_dir, "summary.json")
with open(output_path, "w") as output:
    json.dump(summary, output, indent=2)
print("EVALUATION_JSON=" + output_path, flush=True)
env.close()
simulation_app.close()
