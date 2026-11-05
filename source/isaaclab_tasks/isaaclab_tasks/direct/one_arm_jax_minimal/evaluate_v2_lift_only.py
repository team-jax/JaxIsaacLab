"""Record and score a fixed-seed arm-lift-only checkpoint."""

import argparse
import importlib.metadata
import json
import os
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", choices=("lift", "pregrasp", "grasp", "grasp_v8", "grasp_v9"), default="lift")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--camera", choices=("front", "close", "wide_front"), default="front")
parser.add_argument("--tag", default="")
parser.add_argument("--rotation-scan", action="store_true", help="Sweep RotationBody after the policy raises and opens the hand")
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
    JaxV2LiftOnlyPPORunnerCfg, JaxV2PregraspPosePPORunnerCfg, JaxV2GraspAfterLiftPPORunnerCfg,
    JaxV2SmoothGraspPPORunnerCfg, JaxV2PadSensorGraspPPORunnerCfg,
)
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_lift_only import JaxV2LiftOnlyEnvCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_pregrasp_pose import JaxV2PregraspPoseEnvCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_grasp_after_lift import JaxV2GraspAfterLiftEnvCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_smooth_grasp import JaxV2SmoothGraspEnvCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_pad_sensor_grasp import JaxV2PadSensorGraspEnvCfg

checkpoint = os.path.abspath(args.checkpoint)
cfg = (JaxV2PadSensorGraspEnvCfg() if args.task == "grasp_v9" else
       JaxV2SmoothGraspEnvCfg() if args.task == "grasp_v8" else
       JaxV2GraspAfterLiftEnvCfg() if args.task == "grasp" else
       JaxV2PregraspPoseEnvCfg() if args.task == "pregrasp" else JaxV2LiftOnlyEnvCfg())
if args.rotation_scan:
    cfg.pregrasp_hold_duration_s = 100.0
if hasattr(cfg, "sensor_randomize"):
    cfg.sensor_randomize = False  # v9: evaluate with the nominal RA12P sensor model
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
if args.camera == "front":
    cfg.viewer.eye = (-0.78, 0.08, 1.04)
    cfg.viewer.lookat = (-0.29, 0.08, 0.80)
elif args.camera == "wide_front":
    cfg.viewer.eye = (-1.85, 0.08, 1.34)
    cfg.viewer.lookat = (-0.25, 0.08, 0.76)
else:
    cfg.viewer.eye = (-0.53, 0.39, 1.02)
    cfg.viewer.lookat = (-0.35, 0.08, 0.79)
cfg.viewer.resolution = (960, 540)
task_id = (
    "Isaac-JaxV2-PadSensorGrasp-Direct-v9" if args.task == "grasp_v9" else
    "Isaac-JaxV2-SmoothLiftGrasp-Direct-v8" if args.task == "grasp_v8" else
    "Isaac-JaxV2-LiftThenGrasp-Direct-v7" if args.task == "grasp" else
    "Isaac-JaxV2-LiftThenPregrasp-Direct-v6" if args.task == "pregrasp" else
    "Isaac-JaxV2-ArmLift-LockedFingers-Direct-v5"
)
raw_env = gym.make(task_id, cfg=cfg, render_mode="rgb_array")
video_dir = os.path.join(
    os.path.dirname(checkpoint), "videos",
    f"evaluation_{args.task}_{args.camera}_seed_{args.seed}_{os.path.splitext(os.path.basename(checkpoint))[0]}"
    f"{('_' + args.tag) if args.tag else ''}",
)
os.makedirs(video_dir, exist_ok=True)
video_env = gym.wrappers.RecordVideo(
    raw_env, video_folder=video_dir, step_trigger=lambda step: step == 0,
    video_length=args.steps, disable_logger=True,
)
agent_cfg_class = (JaxV2PadSensorGraspPPORunnerCfg if args.task == "grasp_v9" else
                   JaxV2SmoothGraspPPORunnerCfg if args.task == "grasp_v8" else
                   JaxV2GraspAfterLiftPPORunnerCfg if args.task == "grasp" else
                   JaxV2PregraspPosePPORunnerCfg if args.task == "pregrasp" else JaxV2LiftOnlyPPORunnerCfg)
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg_class(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(video_env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

observation = env.get_observations()
active = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
keys = ("clearance_ready", "raised_ready", "lift_hold", "success", "table_proximity")
if args.task in ("pregrasp", "grasp", "grasp_v8", "grasp_v9"):
    keys += ("lift_transition", "pregrasp_ready", "finger_open_stage")
if args.task in ("grasp", "grasp_v8", "grasp_v9"):
    keys += ("contact_3", "stable", "hold", "unsafe_contact", "early_close")
ever = {key: torch.zeros_like(active) for key in keys}
max_finger = torch.zeros(args.num_envs, device=env.device)
min_tip_clearance = torch.full((args.num_envs,), float("inf"), device=env.device)
max_tip_clearance = torch.full((args.num_envs,), -float("inf"), device=env.device)
min_tip_center_x = torch.full((args.num_envs,), float("inf"), device=env.device)
max_tip_center_x = torch.full((args.num_envs,), -float("inf"), device=env.device)
min_tip_cube_distance = torch.full((args.num_envs,), float("inf"), device=env.device)
min_success_tip_cube_distance = torch.full((args.num_envs,), float("inf"), device=env.device)
max_body_intrusion = torch.zeros(args.num_envs, device=env.device)
max_cube_shift = torch.zeros(args.num_envs, device=env.device)
max_side_touching = torch.zeros(args.num_envs, device=env.device)
side_touching_steps = torch.zeros(args.num_envs, device=env.device)
min_straddle_error = torch.full((args.num_envs,), float("inf"), device=env.device)
opposed_press_steps = torch.zeros(args.num_envs, device=env.device)
enclosed_steps = torch.zeros(args.num_envs, device=env.device)
precise_grasp_steps = torch.zeros(args.num_envs, device=env.device)
max_precise_fingers = torch.zeros(args.num_envs, device=env.device)
max_cube_rotation = torch.zeros(args.num_envs, device=env.device)
first_done_step = torch.full((args.num_envs,), -1, dtype=torch.long, device=env.device)
same_face_steps = torch.zeros(args.num_envs, device=env.device)
max_cube_rise = torch.full((args.num_envs,), -float("inf"), device=env.device)
three_face_steps = torch.zeros(args.num_envs, device=env.device)
first_body_intrusion_step = torch.full((args.num_envs,), -1, dtype=torch.long, device=env.device)
joint_trace = []
rotation_samples = {}
for step in range(args.steps):
    with torch.inference_mode():
        action = policy(observation)
        if args.rotation_scan and step >= 180:
            action = torch.zeros_like(action)
            phase_ready = raw_env.unwrapped.reward_phase == 2
            q_rotation = raw_env.unwrapped.robot.data.joint_pos[:, raw_env.unwrapped.arm_ids[7]]
            action[:, 7] = torch.where(phase_ready & (q_rotation > -3.13), -1.0, 0.0)
        observation, _, done, _ = env.step(action)
        first_done_step = torch.where((first_done_step < 0) & done.flatten().bool(), step, first_done_step)
        joint_trace.append(raw_env.unwrapped.robot.data.joint_pos[0].cpu().numpy().copy())
        metrics = raw_env.unwrapped.extras["metrics"]
        for key in keys:
            ever[key] |= metrics[key] & active
        max_finger = torch.maximum(max_finger, torch.where(active, metrics["finger_max_abs_rad"], 0.0))
        if "cube_shift_m" in metrics:
            max_cube_shift = torch.maximum(max_cube_shift, torch.where(active, metrics["cube_shift_m"], 0.0))
            side = torch.where(active, metrics["fingers_side_touching"], 0.0)
            max_side_touching = torch.maximum(max_side_touching, side)
            side_touching_steps += (side > 0).float()
        if "straddle_error_m" in metrics:
            min_straddle_error = torch.minimum(
                min_straddle_error, torch.where(active, metrics["straddle_error_m"], float("inf"))
            )
            opposed_press_steps += (metrics["opposed_press"] & active).float()
        if "enclosed" in metrics:
            enclosed_steps += (metrics["enclosed"] & active).float()
        if "three_face" in metrics:
            three_face_steps += (metrics["three_face"] & active).float()
        if "cube_rise_m" in metrics:
            max_cube_rise = torch.maximum(max_cube_rise, torch.where(active, metrics["cube_rise_m"], -float("inf")))
        if "fingers_same_face" in metrics:
            same_face_steps += (metrics["fingers_same_face"] & active).float()
        if "precise_grasp" in metrics:
            precise_grasp_steps += (metrics["precise_grasp"] & active).float()
            max_precise_fingers = torch.maximum(
                max_precise_fingers, torch.where(active, metrics["precise_fingers"], 0.0)
            )
            max_cube_rotation = torch.maximum(
                max_cube_rotation, torch.where(active, metrics["cube_rotation_deg"], 0.0)
            )
        if "body_intrusion_m" in metrics:
            intrusion = torch.where(active, metrics["body_intrusion_m"], 0.0)
            max_body_intrusion = torch.maximum(max_body_intrusion, intrusion)
            first_body_intrusion_step = torch.where(
                (first_body_intrusion_step < 0) & (intrusion > 0.0), step, first_body_intrusion_step
            )
        min_tip_clearance = torch.minimum(
            min_tip_clearance, torch.where(active, metrics["tip_clearance_m"], float("inf"))
        )
        max_tip_clearance = torch.maximum(
            max_tip_clearance, torch.where(active, metrics["tip_clearance_m"], -float("inf"))
        )
        tip_center_x = (
            raw_env.unwrapped.robot.data.body_pos_w[:, raw_env.unwrapped.fingertip_ids, 0].mean(dim=1)
            - raw_env.unwrapped.scene.env_origins[:, 0]
        )
        min_tip_center_x = torch.minimum(min_tip_center_x, torch.where(active, tip_center_x, float("inf")))
        max_tip_center_x = torch.maximum(max_tip_center_x, torch.where(active, tip_center_x, -float("inf")))
        tip_center = raw_env.unwrapped.robot.data.body_pos_w[:, raw_env.unwrapped.fingertip_ids].mean(dim=1)
        tip_cube_distance = torch.linalg.vector_norm(
            tip_center - raw_env.unwrapped.object.data.root_pos_w, dim=1
        )
        min_tip_cube_distance = torch.minimum(
            min_tip_cube_distance, torch.where(active, tip_cube_distance, float("inf"))
        )
        min_success_tip_cube_distance = torch.minimum(
            min_success_tip_cube_distance,
            torch.where(active & metrics["success"], tip_cube_distance, float("inf")),
        )
        active &= ~done.flatten().bool()
        if args.rotation_scan:
            q_rotation = float(raw_env.unwrapped.robot.data.joint_pos[0, raw_env.unwrapped.arm_ids[7]])
            for label, target in (("minus_90", -np.pi / 2), ("minus_135", -3 * np.pi / 4), ("minus_180", -np.pi)):
                if label not in rotation_samples and abs(q_rotation - target) < 0.05:
                    tips = raw_env.unwrapped.robot.data.body_pos_w[0, raw_env.unwrapped.fingertip_ids]
                    cube = raw_env.unwrapped.object.data.root_pos_w[0]
                    rotation_samples[label] = {
                        "step": step, "rotation_rad": q_rotation,
                        "tip_positions_world": tips.cpu().tolist(),
                        "tip_minus_cube_world": (tips - cube).cpu().tolist(),
                        "min_tip_table_clearance_m": float(tips[:, 2].min() - cfg.table_top_z),
                    }

summary = {
    "checkpoint": checkpoint,
    "seed": args.seed,
    "num_envs": args.num_envs,
    "steps": args.steps,
    "counts": {key: int(value.sum()) for key, value in ever.items()},
    "max_finger_deviation_rad": max_finger.cpu().tolist(),
    "min_tip_clearance_m": min_tip_clearance.cpu().tolist(),
    "max_tip_clearance_m": max_tip_clearance.cpu().tolist(),
    "min_tip_center_x_m": min_tip_center_x.cpu().tolist(),
    "max_tip_center_x_m": max_tip_center_x.cpu().tolist(),
    "rotation_samples": rotation_samples,
    "min_tip_cube_distance_m": min_tip_cube_distance.cpu().tolist(),
    "min_success_tip_cube_distance_m": min_success_tip_cube_distance.cpu().tolist(),
    "table_center_x": cfg.table_center[0],
    "max_body_intrusion_m": max_body_intrusion.cpu().tolist(),
    "first_body_intrusion_step": first_body_intrusion_step.cpu().tolist(),
    "max_cube_shift_m": max_cube_shift.cpu().tolist(),
    "max_fingers_side_touching": max_side_touching.cpu().tolist(),
    "side_touching_steps": side_touching_steps.cpu().tolist(),
    "min_straddle_error_m": min_straddle_error.cpu().tolist(),
    "opposed_press_steps": opposed_press_steps.cpu().tolist(),
    "enclosed_steps": enclosed_steps.cpu().tolist(),
    "precise_grasp_steps": precise_grasp_steps.cpu().tolist(),
    "max_precise_fingers": max_precise_fingers.cpu().tolist(),
    "max_cube_rotation_deg": max_cube_rotation.cpu().tolist(),
    "first_done_step": first_done_step.cpu().tolist(),
    "fingers_same_face_steps": same_face_steps.cpu().tolist(),
    "max_cube_rise_m": max_cube_rise.cpu().tolist(),
    "three_face_steps": three_face_steps.cpu().tolist(),
    "video_dir": video_dir,
}
output_path = os.path.join(video_dir, "summary.json")
with open(output_path, "w") as output:
    json.dump(summary, output, indent=2)
np.savez(
    os.path.join(video_dir, "motion_trace.npz"),
    joint_q=np.asarray(joint_trace),
    joint_names=np.asarray(raw_env.unwrapped.robot.joint_names),
)
print("EVALUATION_JSON=" + output_path, flush=True)
env.close()
simulation_app.close()
