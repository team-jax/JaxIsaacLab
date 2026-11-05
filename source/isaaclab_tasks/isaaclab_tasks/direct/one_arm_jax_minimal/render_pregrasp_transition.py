"""Render a labelled scripted reference from the URDF arm-down pose.

This is a kinematic demonstration, not an RL evaluation or policy rollout.
"""

import argparse
import json
import math
import os
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True)
parser.add_argument("--elbow-only", action="store_true", help="Bend only the URDF elbow joint from arm-down")
parser.add_argument("--elbow-positive", action="store_true", help="Bend the elbow in its positive direction")
parser.add_argument("--coordinated-shoulder-elbow", action="store_true", help="Gradually raise with shoulder yaw and elbow together")
parser.add_argument("--clear-table-arc", action="store_true", help="For V2: temporarily move the shoulder away, then return above the tabletop")
parser.add_argument("--v2-two-against-one", action="store_true", help="After clearance, display an open two-finger versus one-finger pose around the cube")
parser.add_argument("--v2-urdf", action="store_true", help="Use the separate October 6 JAX V2 URDF asset")
parser.add_argument("--shoulder-pitch-goal", type=float)
parser.add_argument("--retreat-shoulder-pitch", type=float, default=-0.45)
parser.add_argument("--table-center-x", type=float)
parser.add_argument("--object-x", type=float)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if (args.clear_table_arc or args.v2_two_against_one) and not args.v2_urdf:
    parser.error("the clearance and two-against-one modes require --v2-urdf")
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.one_arm_jax_minimal.pregrasp import OneArmJaxPregraspDownEnvCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_pregrasp import JaxV2PregraspDownEnvCfg

cfg = JaxV2PregraspDownEnvCfg() if args.v2_urdf else OneArmJaxPregraspDownEnvCfg()
if args.shoulder_pitch_goal is None:
    args.shoulder_pitch_goal = -0.15 if args.v2_two_against_one else cfg.initial_arm_q[0]
if args.table_center_x is None:
    args.table_center_x = cfg.table_center[0]
if args.object_x is None:
    args.object_x = cfg.object_x[0]
if args.v2_two_against_one:
    cfg.diagnostic_table_contacts = True
    cfg.table_contact_body_names = (
        "MG996R", "RotationBody", "Finger_1_1", "Finger_2_1", "Finger_3_1",
        "Finger_1_2", "Finger_2_2", "Finger_3_2",
    )
cfg.scene.num_envs = 1
cfg.seed = 12345
short_mode = args.elbow_only or args.coordinated_shoulder_elbow or args.clear_table_arc or args.v2_two_against_one
video_frames = 1140 if args.v2_two_against_one else (600 if (args.coordinated_shoulder_elbow or args.clear_table_arc) else (300 if args.elbow_only else 960))
cfg.episode_length_s = 21.0 if args.v2_two_against_one else (12.0 if (args.coordinated_shoulder_elbow or args.clear_table_arc) else (6.0 if args.elbow_only else 18.0))
cfg.posture_hold_duration_s = 100.0  # keep the final pose on screen
cfg.table_center = (args.table_center_x, 0.0, 0.72)
cfg.object_x = (args.object_x, args.object_x)
cfg.object_y = (0.080, 0.080)
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.eye = (-0.20, 1.35, 1.12) if short_mode else (-1.15, 0.90, 1.35)
cfg.viewer.lookat = (-0.20, 0.04, 0.83) if short_mode else (-0.15, 0.04, 0.83)
if args.v2_two_against_one:
    cfg.viewer.eye = (-0.65, 0.70, 1.25)
    cfg.viewer.lookat = (-0.30, 0.08, 0.80)
cfg.viewer.resolution = (960, 540)
raw_env = gym.make("Isaac-OneArmJax-Pregrasp-Down-Direct-v1", cfg=cfg, render_mode="rgb_array")
video_dir = os.path.abspath(args.output_dir)
os.makedirs(video_dir, exist_ok=True)
env = gym.wrappers.RecordVideo(
    raw_env, video_folder=video_dir, step_trigger=lambda step: step == 0,
    video_length=video_frames, disable_logger=True,
)
env.reset()
robot_env = raw_env.unwrapped
if args.v2_two_against_one:
    # Keep the hand neutral during the clearance arc. Arrange the three
    # fingers only after the arm has reached its safe raised posture.
    neutral_q = robot_env.targets.clone()
    neutral_q[:, robot_env.finger_ids] = 0.0
    robot_env.targets.copy_(neutral_q)
    robot_env.robot.write_joint_state_to_sim(
        neutral_q, torch.zeros_like(neutral_q), env_ids=robot_env.robot._ALL_INDICES,
    )
    robot_env.robot.set_joint_position_target(neutral_q)
start_arm = robot_env.robot.data.joint_pos[0, robot_env.arm_ids].clone()
start_fingers = robot_env.robot.data.joint_pos[0, robot_env.finger_ids].clone()
goal_arm = torch.zeros_like(start_arm) if short_mode else torch.tensor(cfg.initial_arm_q, device=robot_env.device)
if args.elbow_only:
    goal_arm[3] = abs(cfg.initial_arm_q[3]) if args.elbow_positive else cfg.initial_arm_q[3]
if args.coordinated_shoulder_elbow or args.clear_table_arc or args.v2_two_against_one:
    goal_arm[0] = args.shoulder_pitch_goal
    goal_arm[2] = -0.50
    # The revised URDF reverses the physical elbow-bend direction.
    goal_arm[3] = 1.45 if args.v2_urdf else -1.45
goal_fingers = torch.tensor(cfg.finger_open, device=robot_env.device)
candidate_arm = torch.tensor(cfg.initial_arm_q, device=robot_env.device)
candidate_fingers = torch.tensor(cfg.finger_open, device=robot_env.device)
max_force_n = 0.0
max_table_contact_force_n = 0.0
table_contact_step_count = 0
table_contact_step_count_during_spread = 0
trace_joint_q = []
trace_object_xyz = []
early_resets = 0
tracked_bodies = [
    robot_env.robot.body_names.index(name)
    for name in (("MG996R", "Finger_1_1", "Finger_2_1", "Finger_3_1", "Finger_1_2", "Finger_2_2", "Finger_3_2")
                 if args.v2_urdf else ("rotation_base", "finger_1_link2", "finger_2_link2", "finger_3_link2"))
]
minimum_body_origin_clearance_m = float("inf")
minimum_behind_edge_gap_m = float("inf")
for frame in range(video_frames):
    seconds = frame / 60.0
    finger_fraction = min(seconds / 0.5, 1.0)
    # Rotate the distal joints while the hand is outside the table footprint,
    # then flex the elbow to bring the hand inward, then finish the shoulder.
    def smooth_fraction(start: float, duration: float) -> float:
        fraction = min(max((seconds - start) / duration, 0.0), 1.0)
        return 0.5 - 0.5 * math.cos(math.pi * fraction)

    arm_target = start_arm.clone()
    if args.elbow_only:
        arm_target[3] = start_arm[3] + smooth_fraction(0.5, 2.0) * (goal_arm[3] - start_arm[3])
    elif args.clear_table_arc or args.v2_two_against_one:
        # First keep the hand behind the near table edge while the elbow folds.
        # The forward shoulder return begins only after the elbow has raised
        # the hand; no policy or grasp behavior is scripted here.
        first = smooth_fraction(0.5, 5.0)
        second = smooth_fraction(5.5, 2.5)
        retreat = args.retreat_shoulder_pitch
        arm_target[0] = start_arm[0] + first * (retreat - start_arm[0]) + second * (goal_arm[0] - retreat)
        arm_target[2] = start_arm[2] + first * (goal_arm[2] - start_arm[2])
        arm_target[3] = start_arm[3] + first * (goal_arm[3] - start_arm[3])
        if args.v2_two_against_one:
            lift = smooth_fraction(7.0, 1.5)
            orient = smooth_fraction(9.5, 3.0)
            lower = smooth_fraction(12.5, 3.0)
            elevated_arm = goal_arm.clone()
            elevated_arm[3] = 1.70
            aligned_arm = candidate_arm.clone()
            aligned_arm[3] = elevated_arm[3]
            arm_target += lift * (elevated_arm - goal_arm)
            arm_target += orient * (aligned_arm - elevated_arm)
            arm_target += lower * (candidate_arm - aligned_arm)
    elif args.coordinated_shoulder_elbow:
        fraction = smooth_fraction(0.5, 6.0)
        arm_target[0] = start_arm[0] + fraction * (goal_arm[0] - start_arm[0])
        arm_target[2] = start_arm[2] + fraction * (goal_arm[2] - start_arm[2])
        arm_target[3] = start_arm[3] + fraction * (goal_arm[3] - start_arm[3])
    else:
        arm_target[4:] = start_arm[4:] + smooth_fraction(0.5, 3.0) * (goal_arm[4:] - start_arm[4:])
        arm_target[3] = start_arm[3] + smooth_fraction(3.5, 5.0) * (goal_arm[3] - start_arm[3])
        arm_target[:3] = start_arm[:3] + smooth_fraction(8.5, 3.0) * (goal_arm[:3] - start_arm[:3])
    robot_env.targets[:, robot_env.arm_ids] = arm_target
    finger_target = start_fingers + (0.0 if args.v2_two_against_one else finger_fraction) * (goal_fingers - start_fingers)
    if args.v2_two_against_one:
        finger_target += smooth_fraction(9.5, 3.0) * (candidate_fingers - start_fingers)
    robot_env.targets[:, robot_env.finger_ids] = finger_target
    _, _, terminated, truncated, _ = env.step(torch.zeros((1, 11), device=robot_env.device))
    if args.v2_two_against_one:
        trace_joint_q.append(robot_env.robot.data.joint_pos[0].cpu().numpy().copy())
        trace_object_xyz.append(robot_env.object.data.root_pos_w[0].cpu().numpy().copy())
    early_resets += int((terminated | truncated).sum().item())
    max_force_n = max(max_force_n, float(robot_env._contacts()[1].amax().item()))
    if args.v2_two_against_one:
        table_force = max(
            float(torch.linalg.vector_norm(robot_env.scene.sensors[f"table_contact_{name}"].data.force_matrix_w[0, 0, 0]).item())
            for name in cfg.table_contact_body_names
        )
        max_table_contact_force_n = max(max_table_contact_force_n, table_force)
        table_contact_step_count += int(table_force > 0.1)
        table_contact_step_count_during_spread += int(9.5 <= seconds < 12.5 and table_force > 0.1)
    tracked_xyz = robot_env.robot.data.body_pos_w[0, tracked_bodies]
    near_edge = args.table_center_x + cfg.table_cfg.size[0] / 2
    above_footprint = tracked_xyz[:, 0] <= near_edge - 0.01
    if bool(above_footprint.any()):
        clearance = (tracked_xyz[above_footprint, 2] - cfg.table_top_z).min().item()
        minimum_body_origin_clearance_m = min(minimum_body_origin_clearance_m, clearance)
    below_top = tracked_xyz[:, 2] < cfg.table_top_z + 0.03
    if bool(below_top.any()):
        gap = (tracked_xyz[below_top, 0] - near_edge).min().item()
        minimum_behind_edge_gap_m = min(minimum_behind_edge_gap_m, gap)

reported_goal_arm = candidate_arm if args.v2_two_against_one else goal_arm
summary = {
    "robot_urdf": "JAXV2_Assembly.urdf" if args.v2_urdf else "One_Arm_JAX_URDF.urdf",
    "type": "scripted V2 two-against-one open pregrasp candidate, not an RL policy" if args.v2_two_against_one else ("scripted two-stage shoulder clearance arc, not an RL policy" if args.clear_table_arc else ("scripted coordinated shoulder-elbow motion, not an RL policy" if args.coordinated_shoulder_elbow else ("scripted elbow-only motion, not an RL policy" if args.elbow_only else "scripted joint-target interpolation, not an RL policy"))),
    "duration_s": video_frames / 60.0,
    "table_center_x": args.table_center_x,
    "table_near_edge_x": args.table_center_x + cfg.table_cfg.size[0] / 2,
    "object_x": args.object_x,
    "start_arm_q": start_arm.cpu().tolist(),
    "goal_arm_q": reported_goal_arm.cpu().tolist(),
    "goal_finger_q": candidate_fingers.cpu().tolist() if args.v2_two_against_one else goal_fingers.cpu().tolist(),
    "finger_spread_start_s": 9.5 if args.v2_two_against_one else None,
    "final_arm_q": robot_env.robot.data.joint_pos[0, robot_env.arm_ids].cpu().tolist(),
    "final_arm_mean_absolute_error_rad": float(
        (robot_env.robot.data.joint_pos[0, robot_env.arm_ids] - reported_goal_arm).abs().mean().item()
    ),
    "max_finger_object_contact_force_n": max_force_n,
    "max_hand_table_contact_force_n": max_table_contact_force_n if args.v2_two_against_one else None,
    "hand_table_contact_steps": table_contact_step_count if args.v2_two_against_one else None,
    "hand_table_contact_steps_during_spread": table_contact_step_count_during_spread if args.v2_two_against_one else None,
    "min_tracked_body_origin_clearance_over_table_m": (
        None if math.isinf(minimum_body_origin_clearance_m) else minimum_body_origin_clearance_m
    ),
    "min_tracked_body_origin_gap_behind_edge_when_low_m": (
        None if math.isinf(minimum_behind_edge_gap_m) else minimum_behind_edge_gap_m
    ),
    "final_fingertip_midpoint_xyz": robot_env._grasp_midpoint()[0].cpu().tolist(),
    "final_fingertip_xyz": robot_env.robot.data.body_pos_w[0, robot_env.fingertip_ids].cpu().tolist(),
    "final_object_center_xyz": robot_env.object.data.root_pos_w[0].cpu().tolist(),
    "final_fingertip_midpoint_to_object_center_m": float(
        torch.linalg.vector_norm(robot_env._grasp_midpoint()[0] - robot_env.object.data.root_pos_w[0]).item()
    ),
    "final_fingertip_midpoint_above_cube_top_m": float(
        robot_env._grasp_midpoint()[0, 2].item()
        - robot_env.object.data.root_pos_w[0, 2].item() - cfg.object_half_height
    ),
    "clearance_metric_note": "Link origins only; collision meshes may extend beyond them.",
    "early_resets": early_resets,
}
with open(os.path.join(video_dir, "summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
if args.v2_two_against_one:
    np.savez_compressed(
        os.path.join(video_dir, "motion_trace.npz"),
        joint_names=np.asarray(robot_env.robot.joint_names),
        joint_q=np.asarray(trace_joint_q),
        object_xyz=np.asarray(trace_object_xyz),
    )
print(json.dumps(summary), flush=True)
env.close()
simulation_app.close()
