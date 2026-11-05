"""One-environment, checkpoint-free physical grasp and lift sanity test."""

import argparse
import csv
import json
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="")
parser.add_argument("--adaptive-fingers", action="store_true")
parser.add_argument("--object-x", type=float, default=-0.465)
parser.add_argument("--object-y", type=float, default=0.0)
parser.add_argument("--align-to-spawn", action="store_true", help="Use Jacobian IK to align palm above the fixed cube before closing")
parser.add_argument("--direct-grasp-pose", action="store_true", help="Start arm at the fixed grasp pose to isolate approach path effects")
parser.add_argument("--optimized-lift-path", action="store_true", help="Use offline URDF IK grasp/lift joint poses at x=-0.255,y=0.085")
parser.add_argument("--gentle-fingers", action="store_true", help="Test lower effort and slower finger closing")
parser.add_argument("--wide-open", action="store_true", help="Spread the proximal finger joints before descent")
parser.add_argument("--soft-fingers", action="store_true", help="Test low-effort finger actuation with slow closure")
parser.add_argument("--maintain-pressure", action="store_true", help="Continue per-finger pressure control during lift and hold")
parser.add_argument("--low-palm", action="store_true", help="Use lower palm URDF IK pose to contact cube sides")
parser.add_argument("--very-low-palm", action="store_true", help="Use palm 2 cm below low-palm pose for side contact")
parser.add_argument("--third-finger-rotation", type=float, default=None, help="Hold rotation_base joint at this angle through grasp and lift")
parser.add_argument("--fitted-side-grasp", action="store_true", help="Use URDF collision-centroid fitted three-side grasp")
parser.add_argument("--fitted-clear-grasp", action="store_true", help="Use side-contact fit with every distal collision mesh above the tabletop")
parser.add_argument("--surface-pregrasp", action="store_true", help="Use collision-free 2 mm side gap, then pressure-close the fingers")
parser.add_argument("--strong-arm", action="store_true", help="Test higher arm tracking gains when measured pose error exceeds 0.1 rad")
parser.add_argument("--surface-gap10", action="store_true", help="Use 4–6 mm measured side gap and a vertical 6 cm wrist path")
parser.add_argument("--surface-gap20", action="store_true", help="Use the offline 20 mm side-clearance pre-grasp")
parser.add_argument("--high-side-grasp", action="store_true", help="Keep all fingertip collision meshes above the tabletop")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args)
simulation_app = app.app

import cv2
import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.one_arm_jax.env import OneArmJaxPickPlaceEnvCfg

cfg = OneArmJaxPickPlaceEnvCfg()
cfg.scene.num_envs = 1
cfg.object_x = (args.object_x, args.object_x)
cfg.object_y = (args.object_y, args.object_y)
cfg.hold_duration_s = 1.0
if args.wide_open:
    cfg.finger_open = (1.40, 1.40, 1.40, 0.00, 0.00, 0.00)
if args.fitted_side_grasp:
    cfg.finger_open = (1.40, 1.40, 1.40, 0.00, 0.00, 0.00)
    cfg.finger_close = (1.02977735, 1.22651601, 0.60131287,
                        0.54349672, 0.02443116, 2.094394)
    cfg.episode_length_s = 14.0
if args.fitted_clear_grasp:
    cfg.finger_open = (1.40, 1.40, 1.40, 0.00, 0.00, 0.00)
    cfg.finger_close = (0.80394616, 0.95260864, 0.54294656,
                        0.58942975, 0.3361551, 1.37847145)
    cfg.episode_length_s = 14.0
if args.surface_pregrasp:
    cfg.finger_open = (1.40259494, 1.53732118, 1.45911901,
                       0.30201789, 0.04150824, 0.14493258)
    cfg.finger_close = (0.90, 1.00, 1.00, 0.65, 0.50, 0.75)
    cfg.episode_length_s = 14.0
if args.surface_gap10:
    cfg.finger_open = (1.40, 1.40, 1.40, 0.00, 0.00, 0.00)
    cfg.finger_close = (0.80, 0.90, 0.80, 0.55, 0.55, 2.09)
    cfg.episode_length_s = 14.0
if args.surface_gap20:
    cfg.finger_open = (1.40, 1.40, 1.40, 0.00, 0.00, 0.00)
    cfg.finger_close = (0.75, 0.85, 0.75, 2.09, 2.09, 2.09)
    cfg.episode_length_s = 14.0
if args.high_side_grasp:
    cfg.finger_open = (1.40, 1.40, 1.40, 0.00, 0.00, 0.00)
    cfg.finger_close = (0.85, 0.90, 0.85, 0.75, 0.75, 0.75)
    cfg.episode_length_s = 14.0
if args.gentle_fingers:
    cfg.episode_length_s = 14.0
    cfg.robot_cfg.actuators["fingers"].effort_limit_sim = 20.0
    cfg.robot_cfg.actuators["fingers"].stiffness = 80.0
    cfg.robot_cfg.actuators["fingers"].damping = 12.0
if args.soft_fingers:
    cfg.episode_length_s = 20.0
    cfg.robot_cfg.actuators["fingers"].effort_limit_sim = 5.0
    cfg.robot_cfg.actuators["fingers"].stiffness = 30.0
    cfg.robot_cfg.actuators["fingers"].damping = 5.0
if args.strong_arm:
    cfg.robot_cfg.actuators["arm"].effort_limit_sim = 200.0
    cfg.robot_cfg.actuators["arm"].stiffness = 500.0
    cfg.robot_cfg.actuators["arm"].damping = 30.0
cfg.viewer.origin_type = "env"
cfg.viewer.eye = (-1.2, 1.1, 1.4)
cfg.viewer.lookat = (-0.3, 0.0, 0.8)
cfg.viewer.resolution = (960, 540)
output_dir = args.output_dir or os.path.abspath(
    os.path.join("artifacts", "scripted_lift_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
)
os.makedirs(output_dir, exist_ok=True)
cfg.log_dir = output_dir
env = gym.make("Isaac-OneArmJax-LiftHold-Direct-v1", cfg=cfg, render_mode="rgb_array")
env.reset(seed=123)
core = env.unwrapped
video_path = os.path.join(output_dir, "scripted_lift.mp4")
writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, cfg.viewer.resolution)
if not writer.isOpened():
    raise RuntimeError("Cannot open video writer")

history = []
phase = "approach"
terminated = False
success_termination = False
arm_ids = core.arm_ids
finger_ids = core.finger_ids
ee_jacobian_index = core.ee_id - 1  # fixed base is absent from PhysX Jacobian body axis
zero_action = torch.zeros((1, 9), device=core.device)
curl = torch.zeros(3, device=core.device)
active_finger_open = torch.tensor(cfg.finger_open, device=core.device)


def record(step):
    force = core._pressure_forces()[0].detach().cpu().tolist()
    force_vectors = [core.scene.sensors[f"finger_{i}"].data.force_matrix_w[0, 0, 0].detach().cpu().tolist()
                     for i in (1, 2, 3)]
    row = {
        "step": step,
        "phase": phase,
        "p1_force_n": force[0], "p2_force_n": force[1], "p3_force_n": force[2],
        "pressure_force_vectors_world_n": force_vectors,
        "object_z_m": float(core.object.data.root_pos_w[0, 2]),
        "object_xyz_m": core.object.data.root_pos_w[0].detach().cpu().tolist(),
        "object_quat_wxyz": core.object.data.root_quat_w[0].detach().cpu().tolist(),
        "object_vz_mps": float(core.object.data.root_lin_vel_w[0, 2]),
        "ee_xyz_m": core.robot.data.body_pos_w[0, core.ee_id].detach().cpu().tolist(),
        "finger_link_xyz_m": core.robot.data.body_pos_w[0, core.fingertip_ids].detach().cpu().tolist(),
        "arm_joint_rad": core.robot.data.joint_pos[0, arm_ids].detach().cpu().tolist(),
        "finger_joint_rad": core.robot.data.joint_pos[0, finger_ids].detach().cpu().tolist(),
        "arm_target_rad": core.targets[0, arm_ids].detach().cpu().tolist(),
        "finger_target_rad": core.targets[0, finger_ids].detach().cpu().tolist(),
    }
    history.append(row)
    if step % 3 == 0:
        frame = env.render()
        if frame is not None:
            bgr = cv2.cvtColor(np.asarray(frame)[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.rectangle(bgr, (0, 0), (960, 55), (20, 20, 20), -1)
            cv2.putText(bgr, f"Scripted {phase} | step {step} | force N: {force[0]:.2f} {force[1]:.2f} {force[2]:.2f}",
                        (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
            writer.write(bgr)


def advance(step):
    global terminated, success_termination
    if args.maintain_pressure and (phase.startswith("lift_") or phase == "hold"):
        force = core._pressure_forces()[0]
        curl.add_(torch.where(force < 3.0, 0.005, torch.where(force > 10.0, -0.005, 0.0)))
        curl.clamp_(0.0, 1.0)
        open_finger = active_finger_open
        close_finger = torch.tensor(cfg.finger_close, device=core.device)
        core.targets[0, finger_ids] = open_finger + torch.cat((curl, curl)) * (close_finger - open_finger)
    _, _, done, timeout, _ = env.step(zero_action)
    success_termination |= bool(core.extras["metrics"]["success"][0])
    record(step)
    terminated = bool(done[0] or timeout[0])
    return not terminated


def move_ee_to(goal, start, count):
    goal_t = torch.tensor(goal, device=core.device)
    for step in range(start, start + count):
        ee = core.robot.data.body_pos_w[0, core.ee_id]
        error = goal_t - ee
        jac = core.robot.root_physx_view.get_jacobians()[0, ee_jacobian_index, :3, arm_ids]
        dq = jac.T @ torch.linalg.solve(jac @ jac.T + 0.02 * torch.eye(3, device=core.device), error)
        dq = dq.clamp(-0.03, 0.03)
        q = core.robot.data.joint_pos[0, arm_ids]
        core.targets[0, arm_ids] = (q + dq).clamp(core.lower[arm_ids], core.upper[arm_ids])
        if not advance(step):
            return step + 1
    return start + count


try:
    object_position = core.object.data.root_pos_w[0].detach().clone()
    object_orientation = core.object.data.root_quat_w[0].detach().cpu().tolist()
    # Kinematic fits for a clear approach and for open second phalanges around the 48 mm cube.
    approach_arm_q = torch.tensor(
        [0.7853979, 0.04558103, -1.0471979, -1.54256151,
         0.85651726, 1.5707959, -0.52995014, 0.2793526], device=core.device
    )
    grasp_arm_q = torch.tensor(
        [0.66526774, -0.0436329, -1.0471979, -1.38850439,
         0.57322713, 1.54101851, -0.65936561, 0.95230107], device=core.device
    )
    optimized_lift_q = None
    if args.optimized_lift_path:
        grasp_arm_q = torch.tensor(
            [0.61936741, -0.03479314, -0.98698333, -1.40757144,
             0.55121264, 1.50558886, -0.69994159, 0.93852028], device=core.device
        )
        optimized_lift_q = torch.tensor(
            [0.785397, -0.00269022, -1.04719699, -1.51999425,
             0.72478146, 1.570795, -0.93259199, 1.31012272], device=core.device
        )
        approach_arm_q = optimized_lift_q.clone()
    if args.low_palm:
        grasp_arm_q = torch.tensor(
            [0.54115203, -0.043632, -0.96493703, -1.31108573,
             0.47625531, 1.42013747, -0.5832422, 0.74927559], device=core.device
        )
        optimized_lift_q = torch.tensor(
            [0.64354473, 0.00399351, -1.04711031, -1.53788557,
             0.5951309, 1.57074222, -0.05408376, 0.39920297], device=core.device
        )
        approach_arm_q = optimized_lift_q.clone()
    if args.very_low_palm:
        grasp_arm_q = torch.tensor(
            [0.51693858, -0.043632, -0.97900159, -1.23282265,
             0.46861325, 1.32154267, -0.55498539, 0.69469648], device=core.device
        )
        optimized_lift_q = torch.tensor(
            [0.57597683, -0.02022471, -1.04549025, -1.49861287,
             0.51903925, 1.5401905, 0.32026581, 0.00000646], device=core.device
        )
        approach_arm_q = optimized_lift_q.clone()
    if args.fitted_side_grasp:
        grasp_arm_q = torch.tensor(
            [0.4824306, -0.043632, -1.047197, -1.41911414,
             0.38628859, 1.47640017, -0.68023494, 0.52533853], device=core.device
        )
        optimized_lift_q = torch.tensor(
            [0.65169584, -0.04362807, -1.047197, -1.54682283,
             0.57349721, 1.570795, -1.45358559, 1.38962149], device=core.device
        )
        approach_arm_q = optimized_lift_q.clone()
    if args.fitted_clear_grasp:
        grasp_arm_q = torch.tensor(
            [0.50252579, -0.043632, -1.047197, -1.46692357,
             0.44223853, 1.56958484, -0.68718286, 0.42926991], device=core.device
        )
        optimized_lift_q = torch.tensor(
            [0.67978879, -0.02929161, -1.047197, -1.570795,
             0.67892916, 1.570795, -1.54803084, 1.38423575], device=core.device
        )
        approach_arm_q = optimized_lift_q.clone()
    if args.surface_pregrasp:
        grasp_arm_q = torch.tensor(
            [0.43650173, -0.04346699, -1.04713654, -1.2963253,
             0.45977271, 1.38189144, -0.62958216, 0.06336259], device=core.device
        )
        optimized_lift_q = torch.tensor(
            [0.57412897, -0.04363196, -1.047197, -1.47093074,
             0.58252663, 1.570795, -1.21407161, 0.75380861], device=core.device
        )
        approach_arm_q = optimized_lift_q.clone()
    if args.surface_gap10:
        grasp_arm_q = torch.tensor(
            [0.35695633, -0.03118658, -0.89226596, -1.39423764,
             0.46906946, 1.42672485, -0.72736827, 0.01658382], device=core.device
        )
        optimized_lift_q = torch.tensor(
            [0.62159237, -0.04362334, -1.0448579, -1.4903238,
             0.68274015, 1.56894486, -1.75885423, 1.29911998], device=core.device
        )
        approach_arm_q = torch.tensor(
            [0.72592483, -0.02220609, -1.047197, -1.570795,
             0.85823423, 1.570795, -1.87081975, 1.49179096], device=core.device
        )
    if args.surface_gap20:
        grasp_arm_q = torch.tensor(
            [0.36538657, -0.03849076, -0.80879382, -1.33852161,
             0.56886728, 1.44148383, -0.81038893, 0.00272290], device=core.device
        )
        optimized_lift_q = torch.tensor(
            [0.62159237,-0.04362334,-1.0448579,-1.4903238,
             0.68274015,1.56894486,-1.75885423,1.29911998], device=core.device
        )
        approach_arm_q = torch.tensor(
            [0.72592483,-0.02220609,-1.047197,-1.570795,
             0.85823423,1.570795,-1.87081975,1.49179096], device=core.device
        )
    if args.high_side_grasp:
        grasp_arm_q = torch.tensor(
            [0.37462126,-0.04014305,-0.92790456,-1.34332478,
             0.45036953,1.39664092,-0.68028580,0.00413832], device=core.device
        )
        optimized_lift_q = torch.tensor(
            [0.62159237,-0.04362334,-1.0448579,-1.4903238,
             0.68274015,1.56894486,-1.75885423,1.29911998], device=core.device
        )
        approach_arm_q = torch.tensor(
            [0.72592483,-0.02220609,-1.047197,-1.570795,
             0.85823423,1.570795,-1.87081975,1.49179096], device=core.device
        )
    if args.third_finger_rotation is not None:
        grasp_arm_q[-1] = args.third_finger_rotation
        approach_arm_q[-1] = args.third_finger_rotation
        if optimized_lift_q is not None:
            optimized_lift_q[-1] = args.third_finger_rotation
    q = core.robot.data.joint_pos.clone()
    q[0, arm_ids] = grasp_arm_q if args.direct_grasp_pose else approach_arm_q
    core.targets[:] = q
    core.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    core.robot.set_joint_position_target(q)
    phase = "approach_above"
    step = 0
    for j in range(12):
        if not advance(j):
            break
    step = j + 1
    if (args.surface_gap10 or args.surface_gap20 or args.high_side_grasp) and not terminated:
        phase = "shape_pre_grasp"
        pregrasp_finger = torch.tensor(
            [1.45263867,1.56306778,1.51794567,
             0.18054861,0.03873469,0.17312320] if args.high_side_grasp else
            [1.11228349, 1.27657480, 1.33673870,
             1.92136924, 1.75725938, 1.87046388] if args.surface_gap20 else
            [1.24910769, 1.37171533, 1.26928166,
             0.23512283, 0.23644301, 1.89732593], device=core.device
        )
        for j in range(100):
            alpha = (j + 1) / 100
            core.targets[0, finger_ids] = active_finger_open * (1.0 - alpha) + pregrasp_finger * alpha
            if not advance(step + j):
                break
        step += j + 1
        active_finger_open = pregrasp_finger
    if args.direct_grasp_pose:
        phase = "lower_to_grasp"
        for j in range(8):
            if not advance(step + j):
                break
        step += j + 1
    elif args.align_to_spawn:
        # Measured palm-to-object offset from the fixed near-spawn trace; hold
        # that offset while checking whether the far cube can be approached.
        desired_hand = [float(object_position[0] + 0.006172), float(object_position[1] + 0.012393),
                        float(object_position[2] + 0.101352)]
        if not terminated:
            phase = "align_above"
            step = move_ee_to([desired_hand[0], desired_hand[1], desired_hand[2] + 0.045], step, 100)
        if not terminated:
            phase = "lower_to_grasp"
            step = move_ee_to(desired_hand, step, 70)
    elif not terminated:
        phase = "lower_to_grasp"
        for j in range(150):
            alpha = (j + 1) / 150.0
            core.targets[0, arm_ids] = approach_arm_q * (1.0 - alpha) + grasp_arm_q * alpha
            if not advance(step + j):
                break
        step += j + 1
        if not terminated:
            phase = "settle_pre_grasp"
            for j in range(60):
                if not advance(step + j):
                    break
            step += j + 1
    goal = core.robot.data.body_pos_w[0, core.ee_id].detach().cpu().tolist()
    hand_quat = core.robot.data.body_quat_w[0, core.ee_id].detach().cpu().tolist()
    hand_position = core.robot.data.body_pos_w[0, core.ee_id].detach().cpu().tolist()
    arm_at_soft_limits = ((core.robot.data.joint_pos[0, arm_ids] - core.lower[arm_ids]).abs() < 0.02).logical_or(
        (core.robot.data.joint_pos[0, arm_ids] - core.upper[arm_ids]).abs() < 0.02
    ).detach().cpu().tolist()
    approach_error = torch.linalg.vector_norm(
        core.robot.data.body_pos_w[0, core.fingertip_ids].mean(dim=0) - object_position
    ).item()
    print("FINGERTIP_CENTER_ERROR_M", approach_error, flush=True)
    if not terminated:
        phase = "close"
        open_finger = active_finger_open
        close_finger = torch.tensor(cfg.finger_close, device=core.device)
        curl.zero_()
        close_steps = 400 if args.soft_fingers else 200 if args.gentle_fingers or args.fitted_side_grasp or args.fitted_clear_grasp else 100
        for j in range(close_steps):
            if args.adaptive_fingers:
                pressure_force = core._pressure_forces()[0]
                curl += torch.where(pressure_force < 3.0, 0.0025 if args.soft_fingers else 0.005 if args.gentle_fingers else 0.015,
                                    torch.where(pressure_force > 10.0, -0.01, 0.0))
                curl.clamp_(0.0, 1.0)
                curl_targets = torch.cat((curl, curl))
            else:
                curl_targets = torch.full((6,), (j + 1) / close_steps, device=core.device)
            core.targets[0, finger_ids] = open_finger * (1.0 - curl_targets) + close_finger * curl_targets
            if not advance(step + j):
                break
        step += j + 1
    for label, dz in (("lift_1cm", 0.01), ("lift_2cm", 0.02), ("lift_4cm", 0.04)):
        if terminated:
            break
        phase = label
        if optimized_lift_q is not None:
            desired_q = grasp_arm_q + (optimized_lift_q - grasp_arm_q) * (dz / 0.05)
            for j in range(30):
                alpha = (j + 1) / 30
                core.targets[0, arm_ids] = core.targets[0, arm_ids] * (1.0 - alpha) + desired_q * alpha
                if not advance(step + j):
                    break
            step += j + 1
        else:
            lift_goal = [goal[0], goal[1], goal[2] + dz]
            step = move_ee_to(lift_goal, step, 30)
    if not terminated:
        phase = "hold"
        for j in range(60):
            if not advance(step + j):
                break
    forces = np.array([[r[f"p{i}_force_n"] for i in (1, 2, 3)] for r in history])
    heights = np.array([r["object_z_m"] for r in history])
    vz = np.array([r["object_vz_mps"] for r in history])
    hold_mask = (forces >= 2.0).all(axis=1) & (heights - object_position[2].item() >= 0.04) & (vz > -0.05)
    hold_run = 0
    max_hold_run = 0
    for holds in hold_mask:
        hold_run = hold_run + 1 if holds else 0
        max_hold_run = max(max_hold_run, hold_run)
    phase_results = {}
    for name in ("approach_above", "shape_pre_grasp", "align_above", "lower_to_grasp", "settle_pre_grasp", "close", "lift_1cm", "lift_2cm", "lift_4cm", "hold"):
        rows = [r for r in history if r["phase"] == name]
        if rows:
            phase_results[name] = {
                "max_force_n": np.array([[r[f"p{i}_force_n"] for i in (1, 2, 3)] for r in rows]).max(axis=0).tolist(),
                "max_object_lift_m": max(r["object_z_m"] for r in rows) - object_position[2].item(),
                "end_object_lift_m": rows[-1]["object_z_m"] - object_position[2].item(),
                "end_hand_lift_m": rows[-1]["ee_xyz_m"][2] - hand_position[2],
                "end_object_xyz_m": rows[-1]["object_xyz_m"],
                "end_finger_link_xyz_m": rows[-1]["finger_link_xyz_m"],
            }
    first_contact_row = next((r for r in history if max(r[f"p{i}_force_n"] for i in (1, 2, 3)) >= 2.0), None)
    three_contact_row = next((r for r in history if min(r[f"p{i}_force_n"] for i in (1, 2, 3)) >= 2.0), None)
    close_rows = [r for r in history if r["phase"] == "close"]
    lift_rows = [r for r in history if r["phase"].startswith("lift_")]
    finger_close_delta = (float(np.linalg.norm(np.array(close_rows[-1]["finger_joint_rad"])
                                               - np.array(close_rows[0]["finger_joint_rad"])))
                          if len(close_rows) > 1 else 0.0)
    arm_upward_motion = (max(r["ee_xyz_m"][2] for r in lift_rows) - hand_position[2]
                         if lift_rows else 0.0)
    physics_ok = bool(np.isfinite(heights).all() and np.isfinite(forces).all()
                      and np.max(forces) < 200.0 and np.min(heights) > cfg.table_top_z - 0.03)
    summary = {
        "video": video_path,
        "object_initial_xyz_m": object_position.detach().cpu().tolist(),
        "object_initial_quat_wxyz": object_orientation,
        "pre_grasp_hand_xyz_m": hand_position,
        "pre_grasp_hand_quat_wxyz": hand_quat,
        "pre_grasp_object_minus_hand_xyz_m": (object_position.detach().cpu() - torch.tensor(hand_position)).tolist(),
        "pre_grasp_desired_hand_xyz_m": desired_hand if args.align_to_spawn else None,
        "pre_grasp_hand_goal_error_m": float(np.linalg.norm(np.array(hand_position) - np.array(desired_hand))) if args.align_to_spawn else None,
        "pre_grasp_arm_at_soft_limits": arm_at_soft_limits,
        "phase_results": phase_results,
        "first_contact_step": first_contact_row["step"] if first_contact_row else None,
        "stable_three_contact_step": three_contact_row["step"] if three_contact_row else None,
        "steps": len(history),
        "fingertip_center_error_m": approach_error,
        "max_force_n": forces.max(axis=0).tolist(),
        "finger_close_delta_rad": finger_close_delta,
        "arm_upward_motion_m": arm_upward_motion,
        "physics_ok": physics_ok,
        "force_p95_n": np.percentile(forces, 95, axis=0).tolist(),
        "max_lift_m": float(heights.max() - object_position[2].item()),
        "final_lift_m": float(heights[-1] - object_position[2].item()),
        "three_finger_contact_steps": int((forces >= 2.0).all(axis=1).sum()),
        "four_cm_hold_steps": max_hold_run,
        "scripted_success": bool(max_hold_run >= core.hold_required_steps),
        "terminated": terminated,
        "rl_success": bool(core._success[0]),
    }
    with open(os.path.join(output_dir, "trace.csv"), "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=history[0].keys())
        writer_csv.writeheader()
        writer_csv.writerows(history)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("SCRIPTED_TEST=" + json.dumps(summary), flush=True)
finally:
    writer.release()
    env.close()
    simulation_app.close()
