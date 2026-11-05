"""Run a v7 policy to its opposed pose, then sweep the two thumb joints in close-up.

The video is recorded with gym's RecordVideo, as in evaluate_v2_lift_only.py and
Isaac Lab's play.py --video, from one fixed close-up camera. An unrecorded first
episode finds where the (deterministic) policy leaves the hand to aim it. Per-frame segment names and measured thumb angles
are written to labels.json next to the video.
"""
import argparse, json, os, importlib.metadata
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--policy_steps", type=int, default=480)
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
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2GraspAfterLiftPPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_grasp_after_lift import JaxV2GraspAfterLiftEnvCfg


# Thumb poses (label, j31 deg, j32 deg). Each target is commanded directly; the
# actuator limits the speed. Once both joints are within SETTLE_TOL_DEG of the
# target the pose is held, fully still, for HOLD_STEPS.
POSES = [
    ("j31=0, j32=0", 0, 0),
    ("j31=+90, j32=0", 90, 0),
    ("j31=0, j32=0", 0, 0),
    ("j31=-90, j32=0", -90, 0),
    ("j31=0, j32=0", 0, 0),
    ("j31=0, j32=+120", 0, 120),
    ("j31=+90, j32=+120", 90, 120),
    ("j31=-90, j32=+120", -90, 120),
]
SETTLE_TOL_DEG = 1.0
MAX_SETTLE_STEPS = 240
HOLD_STEPS = 120  # 2 s at 60 Hz
CAMERA_OFFSET = (-0.10, -0.30, 0.12)  # from the hand, on the side opposite the default view
video_steps = args.policy_steps + len(POSES) * (MAX_SETTLE_STEPS + HOLD_STEPS)

checkpoint = os.path.abspath(args.checkpoint)
cfg = JaxV2GraspAfterLiftEnvCfg()
cfg.scene.num_envs = 1
cfg.episode_length_s = 100.0
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.eye = (-0.53, 0.39, 1.02)
cfg.viewer.lookat = (-0.35, 0.08, 0.79)
cfg.viewer.resolution = (960, 720)
raw_env = gym.make("Isaac-JaxV2-LiftThenGrasp-Direct-v7", cfg=cfg, render_mode="rgb_array")
video_dir = os.path.join(
    os.path.dirname(checkpoint), "videos",
    f"thumb_scan_close_{os.path.splitext(os.path.basename(checkpoint))[0]}{('_' + args.tag) if args.tag else ''}",
)
os.makedirs(video_dir, exist_ok=True)
video_env = gym.wrappers.RecordVideo(
    raw_env, video_folder=video_dir, step_trigger=lambda step: step == args.policy_steps,
    video_length=video_steps, disable_logger=True,
)
agent_cfg = handle_deprecated_rsl_rl_cfg(JaxV2GraspAfterLiftPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(video_env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

u = raw_env.unwrapped
names = u.robot.joint_names
J31, J32 = names.index("rotation_base_to_finger_3_1"), names.index("finger_3_1_to_finger_3_2")
override = {}
orig_apply = u._apply_action


def apply_with_override():
    for idx, rad in override.items():
        u.targets[:, idx] = rad
    orig_apply()


u._apply_action = apply_with_override
hand_id = u.robot.body_names.index("MG996R")
labels = []
obs = env.get_observations()
with torch.inference_mode():
    # Unrecorded episode: locate the hand, then aim the fixed camera at it.
    for _ in range(args.policy_steps):
        obs, _, _, _ = env.step(policy(obs))
    hand_p = u.robot.data.body_pos_w[0, hand_id].cpu().numpy()
    tips = u.robot.data.body_pos_w[0, u.fingertip_ids].cpu().numpy()
    center = (hand_p + tips.mean(axis=0)) / 2
    obs, _ = env.reset()
    u.sim.set_camera_view(eye=(center + np.asarray(CAMERA_OFFSET)).tolist(), target=center.tolist())
    # Recorded episode: the policy lift, then the thumb sweep.
    for _ in range(args.policy_steps):
        obs, _, _, _ = env.step(policy(obs))
        labels.append(("policy", "policy", None, None))
    # Freeze every joint target where the policy left it and move the cube to
    # the far side of the table (still on it, so no "fallen" reset).
    for i in range(len(names)):
        override[i] = float(u.targets[0, i])
    far_pose = u.object.data.root_state_w[:, :7].clone()
    far_pose[:, 0] -= 0.55
    zero_velocity = torch.zeros((1, 6), device=u.device)
    zero_action = torch.zeros_like(u.actions)
    def step_scan(label, phase):
        u.object.write_root_pose_to_sim(far_pose)
        u.object.write_root_velocity_to_sim(zero_velocity)
        env.step(zero_action)
        q = u.robot.data.joint_pos[0]
        q31, q32 = float(np.degrees(float(q[J31]))), float(np.degrees(float(q[J32])))
        labels.append((label, phase, q31, q32))
        return q31, q32

    for label, a, b in POSES:
        override[J31], override[J32] = np.radians(a), np.radians(b)
        for _ in range(MAX_SETTLE_STEPS):
            q31, q32 = step_scan(label, "moving")
            if abs(q31 - a) < SETTLE_TOL_DEG and abs(q32 - b) < SETTLE_TOL_DEG:
                break
        for _ in range(HOLD_STEPS):
            step_scan(label, "hold")
json.dump(labels, open(os.path.join(video_dir, "labels.json"), "w"))
print("THUMB_SCAN_DIR=" + video_dir, flush=True)
env.close()
simulation_app.close()
