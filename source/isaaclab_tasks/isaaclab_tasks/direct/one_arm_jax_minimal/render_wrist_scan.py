"""Run a v8 policy to its grasp pose, then sweep wrist_yaw_to_hand_base in close-up.

Shows how the hand-base yaw lines fingers 1 and 2 up (equal height at about
+63 deg in the contact_center pose). Recorded with gym's RecordVideo, as in
evaluate_v2_lift_only.py and Isaac Lab's play.py --video, from the fixed
"front" evaluation camera. The cube is moved to the far side of the table so
the sweep is unobstructed. Per-frame pose labels, the measured joint angle and
the finger 1 minus finger 2 height are written to labels.json.
"""
import argparse, json, math, os, importlib.metadata
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
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2SmoothGraspPPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_smooth_grasp import JaxV2SmoothGraspEnvCfg

JOINT = "wrist_yaw_to_hand_base"
# Target angles (deg); "policy" holds the angle the policy left.
POSES = ["policy", 36.0, 63.0, 75.0, "policy"]
SETTLE_TOL_DEG = 1.0
MAX_SETTLE_STEPS = 240
HOLD_STEPS = 120  # 2 s at 60 Hz
video_steps = args.policy_steps + len(POSES) * (MAX_SETTLE_STEPS + HOLD_STEPS)

checkpoint = os.path.abspath(args.checkpoint)
cfg = JaxV2SmoothGraspEnvCfg()
cfg.scene.num_envs = 1
cfg.episode_length_s = 100.0
# The sweep moves the cube aside and turns the hand; keep the episode alive.
cfg.cube_move_limit_m = 10.0
cfg.cube_rotation_limit_deg = 1000.0
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.eye = (-0.78, 0.08, 1.04)
cfg.viewer.lookat = (-0.29, 0.08, 0.80)
cfg.viewer.resolution = (960, 540)
raw_env = gym.make("Isaac-JaxV2-SmoothLiftGrasp-Direct-v8", cfg=cfg, render_mode="rgb_array")
video_dir = os.path.join(
    os.path.dirname(checkpoint), "videos",
    f"wrist_scan_front_{os.path.splitext(os.path.basename(checkpoint))[0]}{('_' + args.tag) if args.tag else ''}",
)
os.makedirs(video_dir, exist_ok=True)
video_env = gym.wrappers.RecordVideo(
    raw_env, video_folder=video_dir, step_trigger=lambda step: step == 0,
    video_length=video_steps, disable_logger=True,
)
agent_cfg = handle_deprecated_rsl_rl_cfg(JaxV2SmoothGraspPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(video_env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

u = raw_env.unwrapped
names = u.robot.joint_names
bodies = u.robot.body_names
jid = names.index(JOINT)
f1, f2 = bodies.index("Finger_1_2"), bodies.index("Finger_2_2")
override = {}
orig_apply = u._apply_action


def apply_with_override():
    for idx, rad in override.items():
        u.targets[:, idx] = rad
    orig_apply()


u._apply_action = apply_with_override
labels = []
obs = env.get_observations()
with torch.inference_mode():
    for _ in range(args.policy_steps):
        obs, _, _, _ = env.step(policy(obs))
        labels.append(("policy", None, None))
    for i in range(len(names)):
        override[i] = float(u.robot.data.joint_pos[0, i])
    policy_deg = math.degrees(override[jid])
    far_pose = u.object.data.root_state_w[:, :7].clone()
    far_pose[:, 0] -= 0.4
    zero_velocity = torch.zeros((1, 6), device=u.device)
    zero_action = torch.zeros_like(u.actions)

    def step_scan(label, phase):
        u.object.write_root_pose_to_sim(far_pose)
        u.object.write_root_velocity_to_sim(zero_velocity)
        env.step(zero_action)
        p = u.robot.data.body_pos_w[0]
        q = math.degrees(float(u.robot.data.joint_pos[0, jid]))
        labels.append((f"{label} {phase}", round(q, 1), round(1000 * float(p[f1, 2] - p[f2, 2]), 1)))
        return q

    for pose in POSES:
        target = policy_deg if pose == "policy" else pose
        label = f"{JOINT}={target:+.1f}"
        override[jid] = math.radians(target)
        for _ in range(MAX_SETTLE_STEPS):
            if abs(step_scan(label, "moving") - target) < SETTLE_TOL_DEG:
                break
        for _ in range(HOLD_STEPS):
            step_scan(label, "hold")
json.dump(labels, open(os.path.join(video_dir, "labels.json"), "w"))
print("WRIST_SCAN_DIR=" + video_dir, flush=True)
env.close()
simulation_app.close()
