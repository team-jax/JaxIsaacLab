"""Scripted L-pose raise, then hand over to a v9 grasp policy.

The arm joints are interpolated from the hanging start to the L pose (shoulder
yaw first, then the elbow with the wrist pitched up so the fingertips clear the
table edge, then the wrist back to 0), and the
v9 policy takes over in its grasp phase (reward_phase 2). Recorded with gym's
RecordVideo from the fixed "front" evaluation camera; per-second measurements
go to l_then_grasp.json.
"""
import argparse, json, math, os, importlib.metadata
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--yaw_deg", type=float, default=-30.0)
parser.add_argument("--elbow_deg", type=float, default=90.0)
parser.add_argument("--yaw_s", type=float, default=0.7)
parser.add_argument("--elbow_s", type=float, default=1.3)
parser.add_argument("--hold_s", type=float, default=0.8)
parser.add_argument("--wrist_lift_deg", type=float, default=60.0,
                    help="wrist pitch (+ lifts the fingers) while the elbow rises, returned to 0 during the hold")
parser.add_argument("--policy_steps", type=int, default=840)
parser.add_argument("--num_envs", type=int, default=16)
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
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2PadSensorGraspPPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_pad_sensor_grasp import JaxV2PadSensorGraspEnvCfg

checkpoint = os.path.abspath(args.checkpoint)
cfg = JaxV2PadSensorGraspEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sensor_randomize = False
cfg.episode_length_s = 30.0
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.eye = (-0.78, 0.08, 1.04)
cfg.viewer.lookat = (-0.29, 0.08, 0.80)
cfg.viewer.resolution = (960, 540)
raw_env = gym.make("Isaac-JaxV2-PadSensorGrasp-Direct-v9", cfg=cfg, render_mode="rgb_array")
dt = raw_env.unwrapped.step_dt
yaw_steps, elbow_steps, hold_steps = (max(1, round(s / dt)) for s in (args.yaw_s, args.elbow_s, args.hold_s))
script_steps = yaw_steps + elbow_steps + hold_steps
video_dir = os.path.join(
    os.path.dirname(checkpoint), "videos",
    f"l_then_grasp_front_{os.path.splitext(os.path.basename(checkpoint))[0]}{('_' + args.tag) if args.tag else ''}",
)
os.makedirs(video_dir, exist_ok=True)
video_env = gym.wrappers.RecordVideo(
    raw_env, video_folder=video_dir, step_trigger=lambda step: step == 0,
    video_length=script_steps + args.policy_steps, disable_logger=True,
)
agent_cfg = handle_deprecated_rsl_rl_cfg(JaxV2PadSensorGraspPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(video_env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

u = raw_env.unwrapped
yaw_id, elbow_id, wrist_id = u.arm_ids[2], u.arm_ids[3], u.arm_ids[5]
frozen = None
orig_apply = u._apply_action


def apply_frozen():
    if frozen is not None:
        u.targets[:] = frozen
    orig_apply()


u._apply_action = apply_frozen
rows = []


def record(step, phase):
    cube = u.object.data.root_pos_w - u.scene.env_origins
    tips = u.robot.data.body_pos_w[:, u.fingertip_ids].mean(dim=1) - u.scene.env_origins
    rows.append({"step": step, "phase": phase,
                 "q_deg_env0": {u.robot.joint_names[i]: round(math.degrees(float(u.robot.data.joint_pos[0, i])), 1)
                                for i in u.arm_ids[:7]},
                 "tips_minus_cube_mm_env0": [round(1000 * float(v), 1) for v in (tips[0] - cube[0])],
                 "y_err_mm": [round(1000 * float(v), 1) for v in (tips[:, 1] - cube[:, 1])],
                 "cube_rise_mm": [round(1000 * float(v), 1) for v in u._cube_rise],
                 "pad_held": [bool(v) for v in u._last_held] if hasattr(u, "_last_held") else None})
    print("LGRASP " + json.dumps({k: v for k, v in rows[-1].items() if k != "pad_held"}), flush=True)


obs = env.get_observations()
zero = torch.zeros_like(u.actions)
with torch.inference_mode():
    frozen = u.targets.clone()
    frozen[:, u.finger_ids] = 0.0
    for k in range(script_steps):
        if k < yaw_steps:
            frozen[:, yaw_id] = math.radians(args.yaw_deg) * (k + 1) / yaw_steps
        elif k < yaw_steps + elbow_steps:
            f = (k + 1 - yaw_steps) / elbow_steps
            frozen[:, elbow_id] = math.radians(args.elbow_deg) * f
            frozen[:, wrist_id] = math.radians(args.wrist_lift_deg) * min(1.0, 2.0 * f)
        else:
            f = (k + 1 - yaw_steps - elbow_steps) / hold_steps
            frozen[:, wrist_id] = math.radians(args.wrist_lift_deg) * (1.0 - f)
        obs, _, _, _ = env.step(zero)
    record(script_steps, "l_pose")
    # Hand over: the policy continues from the L pose in its grasp phase.
    u.targets[:] = frozen
    u._filtered_actions[:] = 0.0
    u.reward_phase[:] = 2
    frozen = None
    for k in range(args.policy_steps):
        obs, _, _, _ = env.step(policy(obs))
        if k % 60 == 59:
            record(script_steps + k + 1, "policy")
json.dump(rows, open(os.path.join(video_dir, "l_then_grasp.json"), "w"))
print("L_THEN_GRASP_DIR=" + video_dir, flush=True)
env.close()
simulation_app.close()
