"""Record a v10 L-raise policy with gym's RecordVideo from the fixed "front" camera.

Per-joint arm angles, the L-pose errors and the finger-to-cube y error are
printed every second and saved to l_raise.json next to the video.
"""
import argparse, json, math, os, importlib.metadata
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--camera", choices=("front", "side"), default="front")
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
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2LRaisePPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_l_raise import JaxV2LRaiseEnvCfg

checkpoint = os.path.abspath(args.checkpoint)
cfg = JaxV2LRaiseEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.episode_length_s = args.steps / 60.0 + 1.0
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
if args.camera == "front":
    cfg.viewer.eye = (-0.78, 0.08, 1.04)
    cfg.viewer.lookat = (-0.29, 0.08, 0.80)
else:
    cfg.viewer.eye = (-0.20, 1.05, 1.00)
    cfg.viewer.lookat = (-0.20, 0.15, 0.85)
cfg.viewer.resolution = (960, 540)
raw_env = gym.make("Isaac-JaxV2-LRaise-Direct-v10", cfg=cfg, render_mode="rgb_array")
video_dir = os.path.join(
    os.path.dirname(checkpoint), "videos",
    f"l_raise_{args.camera}_{os.path.splitext(os.path.basename(checkpoint))[0]}{('_' + args.tag) if args.tag else ''}",
)
os.makedirs(video_dir, exist_ok=True)
video_env = gym.wrappers.RecordVideo(
    raw_env, video_folder=video_dir, step_trigger=lambda step: step == 0,
    video_length=args.steps, disable_logger=True,
)
agent_cfg = handle_deprecated_rsl_rl_cfg(JaxV2LRaisePPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(video_env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

u = raw_env.unwrapped
names = [u.robot.joint_names[i] for i in u.arm_ids[:7]]
rows = []
obs = env.get_observations()
with torch.inference_mode():
    for step in range(args.steps):
        obs, _, _, _ = env.step(policy(obs))
        if step % 60 == 59:
            pose_err, y_err = u._l_errors()
            q = torch.rad2deg(u.robot.data.joint_pos[:, u.arm_ids[:7]])
            row = {"t": round((step + 1) / 60, 1),
                   "q_deg_env0": {n: round(float(v), 1) for n, v in zip(names, q[0])},
                   "q_deg_mean": {n: round(float(v), 1) for n, v in zip(names, q.mean(dim=0))},
                   "pose_err_deg": [round(math.degrees(float(v)), 1) for v in pose_err],
                   "y_err_mm": [round(1000 * float(v), 1) for v in y_err],
                   "hand_z": round(float(u.robot.data.body_pos_w[0, u._hand_id, 2]), 3)}
            rows.append(row)
            print("LRAISE " + json.dumps(row), flush=True)
json.dump(rows, open(os.path.join(video_dir, "l_raise.json"), "w"))
print("L_RAISE_DIR=" + video_dir, flush=True)
env.close()
simulation_app.close()
