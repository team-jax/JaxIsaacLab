"""Scripted arm motion in the v12 env: L pose -> a given arm pose, hand shape held.

Arm joints are interpolated from the v12 start (L pose) to --q_deg over
--move_s and held for --hold_s, with the cube in place. Recorded with gym's
RecordVideo from the wide "front" or "side" camera; pad forces, cube motion and
the final finger/thumb positions relative to the cube go to pose_path.json.
"""
import argparse, json, math, os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--q_deg", required=True, help="7 arm joint angles (deg), comma separated")
parser.add_argument("--out_dir", required=True)
parser.add_argument("--camera", choices=("front", "side", "close"), default="front")
parser.add_argument("--move_s", type=float, default=2.0)
parser.add_argument("--hold_s", type=float, default=3.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_pregrasp import JaxV2FrontPregraspEnvCfg

cfg = JaxV2FrontPregraspEnvCfg()
cfg.scene.num_envs = 1
cfg.episode_length_s = 100.0
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.eye, cfg.viewer.lookat = {
    "front": ((-1.35, 0.10, 1.35), (-0.30, 0.12, 0.85)),
    "side": ((-0.25, 1.75, 1.30), (-0.25, 0.12, 0.85)),
    "close": ((-0.62, 0.22, 0.95), (-0.40, 0.08, 0.78)),
}[args.camera]
cfg.viewer.resolution = (960, 540)
raw = gym.make("Isaac-JaxV2-FrontPregrasp-Direct-v12", cfg=cfg, render_mode="rgb_array")
u = raw.unwrapped
dt = u.step_dt
move, hold = round(args.move_s / dt), round(args.hold_s / dt)
os.makedirs(args.out_dir, exist_ok=True)
env = gym.wrappers.RecordVideo(raw, video_folder=args.out_dir, step_trigger=lambda s: s == 0,
                               video_length=move + hold, disable_logger=True)
env.reset()
R = u.robot
dev = u.device
arm = list(u.arm_ids[:7])
zd = torch.zeros(1, dtype=torch.bool, device=dev)
u._get_dones = lambda: (zd, zd)
start = R.data.joint_pos[0, arm].clone()
goal = torch.tensor([math.radians(float(v)) for v in args.q_deg.split(",")], device=dev)
cmd = start.clone()
orig = u._apply_action


def ap():
    u._hold_hand_shape()
    u.targets[0, arm] = cmd
    orig()


u._apply_action = ap
zero = torch.zeros_like(u.actions)
rows = []
with torch.inference_mode():
    for k in range(move + hold):
        cmd[:] = start + (goal - start) * min(1.0, (k + 1) / move)
        env.step(zero)
        if k % 30 == 29:
            cube = u.object.data.root_pos_w[0]
            tips = R.data.body_pos_w[0, list(u.fingertip_ids)] - cube
            rows.append({"t": round((k + 1) * dt, 1), "pad_N": [round(float(v), 2) for v in u.pad_forces()[0]],
                         "cube_shift_mm": round(1000 * float(torch.linalg.vector_norm(u._cube_shift[0:1])), 1),
                         "finger12_thumb_minus_cube_mm": [[round(1000 * float(v), 1) for v in tips[i]] for i in range(3)]})
json.dump(rows, open(os.path.join(args.out_dir, "pose_path.json"), "w"))
print("FINAL " + json.dumps(rows[-1]), flush=True)
print("POSE_PATH_DIR=" + args.out_dir, flush=True)
env.close()
simulation_app.close()
