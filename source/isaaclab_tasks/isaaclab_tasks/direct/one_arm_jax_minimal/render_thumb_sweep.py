"""v12 pre-grasp (frozen policy) with the cube in place, then the thumb's first joint swept.

The policy runs the 4 cm pre-grasp; from --sweep_start_s the arm targets are
frozen (the policy sees the changed thumb as an unseen input and flew the arm
away in the first version) and the thumb first joint
(rotation_base_to_finger_3_1) is set to each --angles_deg value for --hold_s.
Recorded with gym's RecordVideo from the close front camera; per-angle cube
contact and thumb pad direction go to thumb_sweep.json.
"""
import argparse, importlib.metadata, json, math, os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--angles_deg", default="-30,0,20,40,60")
parser.add_argument("--sweep_start_s", type=float, default=4.5)
parser.add_argument("--hold_s", type=float, default=1.5)
parser.add_argument("--camera", choices=("close_front", "close"), default="close_front")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_apply
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2FrontPregraspPPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_pregrasp import JaxV2FrontPregraspEnvCfg

angles = [float(v) for v in args.angles_deg.split(",")]
cfg = JaxV2FrontPregraspEnvCfg()
cfg.scene.num_envs = 1
cfg.episode_length_s = 100.0
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.resolution = (960, 540)
cfg.viewer.eye, cfg.viewer.lookat = {
    "close_front": ((-0.72, 0.24, 0.84), (-0.40, 0.26, 0.78)),
    "close": ((-0.52, 0.52, 0.86), (-0.40, 0.26, 0.78)),
}[args.camera]
raw = gym.make("Isaac-JaxV2-FrontPregrasp-Direct-v12", cfg=cfg, render_mode="rgb_array")
u = raw.unwrapped
dt = u.step_dt
start, hold = round(args.sweep_start_s / dt), round(args.hold_s / dt)
total = start + hold * len(angles)
ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
video_dir = os.path.join(ckpt_dir, "videos", f"thumb_sweep_{args.camera}_{os.path.splitext(os.path.basename(args.checkpoint))[0]}")
os.makedirs(video_dir, exist_ok=True)
video = gym.wrappers.RecordVideo(raw, video_folder=video_dir, step_trigger=lambda s: s == 0, video_length=total, disable_logger=True)
agent_cfg = handle_deprecated_rsl_rl_cfg(JaxV2FrontPregraspPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(video, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
zd = torch.zeros(1, dtype=torch.bool, device=u.device)
u._get_dones = lambda: (zd, zd)
X = torch.tensor([[1.0, 0.0, 0.0]], device=u.device)
names = [f"proximal_{i}" for i in (1, 2, 3)] + [f"finger_{i}" for i in (1, 2, 3)]
rows = []
obs = env.get_observations()
with torch.inference_mode():
    for k in range(total):
        if k == start:
            arm = list(u.arm_ids[:7])
            frozen = u.targets[:, arm].clone()
            orig_pre = u._pre_physics_step

            def pre(actions):
                orig_pre(actions)
                u.targets[:, arm] = frozen

            u._pre_physics_step = pre
        if k >= start:
            i = (k - start) // hold
            u._hand_shape[2] = math.radians(angles[i])
        obs, *_ = env.step(policy(obs))
        if k >= start and (k - start) % hold == hold - 1:
            contact = [round(float(torch.linalg.vector_norm(u.scene.sensors[n].data.force_matrix_w[0, 0, 0])), 2) for n in names]
            pad = quat_apply(u.robot.data.body_quat_w[0:1, u.fingertip_ids[2]], X)[0]
            rows.append({"thumb_first_deg": angles[i], "t_s": round((k + 1) * dt, 2), "cube_contact_N": contact,
                         "thumb_distal_pad_dir": [round(float(v), 2) for v in pad]})
            print("SWEEP " + json.dumps(rows[-1]), flush=True)
json.dump(rows, open(os.path.join(video_dir, "thumb_sweep.json"), "w"), indent=1)
print("THUMB_SWEEP_DIR=" + video_dir, flush=True)
video.close()
simulation_app.close()
