"""Evaluate and record the v20 learned raise (hanging -> L), raise stage only.

Every env starts hanging (no reference-state starts); the policy alone drives
the arm (joint velocity commands); the only script is the thumb ramp. Reports
L-pose success (every joint within raise_settle_err_rad of the v16 start pose
and slower than raise_settle_speed for 0.5 s), table contact per link, keep-out
depth, cube motion, final joint error and speed; the video goes to the
checkpoint's run folder under videos/.
"""

import argparse
import importlib.metadata
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--camera", choices=("front", "side"), default="side")
parser.add_argument("--tag", default="")
parser.add_argument("--v19", action="store_true", help="v19 learned raise into the pre-grasp pose instead of the v20 L")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2RaiseLearnedPPORunnerCfg, JaxV2RaiseToLPPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_arm_raise import JaxV2RaiseLearnedEnvCfg, JaxV2RaiseToLEnvCfg

LINKS = ("Under_Arm", "Wrist_Pitch", "Wrist_yaw", "MG996R", "RotationBody",
         "Finger_1_1", "Finger_2_1", "Finger_3_1", "Finger_1_2", "Finger_2_2", "Finger_3_2")
cfg = JaxV2RaiseLearnedEnvCfg() if args.v19 else JaxV2RaiseToLEnvCfg()
TASK = "Isaac-JaxV2-RaiseLearned-Direct-v19" if args.v19 else "Isaac-JaxV2-RaiseToL-Direct-v20"
RUNNER = JaxV2RaiseLearnedPPORunnerCfg if args.v19 else JaxV2RaiseToLPPORunnerCfg
cfg.scene.num_envs = args.num_envs
cfg.raise_rsi_prob = 0.0
cfg.sensor_randomize = False
cfg.diagnostic_table_contacts = True
cfg.table_contact_body_names = LINKS
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
if args.camera == "front":
    cfg.viewer.eye = (-1.35, 0.10, 1.35)
    cfg.viewer.lookat = (-0.30, 0.12, 0.85)
else:
    cfg.viewer.eye = (-0.25, 1.75, 1.30)
    cfg.viewer.lookat = (-0.25, 0.12, 0.85)
cfg.viewer.resolution = (960, 540)
steps = round(cfg.episode_length_s * 60) - 2  # one episode, no reset
checkpoint = os.path.abspath(args.checkpoint)
raw_env = gym.make(TASK, cfg=cfg, render_mode="rgb_array")
video_dir = os.path.join(os.path.dirname(checkpoint), "videos",
                         f"raise_l_{args.camera}_{os.path.splitext(os.path.basename(checkpoint))[0]}{('_' + args.tag) if args.tag else ''}")
os.makedirs(video_dir, exist_ok=True)
video_env = gym.wrappers.RecordVideo(raw_env, video_folder=video_dir, step_trigger=lambda s: s == 0,
                                     video_length=steps, disable_logger=True)
agent_cfg = handle_deprecated_rsl_rl_cfg(RUNNER(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(video_env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = raw_env.unwrapped
arm = list(u.arm_ids)
goal = u._raise_goal()
obs = env.get_observations()
cube0 = u.object.data.root_pos_w.clone()
n = u.num_envs
calm = torch.zeros(n, dtype=torch.long, device=u.device)
success = torch.zeros(n, dtype=torch.bool, device=u.device)
success_t = torch.full((n,), float("nan"), device=u.device)
keep_max = torch.zeros(n, device=u.device)
link_f = torch.zeros(n, len(LINKS), device=u.device)
cube_max = torch.zeros(n, device=u.device)
vmax = torch.zeros(n, device=u.device)
rows = []
with torch.inference_mode():
    for k in range(steps):
        obs, _, _, _ = env.step(policy(obs))
        t = (k + 1) / 60.0
        err = (u.robot.data.joint_pos[:, arm] - goal).abs().amax(dim=1)
        spd = u.robot.data.joint_vel[:, arm].abs().amax(dim=1)
        ok = (err < cfg.raise_settle_err_rad) & (spd < cfg.raise_settle_speed)
        calm = torch.where(ok, calm + 1, 0)
        newly = (calm >= 30) & ~success
        success_t = torch.where(newly, torch.full_like(success_t, t), success_t)
        success |= calm >= 30
        keep_max = torch.maximum(keep_max, u.raise_keepout())
        f = torch.stack([torch.linalg.vector_norm(u.scene.sensors[f"table_contact_{l}"].data.net_forces_w[:, 0], dim=-1) for l in LINKS], 1)
        link_f = torch.maximum(link_f, f)
        cube_max = torch.maximum(cube_max, (u.object.data.root_pos_w - cube0).norm(dim=1))
        vmax = torch.maximum(vmax, spd)
        if k % 30 == 29:
            rows.append({"t": round(t, 2), "q_deg_env0": [round(float(v), 1) for v in torch.rad2deg(u.robot.data.joint_pos[0, arm])],
                         "err_max_deg_env0": round(float(torch.rad2deg(err[0])), 2), "speed_env0_deg_s": round(float(torch.rad2deg(spd[0])), 2)})
            print("RAISE_L " + json.dumps(rows[-1]), flush=True)
    err_end = torch.rad2deg((u.robot.data.joint_pos[:, arm] - goal).abs())
    spd_end = torch.rad2deg(u.robot.data.joint_vel[:, arm].abs())
res = {
    "checkpoint": checkpoint, "envs": n, "episode_s": round(steps / 60, 2),
    "L_success_rate": round(float(success.float().mean()), 3),
    "L_success_time_s_median": round(float(success_t[success].median()), 2) if success.any() else None,
    "final_err_deg_max_all": round(float(err_end.max()), 2), "final_err_deg_mean": round(float(err_end.mean()), 3),
    "final_err_deg_env0": [round(float(v), 2) for v in err_end[0]],
    "final_err_deg_mean_per_joint": [round(float(v), 2) for v in err_end.mean(0)],
    "final_q_deg_mean": [round(float(v), 2) for v in torch.rad2deg(u.robot.data.joint_pos[:, arm]).mean(0)],
    "goal_deg": [round(float(v), 2) for v in torch.rad2deg(goal)],
    "final_speed_deg_s_max_all": round(float(spd_end.max()), 2),
    "max_joint_speed_deg_s": round(float(torch.rad2deg(vmax.max())), 1),
    "keepout_max_mm": round(float(keep_max.max() * 1000), 1),
    "table_or_any_contact_max_n_per_link": {l: round(float(link_f[:, i].max()), 2) for i, l in enumerate(LINKS)},
    "envs_with_contact_over_0.5n": int((link_f.amax(dim=1) > 0.5).sum()),
    "cube_moved_max_mm": round(float(cube_max.max() * 1000), 2),
}
print("RAISE_L_RESULT " + json.dumps(res), flush=True)
json.dump({"result": res, "trace_env0": rows}, open(os.path.join(video_dir, "raise_l.json"), "w"), indent=1)
print("RAISE_L_DIR=" + video_dir, flush=True)
env.close()
simulation_app.close()
