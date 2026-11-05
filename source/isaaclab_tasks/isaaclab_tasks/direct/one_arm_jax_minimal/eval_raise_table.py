"""Evaluate the v22 table-aware raise policy at a grid of table offsets (raise stage only).

Envs cycle through 9 table offsets (front edge x -2/0/+2 cm, top z -1.5/0/+1.5 cm); every
env starts hanging (no reference-state starts), arm gains nominal unless --gain_scale. Per
offset: arrival (every arm joint within 1.5 deg of the shifted pre-grasp pose and slower
than 0.05 rad/s for 0.5 s), non-cube contact (table), keep-out depth, cube motion, final
error, and tremor (direction reversals per second). Video (env 0) via RecordVideo into the
checkpoint run folder under videos/.
"""

import argparse
import importlib.metadata
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=36)
parser.add_argument("--camera", choices=("front", "side"), default="side")
parser.add_argument("--video_offset", type=int, default=4, help="grid index shown in env 0 (4 = no offset)")
parser.add_argument("--gain_scale", type=float, default=1.0)
parser.add_argument("--tag", default="")
parser.add_argument("--v24", action="store_true", help="L-based path (v24)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2RaiseLTablePPORunnerCfg, JaxV2RaiseTablePPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_arm_raise import JaxV2RaiseLTableEnvCfg, JaxV2RaiseTableEnvCfg

cfg = JaxV2RaiseLTableEnvCfg() if args.v24 else JaxV2RaiseTableEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.raise_rsi_prob = 0.0
cfg.raise_gain_rand = 0.0
cfg.sensor_randomize = False
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
if args.camera == "front":
    cfg.viewer.eye, cfg.viewer.lookat = (-1.35, 0.10, 1.35), (-0.30, 0.12, 0.85)
else:
    cfg.viewer.eye, cfg.viewer.lookat = (-0.25, 1.75, 1.30), (-0.25, 0.12, 0.85)
cfg.viewer.resolution = (960, 540)
steps = round(cfg.episode_length_s * 60) - 2
checkpoint = os.path.abspath(args.checkpoint)
raw_env = gym.make("Isaac-JaxV2-RaiseLTable-Direct-v24" if args.v24 else "Isaac-JaxV2-RaiseTable-Direct-v22", cfg=cfg, render_mode="rgb_array")
video_dir = os.path.join(os.path.dirname(checkpoint), "videos",
                         f"raise_table_{args.camera}_{os.path.splitext(os.path.basename(checkpoint))[0]}{('_' + args.tag) if args.tag else ''}")
os.makedirs(video_dir, exist_ok=True)
video_env = gym.wrappers.RecordVideo(raw_env, video_folder=video_dir, step_trigger=lambda s: s == 0, video_length=steps, disable_logger=True)
agent_cfg = handle_deprecated_rsl_rl_cfg((JaxV2RaiseLTablePPORunnerCfg if args.v24 else JaxV2RaiseTablePPORunnerCfg)(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(video_env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = raw_env.unwrapped
n = u.num_envs
grid = torch.tensor([[dx, dz] for dx in (-0.02, 0.0, 0.02) for dz in (-0.015, 0.0, 0.015)], device=u.device)
order = torch.arange(n, device=u.device) % 9
order = (order + args.video_offset) % 9  # env 0 shows the chosen offset
u._table_off_fixed = grid[order]
u._reset_idx(u.robot._ALL_INDICES)
if args.gain_scale != 1.0:
    arm = list(u.arm_ids)
    u.robot.write_joint_stiffness_to_sim(u.robot.actuators["arm"].stiffness[:, :1].expand(-1, len(arm)) * args.gain_scale, joint_ids=arm)
    u.robot.write_joint_damping_to_sim(u.robot.actuators["arm"].damping[:, :1].expand(-1, len(arm)) * args.gain_scale, joint_ids=arm)
arm = list(u.arm_ids)
names = [f"proximal_{i}" for i in (1, 2, 3)] + [f"finger_{i}" for i in (1, 2, 3)]
cube0 = u.object.data.root_pos_w.clone()
calm = torch.zeros(n, dtype=torch.long, device=u.device)
arrived = torch.zeros(n, dtype=torch.bool, device=u.device)
cmax = torch.zeros(n, device=u.device); kmax = torch.zeros(n, device=u.device); cube_max = torch.zeros(n, device=u.device)
qd_prev = None; rev = torch.zeros(n, 8, device=u.device); qhist = []
obs = env.get_observations()
with torch.inference_mode():
    for k in range(steps):
        obs, _, _, _ = env.step(policy(obs))
        goal = u._raise_goal() + u._table_shift_q(u.episode_length_buf * u.step_dt)
        err = (u.robot.data.joint_pos[:, arm] - goal).abs().amax(dim=1)
        qd = u.robot.data.joint_vel[:, arm]
        # stopped = every arm joint moved < 0.5 deg over the last 0.5 s (the elbow's
        # velocity reading stays at ~4-7 deg/s while its angle is constant)
        qhist.append(u.robot.data.joint_pos[:, arm].clone())
        still = (qhist[-1] - qhist[-31]).abs().amax(dim=1) < 0.0087 if len(qhist) > 30 else torch.zeros_like(arrived)
        arrived |= (err < cfg.raise_settle_err_rad) & still
        if qd_prev is not None:
            rev += ((qd * qd_prev < 0) & (qd.abs() > 0.0175)).float()
        qd_prev = qd.clone()
        cn = torch.stack([(torch.linalg.vector_norm(u.scene.sensors[s].data.net_forces_w[:, 0], dim=-1)
                           - torch.linalg.vector_norm(u.scene.sensors[s].data.force_matrix_w[:, 0, 0], dim=-1)).clamp_min(0.0) for s in names], 1).sum(1)
        cmax = torch.maximum(cmax, cn)
        kmax = torch.maximum(kmax, u.raise_keepout())
        cube_max = torch.maximum(cube_max, (u.object.data.root_pos_w - cube0).norm(dim=1))
    err_end = torch.rad2deg((u.robot.data.joint_pos[:, arm] - goal).abs())
    spd_end = torch.rad2deg(u.robot.data.joint_vel[:, arm].abs())
res = {"checkpoint": checkpoint, "gain_scale": args.gain_scale, "per_offset": []}
for g in range(9):
    m = order == g
    res["per_offset"].append({
        "dx_cm": round(float(grid[g, 0]) * 100, 1), "dz_cm": round(float(grid[g, 1]) * 100, 1),
        "arrived": round(float(arrived[m].float().mean()), 2),
        "final_err_deg_max": round(float(err_end[m].amax()), 2),
        "final_speed_deg_s_max": round(float(spd_end[m].amax()), 2),
        "final_speed_joint": int(spd_end[m].amax(dim=0).argmax()),
        "table_contact_n_max": round(float(cmax[m].max()), 2), "keepout_mm_max": round(float(kmax[m].max()) * 1000, 1),
        "cube_moved_mm_max": round(float(cube_max[m].max()) * 1000, 2),
        "reversals_per_s_max_joint": round(float((rev[m] / (steps / 60)).mean(0).max()), 2)})
res["arrived_all"] = round(float(arrived.float().mean()), 3)
print("RAISE_TABLE_RESULT " + json.dumps(res), flush=True)
json.dump(res, open(os.path.join(video_dir, "raise_table.json"), "w"), indent=1)
print("RAISE_TABLE_DIR=" + video_dir, flush=True)
env.close()
simulation_app.close()
