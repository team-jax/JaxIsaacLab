"""Grasp strength test: let a v8 policy grasp, then lift the hand and see if the cube follows.

After --policy_steps of the policy, every finger target is frozen at the
policy's last value (so the squeeze is kept) and the hand body (MG996R) is
raised straight up by differential IK on the arm joints, keeping its
orientation, at --lift_speed for --lift_m, then held. The cube's rise is
compared with the hand's. Recorded with gym's RecordVideo (as in
evaluate_v2_lift_only.py / Isaac Lab play.py --video) from the fixed "front"
evaluation camera; per-step measurements go to lift_test.json.
"""
import argparse, json, os, importlib.metadata
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--policy_steps", type=int, default=600)
parser.add_argument("--lift_m", type=float, default=0.05)
parser.add_argument("--lift_speed", type=float, default=0.02)
parser.add_argument("--hold_steps", type=int, default=180)
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
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2SmoothGraspPPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_smooth_grasp import JaxV2SmoothGraspEnvCfg

checkpoint = os.path.abspath(args.checkpoint)
cfg = JaxV2SmoothGraspEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.episode_length_s = 100.0
# The test lifts the cube on purpose; do not end the episode for it.
for name, value in (("cube_move_limit_m", 10.0), ("cube_rotation_limit_deg", 1000.0)):
    if hasattr(cfg, name):
        setattr(cfg, name, value)
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.eye = (-0.78, 0.08, 1.04)
cfg.viewer.lookat = (-0.29, 0.08, 0.80)
cfg.viewer.resolution = (960, 540)
raw_env = gym.make("Isaac-JaxV2-SmoothLiftGrasp-Direct-v8", cfg=cfg, render_mode="rgb_array")
lift_steps = int(round(args.lift_m / args.lift_speed * 60))
total_steps = args.policy_steps + lift_steps + args.hold_steps
video_dir = os.path.join(
    os.path.dirname(checkpoint), "videos",
    f"lift_test_front_{os.path.splitext(os.path.basename(checkpoint))[0]}{('_' + args.tag) if args.tag else ''}",
)
os.makedirs(video_dir, exist_ok=True)
video_env = gym.wrappers.RecordVideo(
    raw_env, video_folder=video_dir, step_trigger=lambda step: step == 0,
    video_length=total_steps, disable_logger=True,
)
agent_cfg = handle_deprecated_rsl_rl_cfg(JaxV2SmoothGraspPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(video_env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

u = raw_env.unwrapped
hand_id = u.robot.body_names.index("MG996R")
jac_body = hand_id - 1  # fixed-base articulation: the root body has no Jacobian row
arm_ids = list(u.arm_ids[:7])  # exclude RotationBody (thumb opposition)
frozen = None
orig_apply = u._apply_action


def apply_frozen():
    if frozen is not None:
        u.targets[:] = frozen
    orig_apply()


u._apply_action = apply_frozen
rows = []


def record(step, phase):
    hand_z = u.robot.data.body_pos_w[:, hand_id, 2]
    cube_z = u.object.data.root_pos_w[:, 2]
    rows.append({"step": step, "phase": phase,
                 "hand_rise_mm": (1000 * (hand_z - hand_z0)).tolist(),
                 "cube_rise_mm": (1000 * (cube_z - cube_z0)).tolist()})


obs = env.get_observations()
dt = u.step_dt
with torch.inference_mode():
    for _ in range(args.policy_steps):
        obs, _, _, _ = env.step(policy(obs))
    hand_z0 = u.robot.data.body_pos_w[:, hand_id, 2].clone()
    cube_z0 = u.object.data.root_pos_w[:, 2].clone()
    frozen = u.targets.clone()
    zero = torch.zeros_like(u.actions)
    twist = torch.zeros((u.num_envs, 6), device=u.device)
    twist[:, 2] = args.lift_speed
    for k in range(lift_steps + args.hold_steps):
        lifting = k < lift_steps
        if lifting:
            jac = u.robot.root_physx_view.get_jacobians()[:, jac_body, :, :][:, :, arm_ids]
            dq = torch.linalg.lstsq(jac, (twist * dt)[:, :, None]).solution[:, :, 0]
            frozen[:, arm_ids] = frozen[:, arm_ids] + dq
        env.step(zero)
        if k % 10 == 9 or k == lift_steps + args.hold_steps - 1:
            record(args.policy_steps + k + 1, "lift" if lifting else "hold")
final = rows[-1]
summary = {
    "checkpoint": checkpoint, "lift_m": args.lift_m,
    "final_hand_rise_mm": [round(v, 1) for v in final["hand_rise_mm"]],
    "final_cube_rise_mm": [round(v, 1) for v in final["cube_rise_mm"]],
    "envs_cube_followed_80pct": int(sum(
        c >= 0.8 * h for c, h in zip(final["cube_rise_mm"], final["hand_rise_mm"]))),
    "rows": rows,
}
json.dump(summary, open(os.path.join(video_dir, "lift_test.json"), "w"))
print("LIFT_TEST " + json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)
env.close()
simulation_app.close()
