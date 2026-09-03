# SPDX-License-Identifier: BSD-3-Clause
"""Eval script with per-step EE trajectory logging (console + file).

Usage
-----
# 1. 한 터미널에서 평가 실행 (debug_log 파일에 동시 기록)
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_debug_ee.py \\
        --task Isaac-PnP-Obstacle-Franka-Play-v0 \\
        --num_envs 1 \\
        --verbose_ee \\
        --debug_log logs/debug_trace.log

# 2. 다른 터미널에서 실시간 모니터링
    bash scripts/watch_debug.sh logs/debug_trace.log

Debug log tags
--------------
  [TARGET_EE]   : end-effector position every step
  [OBJECT_POS]  : grasped object position every step
  [OBSTACLE_0]  : slot-0 active obstacle position (zeros if inactive)
  [OBSTACLE_1]  : slot-1 active obstacle position (zeros if inactive)
  [EPISODE]     : episode start / end events

Control-mode notes
------------------
  joint_pos (default for this task):
    Policy outputs joint angle targets. Target EE position is not directly
    available without FK. We log the CURRENT EE position (FrameTransformer
    reading after physics step) — this is the physics result of the previous
    action and is the standard diagnostic signal for joint-position controllers.
    FK on the commanded angles would require an external kinematics library
    (e.g. pytorch_kinematics) and is not implemented here.
  task_space:
    Policy outputs EE position offsets; action[:3] is logged as target_ee_pos
    and the FrameTransformer reading is logged as current_ee_pos.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Eval RSL-RL agent with EE trajectory debugging.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--real-time", action="store_true", default=False)

parser.add_argument(
    "--verbose_ee",
    action="store_true",
    default=False,
    help="Print EE + object + obstacle positions every step to stdout.",
)
parser.add_argument(
    "--verbose_ep_idx",
    type=int,
    default=0,
    help="Parallel env index to monitor (default: 0).",
)
parser.add_argument(
    "--log_csv",
    type=str,
    default=None,
    help="CSV path for EE trajectory. Columns depend on control_mode.",
)
parser.add_argument(
    "--debug_log",
    type=str,
    default=None,
    help="Path to the debug trace log file (tagged output). "
         "Default: logs/debug_trace.log when --verbose_ee is active.",
)
parser.add_argument(
    "--control_mode",
    type=str,
    default="joint_pos",
    choices=["task_space", "joint_pos"],
    help="'task_space': action[:3] = EE target offset. "
         "'joint_pos': action = joint targets (current EE pos logged from FrameTransformer).",
)
parser.add_argument(
    "--ee_frame_name",
    type=str,
    default="ee_frame",
    help="FrameTransformer scene entity that tracks the EE.",
)
parser.add_argument(
    "--dashboard",
    action="store_true",
    default=False,
    help="Write logs/dashboard.json every step for the real-time dashboard server.",
)
parser.add_argument(
    "--dashboard_json",
    type=str,
    default="logs/dashboard.json",
    help="Path to the dashboard JSON state file (used with --dashboard).",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports (after app launch)
# ---------------------------------------------------------------------------
import csv
import importlib.metadata as metadata
import json
import math
import os
import time

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.assets import RigidObject
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

installed_version = metadata.version("rsl-rl-lib")


# ---------------------------------------------------------------------------
# Debug logger: writes tagged lines to stdout AND a log file simultaneously
# ---------------------------------------------------------------------------

class DebugLogger:
    """Writes tagged debug lines to stdout and (optionally) a log file."""

    def __init__(self, log_path: str | None, enabled: bool):
        self.enabled = enabled
        self._file = None
        if enabled and log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)) if os.path.dirname(log_path) else ".", exist_ok=True)
            self._file = open(log_path, "w", buffering=1)  # line-buffered for tail -f

    def log(self, tag: str, msg: str) -> None:
        if not self.enabled:
            return
        line = f"[{tag}] {msg}"
        print(line)
        if self._file:
            self._file.write(line + "\n")

    def close(self) -> None:
        if self._file:
            self._file.close()


# ---------------------------------------------------------------------------
# Scene data helpers
# ---------------------------------------------------------------------------

def _get_ee_pos(env, ee_frame_name: str, ep_idx: int) -> torch.Tensor | None:
    try:
        ee: FrameTransformer = env.unwrapped.scene[ee_frame_name]
        return ee.data.target_pos_w[ep_idx, 0, :]
    except Exception:
        return None


def _get_object_pos(env, ep_idx: int) -> torch.Tensor | None:
    try:
        obj: RigidObject = env.unwrapped.scene["object"]
        return obj.data.root_pos_w[ep_idx]
    except Exception:
        return None


def _get_obstacle_pos(env, slot_name: str, ep_idx: int) -> torch.Tensor | None:
    """Return active variant world pos for the given slot (zeros if inactive)."""
    try:
        from isaaclab_tasks.manager_based.manipulation.pick_place_obstacle.mdp.observations import (
            OBSTACLE_VARIANTS,
        )
        slot_idx = int(slot_name[-1])  # "obstacle_0" → 0
        active = getattr(env.unwrapped, "_obstacle_active_variants", None)
        if active is None:
            return None
        v_idx = int(active[ep_idx, slot_idx].item())
        if v_idx < 0:
            return torch.zeros(3, device=env.unwrapped.device)
        name, _ = OBSTACLE_VARIANTS[slot_idx][v_idx]
        return env.unwrapped.scene[name].data.root_pos_w[ep_idx]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Eval RSL-RL policy with EE / object / obstacle debug logging."""

    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] No pre-trained checkpoint available.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = OnPolicyRunner if agent_cfg.class_name == "OnPolicyRunner" else DistillationRunner
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if version.parse(installed_version) >= version.parse("4.0.0"):
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        policy_nn = runner.alg.policy if version.parse(installed_version) >= version.parse("2.3.0") else runner.alg.actor_critic
        normalizer = getattr(policy_nn, "actor_obs_normalizer", None) or getattr(policy_nn, "student_obs_normalizer", None)
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    ep_idx = args_cli.verbose_ep_idx
    control_mode = args_cli.control_mode
    verbose = args_cli.verbose_ee

    # Resolve debug log path
    debug_log_path = args_cli.debug_log
    if verbose and debug_log_path is None:
        debug_log_path = "logs/debug_trace.log"
    logger = DebugLogger(log_path=debug_log_path, enabled=verbose)

    if verbose:
        print(f"\n[DEBUG] EE trajectory logging — env {ep_idx}, mode={control_mode}")
        print(f"[DEBUG] Log file: {os.path.abspath(debug_log_path) if debug_log_path else 'stdout only'}")
        if control_mode == "joint_pos":
            print("[DEBUG] joint_pos mode: current EE pos (FrameTransformer) is logged.")
            print("[DEBUG] For target EE pos via FK, add pytorch_kinematics to the pipeline.")
        print("[DEBUG] " + "-" * 60)

    # Dashboard JSON setup
    dashboard_path = None
    if args_cli.dashboard:
        dashboard_path = os.path.abspath(args_cli.dashboard_json)
        os.makedirs(os.path.dirname(dashboard_path) if os.path.dirname(dashboard_path) else ".", exist_ok=True)
        print(f"[DASHBOARD] Writing state to: {dashboard_path}")
        print(f"[DASHBOARD] Run: python scripts/dashboard_server.py")

    # Franka joint limits in degrees (for dashboard visualization)
    _JOINT_LIMITS_DEG = [
        (-166, 166), (-101, 101), (-166, 166), (-176, -4),
        (-166, 166), (-1, 215), (-166, 166),
    ]
    _JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "Finger1", "Finger2"]

    # CSV setup
    csv_file = None
    csv_writer = None
    if args_cli.log_csv:
        csv_path = os.path.abspath(args_cli.log_csv)
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        if control_mode == "task_space":
            csv_writer.writerow(["step", "target_x", "target_y", "target_z",
                                  "current_x", "current_y", "current_z", "tracking_error_m"])
        else:
            csv_writer.writerow(["step", "ee_x", "ee_y", "ee_z",
                                  "obj_x", "obj_y", "obj_z"])

    # Main eval loop
    obs = env.get_observations()
    step = 0
    episode_step = 0

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        episode_step += 1

        # -----------------------------------------------------------------
        # Debug logging
        # -----------------------------------------------------------------
        if verbose or csv_writer or dashboard_path:
            ee_pos = _get_ee_pos(env, args_cli.ee_frame_name, ep_idx)
            obj_pos = _get_object_pos(env, ep_idx)
            obs0_pos = _get_obstacle_pos(env, "obstacle_0", ep_idx)
            obs1_pos = _get_obstacle_pos(env, "obstacle_1", ep_idx)

            def fmt3(t):
                if t is None:
                    return "N/A"
                return f"({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})"

            if control_mode == "task_space":
                a = actions[ep_idx]
                tgt = (a[0].item(), a[1].item(), a[2].item())
                cur = (ee_pos[0].item(), ee_pos[1].item(), ee_pos[2].item()) if ee_pos is not None else (float("nan"),) * 3
                err = ((tgt[0]-cur[0])**2 + (tgt[1]-cur[1])**2 + (tgt[2]-cur[2])**2) ** 0.5

                logger.log("TARGET_EE", f"step={episode_step:04d} "
                           f"target=({tgt[0]:+.4f},{tgt[1]:+.4f},{tgt[2]:+.4f}) "
                           f"current=({cur[0]:+.4f},{cur[1]:+.4f},{cur[2]:+.4f}) "
                           f"err={err:.4f}m")
            else:
                # joint_pos: log current EE (FrameTransformer reading after physics step)
                if ee_pos is not None:
                    logger.log("TARGET_EE", f"step={episode_step:04d} "
                               f"ee_pos=({ee_pos[0]:+.4f},{ee_pos[1]:+.4f},{ee_pos[2]:+.4f}) "
                               f"[joint-pos: current EE after action]")
                else:
                    logger.log("TARGET_EE", f"step={episode_step:04d} ee_pos=N/A")

            if obj_pos is not None:
                logger.log("OBJECT_POS", f"step={episode_step:04d} pos={fmt3(obj_pos)}")
            if obs0_pos is not None:
                logger.log("OBSTACLE_0", f"step={episode_step:04d} pos={fmt3(obs0_pos)}")
            if obs1_pos is not None:
                logger.log("OBSTACLE_1", f"step={episode_step:04d} pos={fmt3(obs1_pos)}")

            if csv_writer:
                if control_mode == "task_space" and ee_pos is not None:
                    csv_writer.writerow([episode_step, tgt[0], tgt[1], tgt[2],
                                         cur[0], cur[1], cur[2], err])
                elif ee_pos is not None and obj_pos is not None:
                    csv_writer.writerow([episode_step,
                                         ee_pos[0].item(), ee_pos[1].item(), ee_pos[2].item(),
                                         obj_pos[0].item(), obj_pos[1].item(), obj_pos[2].item()])

        # -----------------------------------------------------------------
        # Dashboard JSON export
        # -----------------------------------------------------------------
        if dashboard_path:
            try:
                raw_env = env.unwrapped
                robot = raw_env.scene["robot"]
                joint_pos_rad = robot.data.joint_pos[ep_idx].cpu().tolist()
                joint_vel = robot.data.joint_vel[ep_idx].cpu().tolist()
                joint_pos_deg = [math.degrees(r) for r in joint_pos_rad]
                default_pos_rad = robot.data.default_joint_pos[ep_idx].cpu().tolist()
                joint_pos_rel = [joint_pos_rad[i] - default_pos_rad[i] for i in range(len(joint_pos_rad))]

                act = actions[ep_idx].cpu().tolist()
                gripper_signal = act[7] if len(act) > 7 else 0.0
                gripper_state = "OPEN" if gripper_signal >= 0 else "CLOSE"

                ee_pos_list = ee_pos.cpu().tolist() if ee_pos is not None else [0, 0, 0]
                obj_pos_list = obj_pos.cpu().tolist() if obj_pos is not None else [0, 0, 0]
                obs0_list = obs0_pos.cpu().tolist() if obs0_pos is not None else [0, 0, 0]
                obs1_list = obs1_pos.cpu().tolist() if obs1_pos is not None else [0, 0, 0]

                dist = math.sqrt(sum((ee_pos_list[i] - obj_pos_list[i])**2 for i in range(3)))

                state = {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "step": step,
                    "episode_step": episode_step,
                    "joint_pos_rad": [round(v, 4) for v in joint_pos_rad],
                    "joint_pos_deg": [round(v, 1) for v in joint_pos_deg],
                    "joint_pos_rel_rad": [round(v, 4) for v in joint_pos_rel],
                    "joint_vel": [round(v, 3) for v in joint_vel],
                    "joint_names": _JOINT_NAMES,
                    "joint_limits_deg": _JOINT_LIMITS_DEG,
                    "ee_pos": [round(v, 4) for v in ee_pos_list],
                    "object_pos": [round(v, 4) for v in obj_pos_list],
                    "obstacle_0_pos": [round(v, 4) for v in obs0_list],
                    "obstacle_1_pos": [round(v, 4) for v in obs1_list],
                    "goal_pos": [0.5, 0.35, 0.25],
                    "dist_ee_obj": round(dist, 4),
                    "actions_arm": [round(v, 4) for v in act[:7]],
                    "action_gripper": round(gripper_signal, 4),
                    "gripper_state": gripper_state,
                    "done": bool(dones[ep_idx].item()),
                }
                with open(dashboard_path, "w") as _f:
                    json.dump(state, _f)
            except Exception as _e:
                pass  # dashboard는 선택 기능 — 에러가 학습을 멈추면 안 됨

        if dones[ep_idx]:
            logger.log("EPISODE", f"ended at step {episode_step} — resetting")
            episode_step = 0

        step += 1
        if args_cli.video and step >= args_cli.video_length:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if csv_file:
        csv_file.close()
        print(f"[DEBUG] Trajectory saved to {args_cli.log_csv}")
    logger.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
