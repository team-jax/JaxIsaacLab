"""Record a v10 L-raise policy with gym's RecordVideo from the fixed "front" camera.

Per-joint arm angles, the L-pose errors and the finger-to-cube y error are
printed every second and saved to front_grasp.json next to the video.
"""
import argparse, json, math, os, importlib.metadata
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--camera", choices=("front", "side", "close", "close_front"), default="front")
parser.add_argument("--tag", default="")
parser.add_argument("--pregrasp", action="store_true", help="v12 pre-grasp task instead of v11")
parser.add_argument("--close", action="store_true", help="v13 pre-grasp then close task")
parser.add_argument("--lift", action="store_true", help="v14 pre-grasp, close, lift task")
parser.add_argument("--lift_rl", action="store_true", help="v15 policy lift task")
parser.add_argument("--grasp_lift", action="store_true", help="v16 grasp and lift task")
parser.add_argument("--intro_s", type=float, default=0.0,
                    help="first show the arm rising from hanging (all joints 0) to the L start pose over this many s")
parser.add_argument("--raise_policy", default="",
                    help="v17 arm-raise checkpoint: start hanging, raise with that policy, then hand over to the "
                         "episode policy with no state reset (continuous joint motion)")
parser.add_argument("--raise_max_s", type=float, default=5.0)
parser.add_argument("--raise_sag_comp_deg", default="1.9,0,0,2.1,0,0.7,-0.2,0",
                    help="script: raise end target = L + this (8 joints, deg), ramped in with the nominal path, "
                         "so the arm settles on the L despite gravity sag")
parser.add_argument("--handover_ramp_s", type=float, default=0.5,
                    help="after the handover, ramp the allowed joint-target speed from 0 to 1.2 rad/s over this time")
parser.add_argument("--intro_policy", default="",
                    help="v10 L-raise checkpoint: the raise is driven by that policy, then blended into the start pose")
parser.add_argument("--raise_cube", default="",
                    help="v25: cube offset 'dx,dy' (m) on the table; raise policy is the cube-aware residual (v25)")
parser.add_argument("--raise_ltable", action="store_true",
                    help="v24: table-aware residual policy on the L-based path")
parser.add_argument("--raise_table", action="store_true",
                    help="v22: --raise_policy is the table-aware residual policy (observes the table edge and top)")
parser.add_argument("--raise_residual", action="store_true",
                    help="v21: --raise_policy is the residual policy on the v18 path")
parser.add_argument("--raise_learned", action="store_true",
                    help="v19: --raise_policy is the learned raise (joint velocity commands) into the pre-grasp pose")
parser.add_argument("--raise_to_pregrasp", action="store_true",
                    help="v18: --raise_policy rises straight into the pre-grasp pose; v16 then starts at its close")
parser.add_argument("--table_diag", action="store_true",
                    help="log table contact force per link (forearm to fingertips) every step")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.raise_table = args.raise_table or args.raise_ltable or bool(args.raise_cube)
args.raise_residual = args.raise_residual or args.raise_table
args.raise_to_pregrasp = args.raise_to_pregrasp or args.raise_learned or args.raise_residual
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2FrontGraspPPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp import JaxV2FrontGraspEnvCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_pregrasp import JaxV2FrontPregraspEnvCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_close import JaxV2FrontGraspCloseEnvCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_lift import JaxV2FrontLiftEnvCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_lift_rl import JaxV2FrontLiftRLEnvCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_lift import JaxV2FrontGraspLiftEnvCfg

checkpoint = os.path.abspath(args.checkpoint)
if args.grasp_lift:
    cfg, task, prefix = JaxV2FrontGraspLiftEnvCfg(), "Isaac-JaxV2-FrontGraspLift-Direct-v16", "front_grasp_lift"
elif args.lift_rl:
    cfg, task, prefix = JaxV2FrontLiftRLEnvCfg(), "Isaac-JaxV2-FrontLiftRL-Direct-v15", "front_lift_rl"
elif args.lift:
    cfg, task, prefix = JaxV2FrontLiftEnvCfg(), "Isaac-JaxV2-FrontLift-Direct-v14", "front_lift"
elif args.close:
    cfg, task, prefix = JaxV2FrontGraspCloseEnvCfg(), "Isaac-JaxV2-FrontGraspClose-Direct-v13", "front_close"
elif args.pregrasp:
    cfg, task, prefix = JaxV2FrontPregraspEnvCfg(), "Isaac-JaxV2-FrontPregrasp-Direct-v12", "front_pregrasp"
else:
    cfg, task, prefix = JaxV2FrontGraspEnvCfg(), "Isaac-JaxV2-FrontGrasp-Direct-v11", "front_grasp"
cfg.sensor_randomize = False
TD_LINKS = ("Under_Arm", "Wrist_Pitch", "Wrist_yaw", "MG996R", "RotationBody", "Finger_1_1", "Finger_2_1", "Finger_3_1", "Finger_1_2", "Finger_2_2", "Finger_3_2")
if args.table_diag:
    cfg.diagnostic_table_contacts = True
    cfg.table_contact_body_names = TD_LINKS
cfg.scene.num_envs = args.num_envs
cfg.episode_length_s = args.steps / 60.0 + 1.0
if args.raise_to_pregrasp:
    cfg.episode_length_s += cfg.close_after_s  # the episode clock starts at the close
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
# Wide views: the whole arm, the table and the cube in frame.
if args.camera == "close":
    cfg.viewer.eye = (-0.52, 0.52, 0.86)
    cfg.viewer.lookat = (-0.40, 0.26, 0.78)
elif args.camera == "close_front":
    cfg.viewer.eye = (-0.72, 0.24, 0.84)
    cfg.viewer.lookat = (-0.40, 0.26, 0.78)
elif args.camera == "front":
    cfg.viewer.eye = (-1.35, 0.10, 1.35)
    cfg.viewer.lookat = (-0.30, 0.12, 0.85)
else:
    cfg.viewer.eye = (-0.25, 1.75, 1.30)
    cfg.viewer.lookat = (-0.25, 0.12, 0.85)
cfg.viewer.resolution = (960, 540)
raw_env = gym.make(task, cfg=cfg, render_mode="rgb_array")
video_dir = os.path.join(
    os.path.dirname(checkpoint), "videos",
    f"{prefix}_{args.camera}_{os.path.splitext(os.path.basename(checkpoint))[0]}{('_' + args.tag) if args.tag else ''}",
)
os.makedirs(video_dir, exist_ok=True)
video_env = gym.wrappers.RecordVideo(
    raw_env, video_folder=video_dir, step_trigger=lambda step: step == 0,
    video_length=args.steps + round(max(args.intro_s, args.raise_max_s if args.raise_policy else 0.0) * 60), disable_logger=True,
)
agent_cfg = handle_deprecated_rsl_rl_cfg(JaxV2FrontGraspPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(video_env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

u = raw_env.unwrapped
if os.environ.get("TEST_ARM_GAIN_SCALE"):
    # robustness test: arm drive kp/kd scaled for the whole run (a real arm will not match the sim)
    _g = float(os.environ["TEST_ARM_GAIN_SCALE"]); _arm = list(u.arm_ids)
    u.robot.write_joint_stiffness_to_sim(u.robot.actuators["arm"].stiffness[:, :1].expand(-1, len(_arm)) * _g, joint_ids=_arm)
    u.robot.write_joint_damping_to_sim(u.robot.actuators["arm"].damping[:, :1].expand(-1, len(_arm)) * _g, joint_ids=_arm)
    print("ARM_GAIN_SCALE", _g, flush=True)
names = [u.robot.joint_names[i] for i in u.arm_ids[:7]]
rows = []
chain = {}
TRACE = []
TQ = []  # (applied torque, joint velocity) per step
TDF = []
TBL = []
VZ = []
if args.raise_policy:
    # Continuous chain (user, 2026-10-29): the only state write is the initial
    # hanging pose (all joints 0, at rest). The v17 raise policy then drives
    # the arm; when it has settled at the v16 start pose (or at raise_max_s), control passes
    # to the episode's policies. Joint positions and velocities are never
    # written again; only the episode clock and action filters restart, as at
    # a training reset.
    from isaaclab_tasks.direct.one_arm_jax_minimal.v2_arm_raise import (
        JaxV2ArmRaiseEnvCfg, JaxV2RaiseLearnedEnvCfg, JaxV2RaiseResidualEnvCfg, JaxV2RaiseTableEnvCfg, JaxV2RaiseLTableEnvCfg, JaxV2RaiseCubeEnvCfg, JaxV2RaiseCubeWideEnvCfg, JaxV2RaiseToPregraspEnvCfg, raise_extra_obs, raise_nominal, raise_position_step, raise_velocity_step)
    from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_pregrasp import JaxV2FrontPregraspEnv
    rc = (JaxV2RaiseCubeWideEnvCfg() if os.environ.get("RAISE_WIDE") else JaxV2RaiseCubeEnvCfg()) if args.raise_cube else JaxV2RaiseLTableEnvCfg() if args.raise_ltable else JaxV2RaiseTableEnvCfg() if args.raise_table else JaxV2RaiseResidualEnvCfg() if args.raise_residual else JaxV2RaiseLearnedEnvCfg() if args.raise_learned else (JaxV2RaiseToPregraspEnvCfg() if args.raise_to_pregrasp else JaxV2ArmRaiseEnvCfg())
    u._rc = rc
    # v39: with a deeper pre-grasp (GL_REACH_DX/DY) the raise path went in deep and the hand hit the cube (5.9 N,
    # cube pushed 10 mm): the raise now ends at the shallow pre-grasp and the open hand moves in afterwards
    REACH = (float(getattr(rc, "raise_reach_dx_m", 0.0)), float(getattr(rc, "raise_reach_dy_m", 0.0)))
    # v63: RAISE_BACK_M (m, +x = toward the robot) ends the raise that much short of the shallow pre-grasp; the reach-in then
    # covers it too (a turned cube's corners stick out up to 10 mm past its faces and the end of the raise hit them)
    BACK = float(os.environ.get("RAISE_BACK_M", "0"))
    # v63 (wide cube box): RAISE_SIDE_M (m, +y = the side of fingers 1-2) ends the raise that much to the side too; the
    # reach-in then moves in diagonally to the same pre-grasp (at y -10 cm the yaw-free hand turns 14 deg and fingers 1-2
    # swept a turned cube's corner, pushing it up to 24 mm; a 2 cm side start cut that to <1 mm in the probe)
    SIDE = float(os.environ.get("RAISE_SIDE_M", "0"))
    # RAISE_SIDE_K (m per m): plus that times how far the cube sits toward the robot's middle (-dy); the hand's heading turns
    # with dy, so a fixed side start that cleared y -10 drove the thumb into the cube at y +6 (15-20 mm)
    _cdy = float(args.raise_cube.split(",")[1]) if args.raise_cube else 0.0
    SIDE += float(os.environ.get("RAISE_SIDE_K", "0")) * max(-_cdy, 0.0)
    if any(REACH):
        rc.raise_reach_dx_m, rc.raise_reach_dy_m = BACK, SIDE
    # v43c: pre-grasp hand roll (GL_ROLL_DEG, wrist-yaw joint), added after the raise like the reach-in
    ROLL = math.radians(float(os.environ.get("GL_ROLL_DEG", "0")))
    PATH_DZ = float(os.environ.get("RAISE_PATH_DZ", "0"))
    # v34: GL_TABLE_DZ_FIXED moves the real table (grasp-lift env); the raise path, its table observation
    # and the keep-out follow it
    PATH_DZ += float(os.environ.get("GL_TABLE_DZ_FIXED", "0"))
    if args.raise_cube:
        # v25: cube moved on the table by (dx, dy); the path shifts the fingertip midpoint with it
        import types
        from isaaclab_tasks.direct.one_arm_jax_minimal.v2_arm_raise import JaxV2RaiseTableEnv
        _co = torch.tensor([[float(v) for v in args.raise_cube.split(",")]], device=u.device).expand(u.num_envs, -1)
        with torch.inference_mode():
            _cp = u.object.data.root_pose_w.clone(); _cp[:, :2] += _co
            u.object.write_root_pose_to_sim(_cp); u.object.write_root_velocity_to_sim(torch.zeros_like(u.object.data.root_vel_w))
        _shim = types.SimpleNamespace(cfg=rc, device=u.device, _table_off=torch.tensor([[0.0, PATH_DZ]], device=u.device).expand(u.num_envs, -1).contiguous(), _cube_off=_co)
        u._table_shift_q = lambda t, ids=None: JaxV2RaiseTableEnv._table_shift_q(_shim, t)
    elif args.raise_table and PATH_DZ:
        # Stop the hand PATH_DZ higher at the pre-grasp: the policy and its path are
        # told the table sits PATH_DZ higher (path shifted by the table-shift table).
        import types
        from isaaclab_tasks.direct.one_arm_jax_minimal.v2_arm_raise import JaxV2RaiseTableEnv
        _shim = types.SimpleNamespace(cfg=rc, device=u.device, _table_off=torch.tensor([[0.0, PATH_DZ]], device=u.device).expand(u.num_envs, -1))
        u._table_shift_q = lambda t, ids=None: JaxV2RaiseTableEnv._table_shift_q(_shim, t)
    # --raise_policy nominal: the scripted path alone (zero residual).
    sd = None if args.raise_policy == "nominal" else torch.load(args.raise_policy, map_location=u.device, weights_only=False)["actor_state_dict"]
    nl = 0 if sd is None else sum(1 for k in sd if k.startswith("mlp.") and k.endswith(".weight"))
    def actor(x):
        if sd is None:
            return torch.zeros(x.shape[0], u.single_action_space.shape[0], device=x.device)
        x = (x - sd["obs_normalizer._mean"]) / (sd["obs_normalizer._std"] + 1e-2)
        for i in range(nl):
            x = x @ sd[f"mlp.{2 * i}.weight"].T + sd[f"mlp.{2 * i}.bias"]
            if i < nl - 1:
                x = torch.nn.functional.elu(x)
        return x
    arm = list(u.arm_ids); fids = list(u.finger_ids)
    goal = torch.tensor(getattr(rc, "raise_goal_q", None) or rc.start_arm_q, device=u.device)
    sag = torch.tensor([math.radians(float(v)) for v in args.raise_sag_comp_deg.split(",")], device=u.device)
    zero_act = torch.zeros(u.num_envs, u.single_action_space.shape[0], device=u.device)
    with torch.inference_mode():
        cube0 = u.object.data.root_pos_w.clone()
        q = u.robot.data.joint_pos.clone(); q[:, arm + fids] = 0.0
        u.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        u.targets[:, arm + fids] = 0.0
        u.robot.set_joint_position_target(u.targets)
        hide = u.cfg.pre_hide_scripted_obs
        u.cfg.pre_hide_scripted_obs = False
        filt = torch.zeros_like(zero_act); calm = 0; rs = {"v": torch.zeros(u.num_envs, 8, device=u.device)}; keep_max = torch.zeros(u.num_envs, device=u.device)
        hand_ids = [u._hand_id] + u.proximal_ids + list(u.fingertip_ids)
        sens = [f"proximal_{i}" for i in (1, 2, 3)] + [f"finger_{i}" for i in (1, 2, 3)]
        contact_max = torch.zeros(u.num_envs, device=u.device)
        n_max = round(args.raise_max_s * 60)
        if getattr(rc, "raise_mode", "") == "position":
            # as at a v19 reset: the filtered action starts at the one targeting the hanging pose
            filt[:, :8] = ((0.0 - 0.5 * goal) / (0.5 * goal.abs() + rc.raise_position_margin_rad)).clamp(-1.0, 1.0)
        for k in range(n_max):
            obs_r = JaxV2FrontPregraspEnv._get_observations(u)["policy"]
            if args.raise_learned or getattr(rc, "raise_residual_obs", False):  # v19/v21 obs: + phase and reference error at the env clock k / 60
                obs_r = torch.cat((obs_r, raise_extra_obs(u, torch.full((u.num_envs,), k / 60.0, device=u.device))), dim=-1)
            if args.raise_table:  # v22: measured table front edge x and top z, robot-base frame (static table here)
                root = u.robot.data.root_pos_w - u.scene.env_origins
                tm = torch.stack((rc.raise_table_edge_x - root[:, 0], torch.full_like(root[:, 0], rc.table_top_z + PATH_DZ) - root[:, 2]), dim=1)
                obs_r = torch.cat((obs_r, tm), dim=-1)
            filt = rc.action_filter_alpha * actor(obs_r).clamp(-1.0, 1.0) + (1.0 - rc.action_filter_alpha) * filt
            t = (k + 1) / 60.0
            def raise_pre(actions, f=filt, t=t):
                if rc.raise_mode == "velocity":
                    tg = raise_velocity_step(u, f, torch.full((u.num_envs,), t - 1.0 / 60.0, device=u.device))
                elif rc.raise_mode == "position":
                    tg = raise_position_step(u, f, torch.full((u.num_envs,), t - 1.0 / 60.0, device=u.device))
                elif rc.raise_mode == "residual":
                    xs = min(t / rc.raise_nominal_s, 1.0)
                    rmask = torch.zeros(8, device=u.device); rmask[list(getattr(rc, "raise_residual_joints", range(8)))] = 1.0
                    env_w = min(t / 0.5, 1.0) if getattr(rc, "raise_residual_envelope", "sin") == "const" else math.sin(math.pi * xs)
                    tg = raise_nominal(rc, torch.full((u.num_envs,), t, device=u.device), goal) + xs * xs * (3.0 - 2.0 * xs) * sag \
                        + (u._table_shift_q(torch.full((u.num_envs,), t, device=u.device)) if hasattr(u, "_table_shift_q") else 0.0) \
                        + rc.raise_residual_rad * env_w * f[:, :8] * rmask
                    if getattr(rc, "raise_target_smooth", 0.0):
                        new = rc.raise_target_smooth * rs["v"] + (1.0 - rc.raise_target_smooth) * tg
                        if getattr(rc, "raise_rate_rad_s", 0.0):
                            st = rc.raise_rate_rad_s / 60.0
                            new = torch.cat([rs["v"][:, :7] + (new[:, :7] - rs["v"][:, :7]).clamp(-st, st), new[:, 7:]], dim=1)
                        rs["v"] = new
                        tg = rs["v"]
                else:
                    tg = goal + rc.raise_range_rad * f[:, :8] if rc.raise_absolute else u.targets[:, arm] + rc.raise_delta_rad * f[:, :8]
                u.targets[:, arm] = torch.minimum(torch.maximum(tg, u.lower[arm]), u.upper[arm])
                t0, t1 = rc.raise_thumb_ramp_s
                x = min(max((t - t0) / (t1 - t0), 0.0), 1.0); w = x * x * (3.0 - 2.0 * x)
                fg = torch.zeros(u.num_envs, len(fids), device=u.device); fg[:, 2] = w * math.radians(rc.hand_shape_deg[2])
                u.targets[:, fids] = fg
            u._pre_physics_step = raise_pre
            env.step(zero_act)
            u.episode_length_buf[:] = 0
            TRACE.append(u.robot.data.joint_pos.clone()); TQ.append((u.robot.data.applied_torque.clone(), u.robot.data.joint_vel.clone()))
            p = u.robot.data.body_pos_w[:, hand_ids] - u.scene.env_origins[:, None]
            _ex = rc.raise_table_edge_x if getattr(rc, "raise_table_edge_real_x", None) is None else rc.raise_table_edge_real_x
            x_in = (_ex + rc.raise_edge_margin_m) - p[..., 0]
            z_in = (rc.table_top_z + PATH_DZ + float(os.environ.get("GL_TABLE_Z0", "0")) + rc.raise_clear_m + rc.raise_plate_drop_m) - p[..., 2]  # v53: real table
            keep_max = torch.maximum(keep_max, torch.minimum(x_in, z_in).clamp_min(0.0).amax(dim=1))
            cn = torch.stack([torch.linalg.vector_norm(u.scene.sensors[n].data.net_forces_w[:, 0], dim=-1) for n in sens], 1).sum(1)
            contact_max = torch.maximum(contact_max, cn)
            if args.table_diag:
                # v44: raise-phase table contact per link (max force and the time it first exceeds 1 N)
                tf_r = torch.stack([torch.linalg.vector_norm(u.scene.sensors[f"table_contact_{n}"].data.net_forces_w[:, 0], dim=-1) for n in TD_LINKS], 1)
                if "rtd" not in dir():
                    rtd = {"max": torch.zeros(len(TD_LINKS), device=u.device), "t1": [None] * len(TD_LINKS)}
                rtd["max"] = torch.maximum(rtd["max"], tf_r.amax(0))
                for li in range(len(TD_LINKS)):
                    if rtd["t1"][li] is None and float(tf_r[:, li].max()) > 1.0:
                        rtd["t1"][li] = round(t, 2)
            _gs = u._table_shift_q(torch.full((u.num_envs,), 99.0, device=u.device)) if args.raise_cube else 0.0  # v25: the start pose moves with the cube
            err = (u.robot.data.joint_pos[:, arm] - goal - _gs).abs().amax(dim=1)
            spd = u.robot.data.joint_vel[:, arm].abs().amax(dim=1)
            # Hand over once the nominal path has ended and every env is within
            # 3 deg of the start pose and slower than 0.15 rad/s for 10 steps
            # (gravity leaves the elbow ~2 deg low, as at a training reset).
            if args.raise_table:
                # User (2026-11-01): finger 2 hit the table and the hand bounced right after the
                # handover; the arm was still settling (hand -53 mm/s). Hand over only 0.5 s after
                # the path ends and once every arm joint moved < 0.2 deg over the last 0.25 s.
                still = len(TRACE) > 15 and bool(((TRACE[-1][:, arm] - TRACE[-16][:, arm]).abs().amax() < math.radians(0.2)).item())
                calm = calm + 1 if (t >= rc.raise_nominal_s + 0.5 and still and bool((err < math.radians(3.0)).all())) else 0
            else:
                calm = calm + 1 if (t >= rc.raise_nominal_s and bool(((err < math.radians(3.0)) & (spd < 0.15)).all())) else 0
            if k % 30 == 29:
                print("RAISE " + json.dumps({"t": round(t, 2), "q_deg_env0": [round(float(v), 1) for v in torch.rad2deg(u.robot.data.joint_pos[0, arm])],
                      "err_max_deg_mean": round(float(torch.rad2deg(err).mean()), 2), "speed_max": round(float(spd.max()), 3)}), flush=True)
            if calm >= 10:
                break
        if any(REACH) or ROLL:
            # v39: move the open hand in to the deeper pre-grasp over 0.5 s (joint-space smoothstep between the two
            # exact-IK pre-grasp poses, fingers held), then hold 0.25 s; the episode then starts there, as in training
            tt = torch.full((u.num_envs,), 99.0, device=u.device)
            q_sh = u._table_shift_q(tt)
            rc.raise_reach_dx_m, rc.raise_reach_dy_m = REACH
            dq_in = u._table_shift_q(tt) - q_sh
            dq_in[:, 6] += ROLL
            base_tg = u.targets[:, arm].clone(); fg_keep = u.targets[:, fids].clone()
            n_in = round(0.5 * 60); c_in = torch.zeros(u.num_envs, device=u.device)
            for k2 in range(n_in + 15):
                xw = min((k2 + 1) / n_in, 1.0); w_in = xw * xw * (3.0 - 2.0 * xw)
                def in_pre(actions, w_in=w_in):
                    u.targets[:, arm] = torch.minimum(torch.maximum(base_tg + w_in * dq_in, u.lower[arm]), u.upper[arm])
                    u.targets[:, fids] = fg_keep
                u._pre_physics_step = in_pre
                env.step(zero_act)
                u.episode_length_buf[:] = 0
                TRACE.append(u.robot.data.joint_pos.clone()); TQ.append((u.robot.data.applied_torque.clone(), u.robot.data.joint_vel.clone()))
                cn = torch.stack([torch.linalg.vector_norm(u.scene.sensors[n].data.net_forces_w[:, 0], dim=-1) for n in sens], 1).sum(1)
                c_in = torch.maximum(c_in, cn)
            contact_max = torch.maximum(contact_max, c_in)
            print("REACH_IN " + json.dumps({"reach_m": REACH, "dq_deg_env0": [round(float(v), 2) for v in torch.rad2deg(dq_in[0])],
                  "contact_max_n": round(float(c_in.max()), 2)}), flush=True)
        if args.table_diag and "rtd" in dir():
            print("RAISE_TABLE_DIAG " + json.dumps({n: {"max_n": round(float(rtd["max"][i]), 1), "first_t_s": rtd["t1"][i]} for i, n in enumerate(TD_LINKS) if float(rtd["max"][i]) > 0.05}), flush=True)
        del u._pre_physics_step  # back to the env's own control
        # User: the arm seemed to jump at the handover. The v12 pre-grasp
        # commands absolute targets, so its first step drove the joints from
        # ~1 deg/s to the 1 rad/s drive limit in one step (0.02 -> 0.92 deg
        # per step). Script: for handover_ramp_s the targets sent to the drives
        # blend from the handover pose to the policies' targets with a
        # smoothstep w(t), q_sent = q_h + w (q_cmd - q_h); at w = 1 they equal
        # the policies' targets, so the blend ends without a jump. (A speed
        # limiter instead lagged and destabilized v16: lifted 7/16.)
        if args.handover_ramp_s > 0:
            # Blend from the targets held at the handover, not the joint positions:
            # the arm hangs ~2 deg below its targets under gravity, and a target equal
            # to the sagged position let it sag again (hand -1-2 mm), so finger 2 hit
            # the table (4.9 N, all envs) and the hand bounced (user, 2026-11-01).
            q_h = u.targets.clone(); blend = {"k": 0}
            def ramped_apply():
                blend["k"] += 1
                x = min(1.0, blend["k"] * u.cfg.sim.dt / args.handover_ramp_s); w = x * x * (3.0 - 2.0 * x)
                u.robot.set_joint_position_target(q_h + w * (u.targets - q_h))
                if x >= 1.0:
                    del u._apply_action
            u._apply_action = ramped_apply
        u.cfg.pre_hide_scripted_obs = hide
        if hasattr(u, "_filtered_actions"):
            u._filtered_actions.zero_()
        u.actions.zero_(); u.previous_actions.zero_()
        u.episode_length_buf[:] = 0
        if args.raise_to_pregrasp:
            # v18 ends in the pre-grasp pose: skip the v12 pre-grasp and start
            # the v16 episode at its close, with the arm holds latched to the
            # handover targets (they latch only before the close).
            u.episode_length_buf[:] = round(u.cfg.close_after_s / u.step_dt)
            a7 = u.targets[:, u.arm_ids[:7]].clone()
            u._arm_hold = a7.clone(); u._lift_hold = a7.clone(); u._arm_smooth = a7.clone()
            u._lift_offset = torch.zeros_like(a7)
            if hasattr(u, "_bal_q"):
                del u._bal_q
        qd = torch.rad2deg(u.robot.data.joint_pos[:, arm]); vd = torch.rad2deg(u.robot.data.joint_vel[:, arm])
        chain = {"handover_t_s": round(t, 2), "handover_by": "converged" if calm >= 10 else "timeout",
                 "q_deg_env0": [round(float(v), 2) for v in qd[0]],
                 "qdot_deg_s_env0": [round(float(v), 2) for v in vd[0]],
                 "err_to_start_deg_max_all_envs": round(float(torch.rad2deg((u.robot.data.joint_pos[:, arm] - goal).abs()).max()), 2),
                 "qdot_deg_s_max_all_envs": round(float(vd.abs().max()), 2),
                 "fingers_deg_env0": [round(float(v), 1) for v in torch.rad2deg(u.robot.data.joint_pos[0, fids])],
                 "raise_keepout_max_mm": round(float(keep_max.max() * 1000), 1),
                 "raise_hand_contact_max_n": round(float(contact_max.max()), 3),
                 "cube_moved_mm": round(float((u.object.data.root_pos_w - cube0).norm(dim=1).max() * 1000), 2)}
        print("HANDOVER " + json.dumps(chain), flush=True)
if args.intro_s > 0:
    # User: the arm starts upright (hanging); show it rising into the L pose
    # before the episode. Scripted joint targets with a smoothstep ramp; the
    # episode clock is held at 0 so the episode then runs unchanged.
    ids = list(u.arm_ids) + list(u.finger_ids)
    goal = u.robot.data.joint_pos[:, ids].clone()
    q = u.robot.data.joint_pos.clone(); q[:, ids] = 0.0
    u.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    u.targets[:, ids] = 0.0
    n_intro = round(args.intro_s * 60)
    orig_pre = u._pre_physics_step
    zero_act = torch.zeros(u.num_envs, u.single_action_space.shape[0], device=u.device)
    if args.intro_policy:
        # User: use the trained arm-raise (v10 L-raise) checkpoint for the raise.
        # Its control law (filtered delta targets on the 7 arm joints, fingers
        # and RotationBody at 0) and its unmasked v9 observation are reproduced
        # here; afterwards the targets blend into the start pose over 1.5 s.
        from isaaclab_tasks.direct.one_arm_jax_minimal.v2_l_raise import JaxV2LRaiseEnvCfg
        from isaaclab_tasks.direct.one_arm_jax_minimal.v2_pad_sensor_grasp import JaxV2PadSensorGraspEnv
        lc = JaxV2LRaiseEnvCfg()
        sd = torch.load(args.intro_policy, map_location=u.device, weights_only=False)["actor_state_dict"]
        arm7 = list(u.arm_ids[:7]); filt = torch.zeros_like(zero_act)
        def actor(x):
            x = (x - sd["obs_normalizer._mean"]) / (sd["obs_normalizer._std"] + 1e-2)
            x = torch.nn.functional.elu(x @ sd["mlp.0.weight"].T + sd["mlp.0.bias"])
            x = torch.nn.functional.elu(x @ sd["mlp.2.weight"].T + sd["mlp.2.bias"])
            return x @ sd["mlp.4.weight"].T + sd["mlp.4.bias"]
        n_raise = n_intro - 90
        with torch.inference_mode():
            for k in range(n_raise):
                a = actor(JaxV2PadSensorGraspEnv._get_observations(u)["policy"])
                filt = lc.action_filter_alpha * a.clamp(-1.0, 1.0) + (1.0 - lc.action_filter_alpha) * filt
                def intro_pre(actions, f=filt):
                    t = u.targets[:, arm7] + lc.arm_delta_scale * f[:, :7]
                    t = torch.minimum(torch.maximum(t, u.lower[arm7]), u.upper[arm7])
                    t[:, 0] = t[:, 0].clamp_min(lc.l_min_shoulder_pitch_rad)
                    t[:, 2] = t[:, 2].clamp(*lc.l_shoulder_yaw_range_rad)
                    t[:, 3] = t[:, 3].clamp(*lc.l_elbow_range_rad)
                    u.targets[:, arm7] = t
                u._pre_physics_step = intro_pre
                env.step(zero_act)
                u.episode_length_buf[:] = 0
            print("INTRO_RAISED_DEG", [round(float(v), 1) for v in torch.rad2deg(u.robot.data.joint_pos[0, ids])], flush=True)
            start = u.targets[:, ids].clone()
            for k in range(90):
                x = (k + 1) / 90.0; w = x * x * (3.0 - 2.0 * x)
                def intro_pre(actions, w=w):
                    u.targets[:, ids] = start + w * (goal - start)
                u._pre_physics_step = intro_pre
                env.step(zero_act)
                u.episode_length_buf[:] = 0
    else:
      with torch.inference_mode():
        for k in range(n_intro):
            x = min(1.0, (k + 1) / (0.85 * n_intro)); w = x * x * (3.0 - 2.0 * x)
            def intro_pre(actions, w=w):
                u.targets[:, ids] = w * goal
            u._pre_physics_step = intro_pre
            env.step(zero_act)
            u.episode_length_buf[:] = 0
    u._pre_physics_step = orig_pre
    # Snap to the exact reset pose at rest (the ramp ends within ~1 deg) so the
    # frozen pre-grasp starts from the same state as in training; wr03/02/01
    # renders without this drifted to another pre-grasp (forearm roll +7 vs -14).
    with torch.inference_mode():
        q = u.robot.data.joint_pos.clone(); q[:, ids] = goal
        u.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        u.targets[:, ids] = goal
        u.episode_length_buf[:] = 0
    if hasattr(u, "_filtered_actions"):
        u._filtered_actions.zero_()
obs = env.get_observations()
with torch.inference_mode():
    for step in range(args.steps):
        obs, _, _, _ = env.step(policy(obs))
        TRACE.append(u.robot.data.joint_pos.clone()); TQ.append((u.robot.data.applied_torque.clone(), u.robot.data.joint_vel.clone()))
        if args.table_diag:
            TDF.append(torch.stack([torch.linalg.vector_norm(u.scene.sensors[f"table_contact_{n}"].data.net_forces_w[:, 0], dim=-1) for n in TD_LINKS], 1))
            # non-cube (table) force on the six finger links, and hand / cube vertical velocity (bounce)
            _fs = [f"proximal_{i}" for i in (1, 2, 3)] + [f"finger_{i}" for i in (1, 2, 3)]
            TBL.append(torch.stack([(torch.linalg.vector_norm(u.scene.sensors[n].data.net_forces_w[:, 0], dim=-1)
                                     - torch.linalg.vector_norm(u.scene.sensors[n].data.force_matrix_w[:, 0, 0], dim=-1)).clamp_min(0.0) for n in _fs], 1))
            VZ.append(torch.stack((u.robot.data.body_lin_vel_w[:, u._hand_id, 2], u.object.data.root_lin_vel_w[:, 2]), 1))
        if args.raise_policy:
            tf = getattr(u, "_table_force", None)
            if tf is not None:
                chain["episode_table_force_max_n"] = max(chain.get("episode_table_force_max_n", 0.0), round(float(tf.max()), 2))
                chain.setdefault("_tf_sum", 0.0); chain["_tf_sum"] += float(tf.mean())
        if hasattr(u, "lift_done"):
            # v30: hand-over state at the moment each env's lift_done latches
            HO = globals().setdefault("HO", [None] * u.num_envs)
            for i in torch.nonzero(u.lift_done).flatten().tolist():
                if HO[i] is None:
                    o = u.scene.env_origins[i]; hq = u.robot.data.body_quat_w[i, u._hand_id]
                    tip = u.robot.data.body_pos_w[i, u.fingertip_ids].mean(dim=0)
                    HO[i] = {"t_after_handover_s": round((step + 1) / 60, 2), "episode_t_s": round(float(u.lift_done_t[i]), 2),
                             "arm_q_deg": [round(float(v), 2) for v in torch.rad2deg(u.robot.data.joint_pos[i, u.arm_ids])],
                             "finger_q_deg": [round(float(v), 2) for v in torch.rad2deg(u.robot.data.joint_pos[i, u.finger_ids])],
                             "hand_pos_m": [round(float(v), 4) for v in u.robot.data.body_pos_w[i, u._hand_id] - o],
                             "hand_quat_wxyz": [round(float(v), 4) for v in hq],
                             "fingertip_mid_pos_m": [round(float(v), 4) for v in tip - o],
                             "cube_pos_m": [round(float(v), 4) for v in u.object.data.root_pos_w[i] - o],
                             "cube_quat_wxyz": [round(float(v), 4) for v in u.object.data.root_quat_w[i]],
                             "cube_minus_fingertip_mid_m": [round(float(v), 4) for v in u.object.data.root_pos_w[i] - tip],
                             "cube_rise_m": round(float(u._cube_rise[i]), 4),
                             "pad_N": [round(float(v), 3) for v in u.pad_forces()[i]],
                             "robot_base_pos_m": [round(float(v), 4) for v in u.robot.data.root_pos_w[i] - o]}
                    if i == 0:
                        print("LIFT_DONE " + json.dumps(HO[i]), flush=True)
        if step % 60 == 59:
            q = torch.rad2deg(u.robot.data.joint_pos[:, u.arm_ids[:7]])
            hand = u.robot.data.body_pos_w[:, u._hand_id]
            d = u.robot.data.body_pos_w[:, u.fingertip_ids[:2]].mean(dim=1) - hand
            d = d / torch.linalg.vector_norm(d, dim=1, keepdim=True)
            row = {"t": round((step + 1) / 60, 1),
                   "q_deg_env0": {n: round(float(v), 1) for n, v in zip(names, q[0])},
                   "hand_dir_env0": [round(float(v), 2) for v in d[0]],
                   "cube_rise_mm": [round(1000 * float(v), 1) for v in u._cube_rise],
                   "pad_N_env0": [round(float(v), 2) for v in u.pad_forces()[0]],
                   "fingers_deg_env0": [round(float(v), 1) for v in torch.rad2deg(u.robot.data.joint_pos[0, u.finger_ids])]}
            rows.append(row)
            print("FGRASP " + json.dumps(row), flush=True)
if globals().get("HO") is not None:
    done = [h is not None for h in HO]
    chain["lift_done_frac"] = round(sum(done) / len(done), 3)
    chain["lift_done_t_after_handover_s"] = [h["t_after_handover_s"] if h else None for h in HO]
    with open(os.path.join(video_dir, "handover_state.json"), "w") as f:
        json.dump({"note": "state at lift_done (arm at the L, still, gripping, cube >= 4 cm up for 0.5 s); env-origin frame, m / deg; "
                           "arm_q_deg = 7 arm joints + RotationBody, finger_q_deg = fingers 1,2,thumb first joints then second joints, "
                           "pad_N = [finger1,finger2,thumb first links, finger1,finger2,thumb distal]", "joint_names_arm": [u.robot.joint_names[j] for j in u.arm_ids],
                   "joint_names_fingers": [u.robot.joint_names[j] for j in u.finger_ids], "envs": HO}, f, indent=1)
if args.raise_policy:
    chain["episode_table_force_mean_n"] = round(chain.pop("_tf_sum", 0.0) / args.steps, 3)
    rise = u._cube_rise
    chain["cube_rise_mm_at_end"] = [round(float(v) * 1000, 1) for v in rise]
    chain["lifted_4cm_frac"] = round(float((rise > 0.04).float().mean()), 3)
    q = torch.stack(TRACE)  # (T, N, J)
    dq = torch.rad2deg((q[1:] - q[:-1]).abs())  # deg per step
    per_step = dq.amax(dim=2).amax(dim=1)
    top = torch.topk(per_step, 8)
    chain["max_joint_step_deg"] = [[int(i) + 1, round((int(i) + 1) / 60, 2), round(float(v), 2),
                                    u.robot.joint_names[int(dq[i].amax(dim=0).argmax())]] for v, i in zip(top.values, top.indices)]
    n_raise = round(chain.get("handover_t_s", 0) * 60)
    chain["handover_step"] = n_raise
    # Smoothness per phase: largest step-to-step change of joint velocity
    # (deg/s per 1/60 s step) over all envs and joints, excluding envs that
    # reset (a step over 3 deg marks a reset teleport).
    vel = (q[1:] - q[:-1]) * 60.0
    dvel = torch.rad2deg((vel[1:] - vel[:-1]).abs()).amax(dim=2)  # (T-2, N)
    ok = ~(dq.amax(dim=2) > 3.0).any(dim=0)  # envs with no reset
    chain["envs_without_reset"] = int(ok.sum())
    phases = {"raise": (0, n_raise), "handover+0.5s": (n_raise - 2, n_raise + 30), "close 0-1s": (n_raise + 30, n_raise + 60),
              "lift 1-7s": (n_raise + 60, len(dvel))} if args.raise_to_pregrasp else {"raise": (0, n_raise), "handover+0.5s": (n_raise - 2, n_raise + 30), "pregrasp": (n_raise + 30, n_raise + 300),
              "close 5-6s": (n_raise + 300, n_raise + 360), "lift 6-12s": (n_raise + 360, len(dvel))}
    dv_j = torch.rad2deg((vel[1:] - vel[:-1]).abs())[:, ok]  # (T-2, N_ok, J)
    chain["max_dvel_where"] = {}
    for k, (a, b) in phases.items():
        if b > a and ok.any():
            w = dv_j[a:b]; flat = int(w.reshape(-1).argmax()); nj = w.shape[2]; ne = w.shape[1]
            st, rem = divmod(flat, ne * nj); _, j = divmod(rem, nj)
            chain["max_dvel_where"][k] = [round((a + st + 2) / 60, 2), u.robot.joint_names[j]]
    chain["max_dvel_deg_s_per_step"] = {k: round(float(dvel[a:b][:, ok].max()), 1) if b > a and ok.any() else None for k, (a, b) in phases.items()}
    chain["max_joint_step_deg_around_handover"] = [round(float(per_step[k]), 2) for k in range(max(n_raise - 4, 0), min(n_raise + 6, len(per_step)))]
    if TDF:
        f = torch.stack(TDF)  # (T, N, L) episode steps after the handover
        on = f > 0.5
        chain["table_diag"] = {n: {"envs_touching": int(on[:, :, i].any(0).sum()), "max_n": round(float(f[:, :, i].max()), 1),
                                   "first_s": round(float(on[:, :, i].any(1).float().argmax()) / 60, 2) if on[:, :, i].any() else None,
                                   "steps_on_env_mean": round(float(on[:, :, i].float().sum(0).mean()), 1)}
                               for i, n in enumerate(TD_LINKS)}
        sec = f.amax(dim=2)  # strongest link per step
        chain["table_force_per_0.5s_max_n"] = [round(float(sec[a:a + 30].max()), 1) for a in range(0, len(sec), 30)]
        torch.save({"links": TD_LINKS, "f": f.cpu(), "q": q.cpu(), "table_f": torch.stack(TBL).cpu(), "vz": torch.stack(VZ).cpu(), "tau": torch.stack([a for a, _ in TQ]).cpu(),
                    "qd": torch.stack([b for _, b in TQ]).cpu(), "names": u.robot.joint_names, "handover": n_raise},
                   os.path.join(video_dir, "table_diag.pt"))
    print("CHAIN " + json.dumps(chain), flush=True)
    json.dump(chain, open(os.path.join(video_dir, "chain.json"), "w"))
json.dump(rows, open(os.path.join(video_dir, "front_grasp.json"), "w"))
print("FRONT_GRASP_DIR=" + video_dir, flush=True)
env.close()
simulation_app.close()
