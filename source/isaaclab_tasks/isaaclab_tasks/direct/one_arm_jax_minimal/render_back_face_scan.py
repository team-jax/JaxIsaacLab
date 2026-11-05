"""Back-face grasp check: fingers 1/2 on the cube face away from the robot, thumb on the near face.

Fingers 1/2 bend only in one plane (no abduction), and in the v9 grasp pose their
pads already face the robot (+x). From a v9 policy's pre-lift pose this script
freezes the arm, hangs fingers 1/2 (pads toward the robot), turns RotationBody
so the thumb pad faces away from the robot, moves the hand by differential IK
(keeping its orientation) up, behind the cube and down so the finger second
links hang past the far face,
closes fingers 1/2 until their pads press, folds the thumb, then lifts.
Recorded with gym's RecordVideo from the fixed "front" evaluation camera; the
per-phase measurements go to back_face_scan.json.
"""
import argparse, json, math, os, importlib.metadata
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--policy_steps", type=int, default=400)
parser.add_argument("--thumb_fold_deg", type=float, default=90.0)
parser.add_argument("--rotation_deg", type=float, default=-178.0, help="RotationBody angle (thumb pad faces -x near the -180 deg limit)")
parser.add_argument("--finger_hang_deg", type=float, default=30.0)
parser.add_argument("--clear_m", type=float, default=0.06)
parser.add_argument("--pos_only", action="store_true", help="position-only IK (hand orientation free)")
parser.add_argument("--thumb_first", action="store_true",
                    help="fold the thumb before placing (top-down grasp: the folded thumb's back is the near-face stop)")
parser.add_argument("--finger_height_m", type=float, default=0.0, help="finger second-link origin height over the cube centre")
parser.add_argument("--behind_margin_m", type=float, default=0.015)
parser.add_argument("--tag", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2PadSensorGraspPPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_pad_sensor_grasp import JaxV2PadSensorGraspEnvCfg

checkpoint = os.path.abspath(args.checkpoint)
cfg = JaxV2PadSensorGraspEnvCfg()
cfg.scene.num_envs = 1
cfg.sensor_randomize = False
cfg.episode_length_s = 100.0
cfg.forced_lift_time_s = 1000.0
cfg.auto_lift_after_s = 1000.0
cfg.cube_move_limit_m = 10.0
cfg.cube_rotation_limit_deg = 1000.0
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.eye = (-0.78, 0.08, 1.04)
cfg.viewer.lookat = (-0.29, 0.08, 0.80)
cfg.viewer.resolution = (960, 540)
raw_env = gym.make("Isaac-JaxV2-PadSensorGrasp-Direct-v9", cfg=cfg, render_mode="rgb_array")
video_dir = os.path.join(
    os.path.dirname(checkpoint), "videos",
    f"back_face_scan_front_{os.path.splitext(os.path.basename(checkpoint))[0]}{('_' + args.tag) if args.tag else ''}",
)
os.makedirs(video_dir, exist_ok=True)
video_env = gym.wrappers.RecordVideo(
    raw_env, video_folder=video_dir, step_trigger=lambda step: step == 0,
    video_length=args.policy_steps + 2400, disable_logger=True,
)
agent_cfg = handle_deprecated_rsl_rl_cfg(JaxV2PadSensorGraspPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(video_env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

u = raw_env.unwrapped
names = u.robot.joint_names
bodies = u.robot.body_names
hand_id = bodies.index("MG996R")
jac_body = hand_id - 1  # fixed-base articulation: the root body has no Jacobian row
arm_ids = list(u.arm_ids[:7])  # exclude RotationBody (thumb opposition)
rot_id = u.arm_ids[7]
f1_id, f2_id, thumb_id = u.finger_ids[:3]
f1_tip, f2_tip, thumb_tip = u.fingertip_ids
thumb_prox = u.proximal_ids[2]
half = cfg.object_size[0] / 2
dev = u.device
X = torch.tensor([1.0, 0.0, 0.0], device=dev)
frozen = None
orig_apply = u._apply_action


def apply_frozen():
    if frozen is not None:
        u.targets[:] = frozen
    orig_apply()


u._apply_action = apply_frozen
zero = torch.zeros_like(u.actions)
dt = u.step_dt
log = {"checkpoint": checkpoint, "phases": []}


def body_local(b):
    """Body position in the cube frame (cube x points away from the robot when unrotated)."""
    cube_p, cube_q = u.object.data.root_pos_w, u.object.data.root_quat_w
    return quat_apply_inverse(cube_q, u.robot.data.body_pos_w[:, b] - cube_p)[0]


def pad_dir(b):
    return quat_apply(u.robot.data.body_quat_w[0, b], X)


def snapshot(label):
    pad_n = u.pad_forces()[0]
    cube = u.object.data.root_pos_w[0] - u.scene.env_origins[0]
    row = {"phase": label,
           "cube_xyz_mm": [round(1000 * float(v), 1) for v in cube],
           "cube_shift_mm": round(1000 * float(torch.linalg.vector_norm(cube[:2] - cube0[:2])), 1),
           "cube_rise_mm": round(1000 * float(cube[2] - cube0[2]), 1),
           "pad_N_first_123_second_123": [round(float(v), 2) for v in pad_n],
           "contact_norm_N_first_123_second_123": [round(float(torch.linalg.vector_norm(
               u.scene.sensors[n].data.force_matrix_w[0, 0, 0])), 2) for n in
               ("proximal_1", "proximal_2", "proximal_3", "finger_1", "finger_2", "finger_3")],
           "f1_2nd_local_mm": [round(1000 * float(v), 1) for v in body_local(f1_tip)],
           "f2_2nd_local_mm": [round(1000 * float(v), 1) for v in body_local(f2_tip)],
           "thumb_2nd_local_mm": [round(1000 * float(v), 1) for v in body_local(thumb_tip)],
           "thumb_1st_local_mm": [round(1000 * float(v), 1) for v in body_local(thumb_prox)],
           "pad_dir_f1_f2_thumb": [[round(float(v), 2) for v in pad_dir(b)] for b in (f1_tip, f2_tip, thumb_tip)],
           "q_deg": {n: round(math.degrees(float(u.robot.data.joint_pos[0, i])), 1)
                     for i, n in enumerate(names) if i in list(u.finger_ids) + [rot_id]}}
    log["phases"].append(row)
    print("SCAN " + json.dumps(row), flush=True)


def move_hand(delta, seconds):
    """Translate the hand by delta (world, m) at constant speed, keeping its orientation."""
    steps = max(1, int(round(seconds / dt)))
    twist = torch.zeros((1, 6), device=dev)
    twist[0, :3] = torch.as_tensor(delta, device=dev) / (steps * dt)
    for _ in range(steps):
        jac = u.robot.root_physx_view.get_jacobians()[:, jac_body, :, :][:, :, arm_ids]
        rows = 3 if args.pos_only else 6
        dq = torch.linalg.lstsq(jac[:, :rows], (twist[:, :rows] * dt)[:, :, None]).solution[:, :, 0]
        frozen[:, arm_ids] = frozen[:, arm_ids] + dq
        env.step(zero)


def ramp(joint, target_deg, seconds):
    steps = max(1, int(round(seconds / dt)))
    start = float(frozen[0, joint])
    for k in range(steps):
        frozen[0, joint] = start + (math.radians(target_deg) - start) * (k + 1) / steps
        env.step(zero)


def hold(seconds):
    for _ in range(int(round(seconds / dt))):
        env.step(zero)


obs = env.get_observations()
with torch.inference_mode():
    for _ in range(args.policy_steps):
        obs, _, _, _ = env.step(policy(obs))
    cube0 = (u.object.data.root_pos_w[0] - u.scene.env_origins[0]).clone()
    frozen = u.targets.clone()
    snapshot("policy_pose")
    # Turning the thumb to the far side sweeps it through the cube: park the
    # cube aside while the hand is placed, then put it back at its pose.
    cube_pose0 = u.object.data.root_state_w[:, :7].clone()
    parked = cube_pose0.clone()
    parked[:, 0] -= 0.4
    zero_vel = torch.zeros((1, 6), device=dev)
    parking = True
    orig_step = env.step

    def step_parked(a):
        if parking:
            u.object.write_root_pose_to_sim(parked)
            u.object.write_root_velocity_to_sim(zero_vel)
        return orig_step(a)

    env.step = step_parked
    # Fingers 1/2 at about 45 deg hang down with the pad toward the robot (+x);
    # at <= 0 deg they point forward, pads down. Hang them slightly open, open
    # the thumb, lift and turn the thumb to face away from the robot (-x).
    for j in (f1_id, f2_id):
        frozen[0, j] = math.radians(args.finger_hang_deg)
    frozen[0, thumb_id] = math.radians(args.thumb_fold_deg if args.thumb_first else -30.0)
    hold(0.4)
    move_hand((0.0, 0.0, args.clear_m), 1.2)
    ramp(rot_id, args.rotation_deg, 1.0)
    hold(0.3)

    # Place: finger second links just past the far (-x) face, centred in y, at cube height.
    cube_w = cube_pose0[0, :3]
    for _ in range(3):
        tips = u.robot.data.body_pos_w[0, [f1_tip, f2_tip]]
        delta = (float(cube_w[0] - half - args.behind_margin_m - tips[:, 0].max()),
                 float(cube_w[1] - tips[:, 1].mean()),
                 float(cube_w[2] + args.finger_height_m - tips[:, 2].mean()))
        move_hand(delta, 1.5)
        hold(0.3)
    p = u.robot.data.body_pos_w[0]
    log["placed_world_offsets_mm"] = {
        name: [round(1000 * float(v), 1) for v in (p[b] - cube_w)]
        for name, b in (("f1_2nd", f1_tip), ("f2_2nd", f2_tip), ("thumb_1st", thumb_prox), ("thumb_2nd", thumb_tip))
    }
    lim = u.robot.data.joint_pos_limits[0]
    log["placed_arm_q_deg"] = {names[i]: [round(math.degrees(float(v)), 1) for v in
                                          (u.robot.data.joint_pos[0, i], frozen[0, i], lim[i, 0], lim[i, 1])] for i in arm_ids}
    print("PLACED " + json.dumps(log["placed_world_offsets_mm"]) + " ARM(q,target,lo,hi) " + json.dumps(log["placed_arm_q_deg"]), flush=True)
    parking = False
    if args.thumb_first:
        # Put the cube midway between the finger second links and the folded
        # thumb (x), centred on the fingers (y), on the table (z).
        fingers = 0.5 * (p[f1_tip] + p[f2_tip])
        cube_pose0[0, 0] = 0.5 * (fingers[0] + p[thumb_tip, 0])
        cube_pose0[0, 1] = fingers[1]
    u.object.write_root_pose_to_sim(cube_pose0)
    u.object.write_root_velocity_to_sim(zero_vel)
    hold(0.5)
    snapshot("cube_restored")

    # Close fingers 1/2 until each pad presses (>= 1 N on either link), then fold the thumb.
    for k in range(240):
        pad_n = u.pad_forces()[0]
        for i, j in ((0, f1_id), (1, f2_id)):
            if max(float(pad_n[i]), float(pad_n[3 + i])) < 1.0:
                frozen[0, j] = min(float(frozen[0, j]) + math.radians(0.5), math.radians(90.0))
        env.step(zero)
    hold(0.3)
    snapshot("fingers_closed")
    ramp(thumb_id, args.thumb_fold_deg, 1.5)
    hold(0.5)
    snapshot("thumb_folded")
    for j in (f1_id, f2_id):
        frozen[0, j] = float(frozen[0, j]) + math.radians(5.0)
    hold(0.5)
    snapshot("squeezed")
    move_hand((0.0, 0.0, 0.04), 2.0)
    hold(2.0)
    snapshot("lifted")
json.dump(log, open(os.path.join(video_dir, "back_face_scan.json"), "w"))
print("BACK_FACE_DIR=" + video_dir, flush=True)
env.close()
simulation_app.close()
