"""Show candidate hand shapes for the two-fingers-one-face / thumb-opposite-face grasp.

The arm is held in the L pose (hand in the air) and the hand joints move through
three shapes, ~3 s each: the v12 shape (RotationBody -90, nothing opposes the
fingers), shape A (thumb pad facing the finger pads) and shape B (folded
thumb's back facing the finger second links). During each hold the 5 cm cube is
placed kinematically midway between the finger pads and the thumb surface,
turned so its faces meet them, to show where it would sit. Recorded with gym's
RecordVideo; the camera follows the hand. Shapes and timings go to shapes.json.
"""
import argparse, json, math, os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", required=True)
parser.add_argument("--view", choices=("front", "side", "top"), default="front")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_apply, quat_from_matrix
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_pregrasp import JaxV2FrontPregraspEnvCfg

# (label, RotationBody, thumb first, thumb second, fingers 1/2 first; finger link, thumb link, thumb side)
SHAPES = [
    ("now_rot-90", -90, 0, 0, 0, "Finger_1_2", "Finger_3_2", 1.0),
    ("A_thumb_pad", -157, -15, 0, 15, "Finger_1_1", "Finger_3_1", 1.0),
    ("B_thumb_back", -37, 90, 0, 60, "Finger_1_2", "Finger_3_2", -1.0),
]
MOVE_S, HOLD_S = 1.0, 2.5

cfg = JaxV2FrontPregraspEnvCfg()
cfg.scene.num_envs = 1
cfg.episode_length_s = 100.0
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.resolution = (960, 540)
raw = gym.make("Isaac-JaxV2-FrontPregrasp-Direct-v12", cfg=cfg, render_mode="rgb_array")
u = raw.unwrapped
dt = u.step_dt
move, hold = round(MOVE_S / dt), round(HOLD_S / dt)
total = len(SHAPES) * (move + hold) + 30
os.makedirs(args.out_dir, exist_ok=True)
env = gym.wrappers.RecordVideo(raw, video_folder=args.out_dir, step_trigger=lambda s: s == 0,
                               video_length=total, disable_logger=True)
env.reset()
R = u.robot
B = R.body_names
dev = u.device
f = u.finger_ids
zero_d = torch.zeros(1, dtype=torch.bool, device=dev)
u._get_dones = lambda: (zero_d, zero_d)
q = R.data.joint_pos.clone()
orig = u._apply_action


def ap():
    u.targets[:] = q
    orig()


u._apply_action = ap
X = torch.tensor([[1.0, 0.0, 0.0]], device=dev)
zero_a = torch.zeros_like(u.actions)
zv = torch.zeros((1, 6), device=dev)
o = u.scene.env_origins[0]


def pad_point(link):
    """First link: midpoint to the second link's origin; second link: half a link past its origin."""
    p = R.data.body_pos_w[0, B.index(link)]
    if link.endswith("_1"):
        return 0.5 * (p + R.data.body_pos_w[0, B.index(link[:-1] + "2")])
    return p + 0.5 * (p - R.data.body_pos_w[0, B.index(link[:-1] + "1")])


def set_camera(target):
    t = (target - o).tolist()
    offset = {"front": (-0.22, 0.05, 0.10), "side": (0.0, 0.30, 0.08), "top": (-0.05, 0.02, 0.30)}[args.view]
    eye = [t[i] + offset[i] for i in range(3)]
    u.viewport_camera_controller.update_view_location(eye=eye, lookat=t)


log = []
with torch.inference_mode():
    parked = u.object.data.root_state_w[:, :7].clone()
    for label, rot, th1, th2, fq, flink, tlink, tsign in SHAPES:
        start = q.clone()
        goal = q.clone()
        goal[0, u.arm_ids[7]] = math.radians(rot)
        goal[0, f[2]] = math.radians(th1)
        goal[0, f[5]] = math.radians(th2)
        goal[0, f[0]] = goal[0, f[1]] = math.radians(fq)
        goal = torch.minimum(torch.maximum(goal, u.lower), u.upper)
        for k in range(move):
            q[:] = start + (goal - start) * (k + 1) / move
            u.object.write_root_pose_to_sim(parked)
            u.object.write_root_velocity_to_sim(zv)
            env.step(zero_a)
            set_camera(R.data.body_pos_w[0, B.index("MG996R")])
        for k in range(hold):
            f1 = pad_point(flink)
            f2 = pad_point(flink.replace("1_", "2_", 1))
            nf = quat_apply(R.data.body_quat_w[0:1, B.index(flink)], X)[0] + quat_apply(
                R.data.body_quat_w[0:1, B.index(flink.replace("1_", "2_", 1))], X)[0]
            nf = nf / nf.norm()
            pt = pad_point(tlink)
            mid = 0.5 * (0.5 * (f1 + f2) + pt)
            # Cube frame: y along the grasp axis, z as close to world up as possible.
            up = torch.tensor([0.0, 0.0, 1.0], device=dev)
            zc = up - (up @ nf) * nf
            zc = zc / zc.norm().clamp_min(1e-6) if zc.norm() > 1e-3 else torch.tensor([1.0, 0, 0], device=dev)
            xc = torch.linalg.cross(nf, zc)
            quat = quat_from_matrix(torch.stack((xc, nf, zc), dim=1)[None])
            pose = torch.cat((mid[None], quat), dim=1)
            u.object.write_root_pose_to_sim(pose)
            u.object.write_root_velocity_to_sim(zv)
            env.step(zero_a)
            set_camera(mid)
        tn = quat_apply(R.data.body_quat_w[0:1, B.index(tlink)], X)[0] * tsign
        log.append({"shape": label, "RotationBody_deg": rot, "thumb_first_deg": th1, "thumb_second_deg": th2,
                    "fingers_first_deg": fq, "thumb_surface": "pad" if tsign > 0 else "back",
                    "oppose": round(float(tn @ nf), 2),
                    "gap_mm": round(1000 * float((pt - 0.5 * (f1 + f2)) @ nf), 1),
                    "start_s": round(len(log) * (MOVE_S + HOLD_S), 1)})
        print("SHAPE " + json.dumps(log[-1]), flush=True)
    for _ in range(30):
        env.step(zero_a)
json.dump(log, open(os.path.join(args.out_dir, "shapes.json"), "w"))
print("SHAPES_DIR=" + args.out_dir, flush=True)
env.close()
simulation_app.close()
