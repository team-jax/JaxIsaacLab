"""v63: the chain's reach-in alone, headless (no camera, no policy), to see what the open hand hits on the turned cube.
The arm starts at the end of the back-off raise, i.e. the geometric pre-grasp at lookup (dx + PROBE_START_DX, dy)
(GL_REACH_DX = PROBE_START_DX places it, roll 0), next to the 50 mm cube turned by a per-env yaw; then, as in
render_front_grasp.py, the targets move by dq_in = S5(dx - 0.025, dy) - S5(dx + PROBE_START_DX, dy) plus the 12 deg
roll on the wrist-yaw joint over 0.5 s (smoothstep) and hold 0.25 s. Per env: max force per pad (first link and distal
of each digit) and when it starts, cube shift (mm) and turn (deg) at the end.
Switches: PROBE_START_DX (m, default 0.02), PROBE_ROLL_AT_START=1 (the 12 deg roll already applied at the start, not
during the reach-in), PROBE_REACH_S (s, default 0.5), PROBE_XS / PROBE_YS (cm lists), PROBE_YAWS (deg list)."""
import sys
sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import os, json, math, torch, gymnasium as gym
START_DX = float(os.environ.get("PROBE_START_DX", "0.02"))
START_DY = float(os.environ.get("PROBE_START_DY", "0"))  # sideways start: the reach-in then moves diagonally
END_DY = float(os.environ.get("PROBE_END_DY", "0"))  # sideways end (the pre-grasp the policy starts from)
ROLL_AT_START = bool(os.environ.get("PROBE_ROLL_AT_START"))
ROLL = math.radians(float(os.environ.get("PROBE_ROLL_DEG", "12")))  # diagnosis only: the policy starts rolled 12 deg
os.environ["GL_CUBE_BOX"] = "1"
os.environ["GL_REACH_DX"] = str(START_DX)
os.environ["GL_REACH_DY"] = str(START_DY)
os.environ["GL_ROLL_DEG"] = str(math.degrees(ROLL)) if ROLL_AT_START else "0"
os.environ["GL_START_NOISE_DEG"] = "0"
for k in ("GL_CUBE_YAW_DEG", "GL_CUBE_YAW_FIXED_DEG", "GL_OBS_CUBE_NOISE"):
    os.environ.pop(k, None)
import isaaclab_tasks  # noqa
from isaaclab.utils.math import quat_from_angle_axis, quat_mul
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_lift import JaxV2FrontGraspLiftEnvCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_arm_raise import grid_weights
XS = [float(v) / 100 for v in os.environ.get("PROBE_XS", "-14 -6 2").split()]
YS = [float(v) / 100 for v in os.environ.get("PROBE_YS", "6 -2 -10").split()]
YAWS = [float(v) for v in os.environ.get("PROBE_YAWS", "-45 -35 -20 -10 0 10 20 35 45").split()]
cases = [(dx, dy, yw) for dx in XS for dy in YS for yw in YAWS]
NE = len(cases)
cfg = JaxV2FrontGraspLiftEnvCfg(); cfg.episode_length_s = 20.0; cfg.scene.num_envs = NE
LINKS = ("Under_Arm", "Wrist_Pitch", "Wrist_yaw", "MG996R", "RotationBody", "Finger_1_1", "Finger_2_1", "Finger_3_1", "Finger_1_2", "Finger_2_2", "Finger_3_2")
cfg.diagnostic_cube_contacts = True; cfg.cube_contact_body_names = LINKS
raw = gym.make("Isaac-JaxV2-FrontGraspLift-Direct-v16", cfg=cfg); u = raw.unwrapped
u._gl_cube_off_fixed = torch.tensor([[c[0], c[1]] for c in cases], device=u.device)
u._reset_idx(u.robot._ALL_INDICES)
arm = list(u.arm_ids); fids = list(u.finger_ids)
sens = [f"proximal_{i}" for i in (1, 2, 3)] + [f"finger_{i}" for i in (1, 2, 3)]
labels = ["f1_first", "f2_first", "thumb_first", "f1_distal", "f2_distal", "thumb_distal"]
with torch.inference_mode():
    # turn each cube where it stands (as the chain's reset does), then let it settle one step with the arm held
    yaw = torch.tensor([math.radians(c[2]) for c in cases], device=u.device)
    zax = torch.tensor([0.0, 0.0, 1.0], device=u.device).expand(NE, -1)
    cs = u.object.data.root_state_w.clone()
    cs[:, 3:7] = quat_mul(quat_from_angle_axis(yaw, zax), u.object.data.default_root_state[:, 3:7]); cs[:, 7:] = 0.0
    u.object.write_root_pose_to_sim(cs[:, :7]); u.object.write_root_velocity_to_sim(cs[:, 7:])
    c_spawn = cs[:, :3].clone()
    gx, gy, G = u._gl_grid
    off = u._gl_cube_off.clone()
    W1 = grid_weights(gx, gy, off + off.new_tensor([-0.025, END_DY])); W0 = grid_weights(gx, gy, off + off.new_tensor([START_DX, START_DY]))
    dq_in = W1 @ G - W0 @ G
    if not ROLL_AT_START:
        dq_in[:, 6] += ROLL
    base_tg = u.targets[:, arm].clone(); fg = u.targets[:, fids].clone()
    zero = torch.zeros(NE, u.single_action_space.shape[0], device=u.device)
    hold = lambda actions: None
    def put(w):
        def pre(actions):
            u.targets[:, arm] = torch.minimum(torch.maximum(base_tg + w * dq_in, u.lower[arm]), u.upper[arm])
            u.targets[:, fids] = fg
        return pre
    # settle 0.25 s at the start pose (the chain hands over a still arm), then the reach-in and a 0.25 s hold
    for k in range(15):
        u._pre_physics_step = put(0.0); u.step(zero); u.episode_length_buf[:] = 0
    c0 = u.object.data.root_pos_w.clone(); q0 = u.object.data.root_quat_w.clone()
    settle_sh = (c0 - c_spawn)[:, :2].norm(dim=1) * 1000
    lid = [u.robot.body_names.index(n) for n in LINKS]
    rel0 = (u.robot.data.body_pos_w[:, lid] - c0[:, None]) * 1000
    n_in = round(float(os.environ.get("PROBE_REACH_S", "0.5")) * 60)
    fmax = torch.zeros(NE, 6, device=u.device); t_first = torch.full((NE,), -1.0, device=u.device)
    hq_s = u.robot.data.body_quat_w[:, u._hand_id].clone(); resets = 0
    for k in range(n_in + 15):
        x = min((k + 1) / n_in, 1.0)
        u._pre_physics_step = put(x * x * (3.0 - 2.0 * x)); u.step(zero); u.episode_length_buf[:] = 0
        resets += int(u.reset_buf.sum()) if hasattr(u, "reset_buf") else 0
        f = torch.stack([torch.linalg.vector_norm(u.scene.sensors[n].data.net_forces_w[:, 0], dim=-1) for n in sens], 1)
        fmax = torch.maximum(fmax, f)
        lc = torch.stack([torch.linalg.vector_norm(u.scene.sensors[f"cube_contact_{n}"].data.force_matrix_w[:, 0, 0], dim=-1) for n in LINKS], 1)
        lmax = lc if k == 0 else torch.maximum(lmax, lc)
        lfirst = torch.full((NE, len(LINKS)), -1.0, device=u.device) if k == 0 else lfirst
        lfirst = torch.where((lfirst < 0) & (lc > 0.2), torch.full_like(lfirst, (k + 1) / 60.0), lfirst)
        newly = (t_first < 0) & (f.sum(1) > 0.3)
        t_first = torch.where(newly, torch.full_like(t_first, (k + 1) / 60.0), t_first)
    sh = (u.object.data.root_pos_w - c0)[:, :2].norm(dim=1) * 1000
    turn = torch.rad2deg(2 * torch.acos((u.object.data.root_quat_w * q0).sum(-1).abs().clamp(max=1.0)))
    hq_e = u.robot.data.body_quat_w[:, u._hand_id]
    def yaw_of(q):  # heading of the hand's x axis in the world xy plane (deg)
        w_, x_, y_, z_ = q.unbind(-1)
        return torch.rad2deg(torch.atan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_)))
    tip = (u.robot.data.body_pos_w[:, list(u.fingertip_ids)].mean(1) - u.object.data.root_pos_w)
    print("PROBE", json.dumps({"start_dx": START_DX, "start_dy": START_DY, "end_dy": END_DY, "roll_deg": round(math.degrees(ROLL), 1), "roll_at_start": ROLL_AT_START, "reach_s": n_in / 60, "resets": resets}), flush=True)
    for i, (dx, dy, yw) in enumerate(cases):
        top = int(fmax[i].argmax())
        print("CASE x%+.0f y%+.0f yaw%+.0f" % (dx * 100, dy * 100, yw),
              "settle_shift_mm %.1f shift_mm %.1f turn_deg %.1f" % (float(settle_sh[i]), float(sh[i]), float(turn[i])),
              "max_pad_N %.2f (%s)" % (float(fmax[i, top]), labels[top] if float(fmax[i, top]) > 0.05 else "-"),
              "first_touch_s %.2f" % float(t_first[i]),
              "hand_yaw_start %.1f end %.1f" % (float(yaw_of(hq_s[i])), float(yaw_of(hq_e[i]))),
              "tipmid_minus_cube_mm %s" % [round(float(v) * 1000) for v in tip[i]],
              "cube_links %s" % {LINKS[j]: [round(float(lmax[i, j]), 2), round(float(lfirst[i, j]), 2)] for j in range(len(LINKS)) if float(lmax[i, j]) > 0.05}, flush=True)
        if yw == 0.0:
            print("LINKS_AT_START x%+.0f y%+.0f (link origin - cube centre, world axes, mm)" % (dx * 100, dy * 100), {LINKS[j]: [round(float(v)) for v in rel0[i, j]] for j in range(len(LINKS))}, flush=True)
    for dx in XS:
        for dy in YS:
            m = torch.tensor([(c[0] == dx and c[1] == dy) for c in cases], device=u.device)
            print("SPOT x%+.0f y%+.0f" % (dx * 100, dy * 100), "shift_mm max %.1f mean %.1f" % (float(sh[m].max()), float(sh[m].mean())),
                  "settle max %.1f" % float(settle_sh[m].max()), "pad_N max %.2f" % float(fmax[m].max()), "touched %d/%d" % (int((fmax[m].amax(1) > 0.3).sum()), int(m.sum())), flush=True)
    print("ALL shift_mm max %.1f | pad>0.3N %d/%d" % (float(sh.max()), int((fmax.amax(1) > 0.3).sum()), NE), flush=True)
sys.stdout.flush()
os._exit(0)  # app.close() hung once at shutdown (next queued run never started)
