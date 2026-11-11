"""v27: v26_eval over the wide cube box (x -8/-3/+2 cm, y -5/0/+6 cm), exact-IK pre-grasp (R unused):
start pose check, then deterministic policy: lifted (>4 cm at the end), table force, cube xy drift."""
import sys
CK = sys.argv[1]; R = float(sys.argv[2]); sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import os, json, math, importlib.metadata, torch, gymnasium as gym
os.environ["GL_CUBE_BOX"] = "1"
import isaaclab_tasks  # noqa
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2FrontGraspLiftPPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_lift import JaxV2FrontGraspLiftEnvCfg
cfg = JaxV2FrontGraspLiftEnvCfg(); cfg.episode_length_s = 5.0 + float(os.environ.get("EVAL_S", "6.5")) + 1.0; NE = int(os.environ.get("NE", "36")); cfg.scene.num_envs = NE; cfg.sensor_randomize = bool(os.environ.get("EVAL_SENSOR_RAND"))
raw = gym.make("Isaac-JaxV2-FrontGraspLift-Direct-v16", cfg=cfg); u = raw.unwrapped
_EG = os.environ.get("EVAL_GRID", "-0.08,-0.03,0.02;-0.05,0.0,0.06")  # v49: "x0,x1,x2;y0,y1,y2" (m)
_EX = [float(v) for v in _EG.split(";")[0].split(",")]; _EY = [float(v) for v in _EG.split(";")[1].split(",")]
grid = torch.tensor([[dx, dy] for dx in _EX for dy in _EY], device=u.device)
if os.environ.get("EVAL_CELLS"):  # v65: explicit cube offsets "x,y;x,y;..." (m) instead of the 3x3 product
    grid = torch.tensor([[float(v) for v in c.split(",")] for c in os.environ["EVAL_CELLS"].split(";")], device=u.device)
NC = grid.shape[0]
u._gl_cube_off_fixed = grid[torch.arange(NE, device=u.device) % NC]
ac = handle_deprecated_rsl_rl_cfg(JaxV2FrontGraspLiftPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(raw, clip_actions=1.0)
runner = OnPolicyRunner(env, ac.to_dict(), log_dir=None, device=ac.device); runner.load(CK); pol = runner.get_inference_policy(device=u.device)
u._reset_idx(u.robot._ALL_INDICES)
obs = env.get_observations()
names = ["proximal_1", "proximal_2", "finger_1", "finger_2"]
with torch.inference_mode():
    q0 = torch.rad2deg(u.robot.data.joint_pos[0, list(u.arm_ids)])
    print("START", json.dumps({"t": round(float(u.episode_length_buf[0]) * u.step_dt, 2), "q_env0": [round(float(v), 1) for v in q0]}), flush=True)
    from isaaclab.utils.math import quat_mul, quat_conjugate, quat_apply
    def ang(q1, q0):
        return torch.rad2deg(2 * torch.acos((q1 * q0).sum(-1).abs().clamp(max=1.0)))
    cq0 = u.object.data.root_quat_w.clone(); hq0 = u.robot.data.body_quat_w[:, u._hand_id].clone()
    crot_lift = torch.zeros(NE, device=u.device); hrot_lift = torch.zeros(NE, device=u.device); crot_ret = torch.zeros(NE, device=u.device)
    rel0 = None; relmax = torch.zeros(NE, device=u.device)
    cube0 = u.object.data.root_pos_w.clone(); tmax = torch.zeros(NE, device=u.device); rmax = torch.zeros(NE, device=u.device); ret_ever = torch.zeros(NE, dtype=torch.bool, device=u.device); grip_lost = torch.zeros(NE, dtype=torch.bool, device=u.device)
    print("Z cube0_z", round(float(cube0[:, 2].mean() - u.scene.env_origins[:, 2].mean()), 4), "table_z", round(float(u.table.data.root_pos_w[:, 2].mean()), 4) if hasattr(u, "table") else None, flush=True)
    for k in range(round(float(os.environ.get("EVAL_S", "6.5")) * 60)):
        _act = pol(obs); obs, _, _, _ = env.step(_act)
        if os.environ.get("FINGER_TQ"):  # v75: per-env max |drive torque| of the 6 finger joints and the 8 arm joints
            if k == 0:
                _ftq_max = torch.zeros((NE, len(u.finger_ids)), device=u.device); _atq_max = torch.zeros((NE, len(u.arm_ids)), device=u.device)
                _ftq_hold = torch.zeros((NE, len(u.finger_ids)), device=u.device); _ftq_n = torch.zeros(NE, device=u.device)
                _asat = torch.zeros((NE, len(u.arm_ids)), device=u.device); _fsat = torch.zeros((NE, len(u.finger_ids)), device=u.device); _nsat = 0
            # v77: share of steps whose drive demand (PD torque before the PhysX clip) reaches the joint's effort limit
            _aid = list(u.arm_ids); _fid = list(u.finger_ids); _nsat += 1
            _asat += (u.robot.data.applied_torque[:, _aid].abs() >= 0.97 * u.robot.data.joint_effort_limits[:, _aid]).float()
            _fsat += (u.robot.data.applied_torque[:, _fid].abs() >= 0.97 * u.robot.data.joint_effort_limits[:, _fid]).float()
            # v77c: thumb chain (RotationBody, thumb 1st, thumb tip) - angle drift from the return start and |angle - target|
            # during the return, and when the cube (once up 4 cm) fell back, in return seconds (-1 = before the return)
            _tj = [_aid[7], _fid[2], _fid[5]]; _qn = u.robot.data.joint_pos[:, _tj]; _on = u._ret_t >= 0 if hasattr(u, "_ret_t") else torch.zeros(NE, dtype=torch.bool, device=u.device)
            if k == 0:
                _tq0 = _qn.clone(); _tdrift = torch.zeros((NE, 3), device=u.device); _terr = torch.zeros((NE, 3), device=u.device); _won = torch.zeros(NE, dtype=torch.bool, device=u.device)
                _was_up = torch.zeros(NE, dtype=torch.bool, device=u.device); _drop_rt = torch.full((NE,), -9.0, device=u.device)
            _tq0 = torch.where((_on & ~_won)[:, None], _qn, _tq0); _won = _on
            _tdrift = torch.where(_on[:, None], torch.maximum(_tdrift, (_qn - _tq0).abs()), _tdrift)
            _terr = torch.where(_on[:, None], torch.maximum(_terr, (_qn - u.targets[:, _tj]).abs()), _terr)
            _upn = (u.object.data.root_pos_w[:, 2] - cube0[:, 2]) > 0.04
            _fell = _was_up & ~_upn & (_drop_rt < -5)
            _drop_rt = torch.where(_fell, torch.where(_on, u._ret_t if hasattr(u, "_ret_t") else -1.0, torch.full_like(_drop_rt, -1.0)), _drop_rt); _was_up |= _upn
            _ft = torch.minimum(u.robot.data.applied_torque[:, list(u.finger_ids)].abs(), u.robot.data.joint_effort_limits[:, list(u.finger_ids)]); _ftq_max = torch.maximum(_ftq_max, _ft)  # PhysX clips at the effort limit
            _atq_max = torch.maximum(_atq_max, u.robot.data.applied_torque[:, list(u.arm_ids)].abs())
            _hold_now = ((u.object.data.root_pos_w[:, 2] - cube0[:, 2]) > 0.04).float()
            _ftq_hold += _ft * _hold_now[:, None]; _ftq_n += _hold_now
        tmax = torch.maximum(tmax, u._table_force)
        if k == 0:
            tsum = torch.zeros(NE, device=u.device); tsum30 = torch.zeros(NE, device=u.device)
        tsum += u._table_force
        if k < 30:
            tsum30 += u._table_force  # v80: the first 0.5 s of the policy stage (fingers closing on the table)
        rmax = torch.maximum(rmax, u.object.data.root_pos_w[:, 2] - cube0[:, 2])
        ret_ever |= (u._ret_t >= 0)
        _pf = u.pad_forces(); grip_lost |= ret_ever & ~((_pf[:, 5] >= 0.3) & ((_pf[:, 3] >= 0.3) | (_pf[:, 4] >= 0.3)))
        try:
            if k == 0:
                drop_t = torch.full((NE,), -1.0, device=u.device); pf_pre = torch.zeros((NE, 6), device=u.device)
            if hasattr(u, "lift_done"):
                _up4 = (u.object.data.root_pos_w[:, 2] - cube0[:, 2]) > 0.04
                newdrop = u.lift_done & ~_up4 & (drop_t < 0)
                drop_t = torch.where(newdrop, (k + 1) / 60.0 - (u.lift_done_t - u.cfg.close_after_s), drop_t)
                pf_pre = torch.where((u.lift_done & _up4)[:, None], _pf, pf_pre)
        except Exception as _e:
            pass
        try:
            if k == 0:
                prox_lift_sum = torch.zeros((NE, 3), device=u.device); prox_lift_n = torch.zeros(NE, device=u.device)
                prox_done = torch.full((NE, 3), -1.0, device=u.device); _done_prev = torch.zeros(NE, dtype=torch.bool, device=u.device)
            _lift_now = ((u.object.data.root_pos_w[:, 2] - cube0[:, 2]) > 0.01) & ~(u._ret_t >= 0)
            prox_lift_sum += _pf[:, :3] * _lift_now[:, None]; prox_lift_n += _lift_now.float()
            if hasattr(u, "lift_done"):
                _new = u.lift_done & ~_done_prev
                prox_done = torch.where(_new[:, None], _pf[:, :3], prox_done); _done_prev = u.lift_done.clone()
                if k == 0:
                    pads_done = torch.full((NE, 6), -1.0, device=u.device)
                pads_done = torch.where(_new[:, None], _pf, pads_done)
                if k == 0:
                    fq_done = torch.zeros((NE, 6), device=u.device); ft_done = torch.zeros((NE, 6), device=u.device)
                    _bid = {n_: u.robot.find_bodies(n_)[0][0] for n_ in ("Finger_1_1", "Finger_2_1", "Finger_3_1", "Finger_1_2", "Finger_2_2", "Finger_3_2")}
                    def _inhand():
                        _hq = u.robot.data.body_quat_w[:, u._hand_id]; _hp = u.robot.data.body_pos_w[:, u._hand_id]
                        _loc = lambda p_: quat_apply(quat_conjugate(_hq), p_ - _hp)
                        _c = _loc(u.object.data.root_pos_w)
                        _p1 = _loc(u.robot.data.body_pos_w[:, _bid["Finger_1_1"]]); _p2 = _loc(u.robot.data.body_pos_w[:, _bid["Finger_2_1"]]); _p3 = _loc(u.robot.data.body_pos_w[:, _bid["Finger_3_1"]])
                        _ax = _p1 - _p2; _L = _ax.norm(dim=1).clamp_min(1e-6)
                        _sl = ((_c - _p2) * _ax).sum(1) / (_L * _L)  # 0 at finger 2's base, 1 at finger 1's base
                        return torch.cat((_c * 1000, _p1 * 1000, _p2 * 1000, _p3 * 1000, _sl[:, None], _L[:, None] * 1000), dim=1)
                    ih_start = _inhand(); ih_done = torch.zeros_like(ih_start)
                ih_done = torch.where(_new[:, None], _inhand(), ih_done)
                fq_done = torch.where(_new[:, None], torch.rad2deg(u.robot.data.joint_pos[:, list(u.finger_ids)]), fq_done)
                ft_done = torch.where(_new[:, None], torch.rad2deg(u.targets[:, list(u.finger_ids)]), ft_done)
        except Exception as _e:
            pass
        cq = u.object.data.root_quat_w; hq = u.robot.data.body_quat_w[:, u._hand_id]
        # v79: in-hand squareness - the cube's turn about its own vertical axis away from "square to the hand" (the
        # orientation a cube with yaw_rel 0 has in the hand at the pre-grasp), folded to +/-45 deg (cube symmetry)
        try:
            if k == 0:
                from isaaclab.utils.math import quat_from_angle_axis as _qfa
                from isaaclab_tasks.direct.one_arm_jax_minimal.v2_arm_raise import grid_weights as _gw
                _tgx, _tgy, _tth = u._gl_theta
                _th = torch.deg2rad(_gw(_tgx, _tgy, u._gl_cube_off) @ _tth)
                _zax = torch.zeros((NE, 3), device=u.device); _zax[:, 2] = 1.0
                _qdef = u.object.data.default_root_state[:, 3:7]
                _rel_sq = quat_mul(quat_conjugate(hq0), quat_mul(_qfa(_th, _zax), _qdef))
                _psi0 = (torch.rad2deg(u._gl_cube_yaw - _th) + 45.0) % 90.0 - 45.0
                _cax = quat_apply(quat_conjugate(_qdef), _zax)
                _sq_lo = torch.full((NE,), 99.0, device=u.device); _sq_done = torch.full((NE,), 99.0, device=u.device)
                _sq_prev_done = torch.zeros(NE, dtype=torch.bool, device=u.device)
            _D = quat_mul(quat_conjugate(_rel_sq), quat_mul(quat_conjugate(hq), cq))
            _tw = torch.rad2deg(2.0 * torch.atan2((_D[:, 1:] * _cax).sum(1), _D[:, 0]))
            _sq = (_tw + 45.0) % 90.0 - 45.0
            _up1 = (u.object.data.root_pos_w[:, 2] - cube0[:, 2]) > 0.01
            if os.environ.get("GL_PRE_YAW") and k == round(float(os.environ["GL_PRE_YAW"].split(",")[0]) * 60) - 1:
                # v79: at the end of the hand turn - commanded turn, the hand's and the cube's turn about world z, cube shift, pads
                _w = lambda a_: torch.remainder(a_ + math.pi, 2 * math.pi) - math.pi
                _tz = lambda q_: _w(2.0 * torch.atan2(q_[:, 3], q_[:, 0]))
                _cmd = torch.rad2deg((torch.remainder(u._gl_yaw_rel + math.pi / 4, math.pi / 2) - math.pi / 4).clamp(-math.radians(45), math.radians(45)))
                _hz = torch.rad2deg(_tz(quat_mul(hq, quat_conjugate(hq0)))); _cz = torch.rad2deg(_tz(quat_mul(cq, quat_conjugate(cq0))))
                _sh = (u.object.data.root_pos_w[:, :2] - cube0[:, :2]).norm(dim=1) * 1000
                _pfs = u.pad_forces().sum(1)
                for _a, _b in ((0, 15), (15, 30), (30, 45.01)):
                    _bm = (_cmd.abs() >= _a) & (_cmd.abs() < _b)
                    if _bm.any():
                        print("PREYAW |cmd| %2d-%2d: cmd %.1f | hand turned %.1f (ratio %.2f) | cube turned %.1f | cube shift mm %.1f | pads N %.2f | sq now %.1f" % (
                            _a, int(_b), float(_cmd[_bm].abs().mean()), float((_hz[_bm] * torch.sign(_cmd[_bm])).mean()), float((_hz[_bm] / _cmd[_bm].clamp_min(1e-3).where(_cmd[_bm] > 0, _cmd[_bm].clamp_max(-1e-3))).mean()),
                            float((_cz[_bm] * torch.sign(_cmd[_bm])).mean()), float(_sh[_bm].mean()), float(_pfs[_bm].mean()), float(_sq[_bm].abs().mean())), flush=True)
            _sq_lo = torch.where(_up1 & (_sq_lo > 90), _sq, _sq_lo)
            if hasattr(u, "lift_done"):
                _bq = u.robot.data.root_quat_w
                _fq = quat_mul(quat_conjugate(_bq), quat_mul(cq, quat_conjugate(_qdef)))
                _fy = torch.rad2deg(2.0 * torch.atan2(_fq[:, 3], _fq[:, 0])); _fy = (_fy + 45.0) % 90.0 - 45.0
                if k == 0:
                    _fr_done = torch.full((NE,), 99.0, device=u.device)
                _fr_done = torch.where(u.lift_done & ~_sq_prev_done, _fy, _fr_done)
                _sq_done = torch.where(u.lift_done & ~_sq_prev_done, _sq, _sq_done); _sq_prev_done = u.lift_done.clone()
        except Exception as _e:
            if k == 0:
                print("SQUARE err", repr(_e), flush=True)
        up = (u.object.data.root_pos_w[:, 2] - cube0[:, 2]) > 0.01
        ret = (u._ret_t >= 0) if hasattr(u, "_ret_t") else torch.zeros_like(up)
        rel = quat_mul(quat_conjugate(hq), cq)  # cube orientation in the hand frame
        if rel0 is None: rel0 = rel.clone()
        rel0 = torch.where((~up)[:, None], rel, rel0)  # latch at lift-off
        lifting_now = up & ~ret
        crot_lift = torch.where(lifting_now, torch.maximum(crot_lift, ang(cq, cq0)), crot_lift)
        hrot_lift = torch.where(lifting_now, torch.maximum(hrot_lift, ang(hq, hq0)), hrot_lift)
        crot_ret = torch.where(up & ret, torch.maximum(crot_ret, ang(cq, cq0)), crot_ret)
        relmax = torch.where(up, torch.maximum(relmax, ang(rel, rel0)), relmax)
        # tilt (cube z axis vs world z) and yaw (about world z), from the start orientation
        from isaaclab.utils.math import quat_apply
        ez = torch.zeros_like(u.object.data.root_pos_w); ez[:, 2] = 1.0
        ex = torch.zeros_like(ez); ex[:, 0] = 1.0
        zc = quat_apply(cq, ez); tilt = torch.rad2deg(torch.acos(zc[:, 2].clamp(-1, 1)))
        xc = quat_apply(cq, ex); x0 = quat_apply(cq0, ex)
        yaw = torch.rad2deg(torch.atan2(xc[:, 1], xc[:, 0]) - torch.atan2(x0[:, 1], x0[:, 0]))
        yaw = (yaw + 180) % 360 - 180
        if os.environ.get("EVAL_YAW_LO"):
            # v58: yaw while lifting measured from the cube's heading at lift-off (1 cm), so squaring a turned cube up on
            # the table does not count as turning during the lift
            if k == 0:
                x_lo = x0.clone()
            x_lo = torch.where((~up)[:, None], xc, x_lo)
            yaw_l = torch.rad2deg(torch.atan2(xc[:, 1], xc[:, 0]) - torch.atan2(x_lo[:, 1], x_lo[:, 0]))
            yaw_l = (yaw_l + 180) % 360 - 180
        else:
            yaw_l = yaw
        if k == 0:
            tilt_close = torch.zeros(NE, device=u.device); yaw_close = torch.zeros(NE, device=u.device)
            tilt_lift = torch.zeros(NE, device=u.device); yaw_lift = torch.zeros(NE, device=u.device)
        tilt_close = torch.where(~up, torch.maximum(tilt_close, tilt), tilt_close); yaw_close = torch.where(~up, torch.maximum(yaw_close, yaw.abs()), yaw_close)
        tilt_lift = torch.where(lifting_now, torch.maximum(tilt_lift, tilt), tilt_lift); yaw_lift = torch.where(lifting_now, torch.maximum(yaw_lift, yaw_l.abs()), yaw_lift)
    rise = u.object.data.root_pos_w[:, 2] - cube0[:, 2]
    xy = (u.object.data.root_pos_w[:, :2] - cube0[:, :2]).norm(dim=1)
for g in range(NC):
    m = (torch.arange(NE, device=u.device) % NC) == g
    print("RES", json.dumps({"dx_cm": round(float(grid[g, 0]) * 100, 1), "dy_cm": round(float(grid[g, 1]) * 100, 1),
          "lifted": round(float((rise[m] > 0.04).float().mean()), 2), "rise_mm": round(float(rise[m].mean()) * 1000, 1),
          "table_N_max": round(float(tmax[m].max()), 2), "xy_drift_mm": round(float(xy[m].mean()) * 1000, 1),
          "table_N_envmax_mean": round(float(tmax[m].mean()), 2), "table_N_time_mean": round(float(tsum[m].mean()) / (k + 1), 2), "table_N_first05s_mean": round(float(tsum30[m].mean()) / 30.0, 2)}), flush=True)
ok = rise > 0.04
print("ROT cube_world_max_during_lift_deg %.1f hand_max_during_lift_deg %.1f cube_world_max_during_return_deg %.1f cube_in_hand_max_deg %.1f (lifted envs, mean)" % tuple(float(v[ok].mean()) if ok.any() else -1 for v in (crot_lift, hrot_lift, crot_ret, relmax)), flush=True)
print("SPLIT on_table(before 1cm): tilt %.1f yaw %.1f | lifting: tilt %.1f yaw %.1f (deg, max, lifted envs mean)" % tuple(float(v[ok].mean()) if ok.any() else -1 for v in (tilt_close, yaw_close, tilt_lift, yaw_lift)), flush=True)
c = u.cfg; pf = u.pad_forces(); arm = list(u.arm_ids)
rise_l = u.object.data.root_pos_w[:, 2] - u._lift_p0[:, 2]
arrived = u._ret_t >= c.lift_return_s
grip = (pf[:, 5] >= 0.3) & ((pf[:, 3] >= 0.3) | (pf[:, 4] >= 0.3))
print("DIAG arrived %.3f rise4 %.3f grip_distal %.3f grip_any %.3f | pad mean N %s | ret_t mean %.2f | done_cnt mean %.1f" % (
    float(arrived.float().mean()), float((rise_l >= 0.04).float().mean()), float(grip.float().mean()),
    float(((pf[:, [2, 5]].amax(1) >= 0.3) & (pf[:, [0, 1, 3, 4]].amax(1) >= 0.3)).float().mean()),
    [round(float(v), 2) for v in pf.mean(0)], float(u._ret_t.mean()), float(u._done_cnt.mean())), flush=True)
for nm, m in (("arr&rise", arrived & (rise_l >= 0.04)), ("arr&rise&grip", arrived & (rise_l >= 0.04) & grip)):
    print("DIAG", nm, round(float(m.float().mean()), 3), flush=True)
nd = ~u.lift_done
print("FAIL n_not_done %d | of them: return_started %.2f, max_rise_mm mean %.1f, rise<50mm %.2f, grip_lost_after_ret %.2f, end_rise_mm %.1f" % (int(nd.sum()), float(ret_ever[nd].float().mean()) if nd.any() else -1, float(rmax[nd].mean()) * 1000 if nd.any() else -1, float((rmax[nd] < 0.05).float().mean()) if nd.any() else -1, float(grip_lost[nd].float().mean()) if nd.any() else -1, float(rise[nd].mean()) * 1000 if nd.any() else -1), flush=True)
qa = u.robot.data.joint_pos[:, arm]; lo = u.robot.data.soft_joint_pos_limits[:, arm, 0]; hi = u.robot.data.soft_joint_pos_limits[:, arm, 1]
near = torch.minimum(qa - lo, hi - qa) < math.radians(3.0)
tg = u.targets[:, arm]; tnear = torch.minimum(tg - u.lower[arm], u.upper[arm] - tg) < math.radians(1.0)
if nd.any():
    print("STALL joints_near_limit(<3deg) frac per arm joint", [round(float(v), 2) for v in near[nd].float().mean(0)], "| targets at limit", [round(float(v), 2) for v in tnear[nd].float().mean(0)], flush=True)
    print("STALL action0(up<0) mean %.2f | lift_offset mean deg %s | ik_pdes-p z mm %.1f" % (float(_act[nd, 0].mean()), [round(math.degrees(float(v)), 1) for v in u._lift_offset[nd].mean(0)],
          float((u._ik_pdes[nd, 2] - u.robot.data.body_pos_w[nd][:, u.fingertip_ids].mean(1)[:, 2]).mean()) * 1000), flush=True)
    print("STALL q deg mean", [round(math.degrees(float(v)), 1) for v in qa[nd].mean(0)], "lo", [round(math.degrees(float(v)), 1) for v in lo[0]], "hi", [round(math.degrees(float(v)), 1) for v in hi[0]], flush=True)
for g in range(NC):
    m = (torch.arange(NE, device=u.device) % NC) == g
    print("CELL", round(float(grid[g, 0]) * 100), round(float(grid[g, 1]) * 100), "done %.2f lifted %.2f maxrise_mm %.0f" % (float(u.lift_done[m].float().mean()), float((rise[m] > 0.04).float().mean()), float(rmax[m].mean()) * 1000), flush=True)
dt_ = u.lift_done_t[u.lift_done] - c.close_after_s
print("DONE_T mean %.2f p90 %.2f max %.2f (s after close start)" % ((float(dt_.mean()), float(dt_.quantile(0.9)), float(dt_.max())) if len(dt_) else (-1, -1, -1)), flush=True)
try:
    _dm = _sq_done < 90; _lm = _sq_lo < 90
    def _sqs(v, m):
        return "n %d mean|.| %.1f p90 %.1f  <=5deg %.2f <=10deg %.2f" % (int(m.sum()), float(v[m].abs().mean()), float(v[m].abs().quantile(0.9)), float((v[m].abs() <= 5).float().mean()), float((v[m].abs() <= 10).float().mean())) if m.any() else "n 0"
    print("SQUARE start |psi0| mean %.1f | at lift-off: %s | at lift_done: %s" % (float(_psi0.abs().mean()), _sqs(_sq_lo, _lm), _sqs(_sq_done, _dm)), flush=True)
    _dd = ((_fr_done - _sq_done + 45.0) % 90.0 - 45.0)[_dm]
    print("FRONT cube turn against the robot base axes at lift_done (folded +/-45): %s | mean signed %.1f | front - in-hand: mean %.1f sd %.1f" % (
        _sqs(_fr_done, _dm), float(_fr_done[_dm].mean()) if _dm.any() else -99, float(_dd.mean()) if _dm.any() else -99, float(_dd.std()) if _dm.sum() > 1 else -1), flush=True)
    _fc = []
    for g in range(NC):
        _gm = ((torch.arange(NE, device=u.device) % NC) == g) & _dm
        if _gm.any():
            _fc.append("(%d,%d) %.1f/%.1f" % (round(float(grid[g, 0]) * 100), round(float(grid[g, 1]) * 100), float(_fr_done[_gm].abs().mean()), float(_fr_done[_gm].abs().max())))
    print("FRONT_CELL mean/max |deg|:", " ".join(_fc), flush=True)
    for _a, _b in ((0, 15), (15, 30), (30, 45.01)):
        _bm = (_psi0.abs() >= _a) & (_psi0.abs() < _b)
        print("FRONT_BIN |psi0| %2d-%2d: %s" % (_a, int(_b), _sqs(_fr_done, _dm & _bm)), flush=True)
    for _a, _b in ((0, 15), (15, 30), (30, 45.01)):
        _bm = (_psi0.abs() >= _a) & (_psi0.abs() < _b)
        print("SQUARE_BIN |psi0| %2d-%2d deg: start mean %.1f | lift-off %s | lift_done %s" % (_a, int(_b), float(_psi0[_bm].abs().mean()) if _bm.any() else -1, _sqs(_sq_lo, _lm & _bm), _sqs(_sq_done, _dm & _bm)), flush=True)
except Exception as _e:
    print("SQUARE err", repr(_e), flush=True)
print("DONE lift_done_frac %.3f" % (float(u.lift_done.float().mean()) if hasattr(u, "lift_done") else -1), flush=True)
try:
    _hm2 = u.lift_done & (rise > 0.04); _fq = torch.rad2deg(u.robot.data.joint_pos[:, list(u.finger_ids)]); _cap = torch.tensor(list(u.cfg.close_first_max_deg), device=u.device)
    _tq = torch.rad2deg(u.targets[:, list(u.finger_ids)])
    _qa = torch.rad2deg(u.robot.data.joint_pos[:, list(u.arm_ids)])
    print("FINGERQ held_end deg (f1_1, f2_1, th_1, f1_2, f2_2, th_2) q %s | target %s | first-joint caps %s | at cap (target within 0.5 deg) %s | rotation_base %.1f shoulder_roll %.1f" % (
        [round(float(v), 1) for v in _fq[_hm2].mean(0)], [round(float(v), 1) for v in _tq[_hm2].mean(0)], [round(float(v), 1) for v in _cap],
        [round(float(v), 2) for v in ((_cap - _tq[:, :3]).abs() < 0.5)[_hm2].float().mean(0)], float(_qa[_hm2, 7].mean()), float(_qa[_hm2, 1].mean())), flush=True)
except Exception as _e:
    print("FINGERQ err", repr(_e), flush=True)
print("HELD done_and_still_lifted_at_end %.3f" % float((u.lift_done & (rise > 0.04)).float().mean()), flush=True)
if os.environ.get("FINGER_TQ"):
    _q = lambda t, a: [round(float(v), 3) for v in torch.quantile(t, a, dim=0)]
    print("FINGER_TQ order (f1_1, f2_1, th_1, f1_2, f2_2, th_2) N*m | per-env max over the episode: median %s p95 %s max %s" % (_q(_ftq_max, 0.5), _q(_ftq_max, 0.95), [round(float(v), 3) for v in _ftq_max.amax(0)]), flush=True)
    _hm3 = _ftq_n > 0
    print("FINGER_TQ_HOLD mean |torque| while the cube is >4 cm up, median over envs %s p95 %s" % (_q((_ftq_hold / _ftq_n.clamp_min(1.0)[:, None])[_hm3], 0.5), _q((_ftq_hold / _ftq_n.clamp_min(1.0)[:, None])[_hm3], 0.95)), flush=True)
    print("ARM_TQ order (sh_pitch, roll, yaw, elbow, forearm, wr_pitch, wr_yaw, rotbody) per-env max: median %s max %s" % (_q(_atq_max, 0.5), [round(float(v), 3) for v in _atq_max.amax(0)]), flush=True)
    _okm = u.lift_done & (rise > 0.04); _r = lambda t, m: [round(float(v), 3) for v in (t[m] / _nsat).mean(0)] if m.any() else None
    print("ARM_LIMIT", [round(float(v), 3) for v in u.robot.data.joint_effort_limits[0, list(u.arm_ids)]], "FINGER_LIMIT", round(float(u.robot.data.joint_effort_limits[0, list(u.finger_ids)][0]), 3), flush=True)
    print("ARM_SAT share of steps at the limit, Isaac 0,1,2,3,4,5,6,9 | held envs %s | failed envs %s (n %d)" % (_r(_asat, _okm), _r(_asat, ~_okm), int((~_okm).sum())), flush=True)
    print("FINGER_SAT (f1_1, f2_1, th_1, f1_2, f2_2, th_2) | held envs %s | failed envs %s" % (_r(_fsat, _okm), _r(_fsat, ~_okm)), flush=True)
    _d = lambda t, m: [round(float(v) * 57.2958, 1) for v in t[m].mean(0)] if m.any() else None
    print("THUMB_RET (RotationBody, th_1, th_2) deg, max over the return | drift from return start: held %s failed %s | |q - target|: held %s failed %s" % (_d(_tdrift, _okm & _won), _d(_tdrift, ~_okm & _won), _d(_terr, _okm & _won), _d(_terr, ~_okm & _won)), flush=True)
    _dr = _drop_rt[_drop_rt > -5]
    print("DROP_AT n %d | before the return %d | return s: %s" % (int(len(_dr)), int((_dr < 0).sum()), [round(float(v), 2) for v in torch.quantile(_dr[_dr >= 0], torch.tensor([0.1, 0.5, 0.9], device=u.device))] if (_dr >= 0).sum() > 0 else None), flush=True)
try:
    _hm = u.lift_done & (rise > 0.04); _pp = u.pad_forces()[:, :3]
    print("PROX held_end_first_joint_N %s sum %.3f" % ([round(float(v), 3) for v in _pp[_hm].mean(0)] if _hm.any() else None, float(_pp[_hm].sum(1).mean()) if _hm.any() else 0.0), flush=True)
except Exception as _e:
    print("PROX err", repr(_e), flush=True)
try:
    _hm = u.lift_done & (rise > 0.04)
    _pl = prox_lift_sum / prox_lift_n.clamp_min(1.0)[:, None]
    print("PROX_LIFT mean_first_joint_N_while_lifting %s sum %.3f" % ([round(float(v), 3) for v in _pl[_hm].mean(0)] if _hm.any() else None, float(_pl[_hm].sum(1).mean()) if _hm.any() else 0.0), flush=True)
    _dm = prox_done[:, 0] >= 0
    print("PROX_DONE first_joint_N_at_lift_done %s sum %.3f" % ([round(float(v), 3) for v in prox_done[_dm].mean(0)] if _dm.any() else None, float(prox_done[_dm].sum(1).mean()) if _dm.any() else 0.0), flush=True)
    with open("/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad/v46_drop.log", "a") as _f:
        _f.write("PROX_LIFT %s | PROX_DONE %s\n" % ([round(float(v), 3) for v in _pl[_hm].mean(0)] if _hm.any() else None, [round(float(v), 3) for v in prox_done[_dm].mean(0)] if _dm.any() else None))
except Exception as _e:
    print("PROX_LIFT err", repr(_e), flush=True)
try:
    # v80: all six pads at lift_done (prox1, prox2, proxT, dist1, dist2, distT), N
    _pm = pads_done[:, 0] >= 0
    if _pm.any():
        _P = pads_done[_pm]
        print("PADS_DONE n %d | mean N %s | share > 0.1 N %s | share > 0.3 N %s" % (int(_pm.sum()), [round(float(v), 2) for v in _P.mean(0)], [round(float(v), 2) for v in (_P > 0.1).float().mean(0)], [round(float(v), 2) for v in (_P > 0.3).float().mean(0)]), flush=True)
        _D = _P[:, :3] + _P[:, 3:]
        print("DIGITS_DONE (f1, f2, thumb) first+distal: mean N %s | share < 0.1 N (not touching) %s | all three >= 0.3 N %.2f | links touching (>0.1 N) mean %.2f of 6" % ([round(float(v), 2) for v in _D.mean(0)], [round(float(v), 2) for v in (_D < 0.1).float().mean(0)], float((_D >= 0.3).all(1).float().mean()), float((_P > 0.1).float().sum(1).mean())), flush=True)
        _f1off = (pads_done[:, 0] + pads_done[:, 3] < 0.1) & _pm; _f1on = (pads_done[:, 0] + pads_done[:, 3] >= 0.1) & _pm
        for _nm, _mm in (("f1_not_touching", _f1off), ("f1_touching", _f1on)):
            if _mm.any():
                print("F1SPLIT %s n %d | finger q deg (f1_1, f2_1, th_1, f1_2, f2_2, th_2) %s | targets %s | f1 first at cap 30 %.2f | |psi0| mean %.1f" % (
                    _nm, int(_mm.sum()), [round(float(v), 1) for v in fq_done[_mm].mean(0)], [round(float(v), 1) for v in ft_done[_mm].mean(0)],
                    float((ft_done[_mm, 0] > 29.5).float().mean()), float(_psi0[_mm].abs().mean()) if "_psi0" in dir() else -1), flush=True)
        for _nm, _mm in (("f1_not_touching", _f1off), ("f1_touching", _f1on)):
            if _mm.any():
                _r = lambda v: [round(float(x), 1) for x in v]
                print("INHAND %s | start: cube mm %s, f1/f2/thumb base mm %s %s %s, place on f2->f1 line %.2f (base gap %.0f mm) | lift_done: cube mm %s, place %.2f" % (
                    _nm, _r(ih_start[_mm, :3].mean(0)), _r(ih_start[_mm, 3:6].mean(0)), _r(ih_start[_mm, 6:9].mean(0)), _r(ih_start[_mm, 9:12].mean(0)),
                    float(ih_start[_mm, 12].mean()), float(ih_start[_mm, 13].mean()), _r(ih_done[_mm, :3].mean(0)), float(ih_done[_mm, 12].mean())), flush=True)
        _cells = []
        for g in range(NC):
            _gm = ((torch.arange(NE, device=u.device) % NC) == g) & _pm
            if _gm.any():
                _cells.append("(%d,%d) %.2f" % (round(float(grid[g, 0]) * 100), round(float(grid[g, 1]) * 100), float(_f1off[_gm].float().mean())))
        print("F1OFF_BY_CELL share of held envs with finger 1 not touching:", " ".join(_cells), flush=True)
        if "_psi0" in dir():
            _pb = []
            for _a, _b in ((-45, -30), (-30, -15), (-15, 0), (0, 15), (15, 30), (30, 45.01)):
                _bm = (_psi0 >= _a) & (_psi0 < _b) & _pm
                if _bm.any():
                    _pb.append("[%d,%d) %.2f" % (_a, int(_b), float(_f1off[_bm].float().mean())))
            print("F1OFF_BY_PSI0 (signed, deg):", " ".join(_pb), flush=True)
except Exception as _e:
    print("PADS_DONE err", repr(_e), flush=True)
print("ALL lifted", round(float((rise > 0.04).float().mean()), 3), flush=True)
try:
    dropped = drop_t >= 0; heldm = u.lift_done & (rise > 0.04)
    a7 = u._ret_a7; a810 = u._ret_a810
    def _m(x, m): return round(float(x[m].mean()), 2) if m.any() else None
    print("DROP n %d of done %d | drop_t after done s %s | cells %s" % (int(dropped.sum()), int(u.lift_done.sum()),
          [round(float(v), 2) for v in drop_t[dropped]], [(round(float(grid[i % NC, 0]) * 100), round(float(grid[i % NC, 1]) * 100)) for i in torch.nonzero(dropped).flatten().tolist()]), flush=True)
    print("DROP squeeze actions at return start (a7 finger, a9 thumb; force 5+4a N) dropped a7 %s a9 %s | held a7 %s a9 %s" % (_m(a7, dropped), _m(a810[:, 1], dropped), _m(a7, heldm), _m(a810[:, 1], heldm)), flush=True)
    print("DROP pads N before drop (prox1,prox2,proxT,dist1,dist2,distT) dropped %s | held end %s" % ([round(float(v), 2) for v in pf_pre[dropped].mean(0)] if dropped.any() else None, [round(float(v), 2) for v in u.pad_forces()[heldm].mean(0)]), flush=True)
except Exception as _e:
    print("DROP err", repr(_e), flush=True)
try:
    with open("/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad/v46_drop.log", "a") as _f:
        _f.write("== %s held %.3f done %.3f\n" % (CK, float((u.lift_done & (rise > 0.04)).float().mean()), float(u.lift_done.float().mean())))
        _f.write("DROP n %d | t %s | cells %s\n" % (int(dropped.sum()), [round(float(v), 2) for v in drop_t[dropped]], [(round(float(grid[i % NC, 0]) * 100), round(float(grid[i % NC, 1]) * 100)) for i in torch.nonzero(dropped).flatten().tolist()]))
        _f.write("SQZ dropped a7 %s a9 %s | held a7 %s a9 %s\n" % (_m(a7, dropped), _m(a810[:, 1], dropped), _m(a7, heldm), _m(a810[:, 1], heldm)))
        _f.write("CURL held distal curl deg (60 x clamp(a, 0, 1)) fingers %s thumb %s\n" % (_m(60.0 * a810[:, 0].clamp(0, 1), heldm), _m(60.0 * a810[:, 2].clamp(0, 1), heldm)))
        _f.write("PADS dropped_before %s | held_end %s\n" % ([round(float(v), 2) for v in pf_pre[dropped].mean(0)] if dropped.any() else None, [round(float(v), 2) for v in u.pad_forces()[heldm].mean(0)]))
except Exception as _e:
    print("DROPLOG err", repr(_e), flush=True)
sys.stdout.flush()
os._exit(0)
