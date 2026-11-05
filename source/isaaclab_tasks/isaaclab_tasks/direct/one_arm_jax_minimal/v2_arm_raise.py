"""v17: raise the arm from hanging into the v16 pre-grasp start pose (the L).

User (2026-10-29): one continuous motion from the hanging arm through the L
into the v16 pre-grasp, grasp and lift, with no state reset in between. The
v10 L-raise never reached its pose (l_success 0 in all eight runs) and aimed at
another L (shoulder yaw -30 deg) and cube placement, so this stage is trained
in the v16 scene for the v16 start pose: arm (0, 0, 0, 90, 0, 0, 0) deg and
RotationBody -155 deg, at rest.

Bending only the elbow swings the hand on a ~0.4 m arc about the elbow
(z 0.898 m); it enters the table's x range (x < -0.34 m) at about z 0.69 m,
below the 0.74 m top, so the fingers would hit the table edge. Hand points
inside a keep-out box over the table are penalized, and the policy uses
shoulder and wrist to clear it.

Control: filtered delta targets on the seven arm joints and RotationBody
(actions 0:8). The thumb first joint follows a scripted ramp to its pre-grasp
-30 deg; the other finger joints stay at 0. Observation: the 43-dim v9
observation, RotationBody unmasked.
"""

from __future__ import annotations

import math
import os
import struct
import xml.etree.ElementTree as ET

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from .v2_front_grasp_lift import JaxV2FrontGraspLiftEnvCfg
from .v2_front_pregrasp import JaxV2FrontPregraspEnv, JaxV2FrontPregraspEnvCfg


def _add_deg(waypoints, offset_deg):
    return tuple((w[0], tuple(a + b for a, b in zip(w[1], offset_deg))) + tuple(w[2:]) for w in waypoints)


_REF_CACHE = {}


def grid_weights(gx: torch.Tensor, gy: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Bilinear weights (n, nx * ny) of points xy (n, 2) on the uniform grid gx x gy (clamped to it)."""
    def axis(g, v):
        v = v.clamp(float(g[0]), float(g[-1]))
        f = (v - g[0]) / (g[1] - g[0])
        i = f.floor().clamp(0, len(g) - 2).long()
        w = torch.zeros((len(v), len(g)), device=v.device)
        w.scatter_(1, i[:, None], (1.0 - (f - i))[:, None]); w.scatter_add_(1, i[:, None] + 1, (f - i)[:, None])
        return w
    return (axis(gx, xy[:, 0])[:, :, None] * axis(gy, xy[:, 1])[:, None, :]).reshape(len(xy), -1)


def pchip_eval(ts: torch.Tensor, Q: torch.Tensor, t: torch.Tensor, catmull: bool = False, stops=None) -> torch.Tensor:
    """Cubic Hermite through knots (ts (K,), Q (K, D)) at t (N,), velocity continuous through the
    inner knots and zero at the first and last. Tangents: monotone PCHIP (Fritsch-Carlson, no
    overshoot, but a joint that reverses stops at that knot) or Catmull-Rom (catmull=True: no stop
    at a reversal, small overshoot)."""
    h = ts[1:] - ts[:-1]
    d = (Q[1:] - Q[:-1]) / h[:, None]
    m = torch.zeros_like(Q)
    if len(ts) > 2 and catmull:
        m[1:-1] = (Q[2:] - Q[:-2]) / (ts[2:] - ts[:-2])[:, None]
    elif len(ts) > 2:
        w1 = (2 * h[1:] + h[:-1])[:, None]; w2 = (h[1:] + 2 * h[:-1])[:, None]
        same = (d[:-1] * d[1:]) > 0
        m[1:-1] = torch.where(same, (w1 + w2) / (w1 / d[:-1].where(same, torch.ones_like(d[:-1]))
                                                  + w2 / d[1:].where(same, torch.ones_like(d[1:]))), torch.zeros_like(d[1:]))
    if stops is not None:
        m[stops] = 0.0  # knots marked "stop": the arm comes to rest there
    tc = t.clamp(ts[0], ts[-1])
    k = (torch.bucketize(tc, ts, right=True) - 1).clamp(0, len(ts) - 2)
    hk = h[k][:, None]; x = ((tc - ts[k]) / h[k])[:, None]
    h00 = 2 * x**3 - 3 * x**2 + 1; h10 = x**3 - 2 * x**2 + x; h01 = -2 * x**3 + 3 * x**2; h11 = x**3 - x**2
    return h00 * Q[k] + h10 * hk * m[k] + h01 * Q[k + 1] + h11 * hk * m[k + 1]


class _TrackCfg:
    """Reference view of a cfg with its raise_track_waypoints as the path."""

    def __init__(self, c):
        self.raise_waypoints = c.raise_track_waypoints
        self.raise_ref_path = ""
        self.raise_nominal_s = c.raise_nominal_s
        self.raise_interp = getattr(c, "raise_interp", "smoothstep")
        self.raise_interp_catmull = getattr(c, "raise_interp_catmull", False)
        self.raise_prefix_ref_path = getattr(c, "raise_prefix_ref_path", "")
        self.raise_prefix_sag_deg = (0.0,) * 8  # tracked path: actual (geometric) poses


def raise_nominal(cfg, t: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Scripted nominal path (N, 8) at episode time t (N,): smoothstep from
    hanging to the goal over raise_nominal_s, or through raise_waypoints
    ((time_s, 8 joint deg), ...), smoothstep within each segment."""
    ref_path = getattr(cfg, "raise_ref_path", "")
    if ref_path:
        # recorded reference (T, 8) at 60 Hz; held at its last pose
        if ref_path not in _REF_CACHE:
            _REF_CACHE[ref_path] = torch.load(os.path.join(os.path.dirname(__file__), ref_path)).float()
        ref = _REF_CACHE[ref_path].to(goal.device)
        i = (t * 60.0).round().long().clamp(0, len(ref) - 1)
        return ref[i]
    wps = getattr(cfg, "raise_waypoints", None)
    if not wps:
        x = (t / cfg.raise_nominal_s).clamp(0.0, 1.0)[:, None]
        return x * x * (3.0 - 2.0 * x) * goal
    if getattr(cfg, "raise_interp", "smoothstep") == "pchip":
        prefix = getattr(cfg, "raise_prefix_ref_path", "")
        knots = list(wps) if prefix else [(0.0, (0.0,) * 8)] + list(wps)
        ts = goal.new_tensor([w[0] for w in knots])
        Q = torch.stack([goal.new_tensor([math.radians(v) for v in w[1]]) for w in knots])
        stops = torch.tensor([len(w) > 2 and w[2] == "stop" for w in knots], device=goal.device)
        out = pchip_eval(ts, Q, t, catmull=getattr(cfg, "raise_interp_catmull", False), stops=stops)
        if prefix:
            # dense recorded path (60 Hz) up to the first knot, plus its sag offsets ramped in
            if prefix not in _REF_CACHE:
                _REF_CACHE[prefix] = torch.load(os.path.join(os.path.dirname(__file__), prefix)).float()
            ref = _REF_CACHE[prefix].to(goal.device)
            i = (t * 60.0).round().long().clamp(0, len(ref) - 1)
            sag = goal.new_tensor([math.radians(v) for v in getattr(cfg, "raise_prefix_sag_deg", (0.0,) * 8)])
            pre = ref[i] + (t / ts[0]).clamp(0.0, 1.0)[:, None] * sag
            out = torch.where((t < ts[0])[:, None], pre, out)
        return out
    ts = [0.0] + [w[0] for w in wps]
    qs = [torch.zeros_like(goal)] + [goal.new_tensor([math.radians(v) for v in w[1]]) for w in wps]
    out = qs[-1].expand(t.shape[0], -1).clone()
    for i in range(len(wps) - 1, -1, -1):
        x = ((t - ts[i]) / (ts[i + 1] - ts[i])).clamp(0.0, 1.0)[:, None]
        seg = qs[i] + x * x * (3.0 - 2.0 * x) * (qs[i + 1] - qs[i])
        out = torch.where((t < ts[i + 1])[:, None], seg, out)
    return out


def raise_extra_obs(env, t: torch.Tensor) -> torch.Tensor:
    """Velocity mode: phase t / raise_nominal_s and the reference-path error (8 joints, rad)."""
    c = env.cfg if hasattr(env, "cfg") and hasattr(env.cfg, "raise_waypoints") else env._rc
    goal = torch.tensor(getattr(c, "raise_goal_q", None) or c.start_arm_q, device=env.device)
    ref = raise_nominal(_TrackCfg(c) if getattr(c, "raise_track_waypoints", None) else c, t, goal)
    if hasattr(env, "_table_shift_q"):
        ref = ref + env._table_shift_q(t)
    q = env.robot.data.joint_pos[:, list(env.arm_ids)]
    return torch.cat(((t / c.raise_nominal_s).clamp(0.0, 1.25)[:, None], (ref - q).clamp(-3.0, 3.0)), dim=-1)


def raise_velocity_step(env, filtered: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Velocity mode: integrate the filtered actions as joint velocity
    commands (raise_vel_rad_s per joint) from the hanging pose, then the
    target low-pass. Returns the 8 arm + RotationBody targets."""
    c = env.cfg if hasattr(env.cfg, "raise_vel_rad_s") else env._rc
    arm = list(env.arm_ids)
    fresh = (t < 0.5 * env.step_dt)[:, None]
    if not hasattr(env, "_raise_cmd"):
        env._raise_cmd = torch.zeros((env.num_envs, 8), device=env.device)
        env._raise_smooth = torch.zeros((env.num_envs, 8), device=env.device)
    vel = torch.tensor(c.raise_vel_rad_s, device=env.device)
    cmd = torch.where(fresh, 0.0, env._raise_cmd) + vel * env.step_dt * filtered[:, :8]
    env._raise_cmd = torch.minimum(torch.maximum(cmd, env.lower[arm]), env.upper[arm])
    b = c.raise_target_smooth
    env._raise_smooth = b * torch.where(fresh, 0.0, env._raise_smooth) + (1.0 - b) * env._raise_cmd
    return env._raise_smooth


def raise_position_step(env, filtered: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Position mode: the filtered actions are absolute joint targets,
    center + half_range x action per joint (the span from hanging to the goal
    plus raise_position_margin_rad each side), passed through the target
    low-pass and capped at raise_vel_rad_s. Returns the 8 targets."""
    c = env.cfg if hasattr(env.cfg, "raise_position_margin_rad") else env._rc
    fresh = (t < 0.5 * env.step_dt)[:, None]
    goal = torch.tensor(getattr(c, "raise_goal_q", None) or c.start_arm_q, device=env.device)
    center = 0.5 * goal
    half = 0.5 * goal.abs() + c.raise_position_margin_rad
    tgt = center + half * filtered[:, :8]
    if not hasattr(env, "_raise_smooth"):
        env._raise_smooth = torch.zeros((env.num_envs, 8), device=env.device)
    prev = torch.where(fresh, 0.0, env._raise_smooth)
    b = c.raise_target_smooth
    new = b * prev + (1.0 - b) * tgt
    step = torch.tensor(c.raise_vel_rad_s, device=env.device) * env.step_dt
    env._raise_smooth = prev + torch.maximum(torch.minimum(new - prev, step), -step)
    return env._raise_smooth


@configclass
class JaxV2ArmRaiseEnvCfg(JaxV2FrontPregraspEnvCfg):
    episode_length_s = 5.0
    pre_hide_scripted_obs = False
    # Same finger drives as v16 (kp 40 / kd 2).
    robot_cfg = JaxV2FrontGraspLiftEnvCfg().robot_cfg
    raise_delta_rad = 0.02       # per step (60 Hz) at action 1: 1.2 rad/s
    # v17_raise model_249 (delta control): mean pose error stayed 40-55 deg;
    # integrating random deltas never covered RotationBody's 155 deg or the
    # elbow's 90. Absolute targets around the goal instead (as the v12
    # pre-grasp): target = start pose + raise_range_rad x action; at action 0
    # the drives (1 rad/s limit) head straight to the L, and the policy has
    # to learn the detour that clears the table.
    raise_absolute = True
    raise_range_rad = 1.0
    # abs model_249 (deterministic chain render): the arm parked in a detour
    # pose (shoulder pitch 42, elbow 105, wrist 58, RotationBody -98 deg;
    # 58 deg from the L) and never came back. Scripted nominal path + learned
    # residual: target(t) = hanging + smoothstep(t / raise_nominal_s) x (L -
    # hanging) + raise_residual_rad x sin(pi s) x action. The residual is 0 at
    # the start and the end, so the arm leaves the hanging pose and arrives
    # at the L by construction; the policy learns the detour around the table.
    raise_mode = "residual"
    raise_nominal_s = 4.0
    raise_residual_rad = 0.6
    raise_residual_cost = 0.2    # per rad^2 of weighted residual (prefer the nominal path)
    # res model_249 (deterministic chain): elbow stopped at 76.5 deg with the
    # finger plates pressing the table edge (43 N) while the keep-out read 0:
    # it tested link origins, and the plates reach >4 cm past them. The hand
    # swings on a circle about the elbow that enters the table's x range
    # below its top, so the wrist has to curl up mid-raise (more than the 0.6
    # rad residual allowed). Keep-out now on sampled collision-mesh vertices
    # of the hand and all six finger links; residual 1.2 rad; contact 2x.
    raise_mesh_keepout = True
    raise_plate_drop_m = 0.0
    raise_residual_rad = 1.2
    raise_contact_scale = 1.0
    # User: no motion may change abruptly. mesh model_499 in the chain: the
    # largest step-to-step velocity change of the whole chain (127 deg/s per
    # step) was the forearm roll in the raise, a residual the table detour
    # does not need. Residual only on shoulder pitch, elbow, wrist pitch and
    # RotationBody (others follow the nominal path), and the raise targets
    # pass the same low-pass as the pre-grasp (raise_target_smooth).
    raise_residual_joints = (0, 3, 5, 7)
    raise_target_smooth = 0.85
    # smooth model_499 (4 residual joints): could not clear the table (125 N,
    # handover by timeout 17 deg off); mesh model_499 had used the forearm roll
    # for the detour. All 8 joints again, smoothing kept.
    raise_residual_joints = (0, 1, 2, 3, 4, 5, 6, 7)
    raise_thumb_ramp_s = (1.0, 3.0)
    # Keep-out over the table: hand points with x below table edge + margin
    # and z below table top + clearance (finger plates hang ~4.5 cm below
    # their link origins).
    raise_table_edge_x = -0.34
    raise_edge_margin_m = 0.03
    raise_clear_m = 0.02
    raise_plate_drop_m = 0.045
    raise_keepout_scale = 30.0
    raise_contact_scale = 0.5    # per N of finger force on anything
    raise_body_scale = 5.0
    raise_cube_scale = 20.0
    raise_pose_lin_rad = 1.6
    raise_pose_sharp_rad = 0.05
    raise_lin_scale = 2.0
    raise_sharp_scale = 3.0
    raise_settle_err_rad = math.radians(1.5)
    raise_settle_speed = 0.03
    raise_settle_scale = 3.0
    raise_vel_scale = 0.01
    raise_rate_scale = 0.01


class JaxV2ArmRaiseEnv(JaxV2FrontPregraspEnv):
    cfg: JaxV2ArmRaiseEnvCfg

    def _raise_goal(self) -> torch.Tensor:
        return torch.tensor(getattr(self.cfg, "raise_goal_q", None) or self.cfg.start_arm_q, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        q = self.robot.data.joint_pos[ids].clone()
        q[:, list(self.arm_ids)] = 0.0
        q[:, self.finger_ids] = 0.0
        targets = self.targets[ids]
        targets[:, list(self.arm_ids)] = 0.0
        targets[:, self.finger_ids] = 0.0
        p = getattr(self.cfg, "raise_rsi_prob", 0.0)
        if p > 0.0:
            # Reference state initialization: start a fraction of episodes at
            # a random point of the reference path, at rest, with the episode
            # clock and the velocity integrator at that point.
            arm = list(self.arm_ids)
            t0 = torch.rand(len(ids), device=self.device) * self.cfg.raise_nominal_s
            t0 = torch.where(torch.rand(len(ids), device=self.device) < p, t0, 0.0)
            t0 = (t0 / self.step_dt).floor() * self.step_dt
            ref = raise_nominal(self.cfg, t0, self._raise_goal()) + self._table_shift_q(t0, ids)
            q[:, arm] = ref
            tt0, tt1 = self.cfg.raise_thumb_ramp_s
            x = ((t0 - tt0) / (tt1 - tt0)).clamp(0.0, 1.0)
            q[:, self.finger_ids[2]] = x * x * (3.0 - 2.0 * x) * math.radians(self.cfg.hand_shape_deg[2])
            targets[:, arm] = ref
            targets[:, self.finger_ids[2]] = q[:, self.finger_ids[2]]
            self.episode_length_buf[ids] = (t0 / self.step_dt).round().long()
            if not hasattr(self, "_raise_cmd"):
                self._raise_cmd = torch.zeros((self.num_envs, 8), device=self.device)
                self._raise_smooth = torch.zeros((self.num_envs, 8), device=self.device)
            self._raise_cmd[ids] = ref
            self._raise_smooth[ids] = ref
            if getattr(self.cfg, "raise_mode", "") == "position":
                # v19_pos6 model_1195 pulled shoulder pitch back to -22 deg even
                # when started at the goal. RSI started the filtered action at 0,
                # i.e. targets at the midpoint pose (elbow 37 deg), so late starts
                # drooped toward the table edge; holding the shoulder back avoided
                # it. Start the filtered action at the one that targets ref.
                g = self._raise_goal()
                a0 = ((ref - 0.5 * g) / (0.5 * g.abs() + self.cfg.raise_position_margin_rad)).clamp(-1.0, 1.0)
                for name in ("_filtered_actions", "actions", "previous_actions"):
                    buf = getattr(self, name, None)
                    if buf is not None:
                        buf[ids, :8] = a0
        if hasattr(self, "_raise_prev_qd"):
            self._raise_prev_qd[ids] = 0.0
        if getattr(self.cfg, "raise_gain_rand", 0.0):
            # domain randomization: arm drive kp/kd scaled per episode
            arm = list(self.arm_ids); r = self.cfg.raise_gain_rand
            kp0 = self.robot.actuators["arm"].stiffness[0, 0]; kd0 = self.robot.actuators["arm"].damping[0, 0]
            sk = 1.0 + r * (2.0 * torch.rand((len(ids), 1), device=self.device) - 1.0)
            sd = 1.0 + r * (2.0 * torch.rand((len(ids), 1), device=self.device) - 1.0)
            self.robot.write_joint_stiffness_to_sim((kp0 * sk).expand(-1, len(arm)), joint_ids=arm, env_ids=ids)
            self.robot.write_joint_damping_to_sim((kd0 * sd).expand(-1, len(arm)), joint_ids=arm, env_ids=ids)
        self.targets[ids] = targets
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
        self.robot.set_joint_position_target(targets, env_ids=ids)

    def raise_targets(self, filtered: torch.Tensor, t: torch.Tensor):
        """Arm + RotationBody delta step and the scripted thumb ramp (also used by the render chain)."""
        arm = list(self.arm_ids)
        if getattr(self.cfg, "raise_mode", "") == "residual":
            x = (t / self.cfg.raise_nominal_s).clamp(0.0, 1.0)[:, None]
            nom = raise_nominal(self.cfg, t, self._raise_goal()) + self._table_shift_q(t)
            mask = torch.zeros(8, device=self.device)
            mask[list(getattr(self.cfg, "raise_residual_joints", range(8)))] = 1.0
            if getattr(self.cfg, "raise_residual_envelope", "sin") == "const":
                envelope = (t / 0.5).clamp(0.0, 1.0)[:, None]  # on after 0.5 s, kept to the end
            else:
                envelope = torch.sin(math.pi * x)
            self._raise_res = self.cfg.raise_residual_rad * envelope * filtered[:, :8] * mask
            tgt = nom + self._raise_res
            if getattr(self.cfg, "raise_target_smooth", 0.0):
                b = self.cfg.raise_target_smooth
                fresh = (t < 0.5 * self.step_dt)[:, None]
                prev = torch.where(fresh, torch.zeros_like(tgt), getattr(self, "_raise_smooth", torch.zeros_like(tgt)))
                new = b * prev + (1.0 - b) * tgt
                if getattr(self.cfg, "raise_rate_rad_s", 0.0):
                    # arm joints only; RotationBody (hand servo) turns 170 deg
                    step = self.cfg.raise_rate_rad_s * self.step_dt
                    new = torch.cat([prev[:, :7] + (new[:, :7] - prev[:, :7]).clamp(-step, step), new[:, 7:]], dim=1)
                self._raise_smooth = new
                tgt = self._raise_smooth
        elif getattr(self.cfg, "raise_mode", "") == "velocity":
            tgt = raise_velocity_step(self, filtered, t)
        elif getattr(self.cfg, "raise_mode", "") == "position":
            tgt = raise_position_step(self, filtered, t)
        elif self.cfg.raise_absolute:
            tgt = self._raise_goal() + self.cfg.raise_range_rad * filtered[:, :8]
        else:
            tgt = self.targets[:, arm] + self.cfg.raise_delta_rad * filtered[:, :8]
        self.targets[:, arm] = torch.minimum(torch.maximum(tgt, self.lower[arm]), self.upper[arm])
        t0, t1 = self.cfg.raise_thumb_ramp_s
        x = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0)
        w = x * x * (3.0 - 2.0 * x)
        fingers = torch.zeros((self.num_envs, len(self.finger_ids)), device=self.device)
        fingers[:, 2] = w * math.radians(self.cfg.hand_shape_deg[2])
        self.targets[:, self.finger_ids] = fingers

    def _pre_physics_step(self, actions: torch.Tensor):
        alpha = self.cfg.action_filter_alpha
        self._filtered_actions = alpha * actions.clamp(-1.0, 1.0) + (1.0 - alpha) * self._filtered_actions
        self.previous_actions.copy_(self.actions)
        self.actions = self._filtered_actions.clone()
        self.raise_targets(self.actions, self.episode_length_buf * self.step_dt)

    def _table_shift_q(self, t: torch.Tensor, ids=None) -> torch.Tensor:
        """Joint shift (n, 8) of the path for this episode's table offset (0 without one)."""
        off = getattr(self, "_table_off", None)
        n = len(t)
        if off is None:
            return torch.zeros((n, 8), device=self.device)
        off = off if ids is None else off[ids]
        if getattr(self, "_cube_off", None) is not None:
            # [hand dx, dy, dz] = cube x (table x + cube on table), cube y, table z
            co = self._cube_off if ids is None else self._cube_off[ids]
            off = torch.stack((off[:, 0] + co[:, 0], co[:, 1], off[:, 1]), dim=1)
        if getattr(self.cfg, "raise_shift_grid_path", ""):
            # v27: exact per-offset IK of every knot on a cube (dx, dy) grid, bilinear between grid
            # points; table z from the linear table (S[..., 2])
            if not hasattr(self, "_shift_G"):
                d = torch.load(os.path.join(os.path.dirname(__file__), self.cfg.raise_shift_grid_path))
                self._shift_G = d["dq"].to(self.device)  # (n_wp, nx, ny, 8)
                self._shift_gx, self._shift_gy = d["gx"].to(self.device), d["gy"].to(self.device)
                self._shift_Sz = torch.load(os.path.join(os.path.dirname(__file__), self.cfg.raise_table_shift_path)).to(self.device)[:, :, 2]
                self._shift_t = d["ts"].to(self.device)
                # v34: the linear z column left the hand ~3.5 mm far / 3 mm high and 4-7 mm off in y (wide cube y)
                # at a 3 cm table drop (+3 cm was exact): below the table height use the exact-IK grid at -3 cm,
                # blended linearly from the 0 cm grid
                gm = getattr(self.cfg, "raise_shift_grid_dzm3_path", "")
                self._shift_Gm = torch.load(os.path.join(os.path.dirname(__file__), gm))["dq"].to(self.device) if gm else None
                gp = getattr(self.cfg, "raise_shift_grid_dzp3_path", "")
                self._shift_Gp = torch.load(os.path.join(os.path.dirname(__file__), gp))["dq"].to(self.device) if gp else None
            G, ts = self._shift_G, self._shift_t
            nk, ng = G.shape[0], G.shape[1] * G.shape[2]
            wps = getattr(self.cfg, "raise_track_waypoints", None) or V18_WAYPOINTS
            # v44: raise_shift_early_s repeats the first knot's shift at that time, so the whole shift is in early in the
            # rise (table 11.5 cm closer: the ramp to the first knot left the rising thumb 3-4 cm short and it hit the
            # table edge at 3.5 s, 18 N)
            early = getattr(self.cfg, "raise_shift_early_s", None)
            def pad(X):
                return torch.cat([torch.zeros_like(X[:1])] + ([X[:1]] if early else []) + [X])
            ts0 = torch.cat((ts.new_zeros(1),) + ((ts.new_tensor([early]),) if early else ()) + (ts,))
            nk1 = len(ts0)
            Q = pad(G).reshape(nk1, ng * 8)
            Qz = pad(self._shift_Sz)
            stops = torch.tensor([False] + ([False] if early else []) + [len(w) > 2 and w[2] == "stop" for w in wps], device=self.device)
            cat = getattr(self.cfg, "raise_interp_catmull", False)
            W = grid_weights(self._shift_gx, self._shift_gy, off[:, :2] + off.new_tensor([getattr(self.cfg, "raise_reach_dx_m", 0.0), getattr(self.cfg, "raise_reach_dy_m", 0.0)]))
            out = (pchip_eval(ts0, Q, t, catmull=cat, stops=stops).reshape(n, ng, 8) * W[:, :, None]).sum(1)
            if self._shift_Gm is None:
                return out + pchip_eval(ts0, Qz, t, catmull=cat, stops=stops) * off[:, 2:3]
            def at(Gx):
                Qx = pad(Gx).reshape(nk1, ng * 8)
                return (pchip_eval(ts0, Qx, t, catmull=cat, stops=stops).reshape(n, ng, 8) * W[:, :, None]).sum(1)
            res = out + (off[:, 2:3] / -0.03).clamp_min(0.0) * (at(self._shift_Gm) - out)
            if self._shift_Gp is None:
                return res + pchip_eval(ts0, Qz, t, catmull=cat, stops=stops) * off[:, 2:3].clamp_min(0.0)
            return res + (off[:, 2:3] / 0.03).clamp_min(0.0) * (at(self._shift_Gp) - out)
        if not hasattr(self, "_shift_S"):
            self._shift_S = torch.load(os.path.join(os.path.dirname(__file__), self.cfg.raise_table_shift_path)).to(self.device)
            self._shift_t = torch.tensor([w[0] for w in (getattr(self.cfg, "raise_track_waypoints", None) or V18_WAYPOINTS)], device=self.device)
        S, ts = self._shift_S, self._shift_t
        if getattr(self.cfg, "raise_interp", "smoothstep") == "pchip":
            wps = getattr(self.cfg, "raise_track_waypoints", None) or V18_WAYPOINTS
            if getattr(self.cfg, "raise_prefix_ref_path", ""):
                ts0, S0 = ts, S.reshape(len(ts), -1)
                stops = torch.tensor([len(w) > 2 and w[2] == "stop" for w in wps], device=self.device)
            else:
                ts0 = torch.cat((ts.new_zeros(1), ts))
                S0 = torch.cat((torch.zeros_like(S[:1]), S)).reshape(len(ts0), -1)
                stops = torch.tensor([False] + [len(w) > 2 and w[2] == "stop" for w in wps], device=self.device)
            out = pchip_eval(ts0, S0, t, catmull=getattr(self.cfg, "raise_interp_catmull", False), stops=stops).reshape(n, 8, S.shape[-1])
            return (out @ off[:, :, None])[:, :, 0]
        # ramp in with the rise (0 at hanging -> first waypoint), then between waypoints
        x = (t / ts[0]).clamp(0.0, 1.0)
        out = (x * x * (3.0 - 2.0 * x))[:, None, None] * S[0]
        for i in range(len(ts) - 1):
            x = ((t - ts[i]) / (ts[i + 1] - ts[i])).clamp(0.0, 1.0)
            out = out + (x * x * (3.0 - 2.0 * x))[:, None, None] * (S[i + 1] - S[i])
        return (out @ off[:, :, None])[:, :, 0]

    def _hand_mesh_points(self) -> torch.Tensor:
        """Env-frame positions (N, P, 3) of sampled collision-mesh vertices of the hand and fingers."""
        if not hasattr(self, "_hm_pts"):
            d = os.path.join(os.path.dirname(__file__), "../../../../isaaclab_assets/data/Robots/JaxV2Assembly1006")
            root = ET.parse(os.path.join(d, "urdf/JAXV2_Assembly.urdf")).getroot()
            want = ("MG996R", "RotationBody", "Finger_1_1", "Finger_2_1", "Finger_3_1", "Finger_1_2", "Finger_2_2", "Finger_3_2")
            ids, pts = [], []
            for link in root.findall("link"):
                name = link.get("name")
                col = link.find("collision")
                if name not in want or col is None:
                    continue
                mesh = col.find("geometry/mesh"); org = col.find("origin")
                fn = mesh.get("filename").split("://")[-1]
                for cand in (fn, os.path.join(d, "urdf", fn), os.path.join(d, "meshes", os.path.basename(fn))):
                    if os.path.exists(cand):
                        fn = cand
                        break
                raw = open(fn, "rb").read(); n = struct.unpack("<I", raw[80:84])[0]
                v = torch.frombuffer(bytearray(raw[84:84 + n * 50]), dtype=torch.uint8).view(n, 50)[:, 12:48]
                v = v.contiguous().view(torch.float32).view(-1, 3).double()
                v = v * torch.tensor([float(x) for x in mesh.get("scale", "1 1 1").split()], dtype=torch.float64)
                r, pi, y = [float(x) for x in (org.get("rpy") if org is not None else "0 0 0").split()]
                cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(pi), math.sin(pi), math.cos(y), math.sin(y)
                R = torch.tensor([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr], [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr], [-sp, cp*sr, cp*cr]], dtype=torch.float64)
                xyz = org.get("xyz") if org is not None else "0 0 0"
                v = torch.unique(v @ R.T + torch.tensor([float(x) for x in xyz.split()], dtype=torch.float64), dim=0)
                keep = v[torch.linspace(0, len(v) - 1, min(128, len(v))).long()].float()
                ids += [self.robot.body_names.index(name)] * len(keep); pts.append(keep)
            self._hm_ids = torch.tensor(ids, device=self.device)
            self._hm_pts = torch.cat(pts).to(self.device)
        q = self.robot.data.body_quat_w[:, self._hm_ids]
        p = self.robot.data.body_pos_w[:, self._hm_ids] - self.scene.env_origins[:, None]
        n, m = q.shape[:2]
        return p + quat_apply(q.reshape(-1, 4), self._hm_pts[None].expand(n, -1, -1).reshape(-1, 3)).view(n, m, 3)

    def raise_keepout(self) -> torch.Tensor:
        """Depth (m) of the deepest hand point inside the box over the table."""
        c = self.cfg
        if getattr(c, "raise_mesh_keepout", False):
            p = self._hand_mesh_points()
        else:
            ids = [self._hand_id] + self.proximal_ids + list(self.fingertip_ids)
            p = self.robot.data.body_pos_w[:, ids] - self.scene.env_origins[:, None]
        off = getattr(self, "_table_off", None)
        dx = off[:, 0:1] if off is not None else 0.0
        dz = off[:, 1:2] if off is not None else 0.0
        x_in = (c.raise_table_edge_x + dx + c.raise_edge_margin_m) - p[..., 0]
        z_in = (c.table_top_z + dz + c.raise_clear_m + c.raise_plate_drop_m) - p[..., 2]
        return torch.minimum(x_in, z_in).clamp_min(0.0).amax(dim=1)

    def _get_observations(self):
        obs = super()._get_observations()
        if getattr(self.cfg, "raise_mode", "") in ("velocity", "position") or getattr(self.cfg, "raise_residual_obs", False):
            obs["policy"] = torch.cat((obs["policy"], raise_extra_obs(self, self.episode_length_buf * self.step_dt)), dim=-1)
        return obs

    def _raise_dqd(self) -> torch.Tensor:
        """Squared change of arm joint velocity over one step (rad/s)^2, summed."""
        qd = self.robot.data.joint_vel[:, list(self.arm_ids)]
        prev = getattr(self, "_raise_prev_qd", qd)
        fresh = (self.episode_length_buf <= 1)[:, None]
        d = torch.where(fresh, 0.0, qd - prev)
        self._raise_prev_qd = qd.clone()
        return d.square().sum(dim=1)

    def raise_errors(self):
        arm = list(self.arm_ids)
        err = (self.robot.data.joint_pos[:, arm] - self._raise_goal()).abs()
        speed = self.robot.data.joint_vel[:, arm].abs().amax(dim=1)
        return err, speed

    def _get_rewards(self):
        c = self.cfg
        cube = self.object.data.root_pos_w
        self._cube_start = torch.where(self._cube_start_valid[:, None], self._cube_start, cube)
        self._cube_start_quat = torch.where(self._cube_start_valid[:, None], self._cube_start_quat, self.object.data.root_quat_w)
        self._cube_start_valid[:] = True
        err, speed = self.raise_errors()
        mean_err = err.mean(dim=1)
        if getattr(c, "raise_track_waypoints", None):
            # residual mode: pose terms follow the geometric path (no sag offsets)
            ref = raise_nominal(_TrackCfg(c), self.episode_length_buf * self.step_dt, self._raise_goal()) \
                + self._table_shift_q(self.episode_length_buf * self.step_dt)
            jerr = (self.robot.data.joint_pos[:, list(self.arm_ids)] - ref).abs()
            mean_err = jerr.mean(dim=1)
            self._ref_err_log = mean_err
            self._ref_jerr = jerr
        elif getattr(c, "raise_mode", "") in ("velocity", "position"):
            # pose terms follow the reference path (which ends at the goal)
            ref = raise_nominal(c, self.episode_length_buf * self.step_dt, self._raise_goal())
            jerr = (self.robot.data.joint_pos[:, list(self.arm_ids)] - ref).abs()
            mean_err = jerr.mean(dim=1)
            self._ref_err_log = mean_err
            self._ref_jerr = jerr
        settled = (err.amax(dim=1) < c.raise_settle_err_rad) & (speed < c.raise_settle_speed)
        keepout = self.raise_keepout()
        names = [f"proximal_{i}" for i in (1, 2, 3)] + [f"finger_{i}" for i in (1, 2, 3)]
        if getattr(c, "raise_contact_table_only", False):
            # everything but the cube (table, etc.): net minus the cube-filtered force
            contact = torch.stack(
                [(torch.linalg.vector_norm(self.scene.sensors[n].data.net_forces_w[:, 0], dim=-1)
                  - torch.linalg.vector_norm(self.scene.sensors[n].data.force_matrix_w[:, 0, 0], dim=-1)).clamp_min(0.0)
                 for n in names], dim=1
            ).sum(dim=1).clamp(max=20.0)
        else:
            contact = torch.stack(
                [torch.linalg.vector_norm(self.scene.sensors[n].data.net_forces_w[:, 0], dim=-1) for n in names], dim=1
            ).sum(dim=1).clamp(max=20.0)
        terms = {
            "lin": c.raise_lin_scale * (1.0 - mean_err / c.raise_pose_lin_rad).clamp(0.0, 1.0),
            "sharp": c.raise_sharp_scale * torch.exp(-mean_err / c.raise_pose_sharp_rad),
            "settle": c.raise_settle_scale * settled.float(),
            "keepout": -c.raise_keepout_scale * keepout,
            "contact": -c.raise_contact_scale * contact,
            "body": -c.raise_body_scale * self._body_intrusion,
            "cube": -c.raise_cube_scale * self._cube_shift,
            "vel": -c.raise_vel_scale * self.robot.data.joint_vel[:, list(self.arm_ids)].square().sum(dim=1),
            "residual": -c.raise_residual_cost * (self._raise_res.square().sum(dim=1) if hasattr(self, "_raise_res") else 0.0 * mean_err),
            "rate": -c.raise_rate_scale * (self.actions[:, :8] - self.previous_actions[:, :8]).square().sum(dim=1),
            "dqd": -getattr(c, "raise_dqd_scale", 0.0) * self._raise_dqd(),
        }
        if getattr(c, "raise_cube_touch_scale", 0.0):
            # v27d: the v27c residual brushed the cube at y -5 cm (12-36 N, cube pushed 2-3 cm in the
            # chain) since contact counted the table only; the cube now costs too
            cube_n = torch.stack([torch.linalg.vector_norm(self.scene.sensors[n].data.force_matrix_w[:, 0, 0], dim=-1)
                                  for n in names], dim=1).sum(dim=1).clamp(max=20.0)
            terms["cube_touch"] = -c.raise_cube_touch_scale * cube_n
            self._cube_touch_log = cube_n
        if getattr(c, "raise_joint_track_scale", 0.0) and hasattr(self, "_ref_jerr"):
            # per joint: one joint far off is not hidden by the mean
            terms["joint"] = c.raise_joint_track_scale * torch.exp(-self._ref_jerr / c.raise_joint_track_rad).mean(dim=1)
            terms["worst"] = c.raise_worst_scale * torch.exp(-self._ref_jerr.amax(dim=1) / c.raise_worst_rad)
        reward = sum(terms.values()) * getattr(c, "raise_reward_scale", 1.0)
        self.extras["log"] = {f"r_{k}": v.mean() for k, v in terms.items()}
        self.extras["log"].update({
            "raise_err_max_deg": torch.rad2deg(err.amax(dim=1)).mean(),
            "raise_err_mean_deg": torch.rad2deg(mean_err).mean(),
            "raise_settled": settled.float().mean(),
            "raise_keepout_mm": 1000.0 * keepout.mean(),
            "raise_keepout_any": (keepout > 0).float().mean(),
            "raise_contact_n": contact.mean(),
            "raise_cube_touch_n": self._cube_touch_log.mean() if hasattr(self, "_cube_touch_log") else 0.0 * contact.mean(),
            "raise_body_mm": 1000.0 * self._body_intrusion.mean(),
            "raise_cube_shift_mm": 1000.0 * self._cube_shift.mean(),
        })
        if hasattr(self, "_ref_err_log"):
            self.extras["log"]["raise_ref_err_deg"] = torch.rad2deg(self._ref_err_log).mean()
        return reward


@configclass
class JaxV2RaiseToPregraspEnvCfg(JaxV2ArmRaiseEnvCfg):
    """v18: hanging straight up into the pre-grasp pose, no L.

    User (2026-10-29): the arm came down hard from the L into the pre-grasp
    ("would break a real robot"); rise to the grasp height instead of
    lowering into it. The goal is the rate07 pre-grasp end pose (chain,
    all envs alike); the elbow ends at 73.7 deg, below the L's 90, so the arm
    only rises. The hand must come up in front of the table edge and move in
    over the table above its top, so the keep-out is the table itself (5 mm
    clearance on the hand mesh) rather than the box over it. Slow nominal path
    (6 s) and arm targets capped at 0.5 rad/s; then v16 starts at its close.
    """

    episode_length_s = 9.0
    raise_goal_q = tuple(math.radians(v) for v in (-7.3, 0.0, 2.5, 73.7, -5.6, 14.7, -6.6, -170.1))
    raise_nominal_s = 6.0
    raise_rate_rad_s = 0.5
    raise_thumb_ramp_s = (1.0, 4.0)
    raise_edge_margin_m = 0.0
    raise_clear_m = 0.005
    raise_plate_drop_m = 0.0
    # v18_raise model_499 (one smoothstep, hanging -> pre-grasp): the joint-
    # space line runs into the table front from 3.5 s on, where the residual
    # envelope sin(pi s) shrinks to 0; the hand stuck below the edge (elbow 66
    # vs 73.7). Two segments instead (v18_line_probe, offline): up into a pose
    # at grasp height with the hand mesh 3.4 cm in front of the table edge
    # (hand 10 cm back, same height and pitch), then in straight at that
    # height through poses every 2 cm. The pre-grasp hand mesh is 2 mm over
    # the top, so the keep-out clearance is 0.
    raise_waypoints = (
        (5.0, (-28.8, 0.0, 2.5, 89.1, -5.6, 20.8, -6.6, -170.1)),
        (5.5, (-24.9, 0.0, 2.5, 87.9, -5.6, 18.0, -6.6, -170.1)),
        (6.0, (-20.8, 0.0, 2.5, 86.2, -5.6, 15.7, -6.6, -170.1)),
        (6.5, (-16.7, 0.0, 2.5, 84.1, -5.6, 13.7, -6.6, -170.1)),
        (7.0, (-12.6, 0.0, 2.5, 81.5, -5.6, 12.2, -6.6, -170.1)),
        (7.5, (-8.3, 0.0, 2.5, 78.4, -5.6, 11.1, -6.6, -170.1)),
        (8.1, (-7.3, 0.0, 2.5, 73.7, -5.6, 14.7, -6.6, -170.1)),
    )
    # v18 two-segment probe: finger 2's plate caught the table's front edge
    # (84 N) at 6.4 s; gravity holds the arm ~2 deg low (as at the L), so the
    # mesh ran below the planned 2 mm. The line in is now 1.2 cm higher, the
    # hand settles the last 1.2 cm over 0.6 s, and every waypoint carries the
    # L sag compensation (shoulder +1.9, elbow +2.1, wrist pitch +0.7, wrist
    # yaw -0.2 deg), so the render passes --raise_sag_comp_deg 0.
    # RAISE_END_HIGH=1: stop at the end of the approach line (1.2 cm above the
    # pre-grasp height) instead of settling into it (test for table margin).
    if os.environ.get("RAISE_END_HIGH"):
        raise_waypoints = raise_waypoints[:-1]
    raise_sag_comp_deg = (1.9, 0.0, 0.0, 2.1, 0.0, 0.7, -0.2, 0.0)
    raise_waypoints = _add_deg(raise_waypoints, raise_sag_comp_deg)
    raise_nominal_s = 7.5 if os.environ.get("RAISE_END_HIGH") else 8.1
    episode_length_s = 10.0
    raise_clear_m = 0.0


V18_WAYPOINTS = (
    (5.0, (-28.8, 0.0, 2.5, 89.1, -5.6, 20.8, -6.6, -170.1)),
    (5.5, (-24.9, 0.0, 2.5, 87.9, -5.6, 18.0, -6.6, -170.1)),
    (6.0, (-20.8, 0.0, 2.5, 86.2, -5.6, 15.7, -6.6, -170.1)),
    (6.5, (-16.7, 0.0, 2.5, 84.1, -5.6, 13.7, -6.6, -170.1)),
    (7.0, (-12.6, 0.0, 2.5, 81.5, -5.6, 12.2, -6.6, -170.1)),
    (7.5, (-8.3, 0.0, 2.5, 78.4, -5.6, 11.1, -6.6, -170.1)),
    (8.1, (-7.3, 0.0, 2.5, 73.7, -5.6, 14.7, -6.6, -170.1)),
)


@configclass
class JaxV2RaiseLearnedEnvCfg(JaxV2RaiseToPregraspEnvCfg):
    """v19: the hanging -> pre-grasp raise as a learned policy.

    User (2026-10-29): keep the cube where it is and learn the arm raise,
    following the v18 script, then join the grasp. The policy commands joint
    velocities (raise_vel_rad_s; integrated from the hanging pose, then the
    target low-pass), so the drives never see a target jump. The pose terms
    reward following the v18 path; its waypoints here are the plain geometric
    poses (no sag compensation; the policy holds against gravity itself).
    Observation: the 43-dim v9 observation + phase + reference-path error (8).
    The thumb ramp to -30 deg stays scripted (fingers otherwise at 0).
    """

    raise_mode = "velocity"
    observation_space = 52
    raise_vel_rad_s = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0)
    raise_waypoints = V18_WAYPOINTS
    raise_sag_comp_deg = (0.0,) * 8
    raise_residual_cost = 0.0
    raise_pose_sharp_rad = 0.03
    raise_settle_err_rad = math.radians(1.0)
    # v19_raise model_400: never learned the path (episode-mean goal error
    # 25 deg vs 35 deg left hanging; the deterministic chain wandered to
    # shoulder yaw -89, elbow -100 deg). Every episode started hanging, so
    # the later path was rarely seen. Reference state initialization (80% of
    # episodes start at a random path time) and a wider tracking kernel.
    raise_rsi_prob = 0.8
    raise_pose_sharp_rad = 0.1
    # User (2026-10-29): the raise must be learned (closed loop), since the
    # real arm will not match the sim exactly. v20 showed velocity commands
    # random-walk; position commands (absolute targets through the low-pass
    # and the speed cap) with the v18 path as reference, RSI kept.
    raise_mode = "position"
    raise_position_margin_rad = 0.6
    raise_vel_rad_s = (0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 1.0)
    # v19_pos4 model_797: every joint within 0.7 deg of the goal except
    # shoulder pitch, held 17 deg back (-24.4 vs -7.3) in all envs; the
    # mean-error kernel hid it (17 deg on one joint is 2 deg mean). Per-joint
    # tracking terms and one on the worst joint.
    raise_joint_track_scale = 3.0
    raise_joint_track_rad = 0.05
    raise_worst_scale = 3.0
    raise_worst_rad = 0.05
    # v19_pos5 model_996: shoulder pitch still 20 deg back. Probe from the
    # waypoint: moving in with the hand ~1 cm low (gravity sag, the policy
    # tracks joint angles) finger 1's plate caught the table's front edge
    # (60 N), and with a 0 clearance keep-out nothing warned before contact;
    # the policy learned to stay back. Approach line 2 cm above the grasp
    # height (v18_line_probe 0.02), settle 2 cm over 1 s, and a 5 mm keep-out
    # clearance over the table top.
    raise_waypoints = (
        (5.0, (-29.6, 0.0, 2.5, 92.0, -5.6, 18.7, -6.6, -170.1)),
        (5.5, (-25.6, 0.0, 2.5, 90.8, -5.6, 15.9, -6.6, -170.1)),
        (6.0, (-21.6, 0.0, 2.5, 89.2, -5.6, 13.5, -6.6, -170.1)),
        (6.5, (-17.4, 0.0, 2.5, 87.0, -5.6, 11.5, -6.6, -170.1)),
        (7.0, (-13.2, 0.0, 2.5, 84.4, -5.6, 9.9, -6.6, -170.1)),
        (7.5, (-8.9, 0.0, 2.5, 81.3, -5.6, 8.7, -6.6, -170.1)),
        (8.5, (-7.3, 0.0, 2.5, 73.7, -5.6, 14.7, -6.6, -170.1)),
    )
    raise_nominal_s = 8.5
    episode_length_s = 10.5
    raise_clear_m = 0.005


@configclass
class JaxV2RaiseToLEnvCfg(JaxV2RaiseLearnedEnvCfg):
    """v20: learned raise from hanging into the v16 start pose (the L).

    User (2026-10-29): keep the L as the hand-off pose (later cubes will sit
    in several places and the arm reaches from the L), and the policy must
    make the raise path itself. Goal: v12/v16 start_arm_q, arm (0, 0, 0, 90,
    0, 0, 0) deg and RotationBody -155 deg; the v12 pre-grasp is trained for
    +/-1.5 deg start error, so success = every joint within 1.5 deg and slower
    than 0.05 rad/s.

    Policy-controlled: all 7 arm joints and RotationBody, as joint velocity
    commands (raise_vel_rad_s) integrated into the drive targets (then the
    target low-pass). Scripted: the thumb ramp to -30 deg (1-3 s), other
    finger joints held at 0. The reference path (data/v20_raise_ref.pt: the
    v17 chain raise, keep-out 0 and contact 0, stretched 4.25 -> 7 s and
    smoothed) enters only the reward and the episode starts (80% start at a
    random path time); it never drives the joints.
    """

    raise_goal_q = None
    raise_waypoints = None
    raise_ref_path = "data/v20_raise_ref.pt"
    raise_nominal_s = 7.0
    episode_length_s = 9.0
    raise_vel_rad_s = (0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 1.0)
    raise_thumb_ramp_s = (1.0, 3.0)
    # v17 keep-out (box over the table, mesh points): x below edge + 3 cm,
    # z below top + 2 cm + 4.5 cm; covers the table's top, edge and underside.
    raise_edge_margin_m = 0.03
    raise_clear_m = 0.02
    raise_plate_drop_m = 0.045
    raise_settle_err_rad = math.radians(1.5)
    raise_settle_speed = 0.05
    # v20_raise model_399 (velocity commands): from hanging, 0/64 reached the
    # L (shoulder yaw ran to 239 deg); integrated exploration noise made the
    # path a random walk. The policy now outputs absolute joint targets
    # (DeepMimic-style), still through the low-pass and the speed cap.
    raise_mode = "position"
    raise_position_margin_rad = 0.6


@configclass
class JaxV2RaiseResidualEnvCfg(JaxV2RaiseToPregraspEnvCfg):
    """v21: residual RL on the v18 raise.

    User (2026-10-29): the raise must react to a real arm that will not match
    the sim. The v18 scripted path (with its sag offsets, low-pass and 0.5
    rad/s cap) stays the base; the policy adds up to +/-8 deg per joint
    (7 arm joints + RotationBody), from 0.5 s to the end, and sees the phase
    and its error to the geometric path. Rewards follow the geometric v18
    path (no sag offsets) into the pre-grasp pose; the residual, action rate
    and joint velocity change (tremor) cost. Arm kp/kd are scaled +/-25% per
    episode so the base path alone misses and feedback has to correct it.
    """

    observation_space = 52
    raise_residual_obs = True
    raise_residual_rad = math.radians(8.0)
    raise_residual_envelope = "const"
    raise_residual_joints = (0, 1, 2, 3, 4, 5, 6, 7)
    raise_residual_cost = 2.0
    raise_track_waypoints = V18_WAYPOINTS
    raise_rsi_prob = 0.5
    raise_pose_sharp_rad = 0.05
    raise_joint_track_scale = 3.0
    raise_joint_track_rad = 0.03
    raise_worst_scale = 3.0
    raise_worst_rad = 0.03
    raise_settle_err_rad = math.radians(1.5)
    raise_settle_speed = 0.05
    raise_rate_scale = 0.5
    raise_dqd_scale = 5.0
    raise_gain_rand = 0.25
    episode_length_s = 10.0
    # v21_res model_299: with kp/kd +/-25% the pre-grasp's 2 mm finger-table
    # gap could not hold for soft drives; the policy kept the hand high
    # (elbow 81.7 vs 73.7 deg) and missed the cube. +/-10%, continued from
    # model_50 (path error 0.9 deg).
    raise_gain_rand = 0.10
    # v21_dr10 (from model_50): contact cost -1.6/step early (3.2 N, mostly
    # fingers brushing the cube at the pre-grasp; cube moved 0.6 mm); the
    # policy removed it by holding the hand high (path error 1.2 -> 2.4 deg).
    # Contact cost on non-cube contact only (the table); cube motion 100/m.
    raise_contact_table_only = True
    raise_cube_scale = 100.0


# User (2026-11-01): the approach stuttered (smoothstep per segment stops at
# every waypoint, every 0.5 s) and the hand moved in, then dropped 1.2 cm as a
# separate step. One diagonal line instead (1.2 cm high at the waypoint 10 cm
# back, 0 at the pre-grasp; v23_diag_probe.py 0.012) through PCHIP: velocity
# continuous through the waypoints, zero only at the start and the end.
V23_WAYPOINTS = (
    (5.0, (-28.8, 0.0, 2.5, 89.1, -5.6, 20.8, -6.6, -170.1)),
    (5.6, (-24.6, 0.0, 2.5, 87.0, -5.6, 18.7, -6.6, -170.1)),
    (6.2, (-20.4, 0.0, 2.5, 84.4, -5.6, 17.0, -6.6, -170.1)),
    (6.8, (-16.1, 0.0, 2.5, 81.4, -5.6, 15.8, -6.6, -170.1)),
    (7.4, (-11.7, 0.0, 2.5, 77.8, -5.6, 15.0, -6.6, -170.1)),
    (8.0, (-7.3, 0.0, 2.5, 73.7, -5.6, 14.7, -6.6, -170.1)),
)


# User (2026-11-01): the L is the base pose for every object, and lowering the
# hand a little from it is fine, but it must not hit the table. Hanging -> L
# along the v17 chain raise (keep-out 0, contact 0; data/v20_raise_ref.pt sampled
# at 1-4 s) -> rest at the L (5.0-5.4 s) -> 3 cm above the pre-grasp (6.8 s) ->
# settle into the pre-grasp (8.0 s, speed 0). Only the knots after the L shift
# with the table (data/v24_table_shift.pt).
V24_WAYPOINTS = (
    (7.0, (0.0, 0.0, 0.0, 90.0, 0.0, 0.0, 0.0, -155.0), "stop"),
    (7.4, (0.0, 0.0, 0.0, 90.0, 0.0, 0.0, 0.0, -155.0), "stop"),
    # at the L fingers 1/2 sit over the cube (tips x -0.405, y 0.282): lowering
    # from there pressed the cube (29 N, pushed 28 mm). Back 5 cm while high
    # (hand over the pre-grasp, 6 cm up), then down through 3 cm up.
    (8.5, (-10.7, 0.0, 2.5, 94.9, -5.6, -3.1, -6.6, -170.1)),
    (9.3, (-9.5, 0.0, 2.5, 84.9, -5.6, 5.7, -6.6, -170.1)),
    (10.2, (-7.3, 0.0, 2.5, 73.7, -5.6, 14.7, -6.6, -170.1)),
)


@configclass
class JaxV2RaiseTableEnvCfg(JaxV2RaiseResidualEnvCfg):
    """v22: v21 with a measured, randomized table.

    User (2026-10-29): the policy needs the table's coordinates to know where
    it would collide. The table is a kinematic body moved per episode (front
    edge x +/-2 cm, top z +/-1.5 cm; the cube moves with it). Observation +2:
    table front-edge x and top z relative to the robot base, with a per-episode
    bias (3 mm) and per-step noise (1 mm) as from a camera. The base path and
    the tracked path shift with the table (per-waypoint joint shift from
    v22_shift_probe.py, data/v22_table_shift.pt); keep-out and contact use
    the episode's table.
    """

    observation_space = 54
    table_cfg = sim_utils.CuboidCfg(
        size=(0.80, 0.70, 0.04),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0, restitution=0.0),
    )
    raise_table_shift_path = "data/v22_table_shift.pt"
    raise_table_dx_m = 0.02
    raise_table_dz_m = 0.015
    raise_table_bias_m = 0.003
    raise_table_noise_m = 0.001
    raise_gain_rand = 0.10
    # v21/v22 degraded after 50-100 iterations (later at lower lr). Per-step
    # reward ~7 made values ~700 while the clipped value loss lets the critic
    # move 0.2 per update: value loss 4000-4600, advantages dominated by critic
    # error. Scale the summed reward (relative weights unchanged).
    raise_reward_scale = 0.02
    # smooth path: diagonal approach through PCHIP (see V23_WAYPOINTS)
    raise_interp = "pchip"
    # PCHIP stopped the whole arm at the rise -> approach waypoint (shoulder pitch
    # and elbow reverse there; 1 deg/s at 5.1 s). Catmull-Rom tangents pass through.
    raise_interp_catmull = True
    raise_track_waypoints = V23_WAYPOINTS
    raise_waypoints = _add_deg(V23_WAYPOINTS, (1.9, 0.0, 0.0, 2.1, 0.0, 0.7, -0.2, 0.0))
    raise_nominal_s = 8.0
    raise_table_shift_path = "data/v23_table_shift.pt"



@configclass
class JaxV2RaiseLTableEnvCfg(JaxV2RaiseTableEnvCfg):
    """v24: v22 (table-aware residual) on the L-based path V24_WAYPOINTS."""

    # 4 knots sampled from the v17 raise (5 s) lost its table detour: 144 N
    # contact, cube pushed 25 mm. The dense v17 path (7 s, 60 Hz) up to the L instead.
    raise_prefix_ref_path = "data/v20_raise_ref.pt"
    raise_prefix_sag_deg = (1.9, 0.0, 0.0, 2.1, 0.0, 0.7, -0.2, 0.0)
    raise_track_waypoints = V24_WAYPOINTS
    raise_waypoints = _add_deg(V24_WAYPOINTS, (1.9, 0.0, 0.0, 2.1, 0.0, 0.7, -0.2, 0.0))
    raise_nominal_s = 10.2
    episode_length_s = 12.0
    raise_rate_rad_s = 0.7  # v17 path peaks ~35 deg/s
    raise_table_shift_path = "data/v24_table_shift.pt"



@configclass
class JaxV2RaiseCubeEnvCfg(JaxV2RaiseTableEnvCfg):
    """v25: v22/v23 (table-aware residual, smooth diagonal approach) with the cube placed
    anywhere in a box on the table (raise_cube_dx_m, raise_cube_dy_m around the old spot).
    The path shifts with the cube in x and y and the table in z (data/v25_shift_xyz.pt:
    hand x, y, z at constant hand roll and pitch, yaw free, shoulder roll held). The cube's
    position is already observed (robot-base frame)."""

    raise_table_shift_path = "data/v25_shift_xyz.pt"
    raise_cube_dx_m = 0.02
    raise_cube_dy_m = 0.02
    # v34 (user 2026-11-04: table heights +/-3 cm): RAISE_TABLE_DZ (m) sets the table top z range
    if os.environ.get("RAISE_TABLE_DZ"):
        raise_table_dz_m = float(os.environ["RAISE_TABLE_DZ"])


# User (2026-11-01): the cube range was too small. The cube sits 6 cm in from the table's
# near edge (x -0.40 vs edge -0.34, y 0.26 vs side 0.35), so it can move +2 cm toward the
# robot, -8 cm away, -8 / +6 cm sideways. The linear v25 table is replaced by exact IK of
# every knot on a 2 cm grid (artifacts/claude_scripts/v27_grid_probe.py: residual 0 mm, no
# joint limit hit up to +/-8 cm), bilinear in between.
# y below -5 cm: the yaw-free hand turns and a finger plate sinks 1-3.5 mm into the cube
# (v27_pose_check.py); a fixed yaw instead needs 60-70 deg forearm/wrist counter-twist.
V27_CUBE_BOX = (-0.08, 0.02, -0.05, 0.06)


@configclass
class JaxV2RaiseCubeWideEnvCfg(JaxV2RaiseCubeEnvCfg):
    """v27: v25 over the wide cube box with the exact-IK shift grid."""

    # v27c: x shift ramps in after the hand is above the table (v27c_regrid.py)
    raise_shift_grid_path = "data/v27c_shift_grid.pt"
    # v34: exact-IK grids at a 3 cm lower / higher table (v34_grid_probe.py + v34_regrid.py)
    raise_shift_grid_dzm3_path = "data/v34c_shift_grid_dzm3.pt"
    raise_shift_grid_dzp3_path = "data/v34c_shift_grid_dzp3.pt"
    # v41: table moved toward the robot (GL_TABLE_EDGE, see v2_front_grasp_lift): path grids for it; the policy keeps
    # seeing the nominal table edge (the path moved with the table), raise_table_edge_real_x is the real one
    raise_table_edge_real_x = None
    if os.environ.get("GL_TABLE_EDGE"):
        raise_shift_grid_path = "data/v41_basec.pt"
        raise_shift_grid_dzm3_path = "data/v41_m3c.pt"
        raise_shift_grid_dzp3_path = "data/v41_p3c.pt"
        if os.environ.get("GL_GRID_SET"):  # v49: wider grids (see v2_front_grasp_lift)
            _gs = os.environ["GL_GRID_SET"]
            raise_shift_grid_path = f"data/{_gs}_basec.pt"
            raise_shift_grid_dzm3_path = f"data/{_gs}_m3c.pt"
            raise_shift_grid_dzp3_path = f"data/{_gs}_p3c.pt"
        raise_table_edge_real_x = -float(os.environ["GL_TABLE_EDGE"])
        raise_shift_early_s = float(os.environ.get("RAISE_SHIFT_EARLY", "2.0"))
    # v35: pre-grasp hand moved by this in x (m, negative = away from the robot), as GL_REACH_DX
    raise_reach_dx_m = float(os.environ.get("GL_REACH_DX", "0"))
    raise_reach_dy_m = float(os.environ.get("GL_REACH_DY", "0"))
    if (raise_reach_dx_m or raise_reach_dy_m) and not os.environ.get("GL_TABLE_EDGE"):  # v41 grids cover the reach
        raise_shift_grid_path = "data/v34wc.pt"
        raise_shift_grid_dzm3_path = "data/v34w_dzm3c.pt"
        raise_shift_grid_dzp3_path = "data/v34w_dzp3c.pt"
    raise_cube_touch_scale = 2.0
    raise_cube_scale = 300.0
    raise_cube_box_m = V27_CUBE_BOX


class JaxV2RaiseTableEnv(JaxV2ArmRaiseEnv):
    cfg: JaxV2RaiseTableEnvCfg

    def _setup_scene(self):
        super()._setup_scene()
        self.table = RigidObject(RigidObjectCfg(prim_path="/World/envs/env_.*/Table", spawn=None))
        self.scene.rigid_objects["table"] = self.table

    def _reset_idx(self, env_ids: torch.Tensor | None):
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        if not hasattr(self, "_table_off"):
            self._table_off = torch.zeros((self.num_envs, 2), device=self.device)
            self._table_bias = torch.zeros((self.num_envs, 2), device=self.device)
        c = self.cfg
        lim = torch.tensor([c.raise_table_dx_m, c.raise_table_dz_m], device=self.device)
        self._table_off[ids] = lim * (2.0 * torch.rand((len(ids), 2), device=self.device) - 1.0)
        fixed = getattr(self, "_table_off_fixed", None)
        if fixed is not None:
            self._table_off[ids] = fixed
        self._table_bias[ids] = c.raise_table_bias_m * torch.randn((len(ids), 2), device=self.device)
        if getattr(c, "raise_cube_dx_m", 0.0) or getattr(c, "raise_cube_dy_m", 0.0):
            # cube offset on the table (sampled before the reference-state start uses it)
            if getattr(self, "_cube_off", None) is None:
                self._cube_off = torch.zeros((self.num_envs, 2), device=self.device)
            lim_c = torch.tensor([c.raise_cube_dx_m, c.raise_cube_dy_m], device=self.device)
            self._cube_off[ids] = lim_c * (2.0 * torch.rand((len(ids), 2), device=self.device) - 1.0)
            if getattr(c, "raise_cube_box_m", None):  # v27: (x_min, x_max, y_min, y_max)
                b = torch.tensor(c.raise_cube_box_m, device=self.device)
                self._cube_off[ids] = b[0::2] + (b[1::2] - b[0::2]) * torch.rand((len(ids), 2), device=self.device)
            fixed_c = getattr(self, "_cube_off_fixed", None)
            if fixed_c is not None:
                self._cube_off[ids] = fixed_c[ids] if fixed_c.shape[0] == self.num_envs else fixed_c
        super()._reset_idx(env_ids)
        off3 = torch.zeros((len(ids), 3), device=self.device)
        off3[:, 0] = self._table_off[ids, 0]
        off3[:, 2] = self._table_off[ids, 1]
        pose = torch.zeros((len(ids), 7), device=self.device)
        pose[:, :3] = self.scene.env_origins[ids] + torch.tensor(c.table_center, device=self.device) + off3
        pose[:, 3] = 1.0
        self.table.write_root_pose_to_sim(pose, env_ids=ids)
        cube = self.object.data.root_state_w[ids].clone()
        cube[:, :3] = cube[:, :3] + off3
        if getattr(self, "_cube_off", None) is not None:
            cube[:, 0] = cube[:, 0] + self._cube_off[ids, 0]
            cube[:, 1] = cube[:, 1] + self._cube_off[ids, 1]
        cube[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(cube[:, :7], env_ids=ids)
        self.object.write_root_velocity_to_sim(cube[:, 7:], env_ids=ids)

    def table_measurement(self) -> torch.Tensor:
        """Table front-edge x and top z relative to the robot base (m), biased and noisy."""
        c = self.cfg
        root = self.robot.data.root_pos_w - self.scene.env_origins
        edge = c.raise_table_edge_x + self._table_off[:, 0] - root[:, 0]
        top = c.table_top_z + self._table_off[:, 1] - root[:, 2]
        meas = torch.stack((edge, top), dim=1) + self._table_bias
        return meas + c.raise_table_noise_m * torch.randn_like(meas)

    def _get_observations(self):
        obs = super()._get_observations()
        if not hasattr(self, "_table_off"):
            self._table_off = torch.zeros((self.num_envs, 2), device=self.device)
            self._table_bias = torch.zeros((self.num_envs, 2), device=self.device)
        obs["policy"] = torch.cat((obs["policy"], self.table_measurement()), dim=-1)
        return obs
