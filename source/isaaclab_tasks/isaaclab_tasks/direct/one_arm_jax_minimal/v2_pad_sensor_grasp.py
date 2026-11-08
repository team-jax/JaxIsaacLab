"""v9: v8 with six pad pressure sensors, matching the planned real hand.

The real hand will carry one pressure sensor on the first (proximal) and one
on the second phalanx of each finger. v8 observed only the three second-link
contact forces. v9 adds cube contact sensors on the three first links and
observes, for all six links, a simulated RA12P reading of the force component
normal to the link's pad; a push on the side of a link does not register, as on
a pad-mounted sensor.

Each finger joint bends about its link's local z axis and closes toward local
+x (measured on the second links: a pad contact puts the cube's force along
local -x), so the pad-normal force is -f_x in the link frame.

Sensor model (Marveldex RA12P, "FSR Sensor Data Sheet", Marveldex technical
specification H201005, p.7): sensing range 5 gf - 4 kgf, resistance infinite
without load and falling non-linearly with force (curve below, read from the
data-sheet graph; its conductance trace equals 50 / R[kOhm] at every point),
read through the data sheet's divider Vout = Vin * Rc / (Ra + Rc). The
observation is that divider output with Rc = 10 kOhm, quantized to a 10-bit
ADC. During training each sensor gets an episode-level gain error (data sheet:
output tolerance max 20%) and per-step noise.

Observation: 40 -> 43 values (pressure 3 -> 6). v8 checkpoints are widened
with convert_v8_to_v9_checkpoint.py (new inputs start at zero weight).

Rewards: by default the v8 rewards, unchanged, so a v8 policy (widened by
convert_v8_to_v9_checkpoint.py) keeps its behaviour and only adapts to the new
pressure inputs. JaxV2PadSensorFirstPhalanxEnvCfg (task ...-v9-FirstPhalanx)
switches to the first-phalanx grasp rewards:

Grasp target (user): the thumb's palm-side joint fully folded (+90 deg) and
each finger's FIRST phalanx pad pressing the middle of a cube side face, the
three on different faces. v8's second-link contact terms are switched off;
v9 pays thumb folding near the cube, centred first-link pad contacts and the
three-face state, guides the first-link pad centres outside the side faces
before contact, and redefines holding (which starts the auto lift) and lift
success on first-link pads.
"""

from __future__ import annotations

import torch

from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from .v2_smooth_grasp import JaxV2SmoothGraspEnv, JaxV2SmoothGraspEnvCfg


# RA12P force (gf) -> resistance (kOhm), read from the data-sheet graph (which
# starts at 55 gf; the 5 gf / 60 kOhm point is an extrapolation).
_RA12P_FORCE_GF = (5.0, 55.0, 100.0, 165.0, 260.0, 350.0, 470.0, 590.0, 750.0, 975.0, 1290.0, 1585.0,
                   1920.0, 2465.0, 3240.0, 4000.0)
_RA12P_RES_KOHM = (60.0, 19.5, 11.5, 8.8, 7.5, 6.8, 5.4, 4.4, 4.0, 3.5, 3.1, 2.9, 2.7, 2.6, 2.55, 2.5)


@configclass
class JaxV2PadSensorGraspEnvCfg(JaxV2SmoothGraspEnvCfg):
    observation_space = 43
    proximal_body_names = ("Finger_1_1", "Finger_2_1", "Finger_3_1")
    sensor_min_force_gf = 5.0
    sensor_rc_kohm = 10.0
    sensor_adc_levels = 1023
    # Training-time sensor randomization (turn off for evaluation).
    sensor_randomize = True
    sensor_gain_tolerance = 0.2
    sensor_noise_std = 0.005

    # False: v8 rewards unchanged (reproduce the pre-sensor policy with the
    # six-sensor observation). True: the first-phalanx grasp rewards below.
    use_first_phalanx_rewards = False
    # Auto-lift trigger: v8 restarts the 0.5 s hold count on any lost step.
    # The transplanted 7050 policy held the cube steadily, but finger 1's
    # ~0.9 N contact flickered on and off every step, so the count never got
    # past 2-3. Smooth "held" with an exponential average (alpha 0.1, about
    # 10 steps) and count steps while it stays above 0.5 (v8-reward mode only).
    soft_hold_alpha = 0.1
    # The flicker ran ~10 steps per cycle with ~32% held (EMA peak 0.51), so
    # 0.5 still never triggered; 0.25 starts the lift on that steady pinch.
    soft_hold_threshold = 0.25
    # Pad grasp (user: hold with the front of the finger segments, which the
    # RA12P pads can read). The adapted 7050 policy pinched with finger 2's
    # side, invisible to pad sensors, and shook at ~4 Hz as finger 1's light
    # pad contact flickered. Pay each finger's larger pad reading (first or
    # second link), penalize side force on all six links, and count "held"
    # (auto-lift trigger) only when the thumb and finger 1 or 2 press a pad.
    pad_reading_scale = 0.05
    # 0.005 left finger 2 pressing with its side (4 N, no pad reading) after
    # 1000 iterations of the facing reward; make side pressing cost more.
    pad_side_force_scale = 0.02
    pad_hold_force_n = 0.5
    # Forced lift test: the policy learned to keep "held" just under the
    # trigger (pad-held ~28%, auto lift ~0.1%) because a failed irreversible
    # lift loses the rest of the episode's contact rewards. With the cube
    # touched, the auto lift now starts at this time regardless.
    forced_lift_time_s = 8.0
    # Facing (forced_lift model_10047: finger 1 and the thumb had the cube on
    # their pad side, +x of the link, but finger 2 stood beside it, cube at
    # x -17 mm / z +26 mm in its frame, pressing with its side). Near the cube,
    # pay each finger for the cube centre lying on its pad side (+x) and not off
    # along its joint axis (z), in its second-link frame.
    facing_scale = 0.2
    # facing_side4x: finger 2 still lifted with its side (~2 N) because the
    # lift reward (0.3/step) pays for any grip that raises the cube. Cancel the
    # lift reward in proportion to side force, fully at this force.
    lift_side_cancel_force_n = 2.0
    facing_z_sharpness = 40.0
    # Sigmoid on x so a finger with the cube behind it (x < 0) still gets a
    # gradient toward turning its pad to the cube.
    facing_x_scale_m = 0.01


@configclass
class JaxV2PadSensorFirstPhalanxEnvCfg(JaxV2PadSensorGraspEnvCfg):
    use_first_phalanx_rewards = True
    # v8 second-link contact terms off (they paid the straight-thumb pinch).
    side_contact_scale = 0.0
    thumb_closed_scale = 0.0
    precise_finger_scale = 0.0
    pad_contact_scale = 0.0
    three_face_scale = 0.0
    side_force_scale = 0.0
    straddle_outside_scale = 0.0
    enclosed_scale = 0.0
    lift_scale = 0.0
    lift_success_scale = 0.0
    # v9 first-link grasp terms.
    thumb_fold_scale = 0.05
    thumb_fold_success = 0.8
    proximal_center_scale = 0.1
    proximal_center_sharpness = 40.0
    proximal_three_face_scale = 0.2
    proximal_guide_scale = 0.08
    proximal_hold_force_n = 0.5
    v9_lift_scale = 0.3
    v9_lift_success_scale = 20.0
    # Curriculum: starting from v8 model_1998 (which pushed the cube 11 cm),
    # the 1.5 cm / 10 deg termination ended every episode at ~270 steps on
    # first touch. Terminate only on gross motion while first-link grasping is
    # learned; the displacement and rotation penalties stay.
    cube_move_limit_m = 0.05
    cube_rotation_limit_deg = 30.0


class JaxV2PadSensorGraspEnv(JaxV2SmoothGraspEnv):
    cfg: JaxV2PadSensorGraspEnvCfg

    def __init__(self, cfg: JaxV2PadSensorGraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.proximal_ids = [self.robot.body_names.index(n) for n in cfg.proximal_body_names]
        self._log_force = torch.log(torch.tensor(_RA12P_FORCE_GF, device=self.device))
        self._log_res = torch.log(torch.tensor(_RA12P_RES_KOHM, device=self.device))
        self._sensor_gain = torch.ones((self.num_envs, 6), device=self.device)
        self._v9_held_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._v9_lift_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if hasattr(self, "_v9_held_steps"):
            ids = self.robot._ALL_INDICES if env_ids is None else env_ids
            self._v9_held_steps[ids] = 0
            self._v9_lift_steps[ids] = 0
        if hasattr(self, "_sensor_gain") and self.cfg.sensor_randomize:
            ids = self.robot._ALL_INDICES if env_ids is None else env_ids
            tol = self.cfg.sensor_gain_tolerance
            self._sensor_gain[ids] = 1.0 + (2.0 * torch.rand((len(ids), 6), device=self.device) - 1.0) * tol

    def sensor_readings(self, pad_force_n: torch.Tensor) -> torch.Tensor:
        """RA12P divider output in [0, 1] (10-bit) for pad-normal forces in N."""
        force_gf = pad_force_n * (1000.0 / 9.80665)
        log_f = torch.log(force_gf.clamp(self.cfg.sensor_min_force_gf, float(_RA12P_FORCE_GF[-1])))
        k = torch.bucketize(log_f, self._log_force).clamp(1, len(_RA12P_FORCE_GF) - 1)
        f0, f1 = self._log_force[k - 1], self._log_force[k]
        r0, r1 = self._log_res[k - 1], self._log_res[k]
        res = torch.exp(r0 + (r1 - r0) * (log_f - f0) / (f1 - f0))
        rc = self.cfg.sensor_rc_kohm
        loaded = force_gf >= self.cfg.sensor_min_force_gf
        out = (rc / (res + rc)) * loaded
        if self.cfg.sensor_randomize:
            # No load means infinite resistance and a zero reading; noise only on loaded sensors.
            out = (out * self._sensor_gain + self.cfg.sensor_noise_std * torch.randn_like(out)) * loaded
        levels = self.cfg.sensor_adc_levels
        return torch.round(out.clamp(0.0, 1.0) * levels) / levels

    def _setup_scene(self):
        super()._setup_scene()
        for i, name in enumerate(self.cfg.proximal_body_names, start=1):
            self.scene.sensors[f"proximal_{i}"] = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{name}",
                filter_prim_paths_expr=["/World/envs/env_.*/Object"],
                update_period=0.0, track_contact_points=True,
            ))

    def pad_forces(self) -> torch.Tensor:
        """Pad-normal cube force (N) on [first links 1-3, second links 1-3]."""
        names = [f"proximal_{i}" for i in (1, 2, 3)] + [f"finger_{i}" for i in (1, 2, 3)]
        forces_w = torch.stack([self.scene.sensors[n].data.force_matrix_w[:, 0, 0] for n in names], dim=1)
        link_ids = self.proximal_ids + list(self.fingertip_ids)
        link_q = self.robot.data.body_quat_w[:, link_ids].reshape(-1, 4)
        forces_link = quat_apply_inverse(link_q, forces_w.reshape(-1, 3)).reshape(-1, 6, 3)
        return (-forces_link[..., 0]).clamp_min(0.0)

    def _get_observations(self):
        object_b, hand_b, object_vel_b = self._state()
        if hasattr(self, "_obs_base_off"):  # v3: cube position relative to the V2 base origin (object - hand unchanged)
            object_b = object_b - self._obs_base_off; hand_b = hand_b - self._obs_base_off
        q = self.robot.data.joint_pos
        lo, hi = (self._obs_lo, self._obs_hi) if hasattr(self, "_obs_lo") else (self.lower, self.upper)  # v3: V2 limits
        normalized_q = 2.0 * (q - lo) / (hi - lo).clamp_min(1e-5) - 1.0
        pressure = self.sensor_readings(self.pad_forces())
        observation = torch.cat((
            normalized_q, self.robot.data.joint_vel * 0.2,
            object_b, object_vel_b, object_b - hand_b, pressure,
        ), dim=-1)
        return {"policy": observation.clamp(-5.0, 5.0)}

    def _get_rewards(self):
        reward = super()._get_rewards()
        if not self.cfg.use_first_phalanx_rewards:
            if not hasattr(self, "_soft_held_steps"):
                self._soft_held_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                self._held_ema = torch.zeros(self.num_envs, device=self.device)
            fresh = self.episode_length_buf <= 1
            a = self.cfg.soft_hold_alpha
            pad_n = self.pad_forces()  # [first 1-3, second 1-3], N
            finger_pad = torch.maximum(pad_n[:, :3], pad_n[:, 3:])
            pad_held = (finger_pad[:, 2] >= self.cfg.pad_hold_force_n) & (
                finger_pad[:, :2] >= self.cfg.pad_hold_force_n
            ).any(dim=1)
            self._held_ema = torch.where(fresh, 0.0, (1 - a) * self._held_ema + a * pad_held.float())
            readings = self.sensor_readings(pad_n)
            pad_reading_reward = self.cfg.pad_reading_scale * torch.maximum(readings[:, :3], readings[:, 3:]).sum(dim=1)
            names = [f"proximal_{i}" for i in (1, 2, 3)] + [f"finger_{i}" for i in (1, 2, 3)]
            forces_w = torch.stack([self.scene.sensors[n].data.force_matrix_w[:, 0, 0] for n in names], dim=1)
            link_q = self.robot.data.body_quat_w[:, self.proximal_ids + list(self.fingertip_ids)].reshape(-1, 4)
            f_link = quat_apply_inverse(link_q, forces_w.reshape(-1, 3)).reshape(-1, 6, 3)
            side_n = f_link[..., 2].abs().sum(dim=1)
            pad_side_penalty = self.cfg.pad_side_force_scale * side_n.clamp(max=20.0)
            cube_p = self.object.data.root_pos_w
            tips_q = self.robot.data.body_quat_w[:, self.fingertip_ids].reshape(-1, 4)
            tips_p = self.robot.data.body_pos_w[:, self.fingertip_ids]
            c_link = quat_apply_inverse(tips_q, (cube_p[:, None] - tips_p).reshape(-1, 3)).reshape(-1, 3, 3)
            facing = torch.sigmoid(c_link[..., 0] / self.cfg.facing_x_scale_m) * torch.exp(
                -self.cfg.facing_z_sharpness * c_link[..., 2].abs()
            )
            tip_cube = torch.linalg.vector_norm(tips_p.mean(dim=1) - cube_p, dim=1)
            near = (self.reward_phase == 2) & (tip_cube < self.cfg.straddle_near_distance_m)
            facing_reward = self.cfg.facing_scale * facing.sum(dim=1) * near
            self.extras["log"]["facing_f1"] = facing[:, 0].mean()
            self.extras["log"]["facing_f2"] = facing[:, 1].mean()
            self.extras["log"]["facing_thumb"] = facing[:, 2].mean()
            cube_rise = (cube_p[:, 2] - self._cube_start[:, 2]).clamp_min(0.0)
            lift_side_cancel = self.cfg.lift_scale * (cube_rise / self.cfg.lift_target_m).clamp(0.0, 1.0) * (
                side_n / self.cfg.lift_side_cancel_force_n
            ).clamp(0.0, 1.0)
            self.extras["log"]["lift_side_cancel"] = lift_side_cancel.mean()
            reward = reward + pad_reading_reward - pad_side_penalty + facing_reward - lift_side_cancel
            self.extras["log"]["pad_held"] = pad_held.float().mean()
            self.extras["log"]["pad_reading_reward"] = pad_reading_reward.mean()
            self.extras["log"]["all_link_side_force_n"] = side_n.mean()
            steady = self._held_ema > self.cfg.soft_hold_threshold
            self._soft_held_steps = torch.where(steady, self._soft_held_steps + 1, 0)
            forced_step = round(self.cfg.forced_lift_time_s / (self.cfg.sim.dt * self.cfg.decimation))
            touched = (self.pad_forces().sum(dim=1) > 0) | (self._last_held)
            touched = touched | (torch.stack([self.scene.sensors[f"finger_{i}"].data.force_matrix_w[:, 0, 0] for i in (1, 2, 3)],
                                             dim=1).norm(dim=-1).sum(dim=1) > 0.5)
            force_lift = (self.episode_length_buf >= forced_step) & touched
            big = torch.full_like(self._soft_held_steps, 10 ** 6)
            self._held_steps = torch.where(force_lift, big, self._soft_held_steps)
            self.extras["log"]["held_ema"] = self._held_ema.mean()
            return reward
        cube_p = self.object.data.root_pos_w
        cube_q = self.object.data.root_quat_w
        half = self.cfg.object_size[0] / 2
        pad_n = self.pad_forces()[:, :3]  # first links: finger 1, finger 2, thumb
        pressed = pad_n >= self.cfg.proximal_hold_force_n
        # First-link contact points in the cube frame.
        points = torch.stack(
            [self.scene.sensors[f"proximal_{i}"].data.contact_pos_w[:, 0, 0] for i in (1, 2, 3)], dim=1
        ).nan_to_num(0.0)
        q3 = cube_q[:, None, :].expand(-1, 3, -1).reshape(-1, 4)
        c = quat_apply_inverse(q3, (points - cube_p[:, None]).reshape(-1, 3)).reshape(-1, 3, 3)
        ax, ay, az = c[..., 0].abs(), c[..., 1].abs(), c[..., 2]
        on_x = ax >= ay
        # Distance from the middle of the face it touches (face plane + in-plane offsets).
        off_face = torch.where(on_x, (ax - half).abs() + ay, (ay - half).abs() + ax) + az.abs()
        center_quality = torch.exp(-self.cfg.proximal_center_sharpness * off_face) * pressed
        face_id = torch.where(on_x, (c[..., 0] < 0).long(), 2 + (c[..., 1] < 0).long())
        face_id = torch.where(pressed & (az.abs() < half - 0.003), face_id, torch.full_like(face_id, -1))
        three_face = (
            (face_id >= 0).all(dim=1) & (face_id[:, 0] != face_id[:, 1])
            & (face_id[:, 0] != face_id[:, 2]) & (face_id[:, 1] != face_id[:, 2])
        )
        tips = self.robot.data.body_pos_w[:, self.fingertip_ids]
        tip_cube = torch.linalg.vector_norm(tips.mean(dim=1) - cube_p, dim=1)
        near = (self.reward_phase == 2) & (tip_cube < self.cfg.straddle_near_distance_m)
        thumb_fold = (self.robot.data.joint_pos[:, self.finger_ids[2]] / self.cfg.finger_close[2]).clamp(0.0, 1.0)
        # Pre-contact guide: first-link pad centres (link midpoints) just
        # outside opposite side faces at cube height, held at max once pressed.
        mids = 0.5 * (self.robot.data.body_pos_w[:, self.proximal_ids] + tips)
        m = quat_apply_inverse(q3, (mids - cube_p[:, None]).reshape(-1, 3)).reshape(-1, 3, 3)
        side_sign = m.new_tensor((1.0, 1.0, -1.0))
        errs = []
        for sign in (1.0, -1.0):
            inside = (half - sign * side_sign * m[..., 1]).clamp_min(0.0)
            errs.append((m[..., 0].abs() + m[..., 2].abs() + inside).mean(dim=1))
        guide_q = torch.where(pressed.any(dim=1), torch.ones_like(errs[0]),
                              torch.exp(-20.0 * torch.minimum(errs[0], errs[1])))
        held = pressed[:, 2] & pressed[:, :2].any(dim=1)
        self._v9_held_steps = torch.where(held, self._v9_held_steps + 1, 0)
        self._held_steps = self._v9_held_steps  # drives the v8 auto lift
        rise = self._cube_rise
        lift_q = (rise / self.cfg.lift_target_m).clamp(0.0, 1.0) * held
        lifted = (
            held & three_face & (thumb_fold >= self.cfg.thumb_fold_success)
            & (rise >= self.cfg.lift_success_m)
            & (self._cube_rotation_deg < self.cfg.lift_success_max_rotation_deg)
        )
        self._v9_lift_steps = torch.where(lifted, self._v9_lift_steps + 1, 0)
        self.success = self._v9_lift_steps >= self.required_hold_steps
        terms = (
            self.cfg.thumb_fold_scale * thumb_fold * near
            + self.cfg.proximal_center_scale * center_quality.sum(dim=1)
            + self.cfg.proximal_three_face_scale * three_face.float()
            + self.cfg.proximal_guide_scale * near * guide_q
            + self.cfg.v9_lift_scale * lift_q
            + self.cfg.v9_lift_success_scale * self.success.float()
        )
        self.extras["log"].update({
            "v9_thumb_fold": thumb_fold.mean(),
            "v9_proximal_pressed": pressed.float().sum(dim=1).mean(),
            "v9_center_quality": center_quality.sum(dim=1).mean(),
            "v9_three_face": three_face.float().mean(),
            "v9_held": held.float().mean(),
            "v9_lift_quality": lift_q.mean(),
            "v9_success": self.success.float().mean(),
        })
        self.extras["metrics"]["success"] = self.success
        self.extras["metrics"]["three_face"] = three_face
        self.extras["metrics"]["v9_thumb_fold"] = thumb_fold
        return reward + terms

