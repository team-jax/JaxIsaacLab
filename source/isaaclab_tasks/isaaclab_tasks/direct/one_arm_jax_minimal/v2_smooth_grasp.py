"""v8: v7 grasp task retuned for training from scratch with smoother motion.

Changes from v7, each from the v7 cycle-3 evaluation:
- Lift: the phase-1 reward pulled the arm toward initial_arm_q (shoulder yaw
  -51.5 deg, inward in front of the body), then phase 2 re-oriented the arm to
  about -27 deg. Phase 1 now shapes only shoulder pitch, elbow and wrist pitch.
- Motion: joints ran at their rate limit and stopped within one step, and every
  joint halted for the 0.5 s still-hold before phase 2. Actions now pass
  through a first-order low-pass filter and the phase-2 hold is shorter and
  looser.
- Table: the 8 cm clearance target is kept while lifting and drops to 1.5 cm
  only in phase 2 (v7 used 1.5 cm in every phase).
- Grasp: contact reward previously required all link positions to match the
  side-grasp anchors, so it never fired. Any finger contact with the cube and
  closing the thumb's palm-side joint near the cube now pay small rewards.
- Body: the asset has self-collision off (turning it on blows up the wrist),
  and the v8 fresh policy put the forearm/hand through the pole. Arm segments
  inside base_link's measured shape are penalized by depth.
- Cube: the body-penalty policy rested its fingertips on the cube top. Contact
  now pays only on a side face, a closed thumb pays every step while a finger
  presses a side, and moving the cube from its reset position is penalized.
- Enclosed: the largest reward is for the thumb and finger 1 or 2 pressing
  opposite side faces with the thumb closing and the cube still; held 0.5 s it
  is the grasp success. A straddle guide pays for second links outside
  opposite side faces at cube height, without penalizing face contact.
- Precise and still: enclosed now needs contact points in the central part of
  the faces (fingers 1 and 2 on one, thumb on the other) and a cube that has
  not moved or turned; moving it 1.5 cm or turning it 10 deg ends the episode.
- Lift: once the cube has been held 0.5 s the environment lifts the hand
  straight up (differential IK) and the policy keeps the fingers.
- Pads: holding and lift success need finger pads (not sides) on the cube.
- Three faces: lift success needs each finger's pad on a different side face.
- Lift success: the grasp slipped when lifted, so success is now lifting the cube
  2.5 cm while the thumb and a finger touch it and holding 0.5 s; cube rise is paid
  every step and only horizontal displacement counts as "moved".
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .v2_grasp_after_lift import JaxV2GraspAfterLiftEnv, JaxV2GraspAfterLiftEnvCfg

from isaaclab.utils.math import quat_apply_inverse

# The V2 asset disables self-collision, and enabling it makes the overlapping
# wrist/hand collision meshes blow up (wrist yaw spun >4000 deg), so the arm can
# pass through base_link. Keep the arm out of the body geometrically instead.
# Pole half-width of base_link.dae (square section) vs height in the base frame.
_POLE_Z = (0.0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
_POLE_HALF_WIDTH = (0.125, 0.125, 0.112, 0.101, 0.090, 0.081, 0.072, 0.065, 0.059, 0.054, 0.050, 0.047,
                    0.046, 0.045)
# Shoulder beam on top of the pole: z 0.75-0.895, |x| <= 0.15, |y| <= 0.06.
_BEAM = (0.75, 0.895, 0.15, 0.06)
# Arm segments (parent body origin -> child body origin) sampled for intrusion.
_ARM_SEGMENTS = (
    ("Upper_Arm", "Under_Arm"), ("Under_Arm", "Wrist_Pitch"), ("Wrist_Pitch", "Wrist_yaw"),
    ("Wrist_yaw", "MG996R"), ("MG996R", "RotationBody"), ("MG996R", "Finger_1_1"), ("MG996R", "Finger_2_1"),
    ("RotationBody", "Finger_3_1"), ("Finger_1_1", "Finger_1_2"), ("Finger_2_1", "Finger_2_2"),
    ("Finger_3_1", "Finger_3_2"),
)


@configclass
class JaxV2SmoothGraspEnvCfg(JaxV2GraspAfterLiftEnvCfg):
    # Shoulder pitch, elbow, wrist pitch (indices into JAX_V2_ARM_JOINTS).
    raised_shaping_arm_indices = (0, 3, 4)
    raised_hold_duration_s = 0.1
    max_hold_arm_speed_rad_s = 0.6
    table_clearance_target_m = 0.08
    grasp_table_clearance_target_m = 0.015
    # filtered = alpha * action + (1 - alpha) * filtered; 0.25 reaches ~90% of a
    # step change in 8 policy steps.
    action_filter_alpha = 0.25
    action_rate_scale = 0.002
    near_cube_distance_m = 0.07
    any_contact_scale = 0.0
    thumb_close_progress_scale = 0.0
    # Approximate link half-thickness added around the body geometry. With a
    # 2 cm margin, scale 5 and a -2 failure penalty, verified lift paths (which
    # come within 1.4 cm of the pole) were penalized more than the 0.02/step
    # lift reward, and a fresh policy learned not to lift at all.
    body_margin_m = 0.01
    body_intrusion_scale = 2.0
    # No termination: resuming the v8 fresh policy (which crosses 4 cm at step
    # ~49 of its lift) ended every episode at step 49, before any lift reward,
    # so the path could never be corrected. None disables it.
    body_intrusion_termination_m: float | None = None
    body_intrusion_failure_penalty = 0.0
    # The body-penalty policy rested its fingertips on the cube top and nudged
    # it. Penalize cube displacement from its reset position beyond a small
    # tolerance, pay contact only for pressing a side face, and pay a closed
    # thumb every step while a finger presses a side (the progress-based thumb
    # reward paid at most 1.0 per episode and never moved the thumb).
    cube_move_tolerance_m = 0.005
    cube_move_scale = 1.0
    # Cap on the penalized displacement: uncapped (6.9 cm -> 0.064/step) it
    # outweighed the straddle reward and the policy avoided the cube sides.
    cube_move_cap_m = 0.03
    side_contact_min_horizontal = 0.7
    side_contact_scale = 0.02
    thumb_closed_scale = 0.02
    # Nothing paid for reaching a straddle before contact, so the cube-still
    # policy hovered over the top. Per step near the cube, pay for second links
    # just outside opposite side faces (fingers 1/2 on one, thumb on the other,
    # either orientation), and pay more while both sides press oppositely.
    straddle_near_distance_m = 0.12
    # The distance-to-anchor straddle reward (0.12/step at its best) peaked
    # with second links 2.5 cm outside the faces, which fought side contact:
    # the straddle-balanced policy cycled fingers open <-> neutral every 1.3 s
    # and never closed the thumb. It now only checks that fingers 1/2 and the
    # thumb are on opposite sides of the cube, as a small guide.
    straddle_scale = 0.0
    straddle_sharpness = 20.0
    straddle_side_scale = 0.0
    # Outside-face straddle: per link, error is |x| and |z| from the cube
    # center plus how far it is inside (or past) its side face along y, so
    # pressing the face costs nothing. Best of the two mirror assignments.
    straddle_outside_scale = 0.08
    opposed_press_scale = 0.0
    # Enclosed: thumb presses one side face, finger 1 or 2 presses the
    # opposite one, the thumb's palm-side joint is past neutral toward closed,
    # and the cube is still. Largest per-step reward; held 0.5 s it is the
    # grasp success (replacing v7's open-pregrasp anchor test, which never
    # fired for the proximal-only grasp).
    # Thumb bend is not required for now (it touched the face at ~0 deg).
    enclose_thumb_min_closure = 0.0
    enclose_max_cube_speed_mps = 0.05
    enclosed_scale = 0.3
    enclosed_success_scale = 10.0
    # Finger joint speed is penalized while any finger presses a side.
    # 0.01 made the oscillating straddle policy drop side contact (0.23 ->
    # 0.008 fingers in 200 iterations) instead of holding still.
    contact_finger_speed_scale = 0.0

    # Precise grasp (from the enclosed_guide evaluation: finger 2 and the thumb
    # pinched the +x edge, finger 1 sat on the top edge, the cube moved 6-9 mm).
    # Contact points (not link origins) must lie in the central part of a side
    # face: fingers 1 and 2 on one y face, the thumb on the opposite one.
    precise_contact_center_m = 0.015
    precise_contact_face_tolerance_m = 0.004
    precise_finger_scale = 0.03
    # The cube must not move: displacement and rotation from its reset pose are
    # penalized and end the episode past the limits; an enclosed state also
    # requires the cube to be still.
    cube_rotation_scale = 0.5
    cube_move_limit_m = 0.015
    cube_rotation_limit_deg = 10.0
    enclose_max_cube_shift_m = 0.005
    enclose_max_cube_rotation_deg = 3.0
    # Lift test (precise_still model_4995: hand raised 33 mm, cube followed
    # 0-28 mm and slipped; 4/16 envs kept >= 80%). The grasp now has to lift
    # the cube: cube rise while the thumb and a finger touch it is
    # paid every step, and success is >= lift_success_m held for the hold
    # time. Displacement limits and penalties use horizontal motion only.
    lift_target_m = 0.03
    lift_success_m = 0.025
    lift_scale = 0.3
    lift_success_scale = 20.0
    # Stepping stone: with the cube held, the policy never moved the arm up
    # (cube rise stayed ~0.03 mm over 200 iterations). Pay hand (MG996R) rise
    # above its height at the first held step, up to lift_target_m, while held.
    # The policy-driven lift (hand_rise run) raised the hand up to 11 mm with no
    # cube rise, while a scripted straight lift with frozen fingers carried the
    # cube 17-28 mm in 10/16 envs. Auto lift: after the cube is held for
    # auto_lift_after_s, the environment raises the hand straight up by
    # differential IK on the arm (orientation kept) while the policy keeps
    # controlling the fingers.
    hand_lift_scale = 0.0
    auto_lift = True
    auto_lift_after_s = 0.5
    auto_lift_speed_mps = 0.02
    auto_lift_height_m = 0.04
    # The cube turned past 10 deg within ~0.1 s of every auto lift (episodes
    # stuck at ~430 steps for 500 iterations), leaving no time to learn. During
    # the auto lift allow up to 30 deg before ending the episode, but a lift
    # only counts as success while the cube is within 15 deg.
    auto_lift_rotation_limit_deg = 30.0
    lift_success_max_rotation_deg = 15.0
    # Pad contact (auto_lift_rot30: finger 2 pressed with its side, force 99%
    # along its bend axis; the thumb and finger 1 used their pads). Each joint
    # bends about its link's local z, closing toward local +x, so the cube's
    # force on a pad points along local -x. Pad quality = -f_x / |f|; holding
    # (which starts the auto lift) and lift success need the thumb and finger 1
    # or 2 on their pads.
    pad_contact_min_quality = 0.7
    pad_contact_scale = 0.05
    # pad_three_face model_6775: finger 1 touched with its pad at 0.5-1.7 N to
    # satisfy "on pad" while finger 2 pressed ~10 N with its side. Three-face
    # pad contacts need this force (applied to holding as well, it stopped the
    # auto lift: held fell from ~60% to <0.3%), and side-pushing force is
    # penalized.
    pad_contact_min_force_n = 2.0
    side_force_scale = 0.005
    # Three-face grasp (user's proposal): each finger's pad on a different side
    # face. Of four side faces any three include an opposite pair, so two
    # fingers squeeze and the third stops the turn/slide seen in auto_lift
    # (6-11 deg, 13-15 mm). Pays every step and is required for lift success;
    # holding (auto lift start) still needs only two fingers.
    side_face_tolerance_m = 0.004
    side_face_top_margin_m = 0.003
    three_face_scale = 0.1


class JaxV2SmoothGraspEnv(JaxV2GraspAfterLiftEnv):
    cfg: JaxV2SmoothGraspEnvCfg

    def __init__(self, cfg: JaxV2SmoothGraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._filtered_actions = torch.zeros((self.num_envs, 11), device=self.device)
        self._previous_thumb_quality = torch.zeros(self.num_envs, device=self.device)
        bodies = self.robot.body_names
        self._base_id = bodies.index("base_link")
        self._segment_ids = torch.tensor(
            [(bodies.index(a), bodies.index(b)) for a, b in _ARM_SEGMENTS], device=self.device
        )
        self._segment_t = torch.linspace(0.0, 1.0, 5, device=self.device)
        self._pole_z = torch.tensor(_POLE_Z, device=self.device)
        self._pole_half_width = torch.tensor(_POLE_HALF_WIDTH, device=self.device)
        self._body_intrusion = torch.zeros(self.num_envs, device=self.device)
        self._cube_start = torch.zeros((self.num_envs, 3), device=self.device)
        self._cube_start_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._enclosed_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._cube_start_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self._cube_shift = torch.zeros(self.num_envs, device=self.device)
        self._cube_rotation_deg = torch.zeros(self.num_envs, device=self.device)
        self._cube_rise = torch.zeros(self.num_envs, device=self.device)
        self._lift_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._hand_id = self.robot.body_names.index("MG996R")
        self._hand_z_at_hold = torch.zeros(self.num_envs, device=self.device)
        self._hand_z_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._held_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._auto_lifting = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._auto_lift_z0 = torch.zeros(self.num_envs, device=self.device)
        self._auto_lift_arm = torch.zeros((self.num_envs, 7), device=self.device)
        self._lift_arm_ids = list(self.arm_ids[:7])  # RotationBody stays with the policy

    def _setup_scene(self):
        super()._setup_scene()
        # v7 turned contact points off blaming them for an 18 s/iteration run;
        # that slowdown was the GPU stuck at P8. Precise grasp needs them.
        for i in (1, 2, 3):
            self.scene.sensors[f"finger_{i}"].cfg.track_contact_points = True

    def _cube_pose_change(self):
        """Displacement (m) and rotation (deg) of the cube from its reset pose."""
        pos = self.object.data.root_pos_w
        quat = self.object.data.root_quat_w
        start_p = torch.where(self._cube_start_valid[:, None], self._cube_start, pos)
        start_q = torch.where(self._cube_start_valid[:, None], self._cube_start_quat, quat)
        shift = torch.linalg.vector_norm((pos - start_p)[:, :2], dim=1)
        self._cube_rise = pos[:, 2] - start_p[:, 2]
        dot = (quat * start_q).sum(dim=1).abs().clamp(max=1.0)
        return shift, torch.rad2deg(2.0 * torch.acos(dot))

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if hasattr(self, "_filtered_actions"):
            ids = self.robot._ALL_INDICES if env_ids is None else env_ids
            self._filtered_actions[ids] = 0.0
            self._previous_thumb_quality[ids] = 0.0
            self._cube_start_valid[ids] = False
            self._enclosed_steps[ids] = 0
            self._lift_steps[ids] = 0
            self._hand_z_valid[ids] = False
            self._held_steps[ids] = 0
            self._auto_lifting[ids] = False

    def _pre_physics_step(self, actions: torch.Tensor):
        alpha = self.cfg.action_filter_alpha
        self._filtered_actions = alpha * actions.clamp(-1.0, 1.0) + (1.0 - alpha) * self._filtered_actions
        super()._pre_physics_step(self._filtered_actions)
        if self.cfg.auto_lift:
            self._apply_auto_lift()

    def _apply_auto_lift(self):
        hold_steps = round(self.cfg.auto_lift_after_s / (self.cfg.sim.dt * self.cfg.decimation))
        hand_z = self.robot.data.body_pos_w[:, self._hand_id, 2]
        start = (~self._auto_lifting) & (self._held_steps >= hold_steps)
        self._auto_lift_z0 = torch.where(start, hand_z, self._auto_lift_z0)
        self._auto_lift_arm = torch.where(
            start[:, None], self.robot.data.joint_pos[:, self._lift_arm_ids], self._auto_lift_arm
        )
        self._auto_lifting |= start
        if not self._auto_lifting.any():
            return
        # Straight-up hand twist while below the lift height, zero after.
        rising = self._auto_lifting & (hand_z - self._auto_lift_z0 < self.cfg.auto_lift_height_m)
        twist = torch.zeros((self.num_envs, 6), device=self.device)
        twist[:, 2] = self.cfg.auto_lift_speed_mps * self.cfg.sim.dt * self.cfg.decimation * rising
        jac = self.robot.root_physx_view.get_jacobians()[:, self._hand_id - 1][:, :, self._lift_arm_ids]
        dq = torch.linalg.lstsq(jac, twist[:, :, None]).solution[:, :, 0]
        self._auto_lift_arm = self._auto_lift_arm + dq * self._auto_lifting[:, None]
        lifted_targets = self.targets[:, self._lift_arm_ids]
        self.targets[:, self._lift_arm_ids] = torch.where(
            self._auto_lifting[:, None], self._auto_lift_arm, lifted_targets
        )

    def _body_intrusion_depth(self) -> torch.Tensor:
        pos = self.robot.data.body_pos_w
        start, end = pos[:, self._segment_ids[:, 0]], pos[:, self._segment_ids[:, 1]]
        t = self._segment_t[None, None, :, None]
        points = (start[:, :, None] + t * (end - start)[:, :, None]).reshape(self.num_envs, -1, 3)
        base_p = pos[:, self._base_id]
        base_q = self.robot.data.body_quat_w[:, self._base_id]
        count = points.shape[1]
        local = quat_apply_inverse(
            base_q[:, None].expand(-1, count, -1).reshape(-1, 4), (points - base_p[:, None]).reshape(-1, 3)
        ).reshape(self.num_envs, count, 3)
        x, y, z = local[..., 0].abs(), local[..., 1].abs(), local[..., 2]
        margin = self.cfg.body_margin_m
        # Piecewise-linear pole half-width at each point height.
        zc = z.clamp(float(self._pole_z[0]), float(self._pole_z[-1]))
        k = torch.bucketize(zc, self._pole_z).clamp(1, len(self._pole_z) - 1)
        z0, z1 = self._pole_z[k - 1], self._pole_z[k]
        w0, w1 = self._pole_half_width[k - 1], self._pole_half_width[k]
        half_width = w0 + (w1 - w0) * (zc - z0) / (z1 - z0)
        pole = (half_width + margin - torch.maximum(x, y)).clamp_min(0.0) * ((z >= 0.0) & (z <= _BEAM[0]))
        beam_z0, beam_z1, beam_x, beam_y = _BEAM
        beam = torch.minimum(
            torch.minimum(beam_x + margin - x, beam_y + margin - y),
            torch.minimum(beam_z1 + margin - z, z - (beam_z0 - margin)),
        ).clamp_min(0.0)
        return torch.maximum(pole, beam).amax(dim=1)

    def _get_dones(self):
        terminated, timeout = super()._get_dones()
        self._body_intrusion = self._body_intrusion_depth()
        self._cube_shift, self._cube_rotation_deg = self._cube_pose_change()
        rotation_limit = torch.where(
            self._auto_lifting,
            torch.full_like(self._cube_rotation_deg, self.cfg.auto_lift_rotation_limit_deg),
            torch.full_like(self._cube_rotation_deg, self.cfg.cube_rotation_limit_deg),
        )
        cube_moved = (self._cube_shift > self.cfg.cube_move_limit_m) | (self._cube_rotation_deg > rotation_limit)
        return terminated | self._body_intruded() | cube_moved, timeout

    def _body_intruded(self) -> torch.Tensor:
        if self.cfg.body_intrusion_termination_m is None:
            return torch.zeros_like(self._body_intrusion, dtype=torch.bool)
        return self._body_intrusion > self.cfg.body_intrusion_termination_m

    def _get_rewards(self):
        reward = super()._get_rewards()
        _, force_n, _, _ = self._contacts()
        touching = force_n >= self.cfg.min_contact_force_n
        any_contact_reward = self.cfg.any_contact_scale * touching.float().mean(dim=1)
        tips = self.robot.data.body_pos_w[:, self.fingertip_ids]
        tip_cube_distance = torch.linalg.vector_norm(tips.mean(dim=1) - self.object.data.root_pos_w, dim=1)
        near = (self.reward_phase == 2) & (tip_cube_distance < self.cfg.near_cube_distance_m)
        thumb_id = self.finger_ids[2]
        thumb_close = self.cfg.finger_close[2]
        thumb_quality = (self.robot.data.joint_pos[:, thumb_id] / thumb_close).clamp(0.0, 1.0) * near
        thumb_reward = self.cfg.thumb_close_progress_scale * (thumb_quality - self._previous_thumb_quality)
        self._previous_thumb_quality = thumb_quality.detach()

        # Side contact: the finger's second link presses the cube mostly
        # horizontally and sits below the cube's top face.
        cube_pos = self.object.data.root_pos_w
        cube_q = self.object.data.root_quat_w[:, None, :].expand(-1, 3, -1).reshape(-1, 4)
        forces_w = torch.stack(
            [self.scene.sensors[f"finger_{i}"].data.force_matrix_w[:, 0, 0] for i in (1, 2, 3)], dim=1
        )
        local_force = quat_apply_inverse(cube_q, forces_w.reshape(-1, 3)).reshape(-1, 3, 3)
        local_link = quat_apply_inverse(cube_q, (tips - cube_pos[:, None]).reshape(-1, 3)).reshape(-1, 3, 3)
        horizontal = torch.linalg.vector_norm(local_force[..., :2], dim=-1) / force_n.clamp_min(1e-6)
        below_top = local_link[..., 2] < self.cfg.object_size[2] / 2
        side_touching = touching & (horizontal > self.cfg.side_contact_min_horizontal) & below_top
        side_contact_reward = self.cfg.side_contact_scale * side_touching.float().mean(dim=1)
        thumb_closed = (self.robot.data.joint_pos[:, thumb_id] / thumb_close).clamp(0.0, 1.0)
        thumb_closed_reward = self.cfg.thumb_closed_scale * thumb_closed * side_touching.any(dim=1) * near

        # Straddle: second-link origins vs anchors just outside opposite side
        # faces in the cube frame, best of the two mirror assignments.
        side = self.cfg.object_size[1] / 2 + self.cfg.pregrasp_face_margin_m
        anchors = local_link.new_tensor(((0.0, side, 0.0), (0.0, side, 0.0), (0.0, -side, 0.0)))
        mirrored = anchors * anchors.new_tensor((1.0, -1.0, 1.0))
        straddle_error = torch.minimum(
            torch.linalg.vector_norm(local_link - anchors, dim=-1).mean(dim=1),
            torch.linalg.vector_norm(local_link - mirrored, dim=-1).mean(dim=1),
        )
        straddle_near = (self.reward_phase == 2) & (tip_cube_distance < self.cfg.straddle_near_distance_m)
        straddle_quality = torch.exp(-self.cfg.straddle_sharpness * straddle_error) * straddle_near
        straddle_reward = self.cfg.straddle_scale * straddle_quality
        # Opposed press: the thumb and finger 1 or 2 press side faces with
        # opposite cube-frame y forces.
        thumb_fy = local_force[:, 2, 1]
        finger_opposed = side_touching[:, :2] & (local_force[:, :2, 1] * thumb_fy[:, None] < 0.0)
        opposed_press = side_touching[:, 2] & finger_opposed.any(dim=1)
        opposed_press_reward = self.cfg.opposed_press_scale * opposed_press.float()

        # Cube pose change from the first step of the episode (computed in
        # _get_dones, which runs before the rewards).
        cube_shift, cube_rotation_deg = self._cube_shift, self._cube_rotation_deg
        self._cube_start = torch.where(self._cube_start_valid[:, None], self._cube_start, cube_pos)
        self._cube_start_quat = torch.where(
            self._cube_start_valid[:, None], self._cube_start_quat, self.object.data.root_quat_w
        )
        self._cube_start_valid[:] = True
        cube_move_penalty = self.cfg.cube_move_scale * (
            (cube_shift - self.cfg.cube_move_tolerance_m).clamp(0.0, self.cfg.cube_move_cap_m)
        ) + self.cfg.cube_rotation_scale * torch.deg2rad(cube_rotation_deg).clamp(max=0.5)

        # Precise contact: contact point in the central part of a y face.
        points_w = torch.stack(
            [self.scene.sensors[f"finger_{i}"].data.contact_pos_w[:, 0, 0] for i in (1, 2, 3)], dim=1
        ).nan_to_num(0.0)
        contact_c = quat_apply_inverse(cube_q, (points_w - cube_pos[:, None]).reshape(-1, 3)).reshape(-1, 3, 3)
        half_y = self.cfg.object_size[1] / 2
        on_y_face = (contact_c[..., 1].abs() - half_y).abs() < self.cfg.precise_contact_face_tolerance_m
        centered = (contact_c[..., 0].abs() < self.cfg.precise_contact_center_m) & (
            contact_c[..., 2].abs() < self.cfg.precise_contact_center_m
        )
        precise = touching & (horizontal > self.cfg.side_contact_min_horizontal) & on_y_face & centered
        face_sign = torch.sign(contact_c[..., 1])
        fingers_same_face = precise[:, 0] & precise[:, 1] & (face_sign[:, 0] == face_sign[:, 1])
        precise_grasp = fingers_same_face & precise[:, 2] & (face_sign[:, 2] == -face_sign[:, 0])
        precise_finger_reward = self.cfg.precise_finger_scale * precise.float().sum(dim=1)

        # Opposite-side guide: fingers 1/2 and the thumb on opposite cube y
        # sides, below the top face.
        fingers_y = local_link[:, :2, 1].mean(dim=1)
        straddle_side = (
            straddle_near & (fingers_y * local_link[:, 2, 1] < 0.0) & below_top.all(dim=1)
        )
        straddle_side_reward = self.cfg.straddle_side_scale * straddle_side.float()

        half_y = self.cfg.object_size[1] / 2
        side_sign = local_link.new_tensor((1.0, 1.0, -1.0))
        outside_error = []
        for sign in (1.0, -1.0):
            y_inside = (half_y - sign * side_sign * local_link[..., 1]).clamp_min(0.0)
            outside_error.append(
                (local_link[..., 0].abs() + local_link[..., 2].abs() + y_inside).mean(dim=1)
            )
        straddle_outside_error = torch.minimum(*outside_error)
        # Pre-contact guide only: closing the thumb's palm-side joint lifts its
        # second link ~6.5 cm toward the hand, so once a finger presses a side
        # face the guide is held at its maximum and closing costs nothing.
        straddle_outside_quality = torch.where(
            side_touching.any(dim=1),
            torch.ones_like(straddle_outside_error),
            torch.exp(-self.cfg.straddle_sharpness * straddle_outside_error),
        )
        straddle_outside_reward = self.cfg.straddle_outside_scale * straddle_near * straddle_outside_quality

        cube_speed = torch.linalg.vector_norm(self.object.data.root_lin_vel_w, dim=1)
        enclosed = (
            precise_grasp & (thumb_closed >= self.cfg.enclose_thumb_min_closure)
            & (cube_speed < self.cfg.enclose_max_cube_speed_mps)
            & (cube_shift < self.cfg.enclose_max_cube_shift_m)
            & (cube_rotation_deg < self.cfg.enclose_max_cube_rotation_deg)
        )
        enclosed_reward = self.cfg.enclosed_scale * enclosed.float() * (0.5 + 0.5 * thumb_closed)
        self._enclosed_steps = torch.where(enclosed, self._enclosed_steps + 1, 0)
        cube_rise = self._cube_rise
        # Held = the thumb and finger 1 or 2 touch the cube. The opposed-press
        # test also needs the thumb's link origin below the cube top, which the
        # precise_still thumb (pressing the top edge) never passed.
        link_q = self.robot.data.body_quat_w[:, self.fingertip_ids].reshape(-1, 4)
        force_link = quat_apply_inverse(link_q, forces_w.reshape(-1, 3)).reshape(-1, 3, 3)
        pad_quality = (-force_link[..., 0] / force_n.clamp_min(1e-6)).clamp(0.0, 1.0) * touching
        on_pad = pad_quality > self.cfg.pad_contact_min_quality
        firm_pad = on_pad & (force_n >= self.cfg.pad_contact_min_force_n)
        side_force = (force_link[..., 2].abs() * touching).sum(dim=1)
        side_force_penalty = self.cfg.side_force_scale * side_force.clamp(max=20.0)
        pad_contact_reward = self.cfg.pad_contact_scale * pad_quality.sum(dim=1)
        # Side face of each contact point: 0 +x, 1 -x, 2 +y, 3 -y, -1 none.
        half = self.cfg.object_size[0] / 2
        cx, cy, cz = contact_c[..., 0], contact_c[..., 1], contact_c[..., 2]
        on_x = (cx.abs() - half).abs() < self.cfg.side_face_tolerance_m
        on_y = (cy.abs() - half).abs() < self.cfg.side_face_tolerance_m
        below_top_face = cz < half - self.cfg.side_face_top_margin_m
        face_id = torch.full_like(cx, -1, dtype=torch.long)
        face_id = torch.where(on_x & ~on_y, (cx < 0).long(), face_id)
        face_id = torch.where(on_y & ~on_x, 2 + (cy < 0).long(), face_id)
        face_id = torch.where(firm_pad & below_top_face, face_id, torch.full_like(face_id, -1))
        three_face = (
            (face_id >= 0).all(dim=1) & (face_id[:, 0] != face_id[:, 1])
            & (face_id[:, 0] != face_id[:, 2]) & (face_id[:, 1] != face_id[:, 2])
        )
        three_face_reward = self.cfg.three_face_scale * three_face.float()
        held = on_pad[:, 2] & on_pad[:, :2].any(dim=1)
        lift_quality = (cube_rise / self.cfg.lift_target_m).clamp(0.0, 1.0) * held
        lift_reward = self.cfg.lift_scale * lift_quality
        hand_z = self.robot.data.body_pos_w[:, self._hand_id, 2]
        first_hold = held & ~self._hand_z_valid
        self._hand_z_at_hold = torch.where(first_hold, hand_z, self._hand_z_at_hold)
        self._hand_z_valid |= held
        self._held_steps = torch.where(held, self._held_steps + 1, 0)
        self._last_held = held
        hand_rise = (hand_z - self._hand_z_at_hold) * self._hand_z_valid
        hand_lift_reward = self.cfg.hand_lift_scale * held * (hand_rise / self.cfg.lift_target_m).clamp(0.0, 1.0)
        lifted = (
            held & three_face & (cube_rise >= self.cfg.lift_success_m)
            & (self._cube_rotation_deg < self.cfg.lift_success_max_rotation_deg)
        )
        self._lift_steps = torch.where(lifted, self._lift_steps + 1, 0)
        self.success = self._lift_steps >= self.required_hold_steps
        enclosed_success_reward = self.cfg.lift_success_scale * self.success.float()
        finger_speed = self.robot.data.joint_vel[:, self.finger_ids].square().sum(dim=1)
        contact_finger_speed_penalty = (
            self.cfg.contact_finger_speed_scale * finger_speed * side_touching.any(dim=1)
        )

        self.extras["log"].update({
            "straddle_side": straddle_side.float().mean(),
            "precise_fingers": precise.float().sum(dim=1).mean(),
            "cube_rise_m": cube_rise.mean(),
            "lift_quality": lift_quality.mean(),
            "lift_held": held.float().mean(),
            "pad_quality_f1": pad_quality[:, 0].mean(),
            "pad_quality_f2": pad_quality[:, 1].mean(),
            "pad_quality_thumb": pad_quality[:, 2].mean(),
            "three_face": three_face.float().mean(),
            "finger_side_force_n": side_force.mean(),
            "fingers_on_side_face": (face_id >= 0).float().sum(dim=1).mean(),
            "hand_rise_m": hand_rise.mean(),
            "auto_lifting": self._auto_lifting.float().mean(),
            "lift_success": self.success.float().mean(),
            "precise_grasp": precise_grasp.float().mean(),
            "cube_rotation_deg": cube_rotation_deg.mean(),
            "straddle_outside_error_m": straddle_outside_error.mean(),
            "straddle_outside_reward": straddle_outside_reward.mean(),
            "enclosed": enclosed.float().mean(),
            "enclosed_success": self.success.float().mean(),
            "thumb_closed_fraction": thumb_closed.mean(),
            "contact_finger_speed_penalty": contact_finger_speed_penalty.mean(),
        })
        self.extras["metrics"]["enclosed"] = enclosed
        self.extras["metrics"]["cube_rise_m"] = cube_rise
        self.extras["metrics"]["three_face"] = three_face
        self.extras["metrics"]["precise_fingers"] = precise.float().sum(dim=1)
        self.extras["metrics"]["precise_grasp"] = precise_grasp
        self.extras["metrics"]["cube_rotation_deg"] = cube_rotation_deg
        self.extras["metrics"]["success"] = self.success
        self.extras["log"].update({
            "any_contact_reward": any_contact_reward.mean(),
            "fingers_touching": touching.float().sum(dim=1).mean(),
            "fingers_side_touching": side_touching.float().sum(dim=1).mean(),
            "side_contact_reward": side_contact_reward.mean(),
            "thumb_close_reward": thumb_reward.mean(),
            "thumb_closed_reward": thumb_closed_reward.mean(),
            "thumb_closure_near_cube": thumb_quality.mean(),
            "cube_shift_m": cube_shift.mean(),
            "cube_move_penalty": cube_move_penalty.mean(),
            "straddle_error_m": straddle_error.mean(),
            "straddle_reward": straddle_reward.mean(),
            "opposed_press": opposed_press.float().mean(),
        })
        self.extras["metrics"]["straddle_error_m"] = straddle_error
        self.extras["metrics"]["opposed_press"] = opposed_press
        self.extras["metrics"]["fingers_side_touching"] = side_touching.float().sum(dim=1)
        self.extras["metrics"]["cube_shift_m"] = cube_shift
        intrusion_penalty = self.cfg.body_intrusion_scale * self._body_intrusion
        intruded = self._body_intruded()
        self.extras["log"].update({
            "body_intrusion_m": self._body_intrusion.mean(),
            "body_intrusion_termination": intruded.float().mean(),
        })
        self.extras["metrics"]["fingers_touching"] = touching.float().sum(dim=1)
        self.extras["metrics"]["body_intrusion_m"] = self._body_intrusion
        return (
            reward + any_contact_reward + thumb_reward - intrusion_penalty
            - self.cfg.body_intrusion_failure_penalty * intruded.float()
            + side_contact_reward + thumb_closed_reward - cube_move_penalty
            + straddle_reward + opposed_press_reward
            + straddle_side_reward + enclosed_reward + enclosed_success_reward - contact_finger_speed_penalty
            + straddle_outside_reward + precise_finger_reward + lift_reward + hand_lift_reward
            + pad_contact_reward + three_face_reward - side_force_penalty
        )
