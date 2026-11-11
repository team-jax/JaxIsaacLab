"""v13: v12 pre-grasp, then close the hand on the cube (no lift).

Same L start, hand shape and arm control as v12. For the first
``close_after_s`` the episode is the v12 pre-grasp (fingers held open, any cube
contact penalized). After that the finger actions (8:11) close the hand: each
sets an absolute fraction of the way from the open shape to ``close_shape_deg``
(fingers 1/2 with both joints, the thumb with its second joint only; its
palm-side joint and RotationBody stay as in the pre-grasp). In the close phase
the thumb and finger 1 or 2 pressing opposite faces with their pads is paid,
side-of-link force and cube motion are penalized, and the v12 placement terms
stay so the arm holds the pre-grasp position.
"""

from __future__ import annotations

import math
import os

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from .v2_pad_sensor_grasp import JaxV2PadSensorGraspEnv
from .v2_front_pregrasp import JaxV2FrontPregraspEnv, JaxV2FrontPregraspEnvCfg


@configclass
class JaxV2FrontGraspCloseEnvCfg(JaxV2FrontPregraspEnvCfg):
    close_after_s = 5.0
    # finger_ids order: f1_1, f2_1, f3_1, f1_2, f2_2, f3_2. The thumb's first
    # joint keeps its pre-grasp value.
    close_shape_deg = (60.0, 60.0, 50.0, 90.0, 90.0, 90.0)
    grasp_force_n = 0.3
    # distal_thumb (from model_1600): grasp_both 54% at 1620 fell to 0.2% by
    # 1849 while the mean episode reward stayed ~2000: closing paid about the
    # same as staying open, so the policy drifted to open. Pay holding more
    # and guide the fingers toward the distal_scan shape.
    grasp_both_scale = 3.0
    # Guide target (deg, finger_ids order): thumb 30/30, fingers 0/30.
    grasp_guide_deg = (0.0, 0.0, 30.0, 30.0, 30.0, 30.0)
    grasp_guide_rad = 0.3
    grasp_guide_scale = 1.0
    scripted_close = True
    # Finger 1 joints (deg) in the close phase, or None to close it with
    # finger 2. With the hand rolled, finger 1 sits above the cube top.
    finger1_close_deg = None
    # User (2026-10-25, lift close-up): the finger segments reach past the
    # side faces and press the far (-x) face; lift probe forces agree
    # (finger 1 / thumb contact force along -x). At close onset move the hand
    # by this world offset (m) over close_shift_s by differential IK, then
    # close the fingers.
    # lift_probe (model_1849, scripted lift, 256 envs) by +x pull-back:
    # 1 cm held 2% at 10.5 s, 3 cm 99% / 45% above 4 cm at 11.5 s,
    # 4 cm 99% held and 99% above 4 cm, 5 cm 91%.
    close_shift_m = (0.04, 0.0, 0.0)
    close_rotation_delta_deg = 0.0
    # 4 cm cube: distal_scan (v12_cube4 model_1544 pre-grasp) - finger 1 sits
    # above the cube and never touches; finger 2 + thumb distal pads hold it.
    # lift_probe (diagnostic IK lift) for guide thumb 10 / rest 0 with a 4 cm
    # pull-back: 86% above 4 cm, 51% clean (rotation < 15 deg); thumb 10/30
    # held worse (36%).
    grasp_guide_deg = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)
    close_shift_m = (0.0, 0.0, 0.0)
    # 5.5 cm cube: the 5 cm distal_scan shape (thumb 30/30, fingers 0/30).
    grasp_guide_deg = (0.0, 0.0, 30.0, 30.0, 30.0, 30.0)
    # User: keep the pre-grasp finger shape (second joints 0), close only the
    # first joints. flat_scan4 (v12_c55_dx5 model_3038, 4 reps): fingers 10,
    # thumb -27 -> all three distal pads 3/4, first links 0/0/0.58 N, cube 2 mm.
    grasp_guide_deg = (10.0, 10.0, -27.0, 0.0, 0.0, 0.0)
    # User: bring the thumb lower on the cube face. thumb_scan5 (60 shapes):
    # RotationBody -15 deg during the close lowers the thumb distal centre
    # from ~23 to ~5-18 mm over the cube centre; thumb second joint barely
    # matters. Thumb first -30, RotationBody -15: all three distal pads,
    # finger first links 0 N.
    grasp_guide_deg = (10.0, 10.0, -30.0, 0.0, 0.0, 0.0)
    close_rotation_delta_deg = -15.0
    # Thumb turn moved into the pre-grasp (pre_rotation_delta_deg); close only.
    close_rotation_delta_deg = 0.0
    close_shift_s = 0.5
    # User (2026-10-27): the thumb and fingers 1/2 each closed on their own
    # schedule (fixed guide angles), so whichever touched first pushed the cube
    # and it wobbled. Balanced close: first joints only (second joints stay
    # straight), every digit advances at close_speed until its pads touch,
    # touching digits hold light contact until all three touch, then all
    # squeeze together to equal and opposite force (thumb = fingers 1+2 = S).
    close_force_balance = False
    close_touch_n = 0.3
    close_squeeze_n = 2.0
    close_gain_rad_per_n = 0.003
    close_speed_rad_s = 0.25
    close_first_max_deg = (30.0, 30.0, 0.0)
    # v16: squeeze total = close_squeeze_n + this x action 7 (0 = fixed).
    close_squeeze_range_n = 0.0
    # v16: thumb first-joint close limit = close_first_max_deg[2] + this x
    # action 8 (deg; 0 = fixed), so the policy can turn the thumb's distal pad
    # to face fingers 1/2.
    close_thumb_max_range_deg = 0.0
    # Low-pass on the pad forces the balanced close reacts to (EMA weight of
    # the previous value per step; 0 = raw). jitter_probe: raw contact forces
    # made the first joints chatter (5-17 deg/s step-to-step velocity change).
    close_force_ema = 0.0
    # balance_probe (c55_tforce2 model_4284, 64 envs): all three digits on the
    # cube at 6 s 100% (guide angles 0%), forces 1.0/1.0/2.0 N, cube x/y 0.1 mm,
    # rotation 0.3 deg (guide 3.2 mm, 4.0 deg).
    close_force_balance = True
    # scripted (from model_1600): with the close scripted, the policy learned
    # to stop the arm 6 cm from the target so the fingers closed on air, while
    # the logged close-phase reward was ~2x the pre-grasp reward. The learner
    # was changing the pre-grasp. Run the pre-grasp phase with this frozen
    # actor (deterministic mean; the learner's actions there have no effect).
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_grasp_close_v13/2026-10-25_14-23-52_v13_close_latch/model_1600.pt"
    # 4 cm cube (2026-10-26): pre-grasp retrained for it (v12_cube4, success
    # 84%, contact 0.2%).
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-26_00-48-06_v12_cube4/model_1544.pt"
    # Pre-grasp aimed 4 cm back (pre_target_dx_m, v12_cube4_dx4: success
    # 79%, no contact), so the close-onset pull-back is off.
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-26_23-39-25_v12_cube4_dx4/model_1793.pt"
    # User: reach in less - pre-grasp aimed 6 cm back (v12_cube4_dx6: success
    # 95%, no contact).
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-27_00-21-12_v12_cube4_dx6/model_2042.pt"
    # Pre-grasp height = finger 1/2 distal centres at cube centre (v12_cube4_distal_h).
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-27_07-59-05_v12_cube4_distal_h/model_2042.pt"
    # 5.5 cm cube, distal-height curriculum (v12_c55_h00: success 67%).
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-27_08-54-09_v12_c55_h00/model_2789.pt"
    # Pre-grasp 1 cm farther back (v12_c55_dx5) for first-joint-only closing.
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-27_16-35-53_v12_c55_dx5/model_3038.pt"
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-27_17-43-43_v12_c55_hide/model_3287.pt"
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-27_17-51-30_v12_c55_hiderb/model_3287.pt"
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-27_18-01-32_v12_c55_plate/model_3536.pt"
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-27_18-11-54_v12_c55_plate40/model_3785.pt"
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-27_18-20-30_v12_c55_plate_h29/model_4034.pt"
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-27_18-29-14_v12_c55_h29b/model_3850.pt"
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-27_22-21-55_v12_c55_mesh/model_3785.pt"
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-27_22-32-12_v12_c55_thumb0/model_3785.pt"
    # Back to the late-thumb-turn pre-grasp (c55_thumb0 collapsed).
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-27_18-01-32_v12_c55_plate/model_3536.pt"
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-27_22-41-25_v12_c55_h31/model_3785.pt"
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-27_22-51-01_v12_c55_tforce/model_3785.pt"
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-27_22-59-44_v12_c55_tforce2/model_4284.pt"
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-27_23-26-24_v12_c55_refpose2/model_4035.pt"
    # User: keep the pre-grasp checkpoint; fixes go to the close and the solver.
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-27_18-01-32_v12_c55_plate/model_3536.pt"
    # User: no abrupt motion. Pre-grasp retrained under the arm target
    # low-pass (smoothing curriculum 0.5 -> 0.7 -> 0.85 from model_3536):
    # success 97%, no contact, finger-on-table force 0.1 N (was ~4 N).
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-29_01-33-25_v12_sm085/model_4283.pt"
    # Robust to +/-1.5 deg start-pose error (reset-noise curriculum from model_4283;
    # a 2 deg offset dropped model_4283 + v16 lifts 97% -> 9%): success 94%.
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-29_02-26-17_v12_noise15/model_4532.pt"
    # Arm targets capped at 0.7 rad/s and finger-table force 0.3/N (rate
    # curriculum from model_4532): success 96%, the descent no longer hits the
    # table (chain: finger 2 tip 16.5 N -> 0), chain lift 64/64.
    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-29_13-54-40_v12_rate07/model_4781.pt"
    if os.environ.get("PRE_POLICY"):
        pre_policy_path = os.environ["PRE_POLICY"]
    close_ramp_s = 1.0
    close_residual_rad = 0.1
    grasp_reading_scale = 0.1
    grasp_side_scale = 0.05
    grasp_close_rate_scale = 0.05
    # close model_1544: once the fingers closed the arm swung away (shoulder
    # yaw 60, hand yaw -38 deg, hand pointing +y) and dragged the cube 8 cm.
    # In the close phase the arm target is the one at close onset plus this x
    # the action, and cube motion costs more.
    close_arm_range_rad = 0.15
    # frozen_pre model_1849: with the pre-grasp frozen, the learner used the
    # close-phase arm range (the latch switches from the frozen actor's
    # action to its own, up to 2 x 0.15 rad) to roll the shoulder +12 deg and
    # pull the hand off the cube before the fingers closed. Hold the arm
    # exactly at the onset target and keep the finger residual small.
    close_arm_range_rad = 0.0
    # Probe of model_1600 (256 envs, close phase): half-closed fingers held
    # the cube 89% of the time but closing pushed it ~4 cm, and 30 x the
    # shift beyond 1 cm (~0.9/step) ate the hold bonus, so open fingers
    # scored about the same. Allow 2 cm and cost less.
    grasp_cube_move_scale = 5.0
    # close_latch model_1793 stopped closing (both 72% at 1600 -> 8%): touching
    # lost the v12 success bonus (it requires no contact) and every mm of
    # cube shift cost. Pay the success bonus back when placed and touching,
    # and penalize shift only beyond 1 cm.
    grasp_cube_move_free_m = 0.02
    # User (2026-10-25): grasp with the whole face of the distal links. Pay
    # thumb and finger 1/2 second-link pad force (true pad normal, also for
    # the thumb) and those pads lying flat on the cube faces; first-link
    # contact costs a little. Action 7 (unused RotationBody slot) now drives
    # the finger 1/2 second joints separately from their first joints
    # (actions 8/9), so a finger can curl at the base and keep its tip flat.
    grasp_distal_force_n = 0.3
    grasp_flat_scale = 1.0
    grasp_proximal_scale = 0.05
    # close_keep model_1849 still stopped closing: curling the fingers moves
    # the pad centres off the v12 target and turns the pads, losing the
    # placement, facing, success and still terms. In the close phase those
    # are replaced by their value at close onset.


class JaxV2FrontGraspCloseEnv(JaxV2FrontPregraspEnv):
    cfg: JaxV2FrontGraspCloseEnvCfg

    def _closing(self) -> torch.Tensor:
        return self.episode_length_buf * self.step_dt >= self.cfg.close_after_s

    def _frozen_pre_actions(self) -> torch.Tensor | None:
        if not self.cfg.pre_policy_path or not hasattr(self, "_last_obs"):
            return None
        if not hasattr(self, "_pre_actor"):
            sd = torch.load(self.cfg.pre_policy_path, map_location=self.device, weights_only=False)["actor_state_dict"]
            self._pre_actor = {k: v.to(self.device) for k, v in sd.items()}
        a = self._pre_actor
        with torch.no_grad():
            x = (self._last_obs - a["obs_normalizer._mean"]) / (a["obs_normalizer._std"] + 1e-2)
            x = torch.nn.functional.elu(x @ a["mlp.0.weight"].T + a["mlp.0.bias"])
            x = torch.nn.functional.elu(x @ a["mlp.2.weight"].T + a["mlp.2.bias"])
            return x @ a["mlp.4.weight"].T + a["mlp.4.bias"]

    def _get_observations(self):
        obs = super()._get_observations()
        self._last_obs = obs["policy"].clone()
        return obs

    def _pre_physics_step(self, actions: torch.Tensor):
        frozen = self._frozen_pre_actions()
        if frozen is not None:
            actions = torch.where(self._closing()[:, None], actions, frozen)
        super()._pre_physics_step(actions)
        if not hasattr(self, "_arm_hold"):
            self._arm_hold = self.targets[:, self.arm_ids[:7]].clone()
        arm = self.arm_ids[:7]
        closing = self._closing()
        # Latch the pre-grasp arm target (minus the close-phase offset of the
        # current action, so the target does not jump) until closing starts.
        self._arm_hold = torch.where(
            closing[:, None], self._arm_hold, self.targets[:, arm] - self.cfg.close_arm_range_rad * self.actions[:, :7]
        )
        held = self._arm_hold + self.cfg.close_arm_range_rad * self.actions[:, :7]
        held = torch.minimum(torch.maximum(held, self.lower[arm]), self.upper[arm])
        self.targets[:, arm] = torch.where(closing[:, None], held, self.targets[:, arm])
        if any(self.cfg.close_shift_m):
            t = self.episode_length_buf * self.step_dt - self.cfg.close_after_s
            moving = closing & (t < self.cfg.close_shift_s)
            if not hasattr(self, "_shift_dq"):
                self._shift_dq = torch.zeros((self.num_envs, 7), device=self.device)
            self._shift_dq = torch.where(closing[:, None], self._shift_dq, 0.0)
            twist = torch.zeros((self.num_envs, 6, 1), device=self.device)
            twist[:, :3, 0] = self.targets.new_tensor(self.cfg.close_shift_m) * (self.step_dt / self.cfg.close_shift_s)
            twist = twist * moving[:, None, None]
            jac = self.robot.root_physx_view.get_jacobians()[:, self._hand_id - 1][:, :, arm]
            self._shift_dq = self._shift_dq + torch.linalg.lstsq(jac, twist).solution[:, :, 0]
            shifted = torch.minimum(torch.maximum(self.targets[:, arm] + self._shift_dq, self.lower[arm]), self.upper[arm])
            self.targets[:, arm] = torch.where(closing[:, None], shifted, self.targets[:, arm])

    def _hold_hand_shape(self):
        super()._hold_hand_shape()
        if not hasattr(self, "_close_shape"):
            self._close_shape = torch.tensor([math.radians(v) for v in self.cfg.close_shape_deg], device=self.device)
        # Fraction closed per joint: finger 1/2 second joints from action 7,
        # their first joints from action 8, thumb first joint (-30 -> +50 deg)
        # from action 9, thumb second joint from action 10. distal_scan
        # (model_1600 pre-grasp, 320 shapes): with the thumb first joint fixed
        # at -30 the thumb's distal pad rarely met the cube; thumb 30/30,
        # fingers 0/30 put all three distal pads flat on the faces (flat
        # 0.89/1.0/0.83, no first-link force, cube shift 1.5 mm).
        if self.cfg.scripted_close:
            # guide probe (model_1600, 256 envs): the guide shape held the cube
            # with all distal pads 94% of the time and paid ~3x staying open,
            # yet five PPO runs drifted back to open. Close to the guide shape
            # over close_ramp_s; the policy only adds a residual per channel.
            guide_q = self.targets.new_tensor([math.radians(v) for v in self.cfg.grasp_guide_deg])
            t = self.episode_length_buf * self.step_dt - self.cfg.close_after_s
            if any(self.cfg.close_shift_m):
                t = t - self.cfg.close_shift_s
            ramp = (t / self.cfg.close_ramp_s).clamp(0.0, 1.0)[:, None]
            res = self.cfg.close_residual_rad * self.actions[:, 7:11][:, [1, 1, 2, 0, 0, 3]]
            closed = self._hand_shape + ramp * (guide_q - self._hand_shape) + res
            closed = torch.minimum(torch.maximum(closed, self.lower[self.finger_ids]), self.upper[self.finger_ids])
        else:
            frac = 0.5 * (self.actions[:, 7:11] + 1.0)
            frac6 = frac[:, [1, 1, 2, 0, 0, 3]]
            closed = self._hand_shape + frac6 * (self._close_shape - self._hand_shape)
        if self.cfg.finger1_close_deg is not None:
            closed[:, 0] = math.radians(self.cfg.finger1_close_deg[0])
            closed[:, 3] = math.radians(self.cfg.finger1_close_deg[1])
        if self.cfg.close_force_balance:
            closed = self._balanced_close()
        self.targets[:, self.finger_ids] = torch.where(self._closing()[:, None], closed, self._hand_shape)
        if self.cfg.close_rotation_delta_deg:
            t = self.episode_length_buf * self.step_dt - self.cfg.close_after_s
            ramp = (t / self.cfg.close_ramp_s).clamp(0.0, 1.0)
            rot = self.cfg.start_arm_q[7] + ramp * math.radians(self.cfg.close_rotation_delta_deg)
            self.targets[:, self.arm_ids[7]] = torch.where(self._closing(), rot, self.targets[:, self.arm_ids[7]])

    def digit_force_from_torque(self) -> torch.Tensor:
        """v33 sensorless: per-digit grip force (N) estimated from the first finger joints' drive torque
        (what a servo with position feedback gives: kp (commanded - measured)), closing direction only,
        over close_torque_lever_m."""
        tau = self.robot.data.applied_torque[:, list(self.finger_ids)[:3]]
        return tau.clamp_min(0.0) / self.cfg.close_torque_lever_m

    def _balanced_close(self) -> torch.Tensor:
        closing = self._closing()
        if not hasattr(self, "_bal_q"):
            self._bal_q = self._hand_shape.expand(self.num_envs, -1).clone()
            self._bal_max = self._hand_shape.new_tensor([math.radians(v) for v in self.cfg.close_first_max_deg])
        self._bal_q = torch.where(closing[:, None], self._bal_q, self._hand_shape.expand(self.num_envs, -1))
        if getattr(self.cfg, "close_force_source", "pad") == "torque":
            f = self.digit_force_from_torque()
        else:
            pad = JaxV2PadSensorGraspEnv.pad_forces(self)
            f = pad[:, :3] + pad[:, 3:]  # per digit (finger 1, finger 2, thumb), N
        if self.cfg.close_force_ema:
            if not hasattr(self, "_bal_fema"):
                self._bal_fema = f.clone()
            self._bal_fema = torch.where(closing[:, None], self.cfg.close_force_ema * self._bal_fema + (1.0 - self.cfg.close_force_ema) * f, f)
            f = self._bal_fema
        on = f >= self.cfg.close_touch_n
        all_on = on.all(dim=1, keepdim=True)
        # v38: latched per episode once all three digits have touched (gates the lift with lift_wait_grip)
        if not hasattr(self, "_grip_latched"):
            self._grip_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not getattr(self.cfg, "lift_wait_squeeze_frac", 0.0):
            self._grip_latched = (self._grip_latched | all_on[:, 0]) & closing
        s_n = (self.cfg.close_squeeze_n + self.cfg.close_squeeze_range_n * self.actions[:, 7:8]).clamp_min(self.cfg.close_touch_n)
        want = torch.where(all_on, s_n * f.new_tensor([0.5, 0.5, 1.0]),
                           torch.full_like(f, self.cfg.close_touch_n))
        rl_grip = getattr(self.cfg, "close_rl_grip", False)
        if rl_grip:
            # v45 (user: "train the grasp"): the policy sets the squeeze of the finger side (action 7, fingers 1 and 2
            # share it) and of the thumb (action 9), close_rl_force_base_n + range x action each; the touch-then-squeeze
            # force loop below stays (like a gripper's force controller)
            a_ = self.actions
            sf = (self.cfg.close_rl_force_base_n + self.cfg.close_rl_force_range_n * a_[:, 7]).clamp_min(self.cfg.close_touch_n)
            st = (self.cfg.close_rl_force_base_n + self.cfg.close_rl_force_range_n * a_[:, 9]).clamp_min(self.cfg.close_touch_n)
            fmax = getattr(self.cfg, "close_rl_force_max_n", None)
            if fmax:
                # v47b: cap the learned squeeze (envs that dropped the cube at the L had squeezed at the 9 N maximum,
                # held ones at 7.3-8 N: the tilted distal pads pushed the cube out)
                sf = sf.clamp_max(fmax); st = st.clamp_max(fmax)
            want = torch.where(all_on, torch.stack((0.5 * sf, 0.5 * sf, st), dim=1), torch.full_like(f, self.cfg.close_touch_n))
        st_cfg = getattr(self.cfg, "close_settle", None)
        if st_cfg:
            # v68 probe (GL_CLOSE_SETTLE): time since all three digits first touched in this close; squeeze capped meanwhile
            if not hasattr(self, "_settle_t"):
                self._settle_t = torch.zeros(self.num_envs, device=self.device)
            started = all_on[:, 0] | (self._settle_t > 0)
            self._settle_t = torch.where(closing & started, self._settle_t + self.step_dt, torch.zeros_like(self._settle_t))
            want = torch.where((self._settle_t < st_cfg[0])[:, None], want.clamp_max(st_cfg[1]), want)
        frac = getattr(self.cfg, "lift_wait_squeeze_frac", 0.0)
        if frac:
            # v43b: the lift gate opens only once every digit holds this share of its squeeze target
            squeezed = all_on[:, 0] & (f >= frac * s_n * f.new_tensor([0.5, 0.5, 1.0])).all(dim=1)
            self._grip_latched = (self._grip_latched | squeezed) & closing
        dmax = self.cfg.close_speed_rad_s * self.step_dt
        step = (self.cfg.close_gain_rad_per_n * (want - f)).clamp(-dmax, dmax)
        step = torch.where(on, step, torch.full_like(step, dmax))
        pre = getattr(self.cfg, "pre_yaw", None)
        pre_wait = None
        if pre:
            # v79 (GL_PRE_YAW): the fingers stay open while the hand turns to the cube
            pre_wait = (self.episode_length_buf * self.step_dt - self.cfg.close_after_s < pre[0])[:, None]
            step = torch.where(pre_wait, torch.zeros_like(step), step)
        q_max = self._bal_max.expand(self.num_envs, -1).clone()
        if self.cfg.close_thumb_max_range_deg:
            q_max[:, 2] = q_max[:, 2] + math.radians(self.cfg.close_thumb_max_range_deg) * self.actions[:, 8]
            q_max[:, 2] = torch.maximum(q_max[:, 2], self._hand_shape[2])
        q1 = torch.minimum(torch.maximum(self._bal_q[:, :3] + step, self._hand_shape[:3]), q_max)
        q2 = self._hand_shape[3:].expand(self.num_envs, -1)
        if rl_grip:
            # v45: second (distal) joints curled by the policy: fingers 1 and 2 from action 8, thumb from action 10,
            # 0 .. close_rl_curl_deg past the pre-grasp shape for action 0 .. 1 (action <= 0: straight, so the action
            # filter starting at 0 after a reset does not curl the fingers)
            curl = math.radians(self.cfg.close_rl_curl_deg) * self.actions[:, [8, 8, 10]].clamp(0.0, 1.0)
            fix_f = getattr(self.cfg, "close_rl_curl_fix_finger_deg", None)
            if fix_f is not None:
                # v48f probe: fingers 1 and 2 distal curl fixed (the trained policies kept it at 0 deg; does curling them
                # bring the first joints onto the cube?)
                curl[:, :2] = math.radians(fix_f)
            if pre_wait is not None:
                curl = torch.where(pre_wait, torch.zeros_like(curl), curl)
            q2 = q2 + curl
        self._bal_q = torch.cat((q1, q2), dim=1)
        self._bal_f = f
        return self._bal_q

    def _get_rewards(self):
        reward = super()._get_rewards()
        log = self.extras["log"]
        closing = self._closing()
        # Undo the v12 no-contact and cube-move penalties in the close phase.
        names = [f"proximal_{i}" for i in (1, 2, 3)] + [f"finger_{i}" for i in (1, 2, 3)]
        contact_n = torch.stack(
            [torch.linalg.vector_norm(self.scene.sensors[n].data.force_matrix_w[:, 0, 0], dim=-1) for n in names], dim=1
        )
        touching = contact_n.amax(dim=1) > self.cfg.pre_contact_force_n
        if not hasattr(self, "_pre_positive_onset"):
            self._pre_positive_onset = torch.zeros(self.num_envs, device=self.device)
        self._pre_positive_onset = torch.where(closing, self._pre_positive_onset, self._pre_positive)
        undo = (
            self.cfg.pre_contact_scale * touching.float()
            + self.cfg.pre_cube_move_scale * self._cube_shift
            - self._pre_positive
            + self._pre_positive_onset
        )

        # Pad-normal force on [first links 1-3, second links 1-3], thumb included
        # (the v11 override counts any thumb force as a pad force).
        pad_n = JaxV2PadSensorGraspEnv.pad_forces(self)
        distal = pad_n[:, 3:] >= self.cfg.grasp_distal_force_n
        thumb_on = distal[:, 2]
        finger_on = distal[:, :2].any(dim=1)
        both = thumb_on & finger_on
        reading = self.sensor_readings(pad_n)[:, 3:].sum(dim=1)
        # Flatness: second-link pad direction (+x) against the cube face it
        # should meet (fingers on the `side` face, thumb on the other).
        cube_q = self.object.data.root_quat_w
        Y = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, 3)
        cube_y = quat_apply(cube_q, Y)
        cube = self.object.data.root_pos_w
        prox = self.robot.data.body_pos_w[:, self.proximal_ids[:2]]
        tips = self.robot.data.body_pos_w[:, self.fingertip_ids]
        side = torch.sign((((0.5 * (prox + tips[:, :2])).mean(dim=1) - cube) * cube_y).sum(dim=1)).clamp(min=-1.0)
        side = torch.where(side == 0, 1.0, side)
        X = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs * 3, 3)
        pad_dir = quat_apply(self.robot.data.body_quat_w[:, list(self.fingertip_ids)].reshape(-1, 4), X).reshape(-1, 3, 3)
        want = torch.stack((-side, -side, side), dim=1)[..., None] * cube_y[:, None]
        flat = (pad_dir * want).sum(dim=-1).clamp(0.0, 1.0)
        flat_paid = (flat * distal.float()).sum(dim=1)
        proximal_n = pad_n[:, :3].sum(dim=1).clamp(max=20.0)
        side_n = (contact_n - pad_n).clamp_min(0.0).sum(dim=1).clamp(max=20.0)
        close_rate = (self.actions[:, 7:11] - self.previous_actions[:, 7:11]).square().sum(dim=1)
        guide_q = torch.tensor([math.radians(v) for v in self.cfg.grasp_guide_deg], device=self.device)
        guide_err = (self.robot.data.joint_pos[:, self.finger_ids] - guide_q).abs().mean(dim=1)
        guide = torch.exp(-guide_err / self.cfg.grasp_guide_rad)
        grasp = (
            undo
            + self.cfg.grasp_guide_scale * guide
            + self.cfg.grasp_both_scale * both.float()
            + self.cfg.grasp_reading_scale * reading
            + self.cfg.grasp_flat_scale * flat_paid
            - self.cfg.grasp_proximal_scale * proximal_n
            - self.cfg.grasp_side_scale * side_n
            - self.cfg.grasp_cube_move_scale * (self._cube_shift - self.cfg.grasp_cube_move_free_m).clamp_min(0.0)
            - self.cfg.grasp_close_rate_scale * close_rate
        )
        reward = reward + closing.float() * grasp
        self._grasp_terms = {
            **self._pre_terms, "undo": undo, "guide": self.cfg.grasp_guide_scale * guide,
            "both": self.cfg.grasp_both_scale * both.float(), "reading": self.cfg.grasp_reading_scale * reading,
            "flat": self.cfg.grasp_flat_scale * flat_paid, "proximal": -self.cfg.grasp_proximal_scale * proximal_n,
            "side": -self.cfg.grasp_side_scale * side_n,
            "move": -self.cfg.grasp_cube_move_scale * (self._cube_shift - self.cfg.grasp_cube_move_free_m).clamp_min(0.0),
            "close_rate": -self.cfg.grasp_close_rate_scale * close_rate, "total": reward,
        }
        nc = closing.float().sum().clamp_min(1.0)
        npre = (~closing).float().sum().clamp_min(1.0)
        for key, v in self._grasp_terms.items():
            log[f"tc_{key}"] = (v * closing).sum() / nc
            log[f"tp_{key}"] = (v * ~closing).sum() / npre
        log["grasp_both"] = (both & closing).float().sum() / closing.float().sum().clamp_min(1.0)
        log["grasp_thumb_on"] = (thumb_on & closing).float().sum() / closing.float().sum().clamp_min(1.0)
        log["grasp_finger_on"] = (finger_on & closing).float().sum() / closing.float().sum().clamp_min(1.0)
        log["grasp_side_n"] = side_n.mean()
        log["grasp_close_frac"] = (0.5 * (self.actions[:, 7:11] + 1.0)).mean()
        log["grasp_flat"] = ((flat * distal.float()).sum(dim=1) * closing).sum() / closing.float().sum().clamp_min(1.0)
        log["grasp_proximal_n"] = proximal_n.mean()
        log["grasp_guide_err_deg"] = torch.rad2deg((guide_err * closing).sum() / closing.float().sum().clamp_min(1.0))
        log["pre_touch_before_close"] = (touching & ~closing).float().mean()
        return reward
