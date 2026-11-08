"""Minimal object-conditioned Stage 1 reach and stable three-finger grasp."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_assets.robots.one_arm_jax import JAX_ARM_JOINTS, JAX_FINGER_JOINTS, ONE_ARM_JAX_CFG


@configclass
class OneArmJaxMinimalEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 8.0
    # Eight arm deltas plus one independent closure delta per finger.
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(11,), dtype=np.float32)
    observation_space = 40
    state_space = 0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=7056, env_spacing=2.0, replicate_physics=True, clone_in_fabric=False,
    )
    # GL_ROBOT_DZ (equivalence check, 2026-11-06): raise the fixed robot base (and pedestal) by this much (m)
    robot_base_z = 1.20 - 0.895 + float(__import__("os").environ.get("GL_ROBOT_DZ", "0"))
    robot_yaw_quat = (0.7071067811865476, 0.0, 0.0, -0.7071067811865476)
    robot_cfg = ONE_ARM_JAX_CFG.replace(
        init_state=ONE_ARM_JAX_CFG.init_state.replace(
            pos=(0.0, 0.0, robot_base_z), rot=robot_yaw_quat,
        )
    )
    arm_joint_names = JAX_ARM_JOINTS
    finger_joint_names = JAX_FINGER_JOINTS
    fingertip_body_names = ("finger_1_link2", "finger_2_link2", "finger_3_link2")
    wrist_body_name = "rotation_base"
    table_cfg = sim_utils.CuboidCfg(
        size=(0.80, float(__import__("os").environ.get("GL_TABLE_W", "0.70")), 0.04),  # GL_TABLE_W: table width (y, m), diagnosis
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.2, dynamic_friction=1.0, restitution=0.0,
        ),
    )
    table_center = (-0.56, 0.0, 0.72)
    table_top_z = 0.74
    pedestal_cfg = sim_utils.CylinderCfg(
        radius=0.125, height=robot_base_z,
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    object_size = (0.050, 0.050, 0.050)
    object_half_height = object_size[2] / 2
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=object_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, solver_position_iteration_count=8,
                solver_velocity_iteration_count=1, max_depenetration_velocity=2.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2, dynamic_friction=1.0, restitution=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.220, 0.080, 0.765)),
    )
    object_x = (-0.225, -0.215)
    object_y = (0.075, 0.085)
    # Collision-free two-against-one pre-grasp, 6.5 cm above the cube-side
    # grasp height so open distal links clear the 0.74 m tabletop.
    initial_arm_q = (0.7412, 0.1759, -1.0472, -1.5251,
                     1.5224, 1.5707, 0.5004, 2.8648)
    arm_delta_scale = 0.035
    gripper_delta_scale = 0.01
    finger_open = (-0.034, -0.4415, -1.45, 0.50, 0.50, 0.50)
    finger_close = (-0.60, -0.80, -1.00, 1.90, 1.82, 1.82)
    # The 20 N pressure normalization is provisional until RA12P calibration.
    pressure_reference_force_n = 20.0
    min_contact_force_n = 0.5
    max_safe_contact_force_n = 30.0
    unsafe_contact_force_n = 100.0
    unsafe_consecutive_steps = 3
    max_relative_speed_mps = 0.05
    hold_duration_s = 0.5
    reach_distance_m = 0.07
    reach_distance_slope = 12.0
    # Bounded state rewards: slow approach and a steady pre-grasp no longer
    # receive a negative shaping reward. Success still dominates one episode.
    reach_scale = 0.005
    success_scale = 10.0
    inner_contact_scale = 0.02
    # Small shaping term for the measured two-against-one finger layout.
    pregrasp_scale = 0.005
    failure_termination_penalty = 2.0
    action_rate_scale = 0.001
    grasp_region_release_distance_m = 0.12
    diagnostic_table_contacts = False
    table_contact_body_names = fingertip_body_names
    # v63: per-link contact force with the cube (probes only; off in training)
    diagnostic_cube_contacts = False
    cube_contact_body_names = ()


class OneArmJaxMinimalEnv(DirectRLEnv):
    cfg: OneArmJaxMinimalEnvCfg

    def __init__(self, cfg: OneArmJaxMinimalEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.arm_ids = [self.robot.joint_names.index(n) for n in cfg.arm_joint_names]
        self.finger_ids = [self.robot.joint_names.index(n) for n in cfg.finger_joint_names]
        self.fingertip_ids = [self.robot.body_names.index(n) for n in cfg.fingertip_body_names]
        self.wrist_id = self.robot.body_names.index(cfg.wrist_body_name)
        self.lower = self.robot.data.soft_joint_pos_limits[0, :, 0]
        self.upper = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.targets = self.robot.data.default_joint_pos.clone()
        self.actions = torch.zeros((self.num_envs, 11), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.hold_steps = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.unsafe_steps = torch.zeros_like(self.hold_steps)
        self.required_hold_steps = round(cfg.hold_duration_s / (cfg.sim.dt * cfg.decimation))
        self.success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.grasp_enabled = torch.zeros_like(self.success)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.cfg.table_cfg.func(
            "/World/envs/env_0/Table", self.cfg.table_cfg, translation=self.cfg.table_center,
        )
        self.cfg.pedestal_cfg.func(
            "/World/envs/env_0/RobotPedestal", self.cfg.pedestal_cfg,
            translation=(0.0, 0.0, self.cfg.robot_base_z / 2),
        )
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        for i in (1, 2, 3):
            self.scene.sensors[f"finger_{i}"] = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{self.cfg.fingertip_body_names[i - 1]}",
                filter_prim_paths_expr=["/World/envs/env_.*/Object"],
                update_period=0.0, track_contact_points=True,
            ))
        if self.cfg.diagnostic_table_contacts:
            for name in self.cfg.table_contact_body_names:
                self.scene.sensors[f"table_contact_{name}"] = ContactSensor(ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{name}",
                    filter_prim_paths_expr=["/World/envs/env_.*/Table"],
                    update_period=0.0,
                ))
        if getattr(self.cfg, "diagnostic_cube_contacts", False):
            for name in self.cfg.cube_contact_body_names:
                self.scene.sensors[f"cube_contact_{name}"] = ContactSensor(ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{name}",
                    filter_prim_paths_expr=["/World/envs/env_.*/Object"],
                    update_period=0.0,
                ))
        sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    def _pre_physics_step(self, actions: torch.Tensor):
        self.previous_actions.copy_(self.actions)
        raw_actions = actions.clamp(-1.0, 1.0)
        # Track the pre-grasp region for evaluation, but let the policy choose
        # when to open or close the fingers from the first step.
        in_region, _, _ = self._grasp_region()
        self.grasp_enabled = in_region
        finger_actions = raw_actions[:, 8:11]
        self.actions = raw_actions
        self.targets[:, self.arm_ids] += self.cfg.arm_delta_scale * self.actions[:, :8]
        self.targets[:, self.arm_ids] = torch.minimum(
            torch.maximum(self.targets[:, self.arm_ids], self.lower[self.arm_ids]), self.upper[self.arm_ids]
        )
        finger_open = torch.tensor(self.cfg.finger_open, device=self.device)
        finger_close = torch.tensor(self.cfg.finger_close, device=self.device)
        six_actions = torch.cat((finger_actions, finger_actions), dim=1)
        self.targets[:, self.finger_ids] += self.cfg.gripper_delta_scale * six_actions * (finger_close - finger_open)
        self.targets[:, self.finger_ids] = torch.minimum(
            torch.maximum(self.targets[:, self.finger_ids], torch.minimum(finger_open, finger_close)),
            torch.maximum(finger_open, finger_close),
        )

    def _apply_action(self):
        self.robot.set_joint_position_target(self.targets)

    def _state(self):
        root = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        object_pos = self.object.data.root_pos_w
        hand_pos = self._grasp_midpoint()
        object_b = quat_apply_inverse(root_quat, object_pos - root)
        hand_b = quat_apply_inverse(root_quat, hand_pos - root)
        object_vel_b = quat_apply_inverse(
            root_quat, self.object.data.root_lin_vel_w - self.robot.data.root_lin_vel_w,
        )
        return object_b, hand_b, object_vel_b

    def _grasp_midpoint(self):
        tips = self.robot.data.body_pos_w[:, self.fingertip_ids]
        return tips.mean(dim=1)

    def _pregrasp_quality(self):
        """Place fingers 1/2 opposite finger 3 around the cube's X faces."""
        tips_w = self.robot.data.body_pos_w[:, self.fingertip_ids]
        object_w = self.object.data.root_pos_w
        object_q = self.object.data.root_quat_w[:, None, :].expand(-1, 3, -1)
        tips_o = quat_apply_inverse(
            object_q.reshape(-1, 4), (tips_w - object_w[:, None, :]).reshape(-1, 3),
        ).reshape(-1, 3, 3)
        half_x = self.cfg.object_size[0] / 2
        # Link origins sit behind their contact surfaces, so targets remain
        # outside the box while the fingers are open.
        anchors = tips_o.new_tensor((
            (half_x + 0.025, -0.018, 0.0),
            (half_x + 0.025, 0.018, 0.0),
            (-half_x - 0.025, 0.0, 0.0),
        ))
        placement = torch.linalg.vector_norm(tips_o - anchors[None], dim=-1).mean(dim=1)
        wrist_w = self.robot.data.body_pos_w[:, self.wrist_id]
        hand_axis = tips_w.mean(dim=1) - wrist_w
        to_object = object_w - wrist_w
        forward = (hand_axis * to_object).sum(dim=1) / (
            torch.linalg.vector_norm(hand_axis, dim=1) * torch.linalg.vector_norm(to_object, dim=1)
        ).clamp_min(1e-6)
        # The verified approach points down between the opposing fingers.
        alignment = (0.5 * (forward + 1.0)).clamp(0.0, 1.0)
        quality = torch.exp(-10.0 * placement) * alignment
        return quality, torch.linalg.vector_norm(object_w - tips_w.mean(dim=1), dim=1)

    def _grasp_region(self):
        quality, distance = self._pregrasp_quality()
        in_region = (quality > 0.08) & (distance < self.cfg.grasp_region_release_distance_m)
        return in_region, self._grasp_midpoint(), distance

    def _contacts(self):
        sensors = [self.scene.sensors[f"finger_{i}"].data for i in (1, 2, 3)]
        forces_w = torch.stack([sensor.force_matrix_w[:, 0, 0] for sensor in sensors], dim=1)
        points_w = torch.stack([sensor.contact_pos_w[:, 0, 0] for sensor in sensors], dim=1)
        force_n = torch.linalg.vector_norm(forces_w, dim=-1)
        object_pos = self.object.data.root_pos_w
        object_q = self.object.data.root_quat_w[:, None, :].expand(-1, 3, -1)
        relative_w = points_w - object_pos[:, None, :]
        relative = quat_apply_inverse(object_q.reshape(-1, 4), relative_w.reshape(-1, 3)).reshape(-1, 3, 3)
        local_force = quat_apply_inverse(object_q.reshape(-1, 4), forces_w.reshape(-1, 3)).reshape(-1, 3, 3)
        finger_pos = self.robot.data.body_pos_w[:, self.fingertip_ids]
        finger_q = self.robot.data.body_quat_w[:, self.fingertip_ids]
        other_side = torch.stack((
            finger_pos[:, 2], finger_pos[:, 2], 0.5 * (finger_pos[:, 0] + finger_pos[:, 1]),
        ), dim=1)
        local_contact = quat_apply_inverse(
            finger_q.reshape(-1, 4), (points_w - finger_pos).reshape(-1, 3),
        ).reshape(-1, 3, 3)
        local_inside = quat_apply_inverse(
            finger_q.reshape(-1, 4), (other_side - finger_pos).reshape(-1, 3),
        ).reshape(-1, 3, 3)
        # Link2's longitudinal axis is local Y. Keep only the lateral X/Z
        # contact position so a touch on the outside cannot count as inner.
        contact_lateral = local_contact[..., (0, 2)]
        inside_lateral = local_inside[..., (0, 2)]
        inner_cosine = (contact_lateral * inside_lateral).sum(dim=-1) / (
            torch.linalg.vector_norm(contact_lateral, dim=-1)
            * torch.linalg.vector_norm(inside_lateral, dim=-1)
        ).clamp_min(1e-6)
        inner = (inner_cosine > 0.2) & (torch.linalg.vector_norm(contact_lateral, dim=-1) > 0.001)
        half_x, half_y, half_z = (size / 2 for size in self.cfg.object_size)
        # Measured grasp approach: fingers 1/2 share the local +X side;
        # finger 3 opposes them on -X.
        # The Z extent rejects contacts from above and below the cube.
        expected_x = relative.new_tensor((half_x, half_x, -half_x))
        on_side = (relative[..., 0] - expected_x).abs() < 0.012
        face_extent = (relative[..., 1].abs() < half_y + 0.005) & (relative[..., 2].abs() < half_z - 0.003)
        face_force = local_force[..., 0].abs() / force_n.clamp_min(1e-6)
        eligible = (
            torch.isfinite(points_w).all(dim=-1) & inner
            & (force_n >= self.cfg.min_contact_force_n)
            & (force_n <= self.cfg.max_safe_contact_force_n)
        )
        valid = eligible & on_side & face_extent & (face_force > 0.25)
        contact_quality = (valid.float().sum(dim=1) / 3.0).square()
        return valid, force_n, valid.all(dim=1), contact_quality

    def _get_observations(self):
        object_b, hand_b, object_vel_b = self._state()
        q = self.robot.data.joint_pos
        normalized_q = 2.0 * (q - self.lower) / (self.upper - self.lower).clamp_min(1e-5) - 1.0
        valid_contact, force_n, _, _ = self._contacts()
        # Provisional RA12P-normalized second-phalanx contact force (P1/P2/P3).
        pressure = (force_n / self.cfg.pressure_reference_force_n).clamp(0.0, 1.0)
        observation = torch.cat((
            normalized_q, self.robot.data.joint_vel * 0.2,
            object_b, object_vel_b, object_b - hand_b, pressure,
        ), dim=-1)
        return {"policy": observation.clamp(-5.0, 5.0)}

    def _get_rewards(self):
        object_b, hand_b, _ = self._state()
        distance = torch.linalg.vector_norm(object_b - hand_b, dim=-1)
        # A bounded proximity reward, as used in Isaac Gym's Franka cube task.
        # The former gamma*potential(next)-potential(now) was negative for
        # stationary/slow approaches and gave no persistent grasp setup signal.
        reach_reward = self.cfg.reach_scale * torch.exp(-self.cfg.reach_distance_slope * distance)
        pregrasp_quality, _ = self._pregrasp_quality()
        pregrasp_reward = self.cfg.pregrasp_scale * pregrasp_quality
        valid_contact, force_n, contact_geometry, contact_quality = self._contacts()
        inner_contact_reward = self.cfg.inner_contact_scale * contact_quality
        hand_velocity = self.robot.data.body_lin_vel_w[:, self.fingertip_ids].mean(dim=1)
        relative_speed = torch.linalg.vector_norm(self.object.data.root_lin_vel_w - hand_velocity, dim=-1)
        grasp_candidate = contact_geometry & (relative_speed < self.cfg.max_relative_speed_mps)
        self.hold_steps = torch.where(grasp_candidate, self.hold_steps + 1, 0)
        self.success = self.hold_steps >= self.required_hold_steps
        unsafe = force_n.amax(dim=1) > self.cfg.unsafe_contact_force_n
        self.unsafe_steps = torch.where(unsafe, self.unsafe_steps + 1, 0)
        action_rate = ((self.actions - self.previous_actions) ** 2).sum(dim=1)
        fallen = self.object.data.root_pos_w[:, 2] < self.cfg.table_top_z - 0.05
        failed = fallen | (self.unsafe_steps >= self.cfg.unsafe_consecutive_steps)
        failure_penalty = self.cfg.failure_termination_penalty * failed.float()
        reward = (
            reach_reward + pregrasp_reward + inner_contact_reward
            + self.cfg.success_scale * self.success
            - failure_penalty - self.cfg.action_rate_scale * action_rate
        )
        self.extras["log"] = {
            "reach_reward": reach_reward.mean(),
            "pregrasp_reward": pregrasp_reward.mean(),
            "pregrasp_quality": pregrasp_quality.mean(),
            "inner_contact_reward": inner_contact_reward.mean(),
            "inner_contact_quality": contact_quality.mean(),
            "success": self.success.float().mean(),
            "valid_contact_3": valid_contact.all(dim=1).float().mean(),
            "grasp_candidate": grasp_candidate.float().mean(),
            "hold_steps": self.hold_steps.float().mean(),
            "unsafe_contact": unsafe.float().mean(),
            "failure_termination": failed.float().mean(),
            "failure_penalty": failure_penalty.mean(),
            "max_contact_force_n": force_n.amax(dim=1).mean(),
            "action_rate_penalty": (self.cfg.action_rate_scale * action_rate).mean(),
            "grasp_region": self.grasp_enabled.float().mean(),
        }
        self.extras["metrics"] = {
            "reach": distance < self.cfg.reach_distance_m,
            "pregrasp_quality": pregrasp_quality,
            "valid_contact": valid_contact,
            "contact_3": valid_contact.all(dim=1),
            "grasp_candidate": grasp_candidate,
            "hold": self.hold_steps > 0,
            "success": self.success,
            "unsafe_contact": unsafe,
            "failure_termination": failed,
            "force_n": force_n,
            "grasp_region": self.grasp_enabled,
        }
        return reward

    def _get_dones(self):
        fallen = self.object.data.root_pos_w[:, 2] < self.cfg.table_top_z - 0.05
        unsafe = self.unsafe_steps >= self.cfg.unsafe_consecutive_steps
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return self.success | fallen | unsafe, timeout

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        count = len(env_ids)
        q = self.robot.data.default_joint_pos[env_ids].clone()
        q[:, self.arm_ids] = torch.tensor(self.cfg.initial_arm_q, device=self.device)
        q[:, self.finger_ids] = torch.tensor(self.cfg.finger_open, device=self.device)
        q = torch.minimum(torch.maximum(q, self.lower), self.upper)
        self.targets[env_ids] = q
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)
        state = self.object.data.default_root_state[env_ids].clone()
        x = torch.empty(count, device=self.device).uniform_(*self.cfg.object_x)
        y = torch.empty(count, device=self.device).uniform_(*self.cfg.object_y)
        state[:, :3] = self.scene.env_origins[env_ids] + torch.stack((
            x, y, torch.full_like(x, self.cfg.table_top_z + self.cfg.object_half_height),
        ), dim=1)
        state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device)
        state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
        self.object.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.hold_steps[env_ids] = 0
        self.unsafe_steps[env_ids] = 0
        self.success[env_ids] = False
        self.grasp_enabled[env_ids] = False
