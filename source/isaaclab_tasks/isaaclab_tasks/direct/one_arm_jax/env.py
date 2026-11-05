"""Object-conditioned JAX arm reach and stable grasp environment (Stage 1)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils.math import quat_apply_inverse

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from isaaclab_assets.robots.one_arm_jax import JAX_ARM_JOINTS, JAX_FINGER_JOINTS, ONE_ARM_JAX_CFG


@configclass
class OneArmJaxPickPlaceEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 8.0
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(9,), dtype=np.float32)
    observation_space = 40
    state_space = 0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=7056, env_spacing=2.0, replicate_physics=True, clone_in_fabric=False,
    )
    # The imported base_link collision envelope reaches local z=0.895 m.
    robot_base_z = 1.20 - 0.895
    # BACK is on the base_link +Y face, so FRONT is -Y. Rotate -Y onto world -X (the table direction).
    robot_yaw_quat = (0.7071067811865476, 0.0, 0.0, -0.7071067811865476)
    robot_cfg = ONE_ARM_JAX_CFG.replace(
        init_state=ONE_ARM_JAX_CFG.init_state.replace(
            pos=(0.0, 0.0, robot_base_z), rot=robot_yaw_quat,
        )
    )
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.080, 0.080, 0.085),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.220, 0.080, 0.7825)),
    )
    table_cfg = sim_utils.CuboidCfg(
        size=(0.80, 0.70, 0.04),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.2, dynamic_friction=1.0, restitution=0.0,
        ),
    )
    # The near edge is x=-0.16 m, clear of the rotated base collision mesh (x>=-0.125 m).
    table_center = (-0.56, 0.0, 0.72)
    table_top_z = 0.74
    pedestal_cfg = sim_utils.CylinderCfg(
        radius=0.125, height=robot_base_z,
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    object_half_height = 0.0425
    # Narrow first-stage spawn around a pose reached by the three-finger hand.
    object_x = (-0.225, -0.215)
    object_y = (0.075, 0.085)
    # Start with room to move instead of parking three arm joints at their URDF limits.
    initial_arm_q = (0.72592483, -0.02220609, -0.927197, -1.450795,
                     0.85823423, 1.450795, -1.87081975, 1.49179096)
    arm_delta_scale = 0.035
    # Max finger target increment is 0.0192 rad per 1/60 s policy step.
    gripper_delta_scale = 0.01
    # URDF kinematics: increasing proximal angles spreads the fingers, while
    # increasing distal angles curls the tips around the object.
    finger_open = (0.90, 0.90, 0.90, 0.00, 0.00, 0.00)
    finger_close = (-0.36, -0.28, -0.68, 1.67, 1.79, 1.92)
    # Stage 1 reward: progress + one geometric preparation signal + verified success.
    reach_progress_scale = 1.0
    reach_progress_distance = 0.02
    grasp_preparation_scale = 0.5
    success_scale = 10.0
    action_rate_scale = 0.01
    # Temporary safety limit: the manual probe measured >100 N during penetration.
    excess_force_scale = 0.1
    force_limit_n = 20.0
    # Policy/STM32 pressure contract: p_i = clamp(F_i / 20 N, 0, 1).
    # The physical RA12P ADC-to-newton curve is calibrated separately per finger.
    pressure_reference_force = 20.0
    pressure_min = 0.1
    pressure_desired = 0.4
    pressure_excess = 0.9
    grasp_stable_steps = 3
    grasp_quality_threshold = 0.35
    grasp_hold_duration_s = 0.5


class OneArmJaxPickPlaceEnv(DirectRLEnv):
    cfg: OneArmJaxPickPlaceEnvCfg

    def __init__(self, cfg: OneArmJaxPickPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.arm_ids = [self.robot.joint_names.index(n) for n in JAX_ARM_JOINTS]
        self.finger_ids = [self.robot.joint_names.index(n) for n in JAX_FINGER_JOINTS]
        self.ee_id = self.robot.body_names.index("rotation_base")
        self.fingertip_ids = [self.robot.body_names.index(f"finger_{i}_link2") for i in (1, 2, 3)]
        self.lower = self.robot.data.soft_joint_pos_limits[0, :, 0]
        self.upper = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.targets = self.robot.data.default_joint_pos.clone()
        self.actions = torch.zeros((self.num_envs, 9), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.grasp_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.grasp_hold_steps = torch.zeros_like(self.grasp_steps)
        self.grasp_hold_required_steps = round(cfg.grasp_hold_duration_s / (cfg.sim.dt * cfg.decimation))
        self.previous_reach_distance = torch.full((self.num_envs,), float('nan'), device=self.device)
        self._success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.cfg.table_cfg.func(
            "/World/envs/env_0/Table", self.cfg.table_cfg,
            translation=self.cfg.table_center,
        )
        self.cfg.pedestal_cfg.func(
            "/World/envs/env_0/RobotPedestal", self.cfg.pedestal_cfg,
            translation=(0.0, 0.0, self.cfg.robot_base_z / 2.0),
        )
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        for i in (1, 2, 3):
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/finger_{i}_link2",
                filter_prim_paths_expr=["/World/envs/env_.*/Object"],
                update_period=0.0,
                track_contact_points=True,
            ))
            self.scene.sensors[f"finger_{i}"] = sensor
        light = sim_utils.DomeLightCfg(intensity=2000.0)
        light.func("/World/Light", light)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.previous_actions.copy_(self.actions)
        self.actions = actions.clamp(-1.0, 1.0)
        self.targets[:, self.arm_ids] += self.cfg.arm_delta_scale * self.actions[:, :8]
        self.targets[:, self.arm_ids] = torch.minimum(
            torch.maximum(self.targets[:, self.arm_ids], self.lower[self.arm_ids]), self.upper[self.arm_ids]
        )
        # One bounded curl target drives all six finger joints. The command cannot open
        # more than one gripper_delta_scale per policy step.
        finger_open = torch.tensor(self.cfg.finger_open, device=self.device)
        finger_close = torch.tensor(self.cfg.finger_close, device=self.device)
        self.targets[:, self.finger_ids] += self.cfg.gripper_delta_scale * self.actions[:, 8:9] * (
            finger_close - finger_open
        )
        self.targets[:, self.finger_ids] = torch.minimum(
            torch.maximum(self.targets[:, self.finger_ids], torch.minimum(finger_open, finger_close)),
            torch.maximum(finger_open, finger_close),
        )

    def _apply_action(self):
        self.robot.set_joint_position_target(self.targets)

    def _base_states(self):
        root = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        object_b = quat_apply_inverse(root_quat, self.object.data.root_pos_w - root)
        # The palm is ~10 cm behind the contact region. Use the center of the
        # three distal fingers for both policy relative position and reach reward.
        ee_b = quat_apply_inverse(root_quat, self.robot.data.body_pos_w[:, self.fingertip_ids].mean(dim=1) - root)
        object_vel = quat_apply_inverse(root_quat, self.object.data.root_lin_vel_w - self.robot.data.root_lin_vel_w)
        return object_b, ee_b, object_vel

    def _pressure_forces(self):
        forces = [self.scene.sensors[f"finger_{i}"].data.force_matrix_w for i in (1, 2, 3)]
        vectors = torch.stack([f[:, 0, 0] for f in forces], dim=1)
        object_quats = self.object.data.root_quat_w[:, None, :].expand(-1, 3, -1)
        local_forces = quat_apply_inverse(object_quats.reshape(-1, 4), vectors.reshape(-1, 3)).reshape(-1, 3, 3)
        # RA12P's 20 N normalization remains temporary; use object-side force,
        # even when the block tilts, rather than world-horizontal force.
        return torch.linalg.vector_norm(local_forces[..., :2], dim=-1)

    def _pressures(self):
        # P1/P2/P3 order matches finger_1/2/3_link2 and the future STM32 message.
        return (self._pressure_forces() / self.cfg.pressure_reference_force).clamp(0.0, 1.0)

    def _get_observations(self):
        obj, ee, vel = self._base_states()
        q = self.robot.data.joint_pos
        qn = 2.0 * (q - self.lower) / (self.upper - self.lower).clamp_min(1e-5) - 1.0
        obs = torch.cat((
            qn, self.robot.data.joint_vel * 0.2,
            obj, vel, obj - ee, self._pressures(),
        ), dim=-1)
        return {"policy": obs.clamp(-5.0, 5.0)}

    def _get_rewards(self):
        obj, ee, vel = self._base_states()
        sensors = [self.scene.sensors[f"finger_{i}"].data for i in (1, 2, 3)]
        force_vectors = torch.stack([sensor.force_matrix_w[:, 0, 0] for sensor in sensors], dim=1)
        object_quats = self.object.data.root_quat_w[:, None, :].expand(-1, 3, -1)
        local_forces = quat_apply_inverse(object_quats.reshape(-1, 4), force_vectors.reshape(-1, 3)).reshape(-1, 3, 3)
        force_n = torch.linalg.vector_norm(local_forces[..., :2], dim=-1)
        pressure = (force_n / self.cfg.pressure_reference_force).clamp(0.0, 1.0)
        contacts = pressure >= self.cfg.pressure_min
        contact_points = torch.stack([sensor.contact_pos_w[:, 0, 0] for sensor in sensors], dim=1)
        finite_contact = torch.isfinite(contact_points).all(dim=-1)
        contact_points = torch.where(finite_contact[..., None], contact_points, self.object.data.root_pos_w[:, None, :])
        relative_world = contact_points - self.object.data.root_pos_w[:, None, :]
        relative_contact = quat_apply_inverse(object_quats.reshape(-1, 4), relative_world.reshape(-1, 3)).reshape(-1, 3, 3)
        radial_xy = relative_contact[..., :2]
        radial_distance = torch.linalg.vector_norm(radial_xy, dim=-1)
        radial_direction = radial_xy / radial_distance.clamp_min(1e-6)[..., None]
        # Each sensor covers a whole second phalanx. Approximate its inner face by
        # the direction from that link toward the other two distal links.
        finger_centers = self.robot.data.body_pos_w[:, self.fingertip_ids, :]
        toward_grip = (finger_centers.sum(dim=1, keepdim=True) - 3.0 * finger_centers)[..., :2]
        contact_offset = (contact_points - finger_centers)[..., :2]
        inner_alignment = (toward_grip * contact_offset).sum(dim=-1) / (
            torch.linalg.vector_norm(toward_grip, dim=-1)
            * torch.linalg.vector_norm(contact_offset, dim=-1)
        ).clamp_min(1e-6)
        inner_quality = ((inner_alignment - 0.1) / 0.5).clamp(0.0, 1.0)
        # The force matrix is the normal contact force on the finger. Its sign
        # depends on body ordering, so use absolute alignment with the radial
        # direction; vertical/top forces and tangential forces score poorly.
        normal_alignment = (
            (local_forces[..., :2] * radial_direction).sum(dim=-1).abs()
            / torch.linalg.vector_norm(local_forces, dim=-1).clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        # A side face must dominate the top/bottom face in the *object* frame.
        side_extent = torch.maximum(relative_contact[..., 0].abs(), relative_contact[..., 1].abs()) / 0.04
        top_extent = relative_contact[..., 2].abs() / self.cfg.object_half_height
        side_dominance = side_extent - top_extent
        side_band = side_dominance > 0.10
        side_quality = (side_dominance / 0.25).clamp(0.0, 1.0)
        # Presence saturates at the minimum usable contact (2 N with the temporary 20 N scale).
        pressure_presence = (pressure / self.cfg.pressure_min).clamp(0.0, 1.0)
        inner_strength = pressure_presence * inner_quality * normal_alignment * side_quality * finite_contact.float()
        inner_strength = inner_strength * (radial_distance > 0.02).float()
        meaningful_contact = (
            contacts & finite_contact & side_band & (radial_distance > 0.02)
            & (normal_alignment >= 0.35) & (inner_alignment >= 0.35)
        )
        contact_count = meaningful_contact.sum(dim=1)
        three_contact = contact_count == 3
        finger_closed = self.robot.data.joint_pos[:, self.finger_ids[3:]].mean(dim=1) > 0.45
        finger_vel_b = quat_apply_inverse(
            self.robot.data.root_quat_w,
            self.robot.data.body_lin_vel_w[:, self.fingertip_ids, :3].mean(dim=1)
            - self.robot.data.root_lin_vel_w,
        )
        relative_speed = torch.linalg.vector_norm(vel - finger_vel_b, dim=-1)
        inner_contact = inner_strength.mean(dim=1)
        wrist = self.robot.data.body_pos_w[:, self.ee_id, :]
        hand_center = finger_centers.mean(dim=1)
        hand_axis = hand_center - wrist
        object_direction = self.object.data.root_pos_w - wrist
        axis_alignment = ((hand_axis * object_direction).sum(dim=-1) / (
            torch.linalg.vector_norm(hand_axis, dim=-1)
            * torch.linalg.vector_norm(object_direction, dim=-1)
        ).clamp_min(1e-6)).clamp(0.0, 1.0)
        hand_distance = torch.linalg.vector_norm(self.object.data.root_pos_w - hand_center, dim=-1)
        finger_closed_fraction = ((self.robot.data.joint_pos[:, self.finger_ids[3:]].mean(dim=1)
                                  - 0.25) / 0.5).clamp(0.0, 1.0)
        alignment = axis_alignment * torch.exp(-8.0 * hand_distance) * (
            1.0 - 0.5 * finger_closed_fraction * (contact_count < 2).float()
        )
        finger_relative = finger_centers - self.object.data.root_pos_w[:, None, :]
        finger_radial = finger_relative[..., :2]
        finger_radius = torch.linalg.vector_norm(finger_radial, dim=-1)
        finger_direction = finger_radial / finger_radius.clamp_min(1e-6)[..., None]
        enclosure_spread = (1.0 - torch.linalg.vector_norm(finger_direction.sum(dim=1), dim=-1) / 3.0).clamp(0.0, 1.0)
        enclosure = enclosure_spread * torch.exp(
            -20.0 * (finger_radius - 0.05).abs().mean(dim=1)
            -20.0 * finger_relative[..., 2].abs().mean(dim=1)
        )
        direction_sum = (radial_direction * inner_strength[..., None]).sum(dim=1)
        strength_sum = inner_strength.sum(dim=1)
        geometry_spread = (1.0 - torch.linalg.vector_norm(direction_sum, dim=-1)
                           / strength_sum.clamp_min(1e-6)).clamp(0.0, 1.0)
        grasp_geometry = geometry_spread * (strength_sum / 3.0).clamp(0.0, 1.0)
        speed_quality = (1.0 - relative_speed / 0.25).clamp(0.0, 1.0)
        grasp_quality = torch.minimum(inner_contact, grasp_geometry) * speed_quality * finger_closed.float()
        grasp = three_contact & (grasp_quality > self.cfg.grasp_quality_threshold)
        self.grasp_steps = torch.where(grasp, self.grasp_steps + 1, 0)
        stable_grasp = self.grasp_steps >= self.cfg.grasp_stable_steps
        self.grasp_hold_steps = torch.where(stable_grasp, self.grasp_hold_steps + 1, 0)
        self._success = self.grasp_hold_steps >= self.grasp_hold_required_steps
        reach_distance = torch.linalg.vector_norm(obj - ee, dim=-1)
        previous_distance = torch.where(
            torch.isfinite(self.previous_reach_distance), self.previous_reach_distance, reach_distance
        )
        reach_progress = ((previous_distance - reach_distance) / self.cfg.reach_progress_distance).clamp(-1.0, 1.0)
        self.previous_reach_distance = reach_distance.detach()
        # Preparation uses one object-relative geometry score; force magnitude is not rewarded.
        grasp_preparation = enclosure
        action_rate = ((self.actions - self.previous_actions) ** 2).sum(dim=1)
        # Use full contact force for safety, including force from pressing on the top face.
        total_force_n = torch.linalg.vector_norm(force_vectors, dim=-1)
        excess_force = ((total_force_n - self.cfg.force_limit_n).clamp_min(0.0)
                        / self.cfg.force_limit_n).clamp_max(5.0).mean(dim=1)
        reward = (
            self.cfg.reach_progress_scale * reach_progress
            + self.cfg.grasp_preparation_scale * grasp_preparation
            + self.cfg.success_scale * self._success
            - self.cfg.action_rate_scale * action_rate
            - self.cfg.excess_force_scale * excess_force
        )
        self.extras["log"] = {
            "reach_progress_reward": (self.cfg.reach_progress_scale * reach_progress).mean(),
            "grasp_preparation_reward": (self.cfg.grasp_preparation_scale * grasp_preparation).mean(),
            "grasp": stable_grasp.float().mean(),
            "grasp_hold": (self.grasp_hold_steps > 0).float().mean(),
            "success": self._success.float().mean(),
            "grasp_quality": grasp_quality.mean(),
            "action_rate_penalty": (self.cfg.action_rate_scale * action_rate).mean(),
            "excess_force_penalty": (self.cfg.excess_force_scale * excess_force).mean(),
            "pressure_1": pressure[:, 0].mean(), "pressure_2": pressure[:, 1].mean(),
            "pressure_3": pressure[:, 2].mean(),
            "force_1_n": force_n[:, 0].mean(),
            "force_2_n": force_n[:, 1].mean(),
            "force_3_n": force_n[:, 2].mean(),
            "pressure_saturation": (pressure >= 1.0).float().mean(),
            "pressure_excess_rate": (pressure > self.cfg.pressure_excess).float().mean(),
            "contact_3": three_contact.float().mean(),
            "grasp_geometry": grasp_geometry.mean(),
            "inner_alignment": torch.nan_to_num(inner_alignment).mean(),
            "meaningful_contact_1": meaningful_contact[:, 0].float().mean(),
            "meaningful_contact_2": meaningful_contact[:, 1].float().mean(),
            "meaningful_contact_3": meaningful_contact[:, 2].float().mean(),
        }
        self.extras["metrics"] = {
            "reach": torch.linalg.vector_norm(obj - ee, dim=-1) < 0.07,
            "pressure_contact": contacts, "side_contact": meaningful_contact,
            "raw_contact_3": contacts.all(dim=1),
            "contact_2": contact_count >= 2, "contact_3": three_contact,
            "grasp_quality": three_contact & (grasp_quality > self.cfg.grasp_quality_threshold),
            "grasp": stable_grasp,
            "aligned_pregrasp": (hand_distance < 0.07) & (alignment > 0.55) & (finger_closed_fraction < 0.2),
            "enclosed_contact": (enclosure > 0.20) & (contact_count >= 2),
            "grasp_hold": self.grasp_hold_steps > 0,
            "pressure_excess": (pressure > self.cfg.pressure_excess).any(dim=1),
            "success": self._success,
        }
        return reward

    def _get_dones(self):
        fallen = self.object.data.root_pos_w[:, 2] < self.cfg.table_top_z - 0.05
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return self._success | fallen, timeout

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # A/B reset curriculum is deliberately gated on a physically verified
        # pre-grasp pose. No such pose has passed the 0.5 s hold test yet.
        super()._reset_idx(env_ids)
        n = len(env_ids)
        q = self.robot.data.default_joint_pos[env_ids].clone()
        q[:, self.arm_ids] = torch.tensor(self.cfg.initial_arm_q, device=self.device)
        q[:, self.arm_ids] += (torch.rand((n, 8), device=self.device) - 0.5) * 0.04
        q[:, self.finger_ids] = torch.tensor(self.cfg.finger_open, device=self.device)
        q = torch.minimum(torch.maximum(q, self.lower), self.upper)
        self.targets[env_ids] = q
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)
        state = self.object.data.default_root_state[env_ids].clone()
        ox = torch.empty(n, device=self.device).uniform_(*self.cfg.object_x)
        oy = torch.empty(n, device=self.device).uniform_(*self.cfg.object_y)
        state[:, :3] = self.scene.env_origins[env_ids] + torch.stack(
            (ox, oy, torch.full_like(ox, self.cfg.table_top_z + self.cfg.object_half_height)), dim=1
        )
        state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device)
        state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
        self.object.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)
        self.grasp_steps[env_ids] = 0
        self.grasp_hold_steps[env_ids] = 0
        self._success[env_ids] = False
        self.previous_reach_distance[env_ids] = float('nan')
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
