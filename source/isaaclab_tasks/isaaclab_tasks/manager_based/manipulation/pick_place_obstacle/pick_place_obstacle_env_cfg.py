# SPDX-License-Identifier: BSD-3-Clause
"""Abstract base configuration for pick-and-place with obstacle avoidance.

Task overview
-------------
The robot arm must:
  1. Approach and grasp a lightweight object (<300 g) on a table.
  2. Lift and carry it to a fixed goal position.
  3. Avoid randomised box obstacles placed between object and goal.

Curriculum (step-based)
-----------------------
  Stage 0: no obstacles   (pure pick-and-place)
  Stage 1: 1 obstacle     (activate after stage_1_threshold steps)
  Stage 2: 2 obstacles    (activate after stage_2_threshold steps)

Obstacle size randomisation
---------------------------
  Each obstacle slot has 5 discrete box size variants (PhysX geometry is baked
  at spawn time; runtime resizing is not possible). Per episode, one variant
  per slot is randomly chosen and placed in the workspace; the other four are
  parked at x ≈ 200 m so they don't affect physics or observations.

Observation format (vision-pipeline compatible)
-----------------------------------------------
  Designed to match depth-camera + detector pipeline output:
    object_pos / quat / dims  → 3-D bounding box from object detector
    obstacle_N_pos / quat / dims → same, per obstacle slot
    goal_pos                  → from task planner
  Quaternion observations can be disabled via ObservationsCfg flags
  (cube symmetry makes orientation ambiguous; real detector may not
  provide reliable orientation).

Observation space dimensions
-----------------------------
  joint_pos (rel)       :  9
  joint_vel (rel)       :  9
  ee_pos                :  3
  ee_quat               :  4
  object_pos            :  3
  object_quat           :  4  (disabled when include_object_quat=False → -4)
  object_dims           :  3
  obstacle_0_pos        :  3
  obstacle_0_quat       :  4  (disabled when include_obstacle_quat=False → -4)
  obstacle_0_dims       :  3
  obstacle_1_pos        :  3
  obstacle_1_quat       :  4  (disabled when include_obstacle_quat=False → -4)
  obstacle_1_dims       :  3
  goal_pos              :  3
  last_action           :  8  (7 arm + 1 gripper)
  Total (all quat on)   : 66
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class PnPObstacleSceneCfg(InteractiveSceneCfg):
    """Table, robot, graspable object, 2×5 obstacle variants, contact sensor, lights."""

    # Robot + EE frame: filled by concrete robot config
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    # Table — top surface at world z ≈ 0
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.707, 0.0, 0.0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Graspable object: 5 cm cube, 50 g.
    # Half-height = 0.025 m → init z = 0.025 places bottom exactly on table.
    # Franka gripper max opening ≈ 8 cm, so 5 cm fits comfortably.
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, -0.1, 0.025], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.9)),
        ),
    )

    # ------------------------------------------------------------------
    # Contact sensor: monitors panda_hand (palm link) for obstacle hits.
    # panda_hand is the body between wrist and fingers — object grasps
    # produce forces primarily on the fingers, so this sensor is largely
    # free of object-grasp false positives.
    # ------------------------------------------------------------------
    ee_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
        history_length=1,
        track_air_time=False,
        update_period=0.0,
    )

    # ------------------------------------------------------------------
    # Obstacle slot 0 — 5 size variants (one chosen per episode)
    #
    #   a: 5×5×10 cm   center_z = 0.050  small pillar
    #   b: 8×8×15 cm   center_z = 0.075  medium pillar
    #   c: 6×6×22 cm   center_z = 0.110  tall pillar
    #   d: 10×10×8 cm  center_z = 0.040  wide squat block
    #   e: 7×7×18 cm   center_z = 0.090  medium-tall pillar
    #
    # kinematic_enabled=True: obstacles don't topple when the arm nudges them,
    # which removes confusing physics noise during early training.
    # ------------------------------------------------------------------
    obstacle_0_a: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle0A",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[200.0, 200.0, 0.050], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.20, 0.20)),
        ),
    )
    obstacle_0_b: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle0B",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[200.0, 200.0, 0.075], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.08, 0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.50, 0.20)),
        ),
    )
    obstacle_0_c: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle0C",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[200.0, 200.0, 0.110], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.22),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.80, 0.20)),
        ),
    )
    obstacle_0_d: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle0D",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[200.0, 200.0, 0.040], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.10, 0.10, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.50, 0.85, 0.30)),
        ),
    )
    obstacle_0_e: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle0E",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[200.0, 200.0, 0.090], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.07, 0.07, 0.18),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.70, 0.20, 0.80)),
        ),
    )

    # ------------------------------------------------------------------
    # Obstacle slot 1 — same 5 sizes, different prim paths / colours
    # ------------------------------------------------------------------
    obstacle_1_a: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle1A",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[200.0, 200.0, 0.050], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.50, 0.90)),
        ),
    )
    obstacle_1_b: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle1B",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[200.0, 200.0, 0.075], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.08, 0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.80, 0.90)),
        ),
    )
    obstacle_1_c: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle1C",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[200.0, 200.0, 0.110], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.22),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.50, 0.20, 0.90)),
        ),
    )
    obstacle_1_d: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle1D",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[200.0, 200.0, 0.040], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.10, 0.10, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.80, 0.50)),
        ),
    )
    obstacle_1_e: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle1E",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[200.0, 200.0, 0.090], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.07, 0.07, 0.18),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.60, 0.10)),
        ),
    )

    # Ground plane and lighting
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@configclass
class ActionsCfg:
    """Joint position actions — filled by concrete robot config."""
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

@configclass
class ObservationsCfg:
    """Vision-pipeline-compatible state observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        # -- Robot state --
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # -- End-effector pose --
        ee_pos = ObsTerm(func=mdp.ee_position, params={"ee_frame_cfg": SceneEntityCfg("ee_frame")})
        ee_quat = ObsTerm(func=mdp.ee_orientation, params={"ee_frame_cfg": SceneEntityCfg("ee_frame")})

        # -- Object state --
        object_pos = ObsTerm(func=mdp.object_position, params={"object_cfg": SceneEntityCfg("object")})
        object_quat = ObsTerm(func=mdp.object_orientation, params={"object_cfg": SceneEntityCfg("object")})
        object_dims = ObsTerm(func=mdp.object_dims_obs, params={"object_dims": [0.05, 0.05, 0.05]})

        # -- Obstacle slot 0 (zeros when inactive) --
        obs0_pos = ObsTerm(func=mdp.obstacle_0_position)
        obs0_quat = ObsTerm(func=mdp.obstacle_0_orientation)
        obs0_dims = ObsTerm(func=mdp.obstacle_0_dims)

        # -- Obstacle slot 1 (zeros when inactive) --
        obs1_pos = ObsTerm(func=mdp.obstacle_1_position)
        obs1_quat = ObsTerm(func=mdp.obstacle_1_orientation)
        obs1_dims = ObsTerm(func=mdp.obstacle_1_dims)

        # -- Goal position --
        goal_pos = ObsTerm(func=mdp.goal_position, params={"goal_pos_w": [0.5, 0.35, 0.25]})

        # -- Last action --
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Events (domain randomisation)
# ---------------------------------------------------------------------------

@configclass
class EventCfg:
    """Randomisation applied at every episode reset."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_object_position = EventTerm(
        func=mdp.reset_object_position_curriculum,
        mode="reset",
        params={
            # 초기: EE 기본 자세 정면 근처의 좁은 범위
            "narrow_range": {"x": (0.38, 0.46), "y": (-0.05, 0.02), "z": (0.0, 0.0)},
            # 최종: 전체 작업 공간
            "full_range":   {"x": (0.30, 0.55), "y": (-0.25, 0.05), "z": (0.0, 0.0)},
            # 200k 스텝까지 좁은 범위 유지, 600k 스텝에 완전히 확장
            "expand_start_steps": 200_000,
            "expand_end_steps":   600_000,
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    # Obstacle slot 0: 5 variants, one chosen randomly per episode.
    # z is omitted from pose_range — default_root_state z (= init_state z = center_z)
    # is kept, then env_origin is added → correct world-frame z for each variant.
    reset_obstacle_0 = EventTerm(
        func=mdp.reset_obstacle_random_variant,
        mode="reset",
        params={
            "obstacle_idx": 0,
            "pose_range": {"x": (0.35, 0.60), "y": (0.03, 0.28)},
            "variant_0_cfg": SceneEntityCfg("obstacle_0_a"),
            "variant_1_cfg": SceneEntityCfg("obstacle_0_b"),
            "variant_2_cfg": SceneEntityCfg("obstacle_0_c"),
            "variant_3_cfg": SceneEntityCfg("obstacle_0_d"),
            "variant_4_cfg": SceneEntityCfg("obstacle_0_e"),
        },
    )

    reset_obstacle_1 = EventTerm(
        func=mdp.reset_obstacle_random_variant,
        mode="reset",
        params={
            "obstacle_idx": 1,
            "pose_range": {"x": (0.38, 0.62), "y": (0.08, 0.32)},
            "variant_0_cfg": SceneEntityCfg("obstacle_1_a"),
            "variant_1_cfg": SceneEntityCfg("obstacle_1_b"),
            "variant_2_cfg": SceneEntityCfg("obstacle_1_c"),
            "variant_3_cfg": SceneEntityCfg("obstacle_1_d"),
            "variant_4_cfg": SceneEntityCfg("obstacle_1_e"),
        },
    )


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

@configclass
class RewardsCfg:
    """Pick-and-place 단계별 보상.

    가중치 설계 원칙:
      접근(1) < 파지시도(5) < 이동(10) < 들어올리기(25) < 놓기(50)
    → 에이전트가 단순히 물체 옆에 머무는 것보다 들어올리는 것이
      압도적으로 유리하도록 설계.
    """

    # ── 1단계: 접근 ───────────────────────────────────────────────
    # EE 끝이 물체에 가까울수록 증가 (std=0.1m → 10cm 이내부터 의미있음)
    # 낮은 가중치로 유지 — "그냥 옆에 있기" 전략이 최적이 되지 않도록
    reach_object = RewTerm(
        func=mdp.ee_object_distance,
        params={"std": 0.10, "object_cfg": SceneEntityCfg("object"), "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=1.0,
    )

    # ── 2단계: 파지 시도 ──────────────────────────────────────────
    # 물체 근처(15cm 이내)에서 그리퍼를 닫으면 보상
    # 커리큘럼으로 물체가 가까이 스폰되므로 파지 동작을 우연히 발견할 확률 ↑
    grasp_effort = RewTerm(
        func=mdp.grasp_effort,
        params={
            "std": 0.08,
            "close_threshold": 0.15,
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=12.0,
    )

    # ── 3단계: 들어올리기 (핵심) ──────────────────────────────────
    # 물체 높이 4cm 이상 → 스텝마다 큰 보상. 학습의 병목 지점
    lift_object = RewTerm(
        func=mdp.object_lifted,
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("object")},
        weight=25.0,
    )

    # ── 4단계: 목표 위치로 이동 ───────────────────────────────────
    # 들어올린 상태일 때만 활성화 (minimal_height 조건)
    move_to_goal = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.20,
            "minimal_height": 0.04,
            "goal_pos_w": [0.5, 0.35, 0.25],
            "object_cfg": SceneEntityCfg("object"),
        },
        weight=10.0,
    )

    # ── 5단계: 놓기 성공 (희소 보너스) ───────────────────────────
    # 목표 6cm 이내에 물체가 위치하면 큰 일회성 보너스
    place_success = RewTerm(
        func=mdp.place_success,
        params={
            "threshold": 0.06,
            "goal_pos_w": [0.5, 0.35, 0.25],
            "object_cfg": SceneEntityCfg("object"),
        },
        weight=50.0,
    )

    # ── 장애물 회피 (비활성화) ────────────────────────────────────
    obstacle_penalty = RewTerm(
        func=mdp.obstacle_proximity_penalty,
        params={
            "safe_dist": 0.12,
            "std": 0.05,
            "obstacle_0_a_cfg": SceneEntityCfg("obstacle_0_a"),
            "obstacle_0_b_cfg": SceneEntityCfg("obstacle_0_b"),
            "obstacle_0_c_cfg": SceneEntityCfg("obstacle_0_c"),
            "obstacle_0_d_cfg": SceneEntityCfg("obstacle_0_d"),
            "obstacle_0_e_cfg": SceneEntityCfg("obstacle_0_e"),
            "obstacle_1_a_cfg": SceneEntityCfg("obstacle_1_a"),
            "obstacle_1_b_cfg": SceneEntityCfg("obstacle_1_b"),
            "obstacle_1_c_cfg": SceneEntityCfg("obstacle_1_c"),
            "obstacle_1_d_cfg": SceneEntityCfg("obstacle_1_d"),
            "obstacle_1_e_cfg": SceneEntityCfg("obstacle_1_e"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=0.0,
    )

    # ── 바닥(테이블) 침범 패널티 ──────────────────────────────────
    # EE Z 좌표가 5cm 미만으로 내려가면 패널티. 테이블에 쳐박히는 동작 억제
    ee_floor_penalty = RewTerm(
        func=mdp.ee_floor_penalty,
        params={"min_height": 0.05, "std": 0.03, "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=-3.0,
    )

    # ── 동작 부드러움 ─────────────────────────────────────────────
    # 관절 속도와 액션 변화량을 억제 → 꾸물거림/진동 방지
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-0.01,
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.10, "asset_cfg": SceneEntityCfg("object")},
    )

    # obstacle_contact = DoneTerm(...)  # 장애물 비활성화로 제거


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

@configclass
class CurriculumCfg:
    """Step-based obstacle curriculum."""

    # obstacle_stage / obstacle_penalty_weight: 장애물 비활성화로 제거
    # action_rate_weight / joint_vel_weight: 커리큘럼 제거, 고정값 사용


# ---------------------------------------------------------------------------
# Top-level env config
# ---------------------------------------------------------------------------

@configclass
class PnPObstacleEnvCfg(ManagerBasedRLEnvCfg):
    """Pick-and-place with obstacle avoidance — abstract base.

    Concrete configs (e.g. FrankaPnPObstacleEnvCfg) provide robot, ee_frame,
    arm_action, and gripper_action.
    """

    # Quaternion observation flags — set False to drop those terms from obs space.
    # Handled in __post_init__ below (cannot live inside ObservationsCfg itself
    # because ObservationManager treats every field there as an ObservationGroupCfg).
    include_object_quat: bool = True
    include_obstacle_quat: bool = True

    scene: PnPObstacleSceneCfg = PnPObstacleSceneCfg(num_envs=2048, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 20.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        # Apply quaternion flags: set disabled terms to None so ObservationManager skips them.
        if not self.include_object_quat:
            self.observations.policy.object_quat = None  # type: ignore[assignment]
        if not self.include_obstacle_quat:
            self.observations.policy.obs0_quat = None  # type: ignore[assignment]
            self.observations.policy.obs1_quat = None  # type: ignore[assignment]
