"""Offline URDF reach check for a fixed cube; does not start Isaac Sim or training."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

JAX_ARM_JOINTS = [
    "shoulder_pitch_to_shoulder_roll", "shoulder_roll_to_shoulder_yaw",
    "shoulder_yaw_to_upper_arm", "upper_arm_to_under_arm",
    "under_arm_to_arm_yaw", "arm_yaw_to_wrist_pitch",
    "wrist_pitch_to_wrist_yaw", "wrist_yaw_to_rotation_base",
]


def transform(xyz, rpy):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    result[:3, 3] = xyz
    return result


def read_urdf(path):
    tree = ET.parse(path)
    child_to_joint = {joint.find("child").attrib["link"]: joint for joint in tree.findall("joint")}
    return child_to_joint


def link_pose(link, joints, q, cache):
    if link in cache:
        return cache[link]
    joint = joints[link]
    parent = joint.find("parent").attrib["link"]
    origin = joint.find("origin")
    xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
    pose = link_pose(parent, joints, q, cache) @ transform(xyz, rpy)
    if joint.attrib["type"] != "fixed":
        axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
        turn = np.eye(4)
        turn[:3, :3] = Rotation.from_rotvec(axis * q.get(joint.attrib["name"], 0.0)).as_matrix()
        pose = pose @ turn
    cache[link] = pose
    return pose


def world_pose(link, joints, q):
    # base_link FRONT is local -Y and faces world -X.
    root = transform((0.0, 0.0, 0.305), (0.0, 0.0, -np.pi / 2))
    return link_pose(link, joints, q, {"base_link": root})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-x", type=float, default=-0.465)
    parser.add_argument("--object-y", type=float, default=0.0)
    parser.add_argument("--starts", type=int, default=30)
    parser.add_argument("--lift-path", action="store_true", help="Optimize a grasp and vertically lifted palm pose with the same orientation")
    parser.add_argument("--palm-goal", type=float, nargs=3)
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[4] / "isaaclab_assets/data/Robots/OneArmJaxUrdf/urdf/One_Arm_JAX_URDF.urdf"
    joints = read_urdf(path)
    lower, upper = [], []
    for name in JAX_ARM_JOINTS:
        limit = next(j for j in joints.values() if j.attrib["name"] == name).find("limit")
        lower.append(float(limit.attrib["lower"]))
        upper.append(float(limit.attrib["upper"]))
    lower, upper = np.array(lower), np.array(upper)
    object_pos = np.array((args.object_x, args.object_y, 0.764))
    # With a vertical palm the grasp pose used by the scripted controller has
    # rotation_base about 10 cm above the cube; also report unconstrained palm reach.
    palm_target = np.array(args.palm_goal) if args.palm_goal else object_pos + np.array((0.006, 0.012, 0.101))
    rng = np.random.default_rng(123)
    best = None
    nominal = np.array((0.66526774, -0.0436329, -1.0471979, -1.38850439,
                        0.57322713, 1.54101851, -0.65936561, 0.95230107))
    nominal_pose = world_pose("rotation_base", joints, dict(zip(JAX_ARM_JOINTS, nominal)))
    for start in range(args.starts):
        q0 = (lower + upper) / 2 if start == 0 else rng.uniform(lower, upper)
        def error(values):
            q = dict(zip(JAX_ARM_JOINTS, values))
            pose = world_pose("rotation_base", joints, q)
            palm = pose[:3, 3]
            if not args.lift_path:
                return palm - palm_target
            rotation_error = Rotation.from_matrix(nominal_pose[:3, :3].T @ pose[:3, :3]).as_rotvec()
            margin = np.minimum(values - lower, upper - values)
            return np.concatenate((palm - palm_target, 0.08 * rotation_error,
                                   0.05 * np.maximum(0.15 - margin, 0),
                                   0.005 * (values - nominal)))
        fit = least_squares(error, q0, bounds=(lower + 1e-6, upper - 1e-6), max_nfev=250)
        distance = np.linalg.norm(fit.fun)
        if best is None or distance < best[0]:
            best = (distance, fit.x)
    distance, values = best
    q = dict(zip(JAX_ARM_JOINTS, values))
    result = {
        "object_xyz_m": object_pos.tolist(),
        "palm_goal_xyz_m": palm_target.tolist(),
        "nearest_palm_xyz_m": world_pose("rotation_base", joints, q)[:3, 3].tolist(),
        "minimum_palm_goal_error_m": float(distance),
        "arm_joint_radians": values.tolist(),
        "near_limits": ((values - lower < 0.02) | (upper - values < 0.02)).tolist(),
    }
    if args.lift_path:
        grasp_pose = world_pose("rotation_base", joints, q)
        lifted_target = grasp_pose.copy()
        lifted_target[2, 3] += 0.05
        def lifted_error(v):
            pose = world_pose("rotation_base", joints, dict(zip(JAX_ARM_JOINTS, v)))
            rotation_error = Rotation.from_matrix(grasp_pose[:3, :3].T @ pose[:3, :3]).as_rotvec()
            return np.concatenate((pose[:3, 3] - lifted_target[:3, 3], 0.08 * rotation_error))
        lift_fit = least_squares(lifted_error, values, bounds=(lower + 1e-6, upper - 1e-6), max_nfev=500)
        result.update({
            "grasp_palm_xyz_m": grasp_pose[:3, 3].tolist(),
            "lift_palm_xyz_m": world_pose("rotation_base", joints, dict(zip(JAX_ARM_JOINTS, lift_fit.x)))[:3, 3].tolist(),
            "lift_palm_position_error_m": float(np.linalg.norm(lift_fit.fun[:3])),
            "lift_orientation_error_rad": float(np.linalg.norm(lift_fit.fun[3:]) / 0.08),
            "lift_arm_joint_radians": lift_fit.x.tolist(),
        })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
