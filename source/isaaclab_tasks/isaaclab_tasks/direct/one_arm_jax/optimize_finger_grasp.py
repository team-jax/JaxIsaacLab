"""Offline URDF collision-centroid fit for three cube side contacts; no training."""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
from kinematic_grasp_diagnostic import JAX_ARM_JOINTS, read_urdf, transform, world_pose

FINGER_JOINTS = [
    "wrist_yaw_to_finger_2_link1", "wrist_yaw_to_finger_3_link1",
    "rotation_base_to_finger_1_link1", "wrist_yaw_to_finger_2_link2",
    "wrist_yaw_to_finger_3_link2", "finger_1_link1_to_finger_1_link2",
]
LINKS = ("finger_1_link2", "finger_2_link2", "finger_3_link2")

parser = argparse.ArgumentParser()
parser.add_argument("--object-x", type=float, default=-0.205)
parser.add_argument("--object-y", type=float, default=0.085)
parser.add_argument("--starts", type=int, default=30)
parser.add_argument("--surface-fit", action="store_true")
parser.add_argument("--gap", type=float, default=0.002)
parser.add_argument("--min-finger-z", type=float, default=0.743)
parser.add_argument("--finger-center-z-offset", type=float, default=0.0)
parser.add_argument("--trace-row", type=str, default="", help="Inspect measured arm/finger joint state from scripted trace CSV")
args = parser.parse_args()
path = Path(__file__).resolve().parents[4] / "isaaclab_assets/data/Robots/OneArmJaxUrdf/urdf/One_Arm_JAX_URDF.urdf"
urdf_root = ET.parse(path).getroot()
joints = read_urdf(path)
names = JAX_ARM_JOINTS + FINGER_JOINTS
lower, upper = [], []
for name in names:
    limit = next(j for j in joints.values() if j.attrib["name"] == name).find("limit")
    lower.append(float(limit.attrib["lower"]))
    upper.append(float(limit.attrib["upper"]))
lower, upper = np.array(lower), np.array(upper)
finger_centers_local = []
finger_vertices_local = []
for link_name in LINKS:
    link = next(x for x in urdf_root.findall("link") if x.attrib["name"] == link_name)
    collision = link.find("collision")
    origin = collision.find("origin")
    mesh_cfg = collision.find("geometry").find("mesh")
    mesh = trimesh.load(path.parent / mesh_cfg.attrib["filename"])
    scale = np.fromstring(mesh_cfg.attrib["scale"], sep=" ")
    centroid = np.mean(mesh.vertices * scale, axis=0)
    local = transform(np.fromstring(origin.attrib["xyz"], sep=" "),
                      np.fromstring(origin.attrib["rpy"], sep=" ")) @ np.r_[centroid, 1]
    finger_centers_local.append(local)
    # Only convex-hull vertices can attain a linear-coordinate minimum.
    hull_vertices = mesh.convex_hull.vertices
    vertices_local = (transform(np.fromstring(origin.attrib["xyz"], sep=" "),
                                np.fromstring(origin.attrib["rpy"], sep=" "))
                      @ np.c_[hull_vertices * scale, np.ones(len(hull_vertices))].T).T
    finger_vertices_local.append(vertices_local)

object_pos = np.array((args.object_x, args.object_y, 0.764))
target_centers = np.array((
    object_pos + (0.030, -0.016, 0.0),
    object_pos + (0.030, +0.016, 0.0),
    object_pos + (-0.030, 0.0, 0.0),
))
nominal_arm = np.array((0.51693858, -0.043632, -0.97900159, -1.23282265,
                        0.46861325, 1.32154267, -0.55498539, 0.69469648))
nominal_finger = np.array((0.65, 0.65, 0.30, 0.90, 0.90, 0.90))
nominal = np.r_[nominal_arm, nominal_finger]


def centers(values):
    q = dict(zip(names, values))
    return np.array([(world_pose(link, joints, q) @ center)[:3]
                     for link, center in zip(LINKS, finger_centers_local)])


def finger_min_z(values):
    q = dict(zip(names, values))
    return np.array([np.min((world_pose(link, joints, q) @ vertices.T)[2])
                     for link, vertices in zip(LINKS, finger_vertices_local)])


def finger_bounds(values):
    q = dict(zip(names, values))
    meshes = [(world_pose(link, joints, q) @ vertices.T).T[:, :3]
              for link, vertices in zip(LINKS, finger_vertices_local)]
    return np.array([m.min(axis=0) for m in meshes]), np.array([m.max(axis=0) for m in meshes])


def residual(values):
    if args.surface_fit:
        mins, maxes = finger_bounds(values)
        # Finger 1/2 approach the +X cube face; finger 3 opposes them on -X.
        side_error = np.r_[mins[:2, 0] - (object_pos[0] + 0.024 + args.gap),
                           maxes[2, 0] - (object_pos[0] - 0.024 - args.gap)]
        cent = centers(values)
        alignment_error = np.r_[cent[:, 1] - target_centers[:, 1],
                                cent[:, 2] - (object_pos[2] + args.finger_center_z_offset)]
        centroid_error = np.r_[side_error, 0.5 * alignment_error]
    else:
        centroid_error = (centers(values) - target_centers).ravel()
    arm_prior = 0.006 * (values[:8] - nominal_arm)
    joint_margin = np.minimum(values - lower, upper - values)
    margin_penalty = 0.003 * np.maximum(0.03 - joint_margin, 0)
    table_penalty = 3.0 * np.maximum(args.min_finger_z - finger_min_z(values), 0)
    return np.r_[centroid_error, arm_prior, margin_penalty, table_penalty]


if args.trace_row:
    import csv
    phase, path = args.trace_row.split(":", 1)
    row = [r for r in csv.DictReader(open(path)) if r["phase"] == phase][-1]
    q = np.r_[json.loads(row["arm_joint_rad"]), json.loads(row["finger_joint_rad"])]
    error = np.linalg.norm(residual(q)[:9])
else:
    rng = np.random.default_rng(123)
    best = None
    for start in range(args.starts):
        x0 = nominal.copy() if start == 0 else nominal + rng.normal(0, 0.3, size=len(names))
        x0 = np.clip(x0, lower + 1e-5, upper - 1e-5)
        result = least_squares(residual, x0, bounds=(lower + 1e-6, upper - 1e-6), max_nfev=500)
        error = np.linalg.norm(residual(result.x)[:9])
        if best is None or error < best[0]:
            best = (error, result.x)
    error, q = best
print(json.dumps({
    "object_xyz_m": object_pos.tolist(),
    "target_centroids_world_m": target_centers.tolist(),
    "actual_centroids_world_m": centers(q).tolist(),
    "finger_collision_min_world_z_m": finger_min_z(q).tolist(),
    "finger_collision_min_world_xyz_m": finger_bounds(q)[0].tolist(),
    "finger_collision_max_world_xyz_m": finger_bounds(q)[1].tolist(),
    "contact_centroid_error_m": float(error),
    "arm_joint_radians": q[:8].tolist(),
    "finger_joint_radians": q[8:].tolist(),
    "near_limits": ((q - lower < 0.02) | (upper - q < 0.02)).tolist(),
}, indent=2))
