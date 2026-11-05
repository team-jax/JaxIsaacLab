"""Check recorded V2 collision-mesh vertices against the table cuboid.

This is an offline geometric check of a scripted trace, not a physics contact
sensor and not a full triangle-versus-box collision test.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def transform(xyz, rpy):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    pose[:3, 3] = xyz
    return pose


def link_pose(name, joints, q, cache):
    if name in cache:
        return cache[name]
    joint = joints[name]
    parent = joint.find("parent").get("link")
    origin = joint.find("origin")
    pose = link_pose(parent, joints, q, cache) @ transform(
        np.fromstring(origin.get("xyz", "0 0 0"), sep=" "),
        np.fromstring(origin.get("rpy", "0 0 0"), sep=" "),
    )
    if joint.get("type") != "fixed":
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        turn = np.eye(4)
        turn[:3, :3] = Rotation.from_rotvec(axis * q.get(joint.get("name"), 0.0)).as_matrix()
        pose = pose @ turn
    cache[name] = pose
    return pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    urdf = root / "isaaclab_assets/data/Robots/JaxV2Assembly1006/urdf/JAXV2_Assembly.urdf"
    xml = ET.parse(urdf).getroot()
    joints = {joint.find("child").get("link"): joint for joint in xml.findall("joint")}
    names = ("MG996R", "RotationBody", "Finger_1_1", "Finger_2_1", "Finger_3_1",
             "Finger_1_2", "Finger_2_2", "Finger_3_2")
    meshes = {}
    for link in xml.findall("link"):
        if link.get("name") not in names:
            continue
        collision = link.find("collision")
        geometry = collision.find("geometry") if collision is not None else None
        mesh = geometry.find("mesh") if geometry is not None else None
        if mesh is None:
            continue
        vertices = trimesh.load(urdf.parent / mesh.get("filename"), force="mesh").vertices
        vertices *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
        origin = collision.find("origin")
        offset = transform(
            np.fromstring(origin.get("xyz", "0 0 0"), sep=" "),
            np.fromstring(origin.get("rpy", "0 0 0"), sep=" "),
        )
        meshes[link.get("name")] = (offset, np.c_[vertices, np.ones(len(vertices))].T)
    trace = np.load(args.video_dir / "motion_trace.npz")
    joint_names = trace["joint_names"].tolist()
    summary = json.loads((args.video_dir / "summary.json").read_text())
    table_center_x = summary["table_center_x"]
    spread_start = summary.get("finger_spread_start_s", 9.5)
    spread_end = summary.get("finger_spread_end_s", 12.5)
    box_min = np.array([table_center_x - 0.4, -0.35, 0.70])
    box_max = np.array([table_center_x + 0.4, 0.35, 0.74])
    root_pose = transform((0, 0, 0.305), (0, 0, -np.pi / 2))
    minimum = (float("inf"), None, None)
    minimum_spread = (float("inf"), None, None)
    for frame, values in enumerate(trace["joint_q"]):
        q = dict(zip(joint_names, values))
        cache = {"base_link": root_pose}
        for name, (offset, vertices) in meshes.items():
            points = (link_pose(name, joints, q, cache) @ offset @ vertices)[:3].T
            outside = np.maximum(np.maximum(box_min - points, points - box_max), 0)
            distances = np.linalg.norm(outside, axis=1)
            inside = (points >= box_min).all(axis=1) & (points <= box_max).all(axis=1)
            if inside.any():
                distances[inside] = -np.minimum(points[inside] - box_min, box_max - points[inside]).min(axis=1)
            distance = float(distances.min())
            if distance < minimum[0]:
                minimum = (distance, frame / 60, name)
            if spread_start <= frame / 60 < spread_end and distance < minimum_spread[0]:
                minimum_spread = (distance, frame / 60, name)
    output = {
        "minimum_collision_vertex_to_table_m": minimum[0],
        "time_s": minimum[1],
        "link": minimum[2],
        "minimum_during_finger_spread_m": minimum_spread[0],
        "spread_time_s": minimum_spread[1],
        "spread_link": minimum_spread[2],
        "method": "URDF collision-mesh vertices versus table cuboid; triangle interiors not tested",
    }
    print(json.dumps(output, indent=2))
    (args.video_dir / "mesh_clearance.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
