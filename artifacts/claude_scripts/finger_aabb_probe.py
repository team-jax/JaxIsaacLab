"""World z-range of finger collision meshes at the pre-grasp (5 s) and the
closed state (8 s), v13 env with its frozen pre-grasp actor and zero actions."""
import argparse, json, math, os, struct, xml.etree.ElementTree as ET
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); args = p.parse_args(); app = AppLauncher(args).app
import gymnasium as gym, numpy as np, torch
import isaaclab_tasks  # noqa
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_close import JaxV2FrontGraspCloseEnvCfg
D = "source/isaaclab_assets/data/Robots/JaxV2Assembly1006"
root = ET.parse(f"{D}/urdf/JAXV2_Assembly.urdf").getroot()
def rpy_mat(r, p_, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p_), math.sin(p_), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr], [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr], [-sp, cp*sr, cp*cr]])
meshes = {}
for l in root.findall("link"):
    n = l.get("name")
    if not n.startswith("Finger_"): continue
    c = l.find("collision"); m = c.find("geometry/mesh"); o = c.find("origin")
    fn = m.get("filename").split("://")[-1]
    for cand in [fn, os.path.join(D, "urdf", fn), os.path.join(D, "meshes", os.path.basename(fn))]:
        if os.path.exists(cand): fn = cand; break
    d = open(fn, "rb").read(); k = struct.unpack("<I", d[80:84])[0]
    v = np.frombuffer(d[84:84 + k * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))["v"].reshape(-1, 3).astype(np.float64)
    v = v * np.array([float(x) for x in m.get("scale", "1 1 1").split()])
    R = rpy_mat(*[float(x) for x in o.get("rpy").split()]); t = np.array([float(x) for x in o.get("xyz").split()])
    meshes[n] = v @ R.T + t
cfg = JaxV2FrontGraspCloseEnvCfg(); cfg.scene.num_envs = 16; cfg.sensor_randomize = False
_it = int(os.environ.get("POS_ITERS", "0"))
if os.environ.get("NO_TABLE"):
    import isaaclab.sim as sim_utils
    cfg.table_cfg = cfg.table_cfg.replace(collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False))
    cfg.object_cfg = cfg.object_cfg.replace(spawn=cfg.object_cfg.spawn.replace(rigid_props=cfg.object_cfg.spawn.rigid_props.replace(kinematic_enabled=True)))
if _it:
    cfg.robot_cfg = cfg.robot_cfg.replace(spawn=cfg.robot_cfg.spawn.replace(articulation_props=cfg.robot_cfg.spawn.articulation_props.replace(solver_position_iteration_count=_it, solver_velocity_iteration_count=4)))
raw = gym.make("Isaac-JaxV2-FrontGraspClose-Direct-v13", cfg=cfg); u = raw.unwrapped; raw.reset()
act = torch.zeros(16, 11, device=u.device)
def report(tag):
    cube = u.object.data.root_pos_w[0]
    out = {"tag": tag, "table_top_z": cfg.table_top_z, "cube_top_z": float(cube[2]) + cfg.object_size[2] / 2}
    for n, v in meshes.items():
        i = u.robot.body_names.index(n)
        pos = u.robot.data.body_pos_w[0, i]; q = u.robot.data.body_quat_w[0, i]
        w = quat_apply(q.expand(len(v), 4), torch.tensor(v, dtype=torch.float32, device=u.device)) + pos
        out[n] = {"z_min_above_table_mm": round(float((w[:, 2].min() - cfg.table_top_z) * 1000), 1),
                  "z_max_above_table_mm": round(float((w[:, 2].max() - cfg.table_top_z) * 1000), 1),
                  "y_min_rel_cube_mm": round(float((w[:, 1].min() - cube[1]) * 1000), 1), "y_max_rel_cube_mm": round(float((w[:, 1].max() - cube[1]) * 1000), 1)}
    for sn in ("proximal_1", "proximal_2", "finger_1", "finger_2"):
        sd = u.scene.sensors[sn].data
        out["net_" + sn] = [round(float(x), 2) for x in sd.net_forces_w[0, 0]]
        out["cube_" + sn] = [round(float(x), 2) for x in sd.force_matrix_w[0, 0, 0]]
    fi = list(u.finger_ids)
    out["target_fingers_deg"] = [round(float(x), 1) for x in u.targets[0, fi].rad2deg()]
    out["torque_fingers_Nm"] = [round(float(x), 3) for x in u.robot.data.applied_torque[0, fi]]
    lim = u.robot.data.joint_pos_limits[0, fi].rad2deg() if hasattr(u.robot.data, "joint_pos_limits") else u.robot.data.soft_joint_pos_limits[0, fi].rad2deg()
    out["limits_deg"] = [[round(float(a), 1), round(float(b), 1)] for a, b in lim]
    out["stiffness"] = [round(float(x), 1) for x in u.robot.data.joint_stiffness[0, fi]] if hasattr(u.robot.data, "joint_stiffness") else None
    out["q_fingers_deg"] = [round(float(x), 1) for x in u.robot.data.joint_pos[0, u.finger_ids].rad2deg()]
    print("AABB", json.dumps(out), flush=True)
with torch.inference_mode():
    for k in range(int(8.0 / u.step_dt)):
        raw.step(act)
        if k == int(4.9 / u.step_dt): report("5s_pregrasp")
    report("8s_closed")
os._exit(0)
