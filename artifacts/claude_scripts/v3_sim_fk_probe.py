"""v3: same joint targets on the V2 and V3 sim robots (GL_ROBOT=v3 or not) - world positions of the hand and fingertips at
the pre-grasp start pose of the grasp-lift env (one cube offset), joint axes as the importer built them. argv: out.json"""
import sys
OUT = sys.argv[1]; sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import os, json, math, torch, gymnasium as gym
os.environ["GL_CUBE_BOX"] = "1"
import isaaclab_tasks  # noqa
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_lift import JaxV2FrontGraspLiftEnvCfg
cfg = JaxV2FrontGraspLiftEnvCfg(); cfg.scene.num_envs = 2
raw = gym.make("Isaac-JaxV2-FrontGraspLift-Direct-v16", cfg=cfg); u = raw.unwrapped
u._gl_cube_off_fixed = torch.tensor([[-0.08, 0.0]], device=u.device).expand(2, -1)
res = {"robot": os.environ.get("GL_ROBOT", "v2"), "joint_names": u.robot.joint_names, "body_names": u.robot.body_names}
with torch.inference_mode():
    u._reset_idx(u.robot._ALL_INDICES)
    for k in range(3):
        u.sim.step(render=False); u.scene.update(u.sim.get_physics_dt())
    o = u.scene.env_origins[0]
    res["lim_deg"] = [[round(math.degrees(float(a)), 2), round(math.degrees(float(b)), 2)] for a, b in zip(u.lower, u.upper)]
    res["q_deg"] = [round(math.degrees(float(v)), 2) for v in u.robot.data.joint_pos[0]]
    for n in ("Shoulder_Pitch", "Wrist_Pitch", "MG996R", "Finger_1_2", "Finger_2_2", "Finger_3_2", "RotationBody"):
        if n in u.robot.body_names:
            i = u.robot.body_names.index(n); res[n] = [round(float(v), 4) for v in (u.robot.data.body_pos_w[0, i] - o)]
    res["cube"] = [round(float(v), 4) for v in (u.object.data.root_pos_w[0] - o)]
    res["masses"] = {n: round(float(m), 4) for n, m in zip(u.robot.body_names, u.robot.root_physx_view.get_masses()[0])}
json.dump(res, open(OUT, "w"), indent=1)
print("FK_PROBE", OUT, flush=True)
os._exit(0)
