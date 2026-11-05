import argparse, json
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); args = p.parse_args(); app = AppLauncher(args).app
import gymnasium as gym, torch
import isaaclab_tasks  # noqa
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_close import JaxV2FrontGraspCloseEnvCfg
cfg = JaxV2FrontGraspCloseEnvCfg(); cfg.scene.num_envs = 2
raw = gym.make("Isaac-JaxV2-FrontGraspClose-Direct-v13", cfg=cfg); u = raw.unwrapped; raw.reset()
R = u.robot; d = R.data; v = R.root_physx_view
names = R.joint_names; bodies = R.body_names
out = {"joints": {}, "bodies": {}}
lim = v.get_dof_limits()[0]; arm = v.get_dof_armatures()[0] if hasattr(v, "get_dof_armatures") else None
fr = v.get_dof_friction_coefficients()[0] if hasattr(v, "get_dof_friction_coefficients") else None
kp = v.get_dof_stiffnesses()[0]; kd = v.get_dof_dampings()[0]; mf = v.get_dof_max_forces()[0]; mv = v.get_dof_max_velocities()[0]
for i, n in enumerate(names):
    if "finger" in n:
        out["joints"][n] = {"lim_deg": [round(float(x) * 57.2958, 1) for x in lim[i]], "kp": float(kp[i]), "kd": float(kd[i]), "max_f": float(mf[i]), "max_v": float(mv[i]),
                            "armature": float(arm[i]) if arm is not None else None, "friction": float(fr[i]) if fr is not None else None}
m = v.get_masses()[0]; I = v.get_inertias()[0]
for i, n in enumerate(bodies):
    if n.startswith("Finger") or n in ("MG996R", "RotationBody"):
        out["bodies"][n] = {"mass": round(float(m[i]), 5), "inertia_diag": [float(f"{float(x):.3g}") for x in I[i].view(3, 3).diagonal()]}
print("FP", json.dumps(out), flush=True)
import os; os._exit(0)
