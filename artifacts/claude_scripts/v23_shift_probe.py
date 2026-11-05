"""v22: per-waypoint joint shift for a table shift (offline). For each geometric v18 waypoint,
solve dq on shoulder pitch, elbow, wrist pitch that moves the hand by +1 cm in x (or z) at
constant hand pitch (pitch sum). Saves data/v23_table_shift.pt: (n_wp, 8, 2) rad per m [dx, dz]."""
import sys; sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import math, torch, gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_arm_raise import JaxV2RaiseToPregraspEnvCfg, V23_WAYPOINTS
cfg = JaxV2RaiseToPregraspEnvCfg(); cfg.scene.num_envs = 1
u = gym.make("Isaac-JaxV2-RaiseToPregrasp-Direct-v18", cfg=cfg).unwrapped; u.reset()
arm = list(u.arm_ids); cols = [arm[0], arm[3], arm[5]]
def put(q8):
    q = u.robot.data.joint_pos.clone(); q[0, arm] = q8; q[0, u.finger_ids[2]] = math.radians(-30)
    u.robot.write_joint_state_to_sim(q, torch.zeros_like(q)); u.sim.forward(); u.robot.update(0.0)
def hand():
    return u.robot.data.body_pos_w[0, u._hand_id] - u.scene.env_origins[0]
out = []
with torch.inference_mode():
    for t, qd in V23_WAYPOINTS:
        q0 = torch.tensor([math.radians(v) for v in qd], device=u.device)
        S = torch.zeros(8, 2, device=u.device)
        for k, d in enumerate(((0.01, 0.0), (0.0, 0.01))):
            put(q0); p0 = hand().clone(); q = q0.clone()
            for it in range(30):
                put(q); p = hand()
                err = torch.tensor([p0[0] + d[0] - p[0], p0[2] + d[1] - p[2], (q0[0] + q0[3] + q0[5]) - (q[0] + q[3] + q[5])], device=u.device)
                J = u.robot.root_physx_view.get_jacobians()[0, u._hand_id - 1]
                Js = torch.stack([J[0, cols], J[2, cols], torch.ones(3, device=u.device)])
                q[[0, 3, 5]] += torch.linalg.lstsq(Js, err[:, None]).solution[:, 0]
            S[:, k] = (q - q0) / 0.01
        out.append(S.cpu())
        print("SHIFT", t, "deg per cm dx:", [round(math.degrees(float(v)) / 100, 2) for v in S[[0, 3, 5], 0]],
              "dz:", [round(math.degrees(float(v)) / 100, 2) for v in S[[0, 3, 5], 1]], flush=True)
torch.save(torch.stack(out), "source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/data/v23_table_shift.pt")
print("SAVED", flush=True)
sys.stdout.flush()
import os; os._exit(0)
