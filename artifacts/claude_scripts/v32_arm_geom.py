"""v32: arm link geometry at the L end pose (old vs straight-hand): world direction of each segment
between consecutive arm link origins, and its angle from vertical / horizontal-forward."""
import sys; sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import math, torch, gymnasium as gym, isaaclab_tasks  # noqa
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_arm_raise import JaxV2RaiseCubeWideEnvCfg
c = JaxV2RaiseCubeWideEnvCfg(); c.scene.num_envs = 1
u = gym.make("Isaac-JaxV2-RaiseCubeWide-Direct-v27", cfg=c).unwrapped; u.reset()
arm = list(u.arm_ids); fids = list(u.finger_ids); R = u.robot
print("BODIES", R.body_names, flush=True)
fing = torch.tensor([math.radians(v) for v in (13.63, 10.58, -30.43, 0, 0, 0)], device=u.device)
poses = {"old_L": (-0.15, 0.0, 0.0, 89.82, 0.15, -0.02, -0.16, -169.6), "straight_L": (-0.14, 0.0, 7.2, 89.83, 0.16, -2.62, -0.17, -169.6),
         "forearm_straight": (0, 0, 11.4, 90, 0, 0, 0, -169.6), "forearm_straight_j4m": (0, 0, 11.4, 90, -4.0, 0, 0, -169.6)}
names = [R.joint_names[j] for j in arm[:7]]
with torch.inference_mode():
    for k, v in poses.items():
        q = R.data.joint_pos.clone(); q[0, arm] = torch.tensor([math.radians(a) for a in v], device=u.device); q[0, fids] = fing
        R.write_joint_state_to_sim(q, torch.zeros_like(q)); u.sim.forward(); R.update(0.0)
        P = R.data.body_pos_w[0] - u.scene.env_origins[0]
        print("==", k)
        chain = [R.body_names.index(b) for b in ("base_link", "Shoulder_Roll", "Shoulder_Yaw", "Upper_Arm", "Under_Arm", "Wrist_Pitch", "Wrist_yaw", "MG996R")]
        last = chain[0]
        for b in chain[1:]:
            d = P[b] - P[last]; n = float(d.norm())
            if n < 0.01:
                continue
            d = d / n
            print("  %-14s -> %-14s len %.3f dir (%+.3f %+.3f %+.3f) tilt_from_vertical %.1f  horiz_yaw_from_-x %.1f  pitch_from_horizontal %.1f" % (R.body_names[last], R.body_names[b], n, *[float(x) for x in d],
                  math.degrees(math.acos(min(1, abs(float(d[2]))))), math.degrees(math.atan2(float(d[1]), -float(d[0]))), math.degrees(math.asin(float(d[2])))), flush=True)
            last = b
        tip = R.data.body_pos_w[0, list(u.fingertip_ids)].mean(0) - u.scene.env_origins[0]; h = P[u._hand_id]
        d = (tip - h) / (tip - h).norm()
        print("  hand->fingertips dir (%+.3f %+.3f %+.3f)" % tuple(float(x) for x in d), flush=True)
import os; os._exit(0)
