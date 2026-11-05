"""v13 pre-grasp (frozen actor) then close only first joints, second joints 0."""
import argparse, json, math
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); args = p.parse_args(); app = AppLauncher(args).app
import gymnasium as gym, torch
import isaaclab_tasks  # noqa
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_close import JaxV2FrontGraspCloseEnvCfg, JaxV2FrontGraspCloseEnv
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_pad_sensor_grasp import JaxV2PadSensorGraspEnv
F = [0, 5, 10]; T = [-25, -20, -15, -10, -5]; REP = 4
COMBOS = [(f, t) for f in F for t in T for _ in range(REP)]
cfg = JaxV2FrontGraspCloseEnvCfg(); cfg.scene.num_envs = len(COMBOS); cfg.sensor_randomize = False
raw = gym.make("Isaac-JaxV2-FrontGraspClose-Direct-v13", cfg=cfg); u = raw.unwrapped
g = torch.tensor([[f, f, t, 0, 0, 0] for f, t in COMBOS], dtype=torch.float32, device=u.device).deg2rad()
orig = JaxV2FrontGraspCloseEnv._hold_hand_shape
def hold(self):
    orig(self)
    t = self.episode_length_buf * self.step_dt - self.cfg.close_after_s
    ramp = (t / self.cfg.close_ramp_s).clamp(0.0, 1.0)[:, None]
    closed = self._hand_shape + ramp * (g - self._hand_shape)
    self.targets[:, self.finger_ids] = torch.where(self._closing()[:, None], closed, self._hand_shape)
u._hold_hand_shape = hold.__get__(u)
raw.reset()
act = torch.zeros(len(COMBOS), 11, device=u.device)
with torch.inference_mode():
    for k in range(int(8.0 / u.step_dt)):
        if k == int(4.9 / u.step_dt): cube0 = u.object.data.root_pos_w.clone()
        raw.step(act)
    pad = JaxV2PadSensorGraspEnv.pad_forces(u)
    cube = u.object.data.root_pos_w; cq = u.object.data.root_quat_w
    cy = quat_apply(cq, torch.tensor([0.0, 1.0, 0.0], device=u.device).expand(len(COMBOS), 3))
    prox = u.robot.data.body_pos_w[:, u.proximal_ids[:2]]; tips = u.robot.data.body_pos_w[:, u.fingertip_ids]
    side = torch.sign((((0.5 * (prox + tips[:, :2])).mean(1) - cube) * cy).sum(1)); side = torch.where(side == 0, 1.0, side)
    X = torch.tensor([1.0, 0.0, 0.0], device=u.device).expand(len(COMBOS) * 3, 3)
    pd = quat_apply(u.robot.data.body_quat_w[:, list(u.fingertip_ids)].reshape(-1, 4), X).reshape(-1, 3, 3)
    want = torch.stack((-side, -side, side), 1)[..., None] * cy[:, None]
    flat = (pd * want).sum(-1)
    q = u.robot.data.joint_pos[:, u.finger_ids].rad2deg()
    for i in range(len(COMBOS)):
        print("FS", json.dumps({"f1": COMBOS[i][0], "thumb1": COMBOS[i][1], "distal_N": [round(float(v), 2) for v in pad[i, 3:]], "prox_N": [round(float(v), 2) for v in pad[i, :3]],
              "flat": [round(float(v), 2) for v in flat[i]], "q": [round(float(v), 1) for v in q[i]], "cube_mm": round(float((cube[i] - cube0[i]).norm() * 1000), 1),
              "tilt": round(float(cy[i, 2]), 2)}), flush=True)
import os; os._exit(0)
