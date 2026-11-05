"""Why does the pre-grasp lift the hand when RotationBody turns -15 deg?
model_3038 policy to 3.5 s, then RotationBody ramps -15 deg over 1 s.
Group A (envs 0-31): policy keeps acting. Group B (32-63): arm targets frozen.
Log reward terms, thumb link heights, contacts, at 3.4 s and 5.0 s."""
import argparse, importlib.metadata, json, math
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--rb", type=float, default=-15.0); p.add_argument("--mask", default="none")
AppLauncher.add_app_launcher_args(p); args = p.parse_args(); app = AppLauncher(args).app
import gymnasium as gym, torch
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2FrontPregraspPPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_pregrasp import JaxV2FrontPregraspEnvCfg, JaxV2FrontPregraspEnv
cfg = JaxV2FrontPregraspEnvCfg(); cfg.scene.num_envs = 64
raw = gym.make("Isaac-JaxV2-FrontPregrasp-Direct-v12", cfg=cfg); u = raw.unwrapped
ac = handle_deprecated_rsl_rl_cfg(JaxV2FrontPregraspPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(raw, clip_actions=ac.clip_actions)
r = OnPolicyRunner(env, ac.to_dict(), log_dir=None, device=ac.device); r.load(args.checkpoint)
pol = r.get_inference_policy(device=u.device); obs = env.get_observations()
B = torch.arange(64, device=u.device) >= 32
state = {"frozen": None}
orig_hold = JaxV2FrontPregraspEnv._hold_hand_shape
def hold(self):
    orig_hold(self)
    t = self.episode_length_buf * self.step_dt - 3.5
    ramp = (t / 1.0).clamp(0.0, 1.0)
    self.targets[:, self.arm_ids[7]] = self.cfg.start_arm_q[7] + ramp * math.radians(args.rb)
    if state["frozen"] is not None:
        arm = self.arm_ids[:7]
        self.targets[:, arm] = torch.where(B[:, None], state["frozen"], self.targets[:, arm])
u._hold_hand_shape = hold.__get__(u)
rb_i = u.robot.joint_names.index("hand_base_to_rotation_base"); nj = len(u.robot.joint_names)
print("WHY", json.dumps({"rb_index": rb_i, "n_joints": nj}))
orig_obs = type(u)._get_observations
def obs_fn(self):
    o = orig_obs(self)["policy"].clone()
    if args.mask in ("rb", "rb_hand") and state.get("rbo") is not None:
        o[:, rb_i] = state["rbo"][:, 0]; o[:, nj + rb_i] = state["rbo"][:, 1]
    state["last_o"] = o
    if args.mask == "rb_hand" and state.get("hand0") is not None:
        o[:, -9:-6] = state["hand0"]
    return {"policy": o}
u._get_observations = obs_fn.__get__(u)
names = [f"proximal_{i}" for i in (1, 2, 3)] + [f"finger_{i}" for i in (1, 2, 3)]
th = [u.robot.body_names.index(n) for n in ("Finger_3_1", "Finger_3_2")]
def snap(tag):
    terms = {k: v for k, v in u._pre_terms.items()}
    cube = u.object.data.root_pos_w
    cn = torch.stack([torch.linalg.vector_norm(u.scene.sensors[n].data.force_matrix_w[:, 0, 0], dim=-1) for n in names], 1)
    tips = u.robot.data.body_pos_w[:, u.fingertip_ids]
    for g, m in (("A_policy", ~B), ("B_frozen", B)):
        out = {"tag": tag, "group": g}
        out.update({k: round(float(v[m].mean()), 3) for k, v in terms.items()})
        out["thumb_links_above_table_mm"] = [round(float((u.robot.data.body_pos_w[m, i, 2] - cfg.table_top_z).mean() * 1000), 1) for i in th]
        out["thumb_tip_rel_cube_mm"] = [round(float(v), 1) for v in ((tips[m, 2] - cube[m]) * 1000).mean(0)]
        out["f12_tip_z_rel_cube_mm"] = [round(float(((tips[m, j, 2] - cube[m, 2]) * 1000).mean()), 1) for j in range(2)]
        out["contact_N(p1,p2,p3,f1,f2,f3)"] = [round(float(v), 2) for v in cn[m].mean(0)]
        out["elbow_deg"] = round(float(u.robot.data.joint_pos[m, u.arm_ids[3]].mean() * 57.3), 1)
        print("WHY", json.dumps(out), flush=True)
with torch.inference_mode():
    for k in range(int(5.0 / u.step_dt)):
        if k == int(3.4 / u.step_dt):
            snap("3.4s_before"); state["frozen"] = u.targets[:, u.arm_ids[:7]].clone(); state["hand0"] = None; state["rbo"] = torch.stack((state["last_o"][:, rb_i], state["last_o"][:, nj + rb_i]), 1).clone()
        obs, *_ = env.step(pol(obs))
    snap("5.0s_after")
import os; os._exit(0)
