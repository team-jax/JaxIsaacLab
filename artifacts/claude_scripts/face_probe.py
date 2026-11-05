import argparse, importlib.metadata, json, math
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); app = AppLauncher(args).app
import gymnasium as gym, torch
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks  # noqa
from isaaclab.utils.math import quat_apply
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2FrontPregraspPPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_pregrasp import JaxV2FrontPregraspEnvCfg
cfg = JaxV2FrontPregraspEnvCfg(); cfg.scene.num_envs = 1
raw = gym.make("Isaac-JaxV2-FrontPregrasp-Direct-v12", cfg=cfg)
ac = handle_deprecated_rsl_rl_cfg(JaxV2FrontPregraspPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(raw, clip_actions=ac.clip_actions)
r = OnPolicyRunner(env, ac.to_dict(), log_dir=None, device=ac.device); r.load(args.checkpoint)
pol = r.get_inference_policy(device=env.unwrapped.device); u = raw.unwrapped; R = u.robot; B = R.body_names
X = torch.tensor([[1.0, 0, 0]], device=u.device)
obs = env.get_observations()
with torch.inference_mode():
    for k in range(300):
        obs, *_ = env.step(pol(obs))
    n = lambda b: quat_apply(R.data.body_quat_w[0:1, B.index(b)], X)[0]
    f1, f2, t1, t2 = n("Finger_1_1"), n("Finger_2_1"), n("Finger_3_1"), n("Finger_3_2")
    nf = (f1 + f2) / (f1 + f2).norm()
    P = R.data.body_pos_w[0]; cube = u.object.data.root_pos_w[0]
    hand = P[B.index("MG996R")]; fm = 0.5 * (P[B.index("Finger_1_2")] + P[B.index("Finger_2_2")])
    d = (fm - hand) / (fm - hand).norm()
    deg = lambda v: round(math.degrees(v), 1)
    print("FACE " + json.dumps({
        "hand_dir": [round(float(v), 2) for v in d], "hand_yaw_off_deg": deg(math.atan2(float(d[1]), float(-d[0]))),
        "finger_pad_normal": [round(float(v), 2) for v in nf], "finger_normal_vs_-y_deg": deg(math.acos(max(-1, min(1, float(-nf[1]))))),
        "thumb1_pad_normal": [round(float(v), 2) for v in t1], "thumb_vs_fingers_dot": round(float(t1 @ nf), 2),
        "thumb_normal_vs_+y_deg": deg(math.acos(max(-1, min(1, float(t1[1]))))),
        "cube_yaw_deg": deg(2 * math.atan2(float(u.object.data.root_quat_w[0, 3]), float(u.object.data.root_quat_w[0, 0]))),
    }), flush=True)
app.close()
