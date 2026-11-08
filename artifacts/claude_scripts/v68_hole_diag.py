"""v68: why does the grasp fail at a cube turn of -35 deg relative to the hand (y 0 cells 56-58%, -30 / -40 deg 96-100%)?
One cell (EVAL_CELLS), fixed turn (GL_CUBE_YAW_FIXED_DEG), the v46_eval_done env; per env at 10 Hz: cube yaw change from the
start, xy shift, rise, the 6 pads, the 6 finger joints, the hand height above the cube; saved to OUT (.pt) with done/held."""
import sys
CK = sys.argv[1]; OUT = sys.argv[2]; sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import os, math, importlib.metadata, torch, gymnasium as gym
os.environ["GL_CUBE_BOX"] = "1"
import isaaclab_tasks  # noqa
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2FrontGraspLiftPPORunnerCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_grasp_lift import JaxV2FrontGraspLiftEnvCfg
from isaaclab.utils.math import quat_apply
cfg = JaxV2FrontGraspLiftEnvCfg(); cfg.episode_length_s = 5.0 + float(os.environ.get("EVAL_S", "12")) + 1.0; NE = int(os.environ.get("NE", "64")); cfg.scene.num_envs = NE
raw = gym.make("Isaac-JaxV2-FrontGraspLift-Direct-v16", cfg=cfg); u = raw.unwrapped
grid = torch.tensor([[float(v) for v in c.split(",")] for c in os.environ["EVAL_CELLS"].split(";")], device=u.device)
u._gl_cube_off_fixed = grid[torch.arange(NE, device=u.device) % grid.shape[0]]
ac = handle_deprecated_rsl_rl_cfg(JaxV2FrontGraspLiftPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(raw, clip_actions=1.0)
runner = OnPolicyRunner(env, ac.to_dict(), log_dir=None, device=ac.device); runner.load(CK); pol = runner.get_inference_policy(device=u.device)
u._reset_idx(u.robot._ALL_INDICES)
obs = env.get_observations()
ex = torch.zeros((NE, 3), device=u.device); ex[:, 0] = 1.0
rec = {k: [] for k in ("yaw", "xy", "rise", "pads", "fq", "tipz", "t")}
with torch.inference_mode():
    cube0 = u.object.data.root_pos_w.clone(); x0 = quat_apply(u.object.data.root_quat_w, ex)
    pads_start = u.pad_forces().clone()
    for k in range(round(float(os.environ.get("EVAL_S", "12")) * 60)):
        obs, _, _, _ = env.step(pol(obs))
        if k % 6 == 5:
            xc = quat_apply(u.object.data.root_quat_w, ex)
            yaw = torch.rad2deg(torch.atan2(xc[:, 1], xc[:, 0]) - torch.atan2(x0[:, 1], x0[:, 0])); yaw = (yaw + 180) % 360 - 180
            tip = u.robot.data.body_pos_w[:, list(u.fingertip_ids)][:, :, 2] - u.object.data.root_pos_w[:, None, 2]
            rec["yaw"].append(yaw.clone()); rec["xy"].append((u.object.data.root_pos_w[:, :2] - cube0[:, :2]).norm(dim=1) * 1000)
            rec["rise"].append((u.object.data.root_pos_w[:, 2] - cube0[:, 2]) * 1000); rec["pads"].append(u.pad_forces().clone())
            rec["fq"].append(torch.rad2deg(u.robot.data.joint_pos[:, u.finger_ids]).clone()); rec["tipz"].append(tip * 1000); rec["t"].append((k + 1) / 60.0)
    rise = u.object.data.root_pos_w[:, 2] - cube0[:, 2]
    out = {k: torch.stack(v).cpu() if k != "t" else torch.tensor(v) for k, v in rec.items()}
    out["done"] = u.lift_done.cpu(); out["held"] = (u.lift_done & (rise > 0.04)).cpu(); out["pads_start"] = pads_start.cpu()
    torch.save(out, OUT)
    print("HOLE_DIAG saved", OUT, "held %.3f" % float(out["held"].float().mean()), flush=True)
import os as _os; _os._exit(0)
