"""v27: raise policy alone over 9 cube offsets (x -8/-3/+2, y +6/0/-5 cm), 4 envs each, deterministic.
argv: CK (or 'nominal' = zero residual). Per position: max cube displacement (mm), max cube-touch and
table-contact force (N) during the raise, final reference error (deg)."""
import sys
CK = sys.argv[1]; DZ = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0; sys.argv = sys.argv[:1] + ["--headless"]
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import json, importlib.metadata, torch, gymnasium as gym, isaaclab_tasks  # noqa
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_arm_raise import JaxV2RaiseCubeWideEnvCfg
from isaaclab_tasks.direct.one_arm_jax_minimal.agents.rsl_rl_ppo_cfg import JaxV2RaiseCubePPORunnerCfg
c = JaxV2RaiseCubeWideEnvCfg(); c.scene.num_envs = 36; c.raise_rsi_prob = 0.0 if hasattr(c, "raise_rsi_prob") else None
for k in ("raise_gain_rand",):
    if hasattr(c, k): setattr(c, k, 0.0)
raw = gym.make("Isaac-JaxV2-RaiseCubeWide-Direct-v27", cfg=c); u = raw.unwrapped
grid = torch.tensor([[x, y] for x in (-0.08, -0.03, 0.02) for y in (0.06, 0.0, -0.05)], device=u.device)
g = torch.arange(36, device=u.device) % 9
u._cube_off_fixed = grid[g]; u._table_off_fixed = torch.tensor([0.0, DZ], device=u.device)
env = RslRlVecEnvWrapper(raw, clip_actions=1.0)
if CK != "nominal":
    ac = handle_deprecated_rsl_rl_cfg(JaxV2RaiseCubePPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, ac.to_dict(), log_dir=None, device=ac.device); runner.load(CK); pol = runner.get_inference_policy(device=u.device)
else:
    pol = lambda o: torch.zeros(36, u.single_action_space.shape[0], device=u.device)
u.episode_length_buf[:] = 0; u._reset_idx(u.robot._ALL_INDICES); obs = env.get_observations()
names = [f"proximal_{i}" for i in (1, 2, 3)] + [f"finger_{i}" for i in (1, 2, 3)]
with torch.inference_mode():
    u.episode_length_buf[:] = 0
    cube0 = u.object.data.root_pos_w.clone(); dmax = torch.zeros(36, device=u.device); cmax = torch.zeros_like(dmax); tmax = torch.zeros_like(dmax)
    for k in range(round(c.raise_nominal_s * 60) + 30):
        obs, _, done, _ = env.step(pol(obs))
        if bool(done.any()): break
        dmax = torch.maximum(dmax, (u.object.data.root_pos_w - cube0).norm(dim=1))
        cn = torch.stack([torch.linalg.vector_norm(u.scene.sensors[n].data.force_matrix_w[:, 0, 0], dim=-1) for n in names], 1).sum(1)
        nn = torch.stack([torch.linalg.vector_norm(u.scene.sensors[n].data.net_forces_w[:, 0], dim=-1) for n in names], 1).sum(1)
        cmax = torch.maximum(cmax, cn); tmax = torch.maximum(tmax, (nn - cn).clamp_min(0))
    err = torch.rad2deg(u._ref_err_log) if hasattr(u, "_ref_err_log") else torch.zeros(36)
out = []
for i in range(9):
    m = g == i
    out.append("(%+.0f,%+.0f)%.0fmm/%.1fN/%.1fN" % (grid[i, 0] * 100, grid[i, 1] * 100, dmax[m].max() * 1000, cmax[m].max(), tmax[m].max()))
print("RAISE_EVAL dz %+.3f" % DZ, "steps", k, "worst_cube_mm %.1f" % (dmax.max() * 1000), "mean_err_deg %.2f" % float(err.mean()), " ".join(out), flush=True)
import os; os._exit(0)
