"""Stage 2 scan: open hand shapes whose thumb pad faces the finger pads squarely (perpendicular to the cube face)."""
import argparse, itertools, json, math
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); app = AppLauncher(args).app
import gymnasium as gym, torch
import isaaclab_tasks  # noqa
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.direct.one_arm_jax_minimal.v2_front_pregrasp import JaxV2FrontPregraspEnvCfg
ROT = list(range(-180, -119, 5)); TH1 = list(range(-60, 31, 5)); TH2 = [0, 20, 40]; FQ = list(range(0, 31, 5))
grid = list(itertools.product(ROT, TH1, TH2, FQ))
cfg = JaxV2FrontPregraspEnvCfg(); cfg.scene.num_envs = len(grid); cfg.scene.env_spacing = 2.0
raw = gym.make("Isaac-JaxV2-FrontPregrasp-Direct-v12", cfg=cfg); u = raw.unwrapped; raw.reset()
R = u.robot; dev = u.device; N = len(grid)
g = torch.tensor(grid, dtype=torch.float32, device=dev) * math.pi / 180
q = R.data.joint_pos.clone(); f = u.finger_ids
q[:, u.arm_ids[7]] = g[:, 0]; q[:, f[2]] = g[:, 1]; q[:, f[5]] = g[:, 2]; q[:, f[0]] = g[:, 3]; q[:, f[1]] = g[:, 3]; q[:, f[3]] = 0; q[:, f[4]] = 0
q = torch.minimum(torch.maximum(q, u.lower), u.upper)
orig = u._apply_action
def ap():
    u.targets[:] = q
    orig()
u._apply_action = ap
zd = torch.zeros(N, dtype=torch.bool, device=dev); u._get_dones = lambda: (zd, zd)
parked = u.object.data.root_state_w[:, :7].clone(); parked[:, 2] -= 0.5
zero = torch.zeros_like(u.actions)
with torch.inference_mode():
    for _ in range(3):
        R.write_joint_state_to_sim(q, torch.zeros_like(q)); u.object.write_root_pose_to_sim(parked); raw.step(zero)
    B = R.body_names; P = lambda n: R.data.body_pos_w[:, B.index(n)]
    X = torch.tensor([1.0, 0, 0], device=dev).expand(N, 3)
    Nn = lambda n: quat_apply(R.data.body_quat_w[:, B.index(n)], X)
    mid = lambda a, b: 0.5 * (P(a) + P(b))
    p1, p2 = mid("Finger_1_1", "Finger_1_2"), mid("Finger_2_1", "Finger_2_2")
    n1, n2 = Nn("Finger_1_1"), Nn("Finger_2_1")
    nf = (n1 + n2); nf = nf / nf.norm(dim=1, keepdim=True); pf = 0.5 * (p1 + p2)
    same = (n1 * n2).sum(1)
    rows = []
    for tk, (pt, nt) in {"thumb1": (mid("Finger_3_1", "Finger_3_2"), Nn("Finger_3_1")),
                         "thumb2": (P("Finger_3_2") + 0.5 * (P("Finger_3_2") - P("Finger_3_1")), Nn("Finger_3_2"))}.items():
        oppose = (nt * nf).sum(1)
        rel = pt - pf; gap = (rel * nf).sum(1); lateral = torch.linalg.vector_norm(rel - gap[:, None] * nf, dim=1)
        ang = torch.rad2deg(torch.acos((-oppose).clamp(-1, 1)))
        for lab, m in (("any", same > 0.95), ("gap>0.04", (same > 0.95) & (gap > 0.04)), ("gap>0.055", (same > 0.95) & (gap > 0.055))):
            a = torch.where(m, ang, torch.full_like(ang, 999)); i = int(torch.argmin(a))
            print("MIN " + json.dumps({"thumb_link": tk, "filter": lab, "cfg": grid[i], "thumb_angle_deg": round(float(a[i]), 1), "gap_mm": round(1000 * float(gap[i]), 1), "lateral_mm": round(1000 * float(lateral[i]), 1)}), flush=True)
        ok = (same > 0.95) & (gap > 0.06) & (gap < 0.09) & (lateral < 0.02)
        for i in torch.nonzero(ok).flatten().tolist():
            rows.append({"thumb_link": tk, "rot": grid[i][0], "th1": grid[i][1], "th2": grid[i][2], "fq": grid[i][3],
                         "thumb_angle_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, -float(oppose[i]))))), 1),
                         "gap_mm": round(1000 * float(gap[i]), 1), "lateral_mm": round(1000 * float(lateral[i]), 1)})
    rows.sort(key=lambda r: (r["thumb_angle_deg"], r["lateral_mm"]))
    print("COUNT", len(rows), flush=True)
    for r in rows[:20]:
        print("BEST " + json.dumps(r), flush=True)
    cur = grid.index((-150, -30, 0, 0))
    print("CURRENT " + json.dumps([r for r in rows if (r["rot"], r["th1"], r["th2"], r["fq"]) == (-150, -30, 0, 0)]), flush=True)
app.close()
