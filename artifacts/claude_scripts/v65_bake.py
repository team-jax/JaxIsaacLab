"""v65 (user 2026-11-07: "cover the whole range of motion"): bake the turned-approach grids, looked up by the cube offset itself.

The heading-free pre-grasp IK turns the hand with the cube position (x -2 cm: -97 deg at y 0, -76 / -118 deg at y -12 / +12).
The cube is a cube turned U(-45, 45) deg, so a turned hand only sees another cube turn it was trained on; what broke the grasp was
that the hand still met the cube at the same world offset and moved in along world -x, so fingers 1-2 (y < 0) or the thumb
(y > 0) swept the cube. Here the whole hand-cube relation turns with the hand: with theta = heading drift of the heading-free
pre-grasp (fixed point, 2 iterations) and v_f = fingertip midpoint - cube centre at the reference pre-grasp (pre-roll, measured
(0.024, 0.019) m),
  final lookup     L_f = (dx - 0.025, dy) + (R(theta) - I) v_f
  raise-end lookup L_r = (dx + 0.020, dy) + (R(theta) - I) (v_f + (0.045, 0))      (2 cm back-off + 2.5 cm reach, turned)
on the +/-30 cm raw grids v65_{base,m1,p1}_raw.pt (exact IK per knot, heading free). Raise knot k uses (a_k L_r.x, L_r.y) as
v49_regrid (x ramp, a_k = 0 at 5.0 s -> 1 at 8.0 s). Valid = raw residual < 1 mm and no joint limit at every grid node used.
Saves data/v65f_{basec,m3c,p3c}.pt (final pose in every knot slot), data/v65r_{basec,m3c,p3c}.pt (raise path) over cube
offsets x -0.20..+0.04, y -0.30..+0.30 (2 cm), m3 = base + 3 (m1 - base), p3 = base + 3 (p1 - base), and data/v65_valid.pt."""
import math
import torch

D = "source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/data/"
VF = torch.tensor([0.024, 0.019])
import os
BACK = float(os.environ.get("BAKE_BACKOFF", "0.045"))  # v66: raise end this far behind the final pose along the turned approach
TAG = os.environ.get("BAKE_TAG", "")  # v66: suffix of the output sets (v65r<TAG>_*, v65f<TAG>_*, v65<TAG>_valid)
raw = {k: torch.load(D + f"v65_{k}_raw.pt") for k in ("base", "m1", "p1")}
b = raw["base"]
ts, gx, gy = b["ts"], b["gx"], b["gy"]
nk = len(ts)
HEAD = b["heading_deg"]


def bil(T, x, y):
    """bilinear on a (nx, ny, ...) table at (x, y) m, clamped; also returns the 4 node indices"""
    fx = float(((x - gx[0]) / (gx[1] - gx[0])).clamp(0, len(gx) - 1)); i = min(int(math.floor(fx)), len(gx) - 2); a = fx - i
    fy = float(((y - gy[0]) / (gy[1] - gy[0])).clamp(0, len(gy) - 1)); j = min(int(math.floor(fy)), len(gy) - 2); c = fy - j
    v = (1 - a) * (1 - c) * T[i, j] + a * (1 - c) * T[i + 1, j] + (1 - a) * c * T[i, j + 1] + a * c * T[i + 1, j + 1]
    return v, [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]


def rot(th, v):
    c, s = math.cos(th), math.sin(th)
    return torch.tensor([c * v[0] - s * v[1], s * v[0] + c * v[1]])


h0 = float(bil(HEAD, torch.tensor(-0.025), torch.tensor(0.0))[0])
cx = torch.arange(-0.20, 0.04 + 1e-6, 0.02); cy = torch.arange(-0.30, 0.30 + 1e-6, 0.02)
out = {k: {"f": torch.zeros(nk, len(cx), len(cy), 8), "r": torch.zeros(nk, len(cx), len(cy), 8)} for k in raw}
valid = torch.zeros(len(cx), len(cy), dtype=torch.bool); theta = torch.zeros(len(cx), len(cy)); THF = torch.zeros(len(cx), len(cy))
LF = torch.zeros(len(cx), len(cy), 2); LR = torch.zeros(len(cx), len(cy), 2)
for i, dx in enumerate(cx.tolist()):
    for j, dy in enumerate(cy.tolist()):
        th = 0.0
        for _ in range(3):
            th_f = th  # the turn the final lookup below was made with
            lf = torch.tensor([dx - 0.025, dy]) + rot(th, VF) - VF
            hd = float(bil(HEAD, lf[0], lf[1])[0]) - h0
            th = math.radians((hd + 180.0) % 360.0 - 180.0)
        lr = torch.tensor([dx - 0.025 + BACK, dy]) + rot(th, VF + torch.tensor([BACK, 0.0])) - (VF + torch.tensor([BACK, 0.0]))
        theta[i, j] = math.degrees(th); THF[i, j] = math.degrees(th_f); LF[i, j] = lf; LR[i, j] = lr
        ok = True
        for kk in raw:
            G = raw[kk]["dq"]
            v, nodes = bil(G[-1], lf[0], lf[1])
            out[kk]["f"][:, i, j] = v
            for k in range(nk):
                ak = float((ts[k] - ts[0]) / (ts[-1] - ts[0]))
                vr, nr_ = bil(G[k], ak * lr[0], lr[1])
                out[kk]["r"][k, i, j] = vr
                if kk == "base":
                    for (a_, c_) in nr_:
                        ok &= bool(b["resid_mm"][k, a_, c_] < 1.0) and not bool(b["at_limit"][k, a_, c_])
            if kk == "base":
                for (a_, c_) in nodes:
                    ok &= bool(b["resid_mm"][-1, a_, c_] < 1.0) and not bool(b["at_limit"][-1, a_, c_])
        valid[i, j] = ok
for kind in ("f", "r"):
    base = out["base"][kind]
    sets = {"basec": base, "m3c": base + 3.0 * (out["m1"][kind] - base), "p3c": base + 3.0 * (out["p1"][kind] - base)}
    for name, dq in sets.items():
        torch.save({"ts": ts, "gx": cx, "gy": cy, "dq": dq}, D + f"v65{kind}{TAG}_{name}.pt")
# v66: reach-in path grid v65i<TAG>: 6 knots on the straight turned approach from the raise end (back-off BACK) to the final
# pose (knot k at back-off BACK (1 - k / 5)), last knot of the raw grids; the reach-in follows them (near-straight fingertip path)
for kk in raw:
    G = raw[kk]["dq"]
    path = torch.zeros(nk, len(cx), len(cy), 8)
    for i, dx in enumerate(cx.tolist()):
        for j, dy in enumerate(cy.tolist()):
            th_r = math.radians(float(theta[i, j])); th_f = math.radians(float(THF[i, j]))  # raise end / final pose turns
            for k in range(nk):
                bk = BACK * (1.0 - k / (nk - 1)); th = th_r + (th_f - th_r) * k / (nk - 1)
                lk = torch.tensor([dx - 0.025 + bk, dy]) + rot(th, VF + torch.tensor([bk, 0.0])) - (VF + torch.tensor([bk, 0.0]))
                path[k, i, j] = bil(G[-1], lk[0], lk[1])[0]
    out[kk]["i"] = path
basei = out["base"]["i"]
for name, dq in {"basec": basei, "m3c": basei + 3.0 * (out["m1"]["i"] - basei), "p3c": basei + 3.0 * (out["p1"]["i"] - basei)}.items():
    torch.save({"ts": ts, "gx": cx, "gy": cy, "dq": dq}, D + f"v65i{TAG}_{name}.pt")
torch.save({"gx": cx, "gy": cy, "valid": valid, "theta_deg": theta, "lookup_final": LF, "lookup_raise_end": LR, "h0_deg": h0, "v_f": VF}, D + f"v65{TAG}_valid.pt")
print("h0 %.1f deg; valid %d / %d cube offsets" % (h0, int(valid.sum()), valid.numel()))
print("cols y (cm): " + " ".join("%d" % round(v * 100) for v in cy.tolist()))
for i in reversed(range(len(cx))):
    print("x %+4d  " % round(float(cx[i]) * 100) + "".join(("." if valid[i, j] else "x") for j in range(len(cy))))
for (x, y) in [(0.0, 0.0), (0.0, -0.10), (0.0, 0.12), (-0.14, 0.10), (-0.14, -0.10), (0.0, 0.28), (0.0, -0.28)]:
    i = int(round((x - float(cx[0])) / 0.02)); j = int(round((y - float(cy[0])) / 0.02))
    print("cube (%+.2f, %+.2f): theta %+.1f deg, final lookup (%+.3f, %+.3f), raise-end lookup (%+.3f, %+.3f), valid %s" % (
        x, y, float(theta[i, j]), float(LF[i, j, 0]), float(LF[i, j, 1]), float(LR[i, j, 0]), float(LR[i, j, 1]), bool(valid[i, j])))
