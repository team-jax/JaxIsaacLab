# v49: v34_regrid with argv[1] = raw grid file, argv[2] = output file (both in data/)
"""v34: same x ramp for the exact-IK grid at a 3 cm lower table (z applied at every knot). v27c: the x (toward/away from the robot) part of the cube shift only after the hand is above the
table: knot k uses the exact-IK shift at (a_k dx, dy), a_k = 0 at the first knot (5.0 s, hand in front
of the table edge; the rise from hanging before it moves only sideways) -> 1 at the pre-grasp (8.0 s).
v27 shifted the whole rise by dx, so for a far cube the hand rose into the table edge (285 N)."""
import torch, sys
sys.path.insert(0, "source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal")
d = torch.load("source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/data/" + sys.argv[1])
ts, gx, gy, G = d["ts"], d["gx"], d["gy"], d["dq"]
def bil(Gk, x, y):
    fx = ((x - gx[0]) / (gx[1] - gx[0])).clamp(0, len(gx) - 1); i = int(fx.floor().clamp(0, len(gx) - 2)); a = fx - i
    fy = ((y - gy[0]) / (gy[1] - gy[0])).clamp(0, len(gy) - 1); j = int(fy.floor().clamp(0, len(gy) - 2)); b = fy - j
    return (1 - a) * (1 - b) * Gk[i, j] + a * (1 - b) * Gk[i + 1, j] + (1 - a) * b * Gk[i, j + 1] + a * b * Gk[i + 1, j + 1]
out = torch.zeros_like(G)
for k in range(len(ts)):
    ak = float((ts[k] - ts[0]) / (ts[-1] - ts[0]))
    for i in range(len(gx)):
        for j in range(len(gy)):
            out[k, i, j] = bil(G[k], ak * gx[i], gy[j])
    print("knot", float(ts[k]), "a", round(ak, 2))
torch.save({"ts": ts, "gx": gx, "gy": gy, "dq": out}, "source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/data/" + sys.argv[2])
print("SAVED", float((out[-1] - G[-1]).abs().max()))
