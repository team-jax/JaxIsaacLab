#!/bin/bash
# v63x: the v53w -3 cm grid cannot reach the far row (lookup x <= -14 cm: arm short, 2-55 mm), and the 1/3 of it blended
# at a 1 cm lower table broke the start pose there (17650 at table -1 cm: x -14 row 34-94%). Exact-IK grids at -1 / +1 cm
# (DZ -0.0675 / -0.0475) instead, stretched x3 into the -3 / +3 cm files: base + 3 (g1 - base). The blend formula
# (README 7.1) is unchanged and exact within the +/-1 cm the policy was trained on. Set v53x = {v53w base, m3x, p3x}.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
for spec in "m1 -0.0675" "p1 -0.0475"; do
  set -- $spec
  GRID_BX=0.115 timeout -k 20 1800 ./isaaclab.sh -p artifacts/claude_scripts/v41_grid_probe.py save 0.20 2 v53w_$1_raw.pt $2 > $SP/v63x_$1_probe.log 2>&1
  echo "## $1: $(grep -E '^WP|^SAVED|Traceback' $SP/v63x_$1_probe.log | tr '\n' ' ')"
  ./_isaac_sim/python.sh artifacts/claude_scripts/v49_regrid.py v53w_$1_raw.pt v53w_$1c.pt 2>&1 | grep -E "SAVED|Traceback|Error"
done
./_isaac_sim/python.sh - <<'PY' 2>&1 | grep -vE "^\[|Warning"
import torch
D = "source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/data/"
b = torch.load(D + "v53w_basec.pt")
for k, src in (("m3", "m1"), ("p3", "p1")):
    g = torch.load(D + f"v53w_{src}c.pt")
    out = dict(b); out["dq"] = b["dq"] + 3.0 * (g["dq"] - b["dq"])
    torch.save(out, D + f"v53x_{k}c.pt")
    old = torch.load(D + f"v53w_{k}c.pt")["dq"]
    print(k, "saved; max |new - old| deg over the whole grid", round(float((out["dq"] - old).abs().max()) * 57.2958, 2))
torch.save(b, D + "v53x_basec.pt"); print("base copied")
PY
echo V63X_GRIDS_DONE
