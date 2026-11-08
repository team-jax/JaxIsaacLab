#!/bin/bash
# v65 (user 2026-11-07: "cover the whole range of motion"): exact-IK raw grids over +/-30 cm (31x31, every knot) at the CAD
# table height and +/-1 cm, with the last-knot hand heading, residuals and limit flags (v65_grid_probe.py).
cd /home/jeongwoo/isaacsim/JaxIsaacLab
for spec in "base -0.0575" "m1 -0.0675" "p1 -0.0475"; do
  set -- $spec
  GRID_BX=0.115 timeout -k 20 3000 ./isaaclab.sh -p artifacts/claude_scripts/v65_grid_probe.py save 0.30 2 v65_$1_raw.pt $2 2>&1 | grep -E "^WP|^SAVED|Traceback" | tr '\n' ' '; echo " <- $1"
done
echo V65_RAW_DONE
