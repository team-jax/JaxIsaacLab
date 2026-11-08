#!/bin/bash
# v63 (user 2026-11-06: "widen only the cube placement range"): the v53 pre-grasp grids (CAD layout, exact IK, x ramp)
# rebuilt over +/-20 cm (21x21) instead of +/-12 cm, same recipe: GRID_BX=0.115 v41_grid_probe.py save R 2 raw DZ ->
# v49_regrid.py. Sets v53w_{base,m3,p3}c.pt (GL_GRID_SET=v53w). Inside +/-12 cm they should match v53.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
for spec in "base -0.0575" "m3 -0.0875" "p3 -0.0275"; do
  set -- $spec
  echo "## grid $1 DZ=$2"
  GRID_BX=0.115 timeout -k 20 3000 ./isaaclab.sh -p artifacts/claude_scripts/v41_grid_probe.py save 0.20 2 v53w_$1_raw.pt $2 2>&1 | grep -vE "^\[|^$|Warning|warn" | tail -60
  ./_isaac_sim/python.sh artifacts/claude_scripts/v49_regrid.py v53w_$1_raw.pt v53w_$1c.pt 2>&1 | grep -E "SAVED|Traceback|Error"
done
echo V63_GRIDS_DONE
