#!/bin/bash
# v63y: as v63_grids.sh but with the hand yaw fixed too (NR=3), so the hand meets the cube at the same heading everywhere
# (the yaw-free v53w hand turned +14 deg at y -10 and its fingers 1-2 pushed turned cubes 15-24 mm in the reach-in).
# v63 (user 2026-11-06: "widen only the cube placement range"): the v53 pre-grasp grids (CAD layout, exact IK, x ramp)
# rebuilt over +/-20 cm (21x21) instead of +/-12 cm, same recipe: GRID_BX=0.115 v41_grid_probe.py save R 2 raw DZ ->
# v49_regrid.py. Sets v53wy_{base,m3,p3}c.pt (GL_GRID_SET=v53wy); differs from v53 wherever the yaw-free hand turned.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
for spec in "base -0.0575" "m3 -0.0875" "p3 -0.0275"; do
  set -- $spec
  echo "## grid $1 DZ=$2"
  GRID_BX=0.115 timeout -k 20 3000 ./isaaclab.sh -p artifacts/claude_scripts/v41_grid_probe.py save 0.20 3 v53wy_$1_raw.pt $2 > /tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad/v63y_$1_probe.log 2>&1; grep -E "^WP|^SAVED|Traceback" /tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad/v63y_$1_probe.log
  ./_isaac_sim/python.sh artifacts/claude_scripts/v49_regrid.py v53wy_$1_raw.pt v53wy_$1c.pt 2>&1 | grep -E "SAVED|Traceback|Error"
done
echo V63Y_GRIDS_DONE
