#!/bin/bash
# v49y: the +/-20 cm pre-grasp grids with the hand yaw fixed too (NR=3); the v49 (yaw-free, as v41) grids turned the
# hand at y +10 cm and v49b 12900 held 50-72% there (97-100% elsewhere). Then a wide-box eval of 12900 on them.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
for z in 0.0 -0.03 0.03; do
  n=$(python3 -c "print({'0.0':'v49y_base','-0.03':'v49y_m3','0.03':'v49y_p3'}['$z'])")
  echo "## grid $n"
  GRID_BX=0.115 timeout -k 20 2400 ./isaaclab.sh -p artifacts/claude_scripts/v41_grid_probe.py save 0.20 3 ${n}_raw.pt $z 2>&1 | grep -E "^WP|SAVED|Error|Traceback"
  ./_isaac_sim/python.sh artifacts/claude_scripts/v49_regrid.py ${n}_raw.pt ${n}c.pt 2>&1 | grep -E "SAVED|Error|Traceback"
done
echo "## eval 12900 on v49y grids"
GL_GRID_SET=v49y GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 ROLL=12 NE=288 EVAL_GRID="-0.14,-0.06,0.02;-0.10,0.0,0.10" bash artifacts/claude_scripts/v46_big_cmp.sh 2026-11-05_13-56-36_v49b_r1/model_12900.pt
echo V49Y_DONE
