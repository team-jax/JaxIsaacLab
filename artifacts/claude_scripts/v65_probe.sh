#!/bin/bash
# v65: the baked turned approach in physics (no policy, no rendering), 1.2 m table: (a) the final pre-grasp alone (does it overlap
# a turned cube?), (b) the reach-in from the baked raise end to the baked final pose; positions inside the reach map.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
COMMON="GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0"
for row in "2|-28 -20 -10 0 10 20 28" "-6|-20 -10 0 10 20" "-14|-12 -6 0 6"; do
  X=${row%%|*}; Y=${row#*|}
  env $COMMON GL_GRID_SET=v65f PROBE_XS="$X" PROBE_YS="$Y" PROBE_START_DX=0 PROBE_ROLL_AT_START=1 timeout -k 20 300 ./isaaclab.sh -p artifacts/claude_scripts/v63_reach_probe.py > $SP/v65_final_x$X.out 2>&1
  echo "## final pose x $X"; grep -E "^SPOT|Traceback" $SP/v65_final_x$X.out | sed -E 's/shift_mm max [0-9.]+ mean [0-9.]+ //'
  env $COMMON GL_GRID_SET=v65r PROBE_FINAL_GRID=v65f PROBE_XS="$X" PROBE_YS="$Y" PROBE_START_DX=0 timeout -k 20 300 ./isaaclab.sh -p artifacts/claude_scripts/v63_reach_probe.py > $SP/v65_reach_x$X.out 2>&1
  echo "## reach-in x $X"; grep -E "^SPOT|Traceback" $SP/v65_reach_x$X.out
done
echo V65_PROBE_DONE
