#!/bin/bash
# v63: during round $1's training (one eval-type Isaac process, no rendering): reach-in probe on v53w with link positions
# and the start-pose push, then sideways starts (+/-1 cm; + = fingers' side, - = thumb side) moving in diagonally.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
COMMON="GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v53w GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0"
until grep -q "== ROUND $1" $SP/v63.out; do sleep 15; done; sleep 100
for v in "base PROBE_START_DY=0" "dyp1 PROBE_START_DY=0.01" "dym1 PROBE_START_DY=-0.01"; do
  set -- $v; name=$1; shift
  env $COMMON PROBE_YS="-2 -10" "$@" timeout -k 20 300 ./isaaclab.sh -p artifacts/claude_scripts/v63_reach_probe.py > $SP/reach3_$name.out 2>&1
  echo "## $name: $(grep -E '^PROBE' $SP/reach3_$name.out)"; grep -E "^SPOT|^ALL|Traceback" $SP/reach3_$name.out
done
echo PROBE_QUEUE3_DONE
