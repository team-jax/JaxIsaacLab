#!/bin/bash
# v63: during round $1's training (one eval-type Isaac process at a time, no rendering): build the yaw-fixed wide grids
# (v63y_grids.sh), then the reach-in probe on them and the 4 cm back-off probe on v53w. Each step is time-limited.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
COMMON="GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0"
until grep -q "== ROUND $1" $SP/v63.out; do sleep 15; done; sleep 100
bash artifacts/claude_scripts/v63y_grids.sh > $SP/v63y_grids.out 2>&1; echo "## grids: $(grep -cE '^SAVED' $SP/v63y_grids.out) saved; $(grep -E '^WP 8.0' $SP/v63y_grids.out | tr '\n' ' ')"
for v in "yawfix GL_GRID_SET=v53wy" "back4 GL_GRID_SET=v53w PROBE_START_DX=0.04"; do
  set -- $v; name=$1; shift
  env $COMMON PROBE_YS="6 -2 -10" "$@" timeout -k 20 300 ./isaaclab.sh -p artifacts/claude_scripts/v63_reach_probe.py > $SP/reach_$name.out 2>&1
  echo "## $name: $(grep -E '^PROBE' $SP/reach_$name.out)"; grep -E "^SPOT|^ALL|Traceback" $SP/reach_$name.out
done
echo PROBE_QUEUE2_DONE
