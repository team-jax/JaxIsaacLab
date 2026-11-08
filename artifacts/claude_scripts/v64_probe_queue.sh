#!/bin/bash
# v64: during round $1's training (one eval-type Isaac process, no rendering): reach-in probe on the +y rows (6, 10, 12 cm)
# with the 0.90 m table, start offsets 0 / -1 / -2 cm (toward the thumb side, -y): at +y the hand turns the other way and
# the thumb side met turned cubes (y +10: 6-24 mm, y +14: 16-18 mm with no offset, 1.0 m table).
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
COMMON="GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v53x GL_TABLE_W=0.90 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0"
until grep -q "== ROUND $1" $SP/v64.out; do sleep 15; done; sleep 100
for sd in 0 -0.01 -0.02; do
  env $COMMON PROBE_YS="6 10 12" PROBE_START_DY=$sd timeout -k 20 300 ./isaaclab.sh -p artifacts/claude_scripts/v63_reach_probe.py > $SP/v64_reach_sd$sd.out 2>&1
  echo "## start_dy $sd: $(grep -E '^PROBE' $SP/v64_reach_sd$sd.out | cut -c1-60)"; grep -E "^SPOT|^ALL|Traceback" $SP/v64_reach_sd$sd.out
done
echo V64_PROBE_DONE
