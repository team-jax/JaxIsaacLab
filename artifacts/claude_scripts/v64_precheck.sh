#!/bin/bash
# v64 (user 2026-11-07: "widen the table and train"): before training, (1) v53x-recipe grid accuracy over the +y extension
# (y +8..+22 cm) at the CAD height and +/-1 cm, (2) the reach-in probe with a 1.0 m table on the y +6 / +10 / +14 cm rows.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
for z in -0.0575 -0.0675 -0.0475; do
  echo "## grid DZ $z"
  GRID_BX=0.115 timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v64_grid_check_y.py check 0.20 2 none.pt $z 2>&1 | grep -E "^BAD|^WP|Traceback" > $SP/v64_grid_$z.out
  grep -E "^WP" $SP/v64_grid_$z.out | tr '\n' ' '; echo; echo "bad points: $(grep -c '^BAD' $SP/v64_grid_$z.out)"
done
COMMON="GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v53x GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0 GL_TABLE_W=1.0"
env $COMMON PROBE_YS="6 10 14" timeout -k 20 400 ./isaaclab.sh -p artifacts/claude_scripts/v63_reach_probe.py > $SP/v64_reach_y.out 2>&1
echo "## reach-in probe, table 1.0 m: $(grep -E '^PROBE' $SP/v64_reach_y.out)"; grep -E "^SPOT|^ALL|Traceback" $SP/v64_reach_y.out
echo PRECHECK_DONE
