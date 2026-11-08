#!/bin/bash
# v63: does the wide-box policy keep the original box? 17375 on the original eval grid (x -8/-3/+2, y -5/0/+6 cm), nominal
# and combined case, 576 envs each, during v63b round $1's training (one eval at a time).
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
until grep -q "== ROUND $1" $SP/v63b.out; do sleep 15; done; sleep 100
COMMON="GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v53w GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_CUBE_YAW_DEG=45 EVAL_YAW_LO=1 EVAL_GRID=-0.08,-0.03,0.02;-0.05,0.0,0.06"
M=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-07_10-42-18_v63_r3/model_17375.pt
for c in "nominal GL_MU=0.4,0.35" "harsh GL_ACT_DELAY=2 GL_MU=0.35,0.3 GL_CUBE_MASS=0.05,0.05 GL_KP_SCALE=1.5,1.5 GL_OBS_CUBE_NOISE=0.002,0.005"; do
  set -- $c; n=$1; shift
  out=$(env $COMMON "$@" NE=576 EVAL_S=12 GL_TABLE_DZ_FIXED=0.0 GL_START_NOISE_DEG=0.75 timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $M 0 2>&1)
  echo "OLDGRID 17375 $n $(echo "$out" | grep '^HELD')"; echo "$out" | grep -E "^CELL" | sed 's/^/  /'
done
echo OLDGRID_DONE
