#!/bin/bash
# v64: per-position nominal results (576 envs, new eval grid y -10 / +1 / +12 cm, 0.90 m table) of 17650 and the current v64
# best during round $1's training, to see where the widened box fails.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
L=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16
until grep -q "== ROUND $1" $SP/v64.out; do sleep 15; done; sleep 100
COMMON="GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v53x GL_TABLE_W=0.90 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_CUBE_YAW_DEG=45 EVAL_YAW_LO=1 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0 EVAL_S=12 GL_START_NOISE_DEG=0.75 NE=576 EVAL_GRID=-0.14,-0.06,0.02;-0.10,0.01,0.12"
read _s P < artifacts/claude_scripts/v64_best.txt
for m in 2026-11-07_14-34-23_v63b_r10/model_17650.pt $P; do
  out=$(env $COMMON timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $L/$m 0 2>&1)
  echo "CELLS $m $(echo "$out" | grep '^HELD' | awk '{print $NF}')"; echo "$out" | grep -E "^CELL|^DROP n|Traceback" | sed 's/^/  /'
done
echo V64_CELLS_DONE
