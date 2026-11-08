#!/bin/bash
# v64: where does the arm stop reaching on the +y side? 17650 per position at x -14 / -10 / -6 x y +8 / +10 / +12 cm
# (576 envs, nominal, 0.90 m table), after the round-$1 final-pose probe, only while round $1 is still training.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
T=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/tasks/byjl7j1ef.output
L=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16
until grep -q "V64_PROBE3_DONE" $T 2>/dev/null; do sleep 10; done
r=$1
while ! ps -eo args | grep -v grep | grep -q "run_name v64_r$r"; do r=$((r + 1)); until grep -q "== ROUND $r" $SP/v64.out; do sleep 15; done; sleep 100; done
COMMON="GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v53x GL_TABLE_W=0.90 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_CUBE_YAW_DEG=45 EVAL_YAW_LO=1 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0 EVAL_S=12 GL_START_NOISE_DEG=0.75 NE=576 EVAL_GRID=-0.14,-0.10,-0.06;0.08,0.10,0.12"
out=$(env $COMMON timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $L/2026-11-07_14-34-23_v63b_r10/model_17650.pt 0 2>&1)
echo "REACHCELLS 17650 (round $r window) $(echo "$out" | grep '^HELD' | awk '{print $NF}')"; echo "$out" | grep -E "^CELL|Traceback" | sed 's/^/  /'
echo V64_REACHCELLS_DONE
