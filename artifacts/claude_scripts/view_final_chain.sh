#!/bin/bash
# Watch the final chain live in an Isaac Sim window (user 2026-11-11: "let me see the test sim"): 25550 + pre-grasp 0.5 cm lower
# (floor -1 cm) + L-return shoulder-yaw turn, the deploy conditions (servo-emulated arm, motor torque limits, finger/cube friction 2.0,
# cube turned at random +/-45 deg). Hanging arm -> raise -> reach-in -> grasp-lift policy -> L return -> hand-over, about 15 s.
#   bash artifacts/claude_scripts/view_final_chain.sh [dx dy [envs]]     cube offset in m from the reference spot (default -0.06 0.0), envs default 4
#   examples: ... 0.02 -0.28   (far left)   ... -0.14 0.06 (near right)   ... 0.02 0.28 4
# Spots inside the reach: dx -0.14 .. +0.02, dy -0.28 .. +0.28 (see README 10). Close the window or wait for the end.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
DX=${1:--0.06}; DY=${2:-0.0}; NE=${3:-4}
C=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-11_04-57-42_v78d_r9/model_25550.pt
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad; mkdir -p $SP
eval "$(grep -E '^(CLOSE|COMMON)=' artifacts/claude_scripts/v80d_full.sh)"
COMMON=${COMMON//GL_RET_YAW=0,45 /}   # 2026-11-11 evening: the L-return shoulder-yaw turn cancelled (the same L for all)
F=source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/render_front_grasp.py
echo "cube offset dx $DX dy $DY, $NE envs; waiting for the GPU lock if another Isaac job runs..."
flock -o $SP/isaac_eval.lock env $COMMON CHAIN_NO_VIDEO=1 GL_SERVO_ARM=5,5 GL_FINGER_EFFORT=0.177,0.177 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 \
  GL_TABLE_DZ_FIXED=0.0 RAISE_GRID_SET=v65r REACH_FINAL_GRID_SET=v65f CHAIN_GRAV_COMP=1 CHAIN_GRAV_FULL=1 TEST_ARM_GAIN_SCALE=5 RAISE_WIDE=1 \
  GL_WRIST_RANGE=0.4 GL_LIFT_EXTRA=0 \
  ./isaaclab.sh -p $F --grasp_lift --num_envs $NE --steps 660 --camera front --tag view_${DX}_${DY} --raise_cube=$DX,$DY \
  --raise_max_s 11 --raise_sag_comp_deg 0,0,0,0,0,0,0,0 --raise_policy nominal --checkpoint $C 2>&1 | grep -E --line-buffered "LIFT_DONE|lifted|CHAIN |Traceback|Error"
