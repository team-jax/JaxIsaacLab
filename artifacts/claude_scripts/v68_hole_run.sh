#!/bin/bash
# v68: hole diagnosis at the (-8, 0) cell (hand heading ~0): fixed cube turns -35 (hole) and -30 (fine), 64 envs, 18525
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
LOCK=$SP/isaac_eval.lock
CK=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-07_23-50-53_v66_r4/model_18525.pt
COMMON="GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=0 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0 GL_START_NOISE_DEG=0.75 NE=64 EVAL_S=12 EVAL_CELLS=-0.08,0.0"
for y in -35 -30; do
  flock $LOCK env $COMMON GL_CUBE_YAW_FIXED_DEG=$y timeout -k 20 600 ./isaaclab.sh -p artifacts/claude_scripts/v68_hole_diag.py $CK $SP/hole_y$y.pt 2>&1 | grep -E "HOLE_DIAG|Traceback|Error" | head -5
done
./_isaac_sim/python.sh artifacts/claude_scripts/v68_hole_ana.py $SP/hole_y-35.pt $SP/hole_y-30.pt 2>&1 | grep -v "^\[" | head -40
echo V68_HOLE_DONE
