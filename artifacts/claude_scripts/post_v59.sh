#!/bin/bash
# after v59: chain render of the v59 best (50 mm cube, 40 g, turned U(-45, 45) deg per env; GPU free, no training),
# then v60 training; the 1152-env checks and the 288-env sweep of the v59 best run in v60's training windows (post_v59_eval.sh)
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
until grep -q "DONE best" $SP/v59.out; do sleep 10; done
sleep 10
read S0 BEST < artifacts/claude_scripts/v59_best.txt
echo "V59_DONE best $BEST score $S0"
echo "## chain render $BEST (50 mm, turn +/-45)"
(export GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_CUBE_YAW_DEG=45 GL_GRID_SET=v53 GL_TABLE_Z0=-0.0575 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_IK_POS=2 GL_GRAV_COMP=1 GL_TABLE_EDGE=0.225 GL_MU=0.4,0.35 GL_SQUEEZE=6,0 GL_TABLE_DZ_FIXED=0.0 GL_TABLE_REL=1; RETL=1 STEPS=660 LIFTIK=1 IKP= IKVZ=0.035 TAGSUF=_c50_yaw45 bash artifacts/claude_scripts/v27_montage.sh logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/$BEST front)
echo CHAIN_C50_DONE
nohup bash $SP/post_v59_eval.sh > $SP/post_v59_eval.out 2>&1 &
bash artifacts/claude_scripts/v60_full.sh
