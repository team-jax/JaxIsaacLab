#!/bin/bash
# v65: the 9-spot chain again with the scripted raise path alone (no v25 residual; it pushed the hand into the table at far |y|)
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
L=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16
EV="GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_REACH_DX=0 GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_TABLE_REL=1"
(export $EV GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0 RAISE_GRID_SET=v65r REACH_FINAL_GRID_SET=v65f; RAISE_POL=nominal SKIP_DONE=1 RETL=1 STEPS=660 LIFTIK=1 IKP= IKVZ=0.035 SPOTS="0.02,-0.28 0.02,0.0 0.02,0.28 -0.06,-0.20 -0.06,0.0 -0.06,0.20 -0.14,-0.08 -0.14,0.0 -0.14,0.06" TAGSUF=_v65_reach_nom bash artifacts/claude_scripts/v65_montage.sh $L/$(cat $SP/v65_pick.txt) front 2>&1 | grep -E "lifted|SKIP|MONTAGE|Traceback")
echo CHAIN_NOM_DONE
