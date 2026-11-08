#!/bin/bash
# v3 video (user 2026-11-08 "stop, render a video"): the full chain on the V3 robot (URDF_1108) at the 9 reach-map spots,
# policy 22050 (cube-turn input), fixed gravity compensation, table 0.735 m / shoulder 1.2125 m, 8 envs per spot, 3x3 montage.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
L=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16
export GL_ROBOT=v3 GL_OBS_YAW=1 GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_REACH_DX=0 GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_TABLE_REL=1 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=-0.005 RAISE_GRID_SET=v65r REACH_FINAL_GRID_SET=v65f CHAIN_GRAV_COMP=1
RAISE_POL=nominal RETL=1 STEPS=660 LIFTIK=1 IKP= IKVZ=0.035 NE=8 SPOTS="0.02,-0.28 0.02,0.0 0.02,0.28 -0.06,-0.20 -0.06,0.0 -0.06,0.20 -0.14,-0.08 -0.14,0.0 -0.14,0.06" TAGSUF=_v3_urdf1108 bash artifacts/claude_scripts/v3_montage.sh $L/2026-11-08_13-33-41_v71_r6/model_22050.pt front 2>&1 | grep -E "lifted|SKIP|MONTAGE|Traceback"
echo V3_VIDEO_DONE
