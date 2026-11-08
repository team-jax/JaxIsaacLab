#!/bin/bash
# deploy V3 final videos: the full chain on the V3 robot (URDF_1108), policy v72_r5/model_23025 (cube-turn input), chain gravity
# compensation, table 0.735 m / shoulder 1.2125 m (URDF): (1) 9 reach-map spots x 8 envs, front camera, 3x3 montage;
# (2) close front view at (-6, 0), 4 envs.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
L=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16
C=$L/2026-11-08_16-21-06_v72_r5/model_23025.pt
export GL_ROBOT=v3 GL_OBS_YAW=1 GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_REACH_DX=0 GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_TABLE_REL=1 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=-0.005 RAISE_GRID_SET=v65r REACH_FINAL_GRID_SET=v65f CHAIN_GRAV_COMP=1
RAISE_POL=nominal RETL=1 STEPS=660 LIFTIK=1 IKP= IKVZ=0.035 NE=8 SPOTS="0.02,-0.28 0.02,0.0 0.02,0.28 -0.06,-0.20 -0.06,0.0 -0.06,0.20 -0.14,-0.08 -0.14,0.0 -0.14,0.06" TAGSUF=_v3_final bash artifacts/claude_scripts/v3_montage.sh $C front 2>&1 | grep -E "lifted|SKIP|MONTAGE|Traceback"
F=source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/render_front_grasp.py
GL_RETURN_L=1 GL_LIFT_IK=1 GL_IK_VZ=0.035 RAISE_WIDE=1 GL_WRIST_RANGE=0.4 timeout -k 20 900 ./isaaclab.sh -p $F --headless --grasp_lift --num_envs 4 --steps 660 --camera close_front --tag v3_final_close --raise_cube=-0.06,0.0 --raise_max_s 11 --raise_sag_comp_deg 0,0,0,0,0,0,0,0 --raise_policy nominal --checkpoint $C > /tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad/v3_final_close.log 2>&1
echo "CLOSE $(ls -d $(dirname $C)/videos/*v3_final_close* | tail -1)"
echo V3_FINAL_VIDEO_DONE
