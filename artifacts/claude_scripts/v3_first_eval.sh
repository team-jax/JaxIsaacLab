#!/bin/bash
# v3 (URDF_1108): first look - 22050 (cube-turn input) and 18125 (deployed) on the V3 robot, nominal, 576 envs over the 12 cells
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
L=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16
EV="GL_ROBOT=v3 GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_REACH_DX=0 GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 EVAL_YAW_LO=1 EVAL_S=12 GL_START_NOISE_DEG=0.75 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0 NE=576"
export EVAL_CELLS="0.02,-0.28;0.02,0.0;0.02,0.28;-0.04,-0.24;-0.04,0.24;-0.08,-0.18;-0.08,0.0;-0.08,0.18;-0.12,-0.14;-0.12,0.12;-0.14,-0.08;-0.14,0.06"
for mm in "22050|2026-11-08_13-33-41_v71_r6/model_22050.pt|GL_OBS_YAW=1" "18125|2026-11-07_20-52-49_v65_r6/model_18125.pt|"; do
  IFS='|' read nm m ox <<< "$mm"
  flock $SP/isaac_eval.lock env $EV $ox timeout -k 20 1500 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $L/$m 0 > $SP/v3first_$nm.log 2>&1
  echo "== V3 $nm"; grep -E "^(START|Z |HELD|ALL|DONE lift|CELL|FAIL|DIAG arrived|SPLIT)|Traceback|Error" $SP/v3first_$nm.log | cut -c1-250 | head -30
done
echo V3_FIRST_DONE
