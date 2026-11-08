#!/bin/bash
# v69: with the fixed gravity compensation the outer +y chains still push the cube 10-13 mm before the grasp ((+2,+28): 12.9 mm
# with 20875, 9.5 with 18525). Which link, when? Headless 64-env chains with CHAIN_CUBE_DIAG at (+2,+28), (+2,+20), (-6,+20).
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
LOCK=$SP/isaac_eval.lock
F=source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/render_front_grasp.py
C=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-08_07-57-41_v69c_r6/model_20875.pt
CH="CHAIN_NO_VIDEO=1 CHAIN_CUBE_DIAG=1 GL_OBS_YAW=1 GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_REACH_DX=0 GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_TABLE_REL=1 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0 RAISE_GRID_SET=v65r REACH_FINAL_GRID_SET=v65f GL_RETURN_L=1 GL_LIFT_IK=1 GL_IK_VZ=0.035 RAISE_WIDE=1 GL_WRIST_RANGE=0.4 CHAIN_GRAV_COMP=1"
for spot in 0.02,0.28 0.02,0.20 -0.06,0.20; do
  tag=v69push_${spot/,/_}
  flock $LOCK env $CH timeout -k 20 600 ./isaaclab.sh -p $F --headless --grasp_lift --num_envs 64 --steps 660 --camera front --tag $tag --raise_cube=$spot --raise_max_s 11 --raise_sag_comp_deg 0,0,0,0,0,0,0,0 --raise_policy nominal --table_diag --checkpoint $C > $SP/$tag.log 2>&1
  echo "## $spot"; grep -E "^CUBE_DIAG|^RAISE_TABLE_DIAG|Traceback" $SP/$tag.log | cut -c1-500
  J=$(ls -d $(dirname $C)/videos/*_$tag 2>/dev/null | tail -1)/chain.json
  echo "PUSH $spot $(python3 -c "import json;d=json.load(open('$J'));print('done',d.get('lift_done_frac'),'pad_N',d['raise_hand_contact_max_n'],'moved',d['cube_moved_mm'])" 2>&1 | tail -1)"
done
echo V69_PUSH_DONE
