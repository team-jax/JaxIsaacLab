#!/bin/bash
# v69c: headless 64-env chains with the cube-turn policy 20875 (GL_OBS_YAW=1) and the fixed gravity compensation, at the
# spots where 18525 (comp) gave 62 / 61 / 63 / 64 / 63 of 64. Holds the eval lock.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
LOCK=$SP/isaac_eval.lock
F=source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/render_front_grasp.py
C=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-08_07-57-41_v69c_r6/model_20875.pt
CH="CHAIN_NO_VIDEO=1 GL_OBS_YAW=1 GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_REACH_DX=0 GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_TABLE_REL=1 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0 RAISE_GRID_SET=v65r REACH_FINAL_GRID_SET=v65f GL_RETURN_L=1 GL_LIFT_IK=1 GL_IK_VZ=0.035 RAISE_WIDE=1 GL_WRIST_RANGE=0.4 CHAIN_GRAV_COMP=1"
for spot in -0.06,0.0 0.02,0.0 0.02,-0.28 0.02,0.28 -0.14,0.0; do
  tag=v69c64_${spot/,/_}
  flock $LOCK env $CH timeout -k 20 600 ./isaaclab.sh -p $F --headless --grasp_lift --num_envs 64 --steps 660 --camera front --tag $tag --raise_cube=$spot --raise_max_s 11 --raise_sag_comp_deg 0,0,0,0,0,0,0,0 --raise_policy nominal --checkpoint $C > $SP/$tag.log 2>&1
  J=$(ls -d $(dirname $C)/videos/*_$tag 2>/dev/null | tail -1)/chain.json
  echo "C69 $spot $(python3 -c "import json;d=json.load(open('$J'));print('lifted',d['lifted_4cm_frac'],'done',d.get('lift_done_frac'),'pad_N',d['raise_hand_contact_max_n'],'moved',d['cube_moved_mm'],'table_N',d['episode_table_force_max_n'])" 2>&1 | tail -1)"
  grep -E "Traceback|size mismatch" -A 3 $SP/$tag.log | head -6
done
echo V69_CHAIN_DONE
