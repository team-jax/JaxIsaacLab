#!/bin/bash
# v66b: chain at (+2,-28) with the gravity term added after the joint-range clamp (shoulder roll held at its 0 deg stop as in
# the training env's start pose), 256 envs, deployed 18125 - against 84.8% (old clamped comp) and 88.7% (no comp) there.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
LOCK=$SP/isaac_eval.lock
until grep -q "^DONE best" $SP/v66.out; do sleep 10; done
F=source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/render_front_grasp.py
C=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-07_20-52-49_v65_r6/model_18125.pt
tag=v66g256_m28_g1fix
flock $LOCK env CHAIN_NO_VIDEO=1 GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_REACH_DX=0 GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_TABLE_REL=1 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0 RAISE_GRID_SET=v65r REACH_FINAL_GRID_SET=v65f CHAIN_GRAV_COMP=1 GL_RETURN_L=1 GL_LIFT_IK=1 GL_IK_VZ=0.035 RAISE_WIDE=1 GL_WRIST_RANGE=0.4 timeout -k 20 400 ./isaaclab.sh -p $F --headless --grasp_lift --num_envs 256 --steps 660 --camera front --tag $tag --raise_cube=0.02,-0.28 --raise_max_s 11 --raise_sag_comp_deg 0,0,0,0,0,0,0,0 --raise_policy nominal --checkpoint $C > $SP/$tag.log 2>&1
J=$(ls -d $(dirname $C)/videos/*_$tag | tail -1)/chain.json
echo "G256 (+2,-28) grav=1fix $(python3 -c "import json;d=json.load(open('$J'));print('lifted',d['lifted_4cm_frac'],'done',d.get('lift_done_frac'),'pad_N',d['raise_hand_contact_max_n'],'moved',d['cube_moved_mm'],'q_ho',d['q_deg_env0'])" 2>&1 | tail -1)"
grep -E "Traceback" -A 8 $SP/$tag.log | head -12
echo V66_ROLLFIX_DONE
