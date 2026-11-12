#!/bin/bash
# v81b: the team-layout chain (CHAIN_TEAM=1: L -> 8 cm above the pre-grasp -> straight down with the fingers opened wider -> fingers back
# -> policy) at several cube spots, 8 envs each, no video. $1 = checkpoint, SPOTS="dx,dy ..." (m).
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
F=source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/render_front_grasp.py
eval "$(grep -E '^(CLOSE|COMMON)=' artifacts/claude_scripts/v81b_full.sh)"
V=$(dirname $1)/videos; M=$(basename $1 .pt); TOT=0; OKS=0
SP_LIST=${SPOTS:-0.04,0.22 0.10,0.22 0.24,0.22 0.06,0.25 0.16,0.25 0.10,0.28 0.22,0.28 0.18,0.295 0.25,0.295}
for xy in $SP_LIST; do
  x=${xy%,*}; y=${xy#*,}; tag=tc_${x}_${y}${TAGSUF:-}
  flock -o $SP/isaac_eval.lock env $COMMON CHAIN_TEAM=1 CHAIN_NO_VIDEO=${NOVID-1} GL_SERVO_ARM=5,5 GL_FINGER_EFFORT=0.177,0.177 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 \
    GL_TABLE_DZ_FIXED=0.0 CHAIN_GRAV_COMP=1 CHAIN_GRAV_FULL=1 TEST_ARM_GAIN_SCALE=5 RAISE_WIDE=1 GL_WRIST_RANGE=0.4 GL_LIFT_EXTRA=0 RAISE_GRID_SET=v65r \
    REACH_FINAL_GRID_SET=teamf timeout -k 20 900 ./isaaclab.sh -p $F --headless --grasp_lift --num_envs ${NE:-8} --steps 660 --camera ${CAM:-front} --tag $tag \
    --raise_cube=$x,$y --raise_max_s 11 --raise_sag_comp_deg 0,0,0,0,0,0,0,0 --raise_policy nominal --checkpoint $1 > $SP/$tag.log 2>&1
  J=$(ls -d $V/*${M}_${tag} 2>/dev/null | tail -1)/chain.json
  python3 -c "
import json; d = json.load(open('$J'))
print('SPOT ($x, $y) done %.3f lifted %.3f approach_contact_N %.1f cube_moved_mm %.1f table_N_max %.1f' % (d.get('lift_done_frac', -1), d['lifted_4cm_frac'], d['raise_hand_contact_max_n'], d['cube_moved_mm'], d.get('episode_table_force_max_n', -1)))" 2>&1 | tail -1
done
echo CHAIN_CHECK_DONE
