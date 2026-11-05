#!/bin/bash
# v41: chain check in the 22.5 cm table layout (GL_TABLE_EDGE), friction 0.4, squeeze 6 N, three cube spots
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
F=source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/render_front_grasp.py
R=logs/rsl_rl/jax_v2_1006_raise_cube_v25/2026-11-01_21-25-47_v25_cube/model_1197.pt
C=${1:-logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-04_18-22-49_v38f_r1/model_7825.pt}; CAM=${2:-front}
D=$(dirname $C)/videos; M=$(basename $C .pt)
for xy in "0.0,0.0" "0.02,-0.05" "-0.08,0.06"; do
  tag=v41_${xy/,/_}_${CAM}
  GL_TABLE_EDGE=0.225 GL_MU=0.4,0.35 GL_SQUEEZE=6,0 GL_TABLE_DZ_FIXED=0.0 GL_TABLE_REL=1 GL_RETURN_L=1 GL_LIFT_IK=1 GL_IK_PITCH=0.3 GL_IK_VZ=0.035 RAISE_WIDE=1 GL_WRIST_RANGE=0.4 \
    timeout -k 20 900 ./isaaclab.sh -p $F --headless --grasp_lift --steps 660 --camera $CAM --tag $tag --raise_cube=$xy --raise_max_s 11 --raise_sag_comp_deg 0,0,0,0,0,0,0,0 --raise_policy $R --checkpoint $C > $S/v41_$tag.log 2>&1
  J=$D/front_grasp_lift_${CAM}_${M}_${tag}/chain.json
  python3 -c "import json;d=json.load(open('$J'));print('SPOT $xy lifted',d['lifted_4cm_frac'],'done',d.get('lift_done_frac'),'raise_contact_N',d['raise_hand_contact_max_n'],'keepout_mm',d['raise_keepout_max_mm'],'cube_moved_mm',d['cube_moved_mm'],'table_N',d['episode_table_force_max_n'])" 2>&1 | tail -1
done
