#!/bin/bash
# v38: chain video of CK at friction MU (squeeze 6 N, table dz 0): 3 cube spots (x -3 cm; y +6 / 0 / -5 cm) side by side.
# usage: v38_stage_video.sh CK MU TAG
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
F=source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/render_front_grasp.py
R=logs/rsl_rl/jax_v2_1006_raise_cube_v25/2026-11-01_21-25-47_v25_cube/model_1197.pt
C=$1; MU=$2; TAG=$3; D=$(dirname $C)/videos; M=$(basename $C .pt); ARGS=""
for y in 0.06 0.0 -0.05; do
  tag=${TAG}_y$y
  GL_MU=$MU GL_SQUEEZE=6,0 GL_TABLE_DZ_FIXED=0.0 GL_TABLE_REL=1 GL_RETURN_L=1 GL_LIFT_IK=1 GL_IK_PITCH=0.3 GL_IK_VZ=0.035 RAISE_WIDE=1 GL_WRIST_RANGE=0.4 \
    timeout -k 20 900 ./isaaclab.sh -p $F --headless --grasp_lift --steps 660 --camera front --tag $tag --raise_cube=-0.03,$y --raise_max_s 11 --raise_sag_comp_deg 0,0,0,0,0,0,0,0 --raise_policy $R --checkpoint $C > $S/v38v_$tag.log 2>&1
  J=$D/front_grasp_lift_front_${M}_${tag}/chain.json
  python3 -c "import json;d=json.load(open('$J'));print('VID y$y lifted',d['lifted_4cm_frac'],'done',d.get('lift_done_frac'))" 2>&1 | tail -1
  ARGS="$ARGS mu${MU%%,*}_x-3,y$y=$D/front_grasp_lift_front_${M}_${tag}/rl-video-step-0.mp4"
done
SCALE=1.0 ./isaaclab.sh -p artifacts/claude_scripts/montage_scaled.py $D/stage_${TAG}_${M}.mp4 3 $ARGS 2>&1 | grep MONTAGE
