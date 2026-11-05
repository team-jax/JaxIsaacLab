#!/bin/bash
# chain renders (v27b raise + grasp CK) at 9 cube offsets, camera $2, then one 3x3 montage
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
F=source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/render_front_grasp.py
R=logs/rsl_rl/jax_v2_1006_raise_cube_v25/2026-11-01_21-25-47_v25_cube/model_1197.pt; C=$1; CAM=$2
D=$(dirname $C)/videos; M=$(basename $C .pt); ARGS=""
for x in -0.08 -0.03 0.02; do for y in 0.06 0.0 -0.05; do
  tag=v27_x${x}_y${y}; lab=$(python3 -c "print('x%+.0fcm,y%+.0fcm'%($x*100,$y*100))")
  J0=$D/front_grasp_lift_${CAM}_${M}_${tag}${TAGSUF:-}/chain.json; [ -f $J0 ] || GL_RETURN_L=${RETL:-} GL_LIFT_IK=${LIFTIK:-} GL_IK_PITCH=${IKP:-} GL_IK_VZ=${IKVZ:-} GL_IK_VXY=${IKVXY:-} GL_LIFT_EXTRA=${EXTRA:-0} RAISE_WIDE=1 GL_WRIST_RANGE=0.4 timeout -k 20 900 ./isaaclab.sh -p $F --headless --grasp_lift --steps ${STEPS:-420} --camera $CAM --tag ${tag}${TAGSUF:-} --raise_cube=$x,$y --raise_max_s 11 --raise_sag_comp_deg 0,0,0,0,0,0,0,0 --raise_policy $R --checkpoint $C > $S/m_$tag.log 2>&1
  J=$D/front_grasp_lift_${CAM}_${M}_${tag}${TAGSUF:-}/chain.json
  python3 -c "import json;d=json.load(open('$J'));print('$lab', 'lifted',d['lifted_4cm_frac'],'done',d.get('lift_done_frac'),'handover',d['handover_by'],d['handover_t_s'],'raise_contact_N',d['raise_hand_contact_max_n'],'cube_moved_mm',d['cube_moved_mm'],'table_N',d['episode_table_force_max_n'])" 2>&1 | tail -1
  ARGS="$ARGS $lab=$D/front_grasp_lift_${CAM}_${M}_${tag}${TAGSUF:-}/rl-video-step-0.mp4"
done; done
OUT=$D/montage_v27_${CAM}_${M}${TAGSUF:-}.mp4
./isaaclab.sh -p artifacts/claude_scripts/montage.py $OUT 3 $ARGS 2>&1 | grep MONTAGE
