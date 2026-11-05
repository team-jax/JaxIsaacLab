#!/bin/bash
# v34 (user 2026-11-04: table heights +/-3 cm): grasp-lift retrain with the table z randomized, then
# deterministic eval of every 3rd checkpoint at table dz -3 / 0 / +3 cm (9 cube spots x 4 each).
# usage: v34_round.sh LOAD_RUN CK TAG ITERS [DZ]
cd /home/jeongwoo/isaacsim/JaxIsaacLab
V="GL_LIFT_IK=1 GL_IK_PITCH=0.3 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=${REL:-} GL_RETURN_L=${RET:-}"
echo "== GRASP"
env $V GL_TABLE_DZ=${5:-0.03} GL_HARD_FRAC=0.15 GL_HARD_BOX=-0.08,-0.06,-0.05,0.06 GL_START_NOISE_DEG=0.75 GL_CUBE_BOX=1 GL_RISE=8 GL_LATE=8 GL_XY=50 GL_XY_VEL=20 GL_DRAG=300 GL_EXTRA_COST=0 GL_REWARD_SCALE=0.02 GL_TABLE=3 GL_NO_VCLIP=1 GL_LR=5e-5 bash artifacts/claude_scripts/gl_cycle_nr.sh $1 $2 $3 $4
D=$(ls -d logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/*_$3 | tail -1)
for f in $(ls $D/model_*.pt | sort -V | awk 'NR%3==0'); do
  line="$(basename $f)"
  for dz in ${EDZS:--0.03 0.0 0.03}; do
    a=$(env $V GL_TABLE_DZ_FIXED=$dz GL_START_NOISE_DEG=0.75 timeout -k 20 400 ./isaaclab.sh -p artifacts/claude_scripts/v34_eval.py $f 0 2>&1 | grep "^ALL" | awk '{print $NF}')
    line="$line dz$dz=${a:-ERR}"
  done
  echo "EVAL $line"
done
