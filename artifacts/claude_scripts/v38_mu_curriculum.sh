#!/bin/bash
# v38 (user 2026-11-04: friction 0.4, train as before): friction curriculum 0.8 -> 0.6 -> 0.5 -> 0.4 (static; dynamic 0.05-0.1 lower),
# squeeze fixed at 6 N, table +/-1 cm, table-relative cube height, L return; each stage continues from the last
# checkpoint; after each stage a 144-env eval of that checkpoint at friction 0.4,0.35 (table dz 0).
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
RUN=2026-11-04_13-41-42_v34_relret; CK=model_7550.pt
for st in "0.8,0.7" "0.6,0.5" "0.5,0.4" "0.4,0.35"; do
  tag=v38_mu$(echo $st | cut -d, -f1 | tr -d .)
  echo "== STAGE mu $st from $RUN/$CK"
  GL_MU=$st GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_PITCH=0.3 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 \
    GL_TABLE_DZ=0.01 GL_HARD_FRAC=0.15 GL_HARD_BOX=-0.08,-0.06,-0.05,0.06 GL_START_NOISE_DEG=0.75 GL_CUBE_BOX=1 GL_RISE=8 GL_LATE=8 GL_XY=50 GL_XY_VEL=20 GL_DRAG=300 GL_EXTRA_COST=0 GL_REWARD_SCALE=0.02 GL_TABLE=3 GL_NO_VCLIP=1 GL_LR=5e-5 \
    bash artifacts/claude_scripts/gl_cycle_nr.sh $RUN $CK $tag 200 | grep -E "gl_lifted|HUNG|Traceback" | sed 's/.*gl_lifted:=\([0-9.]*\).*@\([0-9]*\)/TRAIN gl_lifted \1 @\2/'
  D=$(ls -d logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/*_$tag | tail -1)
  M=$(ls $D/model_*.pt | sort -V | tail -1)
  r=$(GL_MU=0.4,0.35 GL_SQUEEZE=6,0 DZS="0.0" bash artifacts/claude_scripts/v34_cmp.sh $M | awk '{print $NF}')
  echo "EVAL mu0.4 $(basename $D)/$(basename $M) $r"
  RUN=$(basename $D); CK=$(basename $M)
done
