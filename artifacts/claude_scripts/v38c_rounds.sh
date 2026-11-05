#!/bin/bash
# v38c (user 2026-11-04: "just keep it running"): after the friction curriculum (v38b), more rounds at friction
# 0.4,0.35 (squeeze 6 N, table +/-1 cm, table-relative cube height, L return). Each round: 300 iterations from the
# previous round's best checkpoint, then a 144-env eval (friction 0.4, table dz 0) of every 4th checkpoint
# (100 iterations apart). The best so far goes to v38_best.txt.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
B=artifacts/claude_scripts/v38_best.txt
last=$(grep "^EVAL mu0.4" $S/v38b.out | tail -1)
RUN=$(echo "$last" | awk '{print $3}' | cut -d/ -f1); CK=$(echo "$last" | awk '{print $3}' | cut -d/ -f2)
bs=$(echo "$last" | awk '{print $NF}' | sed 's/.*=//'); [ -f $B ] || echo "$bs $RUN/$CK" > $B
for r in 1 2 3 4 5 6; do
  tag=v38c_r$r
  echo "== ROUND $r from $RUN/$CK"
  GL_MU=0.4,0.35 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_PITCH=0.3 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 \
    GL_TABLE_DZ=0.01 GL_HARD_FRAC=0.15 GL_HARD_BOX=-0.08,-0.06,-0.05,0.06 GL_START_NOISE_DEG=0.75 GL_CUBE_BOX=1 GL_RISE=8 GL_LATE=8 GL_XY=50 GL_XY_VEL=20 GL_DRAG=300 GL_EXTRA_COST=0 GL_REWARD_SCALE=0.02 GL_TABLE=3 GL_NO_VCLIP=1 GL_LR=5e-5 \
    bash artifacts/claude_scripts/gl_cycle_nr.sh $RUN $CK $tag 300 | grep -E "gl_lifted|HUNG|Traceback" | sed 's/.*gl_lifted:=\([0-9.]*\).*@\([0-9]*\)/TRAIN gl_lifted \1 @\2/'
  D=$(ls -d logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/*_$tag | tail -1)
  rb=-1; rk=""
  for f in $(ls $D/model_*.pt | sort -V | awk 'NR%4==0'); do
    v=$(GL_MU=0.4,0.35 GL_SQUEEZE=6,0 DZS="0.0" bash artifacts/claude_scripts/v34_cmp.sh $f | awk '{print $NF}' | sed 's/.*=//')
    echo "EVAL mu0.4 $(basename $D)/$(basename $f) ${v:-ERR}"
    [ -z "$v" ] || [ "$v" = "ERR" ] && continue
    if python3 -c "import sys; sys.exit(0 if float('$v')>float('$rb') else 1)"; then rb=$v; rk=$(basename $f); fi
  done
  [ -z "$rk" ] && rk=$(ls $D/model_*.pt | sort -V | tail -1 | xargs basename)
  echo "ROUND $r best $(basename $D)/$rk $rb"
  if python3 -c "import sys; sys.exit(0 if float('$rb')>float('$(cut -d' ' -f1 $B)') else 1)"; then echo "$rb $(basename $D)/$rk" > $B; echo "NEW_BEST $rb $(basename $D)/$rk"; fi
  RUN=$(basename $D); CK=$rk
done
echo "DONE best $(cat $B)"
