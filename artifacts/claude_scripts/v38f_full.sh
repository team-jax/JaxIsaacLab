#!/bin/bash
# v38f (user 2026-11-04: friction 0.4, keep it running, fix as needed): rounds at friction 0.4,0.35 straight from
# 7550 (300 it each, from the previous round's best), lift state refreshed per episode (v38 _lifting fix), squeeze
# fixed 6 N, table +/-1 cm, table-relative cube height, L return. Evals: 144 envs, friction 0.4, table dz 0.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
B=artifacts/claude_scripts/v38_best.txt; echo "0.389 2026-11-04_13-41-42_v34_relret/model_7550.pt" > $B
COMMON="GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_PITCH=0.3 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1"
TRAIN="GL_TABLE_DZ=0.01 GL_HARD_FRAC=0.15 GL_HARD_BOX=-0.08,-0.06,-0.05,0.06 GL_START_NOISE_DEG=0.75 GL_CUBE_BOX=1 GL_RISE=8 GL_LATE=8 GL_XY=50 GL_XY_VEL=20 GL_DRAG=300 GL_EXTRA_COST=0 GL_REWARD_SCALE=0.02 GL_TABLE=3 GL_NO_VCLIP=1 GL_LR=5e-5"
RUN=2026-11-04_13-41-42_v34_relret; CK=model_7550.pt
evalck() { env $COMMON GL_MU=0.4,0.35 DZS="0.0" bash artifacts/claude_scripts/v34_cmp.sh $1 | awk '{print $NF}' | sed 's/.*=//'; }
stage() {  # mu tag iters every
  echo "== STAGE mu $1 tag $2 from $RUN/$CK"
  env $COMMON $TRAIN GL_MU=$1 bash artifacts/claude_scripts/gl_cycle_nr.sh $RUN $CK $2 $3 | grep -E "gl_lifted|HUNG|Traceback" | sed 's/.*gl_lifted:=\([0-9.]*\).*@\([0-9]*\)/TRAIN gl_lifted \1 @\2/'
  D=$(ls -d logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/*_$2 | tail -1); rb=-1; rk=""
  for f in $(ls $D/model_*.pt | sort -V | awk -v e=$4 'NR%e==0'); do
    v=$(evalck $f); echo "EVAL mu0.4 $(basename $D)/$(basename $f) ${v:-ERR}"
    [ -z "$v" ] && continue
    if python3 -c "import sys; sys.exit(0 if float('$v')>float('$rb') else 1)"; then rb=$v; rk=$(basename $f); fi
  done
  [ -z "$rk" ] && rk=$(ls $D/model_*.pt | sort -V | tail -1 | xargs basename)
  echo "BEST_OF $2 $(basename $D)/$rk $rb"
  if python3 -c "import sys; sys.exit(0 if float('$rb')>float('$(cut -d' ' -f1 $B)') else 1)"; then echo "$rb $(basename $D)/$rk" > $B; echo "NEW_BEST $rb $(basename $D)/$rk"; fi
  RUN=$(basename $D); CK=$rk
}
for r in 1 2 3 4 5 6 7 8; do stage 0.4,0.35 v38f_r$r 300 4; done
echo "DONE best $(cat $B)"
