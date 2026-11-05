#!/bin/bash
# v39b (user 2026-11-04: "keep training"): v39 with the cube drag cost 300 -> 100 per m/s (v39 r1 drifted toward
# letting the cube go: deeper closing pushes the cube, and drag paid more than the grasp). v39 (user "grasp deeper"): pre-grasp hand 2 cm deeper and 0.8 cm toward the thumb side
# (GL_REACH_DX/DY, widened grids), from v38f_r1 model_7825 (friction 0.4: 85%, chain 100%). Rounds at friction
# 0.4,0.35, 300 it each from the previous round's best, squeeze 6 N, table +/-1 cm, table-relative cube height,
# L return, lift state refreshed per episode. Evals: 144 envs, friction 0.4, table dz 0, same deeper pre-grasp.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
B=artifacts/claude_scripts/v39b_best.txt
COMMON="GL_REACH_DX=-0.02 GL_REACH_DY=-0.008 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_PITCH=0.3 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1"
TRAIN="GL_TABLE_DZ=0.01 GL_HARD_FRAC=0.15 GL_HARD_BOX=-0.08,-0.06,-0.05,0.06 GL_START_NOISE_DEG=0.75 GL_CUBE_BOX=1 GL_RISE=8 GL_LATE=8 GL_XY=50 GL_XY_VEL=20 GL_DRAG=100 GL_EXTRA_COST=0 GL_REWARD_SCALE=0.02 GL_TABLE=3 GL_NO_VCLIP=1 GL_LR=5e-5"
RUN=2026-11-04_18-22-49_v38f_r1; CK=model_7825.pt
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
echo "0.632 $RUN/$CK" > $B  # 7825 at the deeper pre-grasp, friction 0.4 (v39 start eval)
for r in 1 2 3 4 5 6; do stage 0.4,0.35 v39b_r$r 300 4; done
echo "DONE best $(cat $B)"
