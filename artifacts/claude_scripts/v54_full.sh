#!/bin/bash
# v54 (CAD layout, user 2026-11-05 "until it works, change the reward if needed"): v53 plateaued at a combined-case score
# of ~0.73 (288 envs: nominal 95.1%, gains x1.5 77.8%, x2.0 67.7%). The weak row is y +6 cm (gains x2.0: 22-38% lifted,
# other cells 72-94%). Timeline traces of v53 r4 14500 there (2026-11-06): the failed envs held the cube with the thumb's
# first link (4-10 N, its distal pad 0 N; at gains x2.0 also the fingers' first links, 12-15 N), which the per-digit
# hold check counts as holding, and the cube slid out of the grip while lifting (hand 31 mm up, cube 19 mm at 1.5 s) or
# during the L return. Changes vs v53 (training only, the eval is unchanged):
#   GL_DISTAL=2 (sat 1.5 N): + 2 x hold x 3 x min(thumb distal, stronger finger distal) of min(F / 1.5 N, 1)
#   GL_SLIP=200: - 200 per m of grasp-point rise above the cube rise beyond 3 mm, while lifting (not during the return)
#   hard box: the y +4..+6 cm strip over the whole x range, 30% of episodes (was the far x column, 15%)
# Same ranking (combined case at 576 envs) and 288-env sweep at each new best. Starts from the v53 best when v53 reaches
# round 8 (its round-8 run is stopped by PID).
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
until grep -q "== ROUND 8\|DONE best" $SP/v53.out; do sleep 10; done
kill 4524 2>/dev/null; sleep 2
for p in $(ps -eo pid,args | grep -E "v53_r8" | grep -v grep | awk '{print $1}'); do kill $p 2>/dev/null; done
sleep 20
for p in $(ps -eo pid,args | grep -E "v53_r8" | grep -v grep | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
sleep 10
echo "STOPPED v53 at round 8 start; v53 best $(cat artifacts/claude_scripts/v53_best.txt)"
B=artifacts/claude_scripts/v54_best.txt
COMMON="GL_GRID_SET=v53 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1"
TRAIN="GL_ACT_DELAY=0,4 GL_MU_RAND=0.25,0.5 GL_KP_SCALE=0.6,2.0 GL_CUBE_MASS=0.04,0.10 GL_OBS_CUBE_NOISE=0.002,0.005 GL_PROX=1 GL_PROX_MIN=2 GL_RET_FREE=1 GL_ROT=30 GL_TABLE_DZ=0.01 GL_HARD_FRAC=0.3 GL_HARD_BOX=-0.08,0.02,0.04,0.06 GL_START_NOISE_DEG=0.75 GL_CUBE_BOX=1 GL_RISE=8 GL_LATE=8 GL_XY=50 GL_XY_VEL=20 GL_DRAG=300 GL_EXTRA_COST=0 GL_REWARD_SCALE=0.02 GL_TABLE=3 GL_NO_VCLIP=1 GL_LR=2e-5 GL_DISTAL=2 GL_SLIP=200"
HARSH="GL_ACT_DELAY=2 GL_MU=0.35,0.3 GL_CUBE_MASS=0.08,0.08 GL_KP_SCALE=1.5,1.5 GL_OBS_CUBE_NOISE=0.002,0.005"
read S0 P0 < artifacts/claude_scripts/v53_best.txt
RUN=$(dirname $P0); CK=$(basename $P0)
evalck() {  # -> "held done lifted tilt_lifting yaw_lifting proxdone-sum proxdone-min"
  out=$(env $COMMON $HARSH NE=576 EVAL_S=12 GL_TABLE_DZ_FIXED=0.0 GL_START_NOISE_DEG=0.75 timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $1 0 2>&1)
  h=$(echo "$out" | grep "^HELD" | awk '{print $NF}'); d=$(echo "$out" | grep "^DONE lift" | awk '{print $NF}'); a=$(echo "$out" | grep "^ALL" | awk '{print $NF}')
  t=$(echo "$out" | grep "^SPLIT" | sed -E 's/.*lifting: tilt ([0-9.-]+) yaw ([0-9.-]+).*/\1 \2/')
  p=$(echo "$out" | grep "^PROX_DONE" | sed -E 's/.*\[([0-9.e-]+), ([0-9.e-]+), ([0-9.e-]+)\] sum ([0-9.e-]+).*/\4 \1 \2 \3/' | awk 'NF == 4 { m = $2; if ($3 < m) m = $3; if ($4 < m) m = $4; print $1, m }'); [ -z "$p" ] && p="0 0"
  echo "$h $d $a $t $p"
}
score() { echo "$1" | awk '{ if (NF >= 7) { ps = $6 / 3.0; if (ps > 1) ps = 1; pm = $7 / 0.5; if (pm > 1) pm = 1; printf "%.4f", $1 - 0.01 * ($4 + $5) + 0.025 * ps + 0.05 * pm } }'; }
sweep() {  # 288-env checks of a new best (run/model path relative to the task log dir)
  for c in "NOMINAL=1" "GL_KP_SCALE=1.5,1.5" "GL_KP_SCALE=2.0,2.0" "MU=0.3,0.25" "GL_CUBE_MASS=0.10,0.10" "GL_ACT_DELAY=4"; do
    echo "SWEEP $1 $c $(env $c GL_GRID_SET=v53 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 ROLL=12 NE=288 bash artifacts/claude_scripts/v46_big_cmp.sh $1 | grep -E '^BIG' | sed -E 's/^BIG [^ ]+ //')"
  done
}
echo "$S0 $RUN/$CK" > $B; echo "START $RUN/$CK score $S0 (v53 ranking, same eval)"
for r in 1 2 3 4 5 6 7 8; do
  tag=v54_r$r
  echo "== ROUND $r from $RUN/$CK"
  SEED=$((5400 + r)) env $COMMON $TRAIN GL_MU=0.4,0.35 bash artifacts/claude_scripts/gl_cycle_nr.sh $RUN $CK $tag 200 | grep -E "gl_lifted|HUNG|Traceback" | sed 's/.*gl_lifted:=\([0-9.]*\).*gl_rot_deg:=\([0-9.]*\).*@\([0-9]*\)/TRAIN gl_lifted \1 rot \2 @\3/'
  D=$(ls -d logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/*_$tag | tail -1); rb=-1; rk=""
  for f in $(ls $D/model_*.pt | sort -V | awk 'NR%2==0'); do
    v=$(evalck $f); [ $(echo $v | wc -w) -lt 7 ] && { echo "RETRY_EVAL $f"; sleep 30; v=$(evalck $f); }; echo "EVAL $(basename $D)/$(basename $f) held_done_lifted_tilt_yaw_proxdone-sum_proxdone-min $v"
    s=$(score "$v"); [ -z "$s" ] && continue
    if python3 -c "import sys; sys.exit(0 if float('$s')>float('$rb') else 1)"; then rb=$s; rk=$(basename $f); fi
  done
  [ -z "$rk" ] && rk=$(ls $D/model_*.pt | sort -V | tail -1 | xargs basename)
  echo "BEST_OF $tag $(basename $D)/$rk $rb"
  if python3 -c "import sys; sys.exit(0 if float('$rb')>float('$(cut -d' ' -f1 $B)') else 1)"; then
    echo "$rb $(basename $D)/$rk" > $B; echo "NEW_BEST $rb $(basename $D)/$rk"; sweep $(basename $D)/$rk
  fi
  RUN=$(basename $D); CK=$rk  # keep going from this round's best
done
echo "DONE best $(cat $B)"
