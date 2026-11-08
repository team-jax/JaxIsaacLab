#!/bin/bash
# v62: back to the v59 recipe (50 mm cube, 30-50 g, turned U(-45, 45) deg, no extra sampling, no tilt cost) from the deployed
# v59 r8 17100 with the learning rate halved (2e-5 -> 1e-5) to refine without drifting: v60 (20% at -45..-25 deg) and v61
# (tilt cost 30/rad) both plateaued below 17100 (v61 lowered the tilt 3.3 -> 2.7 deg but lost 1-3%p). v61 is stopped by PID.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
CLOSE="GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15"
B=artifacts/claude_scripts/v62_best.txt
COMMON="GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v53 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 $CLOSE"
TRAIN="GL_ACT_DELAY=0,4 GL_MU_RAND=0.25,0.5 GL_KP_SCALE=0.6,2.0 GL_CUBE_MASS=0.03,0.05 GL_OBS_CUBE_NOISE=0.002,0.005 GL_PROX=1 GL_PROX_MIN=2 GL_RET_FREE=1 GL_ROT=30 GL_TABLE_DZ=0.01 GL_HARD_FRAC=0.3 GL_HARD_BOX=-0.08,0.02,0.04,0.06 GL_START_NOISE_DEG=0.75 GL_CUBE_BOX=1 GL_RISE=8 GL_LATE=8 GL_XY=50 GL_XY_VEL=20 GL_DRAG=300 GL_EXTRA_COST=0 GL_REWARD_SCALE=0.02 GL_TABLE=3 GL_NO_VCLIP=1 GL_LR=1e-5 GL_DISTAL=2 GL_SLIP=200 GL_PROX_NEED_DISTAL=1 GL_ROT_FROM_LIFTOFF=1"
HARSH="GL_ACT_DELAY=2 GL_MU=0.35,0.3 GL_CUBE_MASS=0.05,0.05 GL_KP_SCALE=1.5,1.5 GL_OBS_CUBE_NOISE=0.002,0.005"
read S0 P0 < artifacts/claude_scripts/v59_best.txt
RUN=$(dirname $P0); CK=$(basename $P0)
YAWEVAL="GL_CUBE_YAW_DEG=45 EVAL_YAW_LO=1"
evalone() {  # $1 = checkpoint, $2 = case flags -> "held tilt yaw proxsum proxmin"
  out=$(env $COMMON $YAWEVAL $2 NE=576 EVAL_S=12 GL_TABLE_DZ_FIXED=0.0 GL_START_NOISE_DEG=0.75 timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $1 0 2>&1)
  h=$(echo "$out" | grep "^HELD" | awk '{print $NF}')
  t=$(echo "$out" | grep "^SPLIT" | sed -E 's/.*lifting: tilt ([0-9.-]+) yaw ([0-9.-]+).*/\1 \2/')
  p=$(echo "$out" | grep "^PROX_DONE" | sed -E 's/.*\[([0-9.e-]+), ([0-9.e-]+), ([0-9.e-]+)\] sum ([0-9.e-]+).*/\4 \1 \2 \3/' | awk 'NF == 4 { m = $2; if ($3 < m) m = $3; if ($4 < m) m = $4; print $1, m }'); [ -z "$p" ] && p="0 0"
  echo "$h $t $p"
}
evalck() {  # -> "held_nom tilt_nom yaw_nom held_harsh tilt_harsh yaw_harsh proxsum proxmin"
  n=$(evalone $1 "GL_MU=0.4,0.35"); [ $(echo $n | wc -w) -lt 5 ] && { sleep 30; n=$(evalone $1 "GL_MU=0.4,0.35"); }
  h=$(evalone $1 "$HARSH"); [ $(echo $h | wc -w) -lt 5 ] && { sleep 30; h=$(evalone $1 "$HARSH"); }
  echo "$(echo $n | awk '{print $1, $2, $3}') $h"
}
score() { echo "$1" | awk '{ if (NF >= 8) { ps = $7 / 3.0; if (ps > 1) ps = 1; pm = $8 / 0.5; if (pm > 1) pm = 1; printf "%.4f", 0.5 * ($1 + $4) - 0.005 * ($2 + $3 + $5 + $6) + 0.025 * ps + 0.05 * pm } }'; }
sweep() {  # 288-env checks of a new best: fixed cube turns (nominal), random +/-45 deg with gains x2.0 / friction 0.3
  for c in "GL_CUBE_YAW_FIXED_DEG=0" "GL_CUBE_YAW_FIXED_DEG=20" "GL_CUBE_YAW_FIXED_DEG=-20" "GL_CUBE_YAW_FIXED_DEG=35" "GL_CUBE_YAW_FIXED_DEG=-35" "GL_CUBE_YAW_FIXED_DEG=45" "GL_CUBE_YAW_FIXED_DEG=-45" "GL_CUBE_YAW_DEG=45 GL_KP_SCALE=2.0,2.0" "GL_CUBE_YAW_DEG=45 MU=0.3,0.25"; do
    echo "SWEEP $1 $c $(env $c EVAL_YAW_LO=1 GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 $CLOSE GL_GRID_SET=v53 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 ROLL=12 NE=288 bash artifacts/claude_scripts/v46_big_cmp.sh $1 | grep -E '^BIG' | sed -E 's/^BIG [^ ]+ //')"
  done
}
L=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16
for p in $(ps -eo pid,args | grep -E "v61_full.sh" | grep -v grep | awk '{print $1}'); do kill $p 2>/dev/null; done; sleep 2
for p in $(ps -eo pid,args | grep -E "v61_r[0-9]" | grep -v grep | awk '{print $1}'); do kill $p 2>/dev/null; done
sleep 20
for p in $(ps -eo pid,args | grep -E "v61_r[0-9]" | grep -v grep | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
sleep 10
echo "STOPPED v61"
echo "$S0 $RUN/$CK" > $B; echo "START $RUN/$CK score $S0 (v59 best, LR 1e-5)"
for r in 1 2 3 4 5 6 7 8; do
  tag=v62_r$r
  YT=45
  echo "== ROUND $r from $RUN/$CK (cube turn +/-$YT deg)"
  SEED=$((6300 + r)) env $COMMON $TRAIN GL_CUBE_YAW_DEG=$YT GL_MU=0.4,0.35 bash artifacts/claude_scripts/gl_cycle_nr.sh $RUN $CK $tag 200 | grep -E "gl_lifted|HUNG|Traceback" | sed 's/.*gl_lifted:=\([0-9.]*\).*gl_rot_deg:=\([0-9.]*\).*@\([0-9]*\)/TRAIN gl_lifted \1 rot \2 @\3/'
  D=$(ls -d $L/*_$tag | tail -1); rb=-1; rk=""
  for f in $(ls $D/model_*.pt | sort -V | awk 'NR%2==0'); do
    v=$(evalck $f); [ $(echo $v | wc -w) -lt 8 ] && { echo "RETRY_EVAL $f"; sleep 30; v=$(evalck $f); }; echo "EVAL $(basename $D)/$(basename $f) nom(held tilt yaw) harsh(held tilt yaw proxsum proxmin) $v"
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
