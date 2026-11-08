#!/bin/bash
# v63c: the deploy check of 17650 found the far row failing at a 1 cm lower table (34-94%): the v53w -3 cm grid cannot
# reach there and its 1/3 blend broke the start pose. Grid set v53x (v63x_grids.sh: exact +/-1 cm grids stretched x3 into
# the -3 / +3 cm files, formula unchanged) fixed most of it without training (92.7%, corners 72-78%). v63c fine-tunes on
# v53x from 17650: every round from the overall best, 2 checkpoints evaluated per round, and the score adds a 1 cm lower
# table case: mean(held nominal, combined, table -1 cm) - 0.005 (tilt + yaw, nominal + combined) + the first-joint terms.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
CLOSE="GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15"
B=artifacts/claude_scripts/v63c_best.txt
export EVAL_GRID="-0.14,-0.06,0.02;-0.10,-0.02,0.06"
COMMON="GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v53x GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 $CLOSE"
TRAIN="GL_ACT_DELAY=0,4 GL_MU_RAND=0.25,0.5 GL_KP_SCALE=0.6,2.0 GL_CUBE_MASS=0.03,0.05 GL_OBS_CUBE_NOISE=0.002,0.005 GL_PROX=1 GL_PROX_MIN=2 GL_RET_FREE=1 GL_ROT=30 GL_TABLE_DZ=0.01 GL_HARD_FRAC=0.15 GL_HARD_BOX=-0.14,0.02,0.04,0.06 GL_START_NOISE_DEG=0.75 GL_CUBE_BOX=-0.14,0.02,-0.10,0.06 GL_RISE=8 GL_LATE=8 GL_XY=50 GL_XY_VEL=20 GL_DRAG=300 GL_EXTRA_COST=0 GL_REWARD_SCALE=0.02 GL_TABLE=3 GL_NO_VCLIP=1 GL_LR=2e-5 GL_DISTAL=2 GL_SLIP=200 GL_PROX_NEED_DISTAL=1 GL_ROT_FROM_LIFTOFF=1"
HARSH="GL_ACT_DELAY=2 GL_MU=0.35,0.3 GL_CUBE_MASS=0.05,0.05 GL_KP_SCALE=1.5,1.5 GL_OBS_CUBE_NOISE=0.002,0.005"
RUN=2026-11-06_10-58-26_v59_r8; CK=model_17100.pt
YAWEVAL="GL_CUBE_YAW_DEG=45 EVAL_YAW_LO=1"
evalone() {  # $1 = checkpoint, $2 = case flags -> "held tilt yaw proxsum proxmin"
  out=$(env $COMMON $YAWEVAL NE=576 EVAL_S=12 GL_TABLE_DZ_FIXED=0.0 GL_START_NOISE_DEG=0.75 $2 timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $1 0 2>&1)
  h=$(echo "$out" | grep "^HELD" | awk '{print $NF}')
  t=$(echo "$out" | grep "^SPLIT" | sed -E 's/.*lifting: tilt ([0-9.-]+) yaw ([0-9.-]+).*/\1 \2/')
  p=$(echo "$out" | grep "^PROX_DONE" | sed -E 's/.*\[([0-9.e-]+), ([0-9.e-]+), ([0-9.e-]+)\] sum ([0-9.e-]+).*/\4 \1 \2 \3/' | awk 'NF == 4 { m = $2; if ($3 < m) m = $3; if ($4 < m) m = $4; print $1, m }'); [ -z "$p" ] && p="0 0"
  echo "$h $t $p"
}
evalck() {  # -> "held_nom tilt_nom yaw_nom held_harsh tilt_harsh yaw_harsh proxsum proxmin held_dzm1"
  n=$(evalone $1 "GL_MU=0.4,0.35"); [ $(echo $n | wc -w) -lt 5 ] && { sleep 30; n=$(evalone $1 "GL_MU=0.4,0.35"); }
  h=$(evalone $1 "$HARSH"); [ $(echo $h | wc -w) -lt 5 ] && { sleep 30; h=$(evalone $1 "$HARSH"); }
  d=$(evalone $1 "GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=-0.01"); [ $(echo $d | wc -w) -lt 5 ] && { sleep 30; d=$(evalone $1 "GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=-0.01"); }
  echo "$(echo $n | awk '{print $1, $2, $3}') $h $(echo $d | awk '{print $1}')"
}
score() { echo "$1" | awk '{ if (NF >= 9) { ps = $7 / 3.0; if (ps > 1) ps = 1; pm = $8 / 0.5; if (pm > 1) pm = 1; printf "%.4f", ($1 + $4 + $9) / 3.0 - 0.005 * ($2 + $3 + $5 + $6) + 0.025 * ps + 0.05 * pm } }'; }
sweep() {  # 288-env checks of a new best: table -1 / +1 cm, the weak turns -35 / +45 deg, drive gains x2.0
  for c in "GL_CUBE_YAW_DEG=45 DZ=-0.01" "GL_CUBE_YAW_DEG=45 DZ=0.01" "GL_CUBE_YAW_FIXED_DEG=-35" "GL_CUBE_YAW_FIXED_DEG=45" "GL_CUBE_YAW_DEG=45 GL_KP_SCALE=2.0,2.0"; do
    echo "SWEEP $1 $c $(env $c EVAL_YAW_LO=1 GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 $CLOSE GL_GRID_SET=v53x GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 ROLL=12 NE=288 bash artifacts/claude_scripts/v46_big_cmp.sh $1 | grep -E '^BIG' | sed -E 's/^BIG [^ ]+ //')"
  done
}
L=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16
RUN=2026-11-07_14-34-23_v63b_r10; CK=model_17650.pt
v=$(evalck $L/$RUN/$CK); [ $(echo $v | wc -w) -lt 9 ] && { sleep 30; v=$(evalck $L/$RUN/$CK); }
s0=$(score "$v"); echo "EVAL $RUN/$CK (start, v53x) nom(held tilt yaw) harsh(held tilt yaw proxsum proxmin) dzm1(held) $v"
echo "${s0:-0.0} $RUN/$CK" > $B; echo "START $RUN/$CK score ${s0:-0.0} (every round restarts from the overall best)"
for r in 1 2 3 4; do
  tag=v63c_r$r
  YT=45
  echo "== ROUND $r from $RUN/$CK (cube turn +/-$YT deg)"
  SEED=$((6600 + r)) env $COMMON $TRAIN GL_CUBE_YAW_DEG=$YT GL_MU=0.4,0.35 bash artifacts/claude_scripts/gl_cycle_nr.sh $RUN $CK $tag 200 | grep -E "gl_lifted|HUNG|Traceback" | sed 's/.*gl_lifted:=\([0-9.]*\).*gl_rot_deg:=\([0-9.]*\).*@\([0-9]*\)/TRAIN gl_lifted \1 rot \2 @\3/'
  D=$(ls -d $L/*_$tag | tail -1); rb=-1; rk=""
  for f in $(ls $D/model_*.pt | sort -V | awk 'NR%4==0'); do
    v=$(evalck $f); [ $(echo $v | wc -w) -lt 9 ] && { echo "RETRY_EVAL $f"; sleep 30; v=$(evalck $f); }; echo "EVAL $(basename $D)/$(basename $f) nom(held tilt yaw) harsh(held tilt yaw proxsum proxmin) dzm1(held) $v"
    s=$(score "$v"); [ -z "$s" ] && continue
    if python3 -c "import sys; sys.exit(0 if float('$s')>float('$rb') else 1)"; then rb=$s; rk=$(basename $f); fi
  done
  [ -z "$rk" ] && rk=$(ls $D/model_*.pt | sort -V | tail -1 | xargs basename)
  echo "BEST_OF $tag $(basename $D)/$rk $rb"
  if python3 -c "import sys; sys.exit(0 if float('$rb')>float('$(cut -d' ' -f1 $B)') else 1)"; then
    echo "$rb $(basename $D)/$rk" > $B; echo "NEW_BEST $rb $(basename $D)/$rk"; sweep $(basename $D)/$rk
  fi
  read _s P < $B; RUN=$(dirname $P); CK=$(basename $P)  # saturated: restart from the overall best
done
echo "DONE best $(cat $B)"
