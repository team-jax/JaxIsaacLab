#!/bin/bash
# v75m (user 2026-11-10: the finger servos are MG90S - stall 0.177 N*m at 4.8 V, 0.216 N*m at 6 V; first assumed MG996R, the
# v75 run at 0.6-1.0 N*m was stopped in round 1). Finger joints get a PhysX torque limit GL_FINGER_EFFORT=0.13,0.22 per episode
# in training (75 % of the 4.8 V stall up to the 6 V stall) and 0.177 in evals. 25150 held the thumb first joint at 0.82 N*m
# median while lifting (4.6x the MG90S stall). From 25150 (scored twice under the limit first), else as v74 (target-feedback
# joint inputs, servo-emulated arm, URDF_1109, table at 47.25 cm), rounds accumulate at lr 2e-5. Best in v75m_best.txt.
# v74 (user 2026-11-09: "commit, then keep training the weak parts; our feedback is the target angle we send, not the encoder"):
# fine-tune v72_r5/model_23025 for the final spec - URDF_1109 robot (task URDF), AK45 arm in servo mode, shoulder -> table 47.25 cm
# (table 0.735 m) - and the real feedback:
#  - GL_OBS_Q_TARGET=1: joint inputs 0-13 = the sent joint targets, 14-27 = their change over the control step / dt (x 0.2)
#  - GL_SERVO_ARM=2,10 (training) / 5,5 (evals): arm drives x U(2, 10) stiffer, start pose held at the actual kp (sits on the
#    geometric pose like a servo); GL_RET_Q_DEG = the geometric L (servo command)
#  - table centred at 47.25 cm (GL_TABLE_DZ_MEAN=0.0) +/- 1 cm; evals at 0.0, the table -1 cm case at -0.01 (48.25 cm)
# Weak parts at the start (23025, servo emulation): combined 92.2, friction 0.3 92.5, table -1 cm 94.1, chain front row.
# Same 5-case score and recheck as v72b, rounds accumulate (each from the last checkpoint), best in v74_best.txt; the start is
# scored twice under the new conditions alongside round 1 training. Evals hold the lock with flock -o (Omniverse Hub).
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
CLOSE="GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15"
B=artifacts/claude_scripts/v75m_best.txt
LOCK=$SP/isaac_eval.lock
L=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16
export EVAL_CELLS="0.02,-0.28;0.02,0.0;0.02,0.28;-0.04,-0.24;-0.04,0.24;-0.08,-0.18;-0.08,0.0;-0.08,0.18;-0.12,-0.14;-0.12,0.12;-0.14,-0.08;-0.14,0.06"
COMMON="GL_ROBOT=v3 GL_OBS_YAW=1 GL_OBS_Q_TARGET=1 GL_RET_Q_DEG=0,0,7.2,90.0,0,-2.6,0 GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=0 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 $CLOSE"
TRAIN="GL_FINGER_EFFORT=0.13,0.22 GL_SERVO_ARM=2,10 GL_ACT_DELAY=0,4 GL_MU_RAND=0.25,0.5 GL_KP_SCALE=0.6,2.0 GL_CUBE_MASS=0.03,0.05 GL_OBS_CUBE_NOISE=0.002,0.005 GL_PROX=1 GL_PROX_MIN=2 GL_RET_FREE=1 GL_ROT=30 GL_TABLE_DZ=0.01 GL_START_NOISE_DEG=0.75 GL_CUBE_BOX=-0.14,0.02,-0.28,0.28 GL_CUBE_MASK=data/v65_mask.pt GL_HARSH_FRAC=0.25 GL_TABLE_DZ_MEAN=0.0 GL_CUBE_YAW_HARD=0.2,30,40,1 GL_RISE=8 GL_LATE=8 GL_XY=50 GL_XY_VEL=20 GL_DRAG=300 GL_EXTRA_COST=0 GL_REWARD_SCALE=0.02 GL_TABLE=3 GL_NO_VCLIP=1 GL_LR=2e-5 GL_ENT=0 GL_DISTAL=2 GL_SLIP=200 GL_PROX_NEED_DISTAL=1 GL_ROT_FROM_LIFTOFF=1"
HARSH="GL_ACT_DELAY=2 GL_MU=0.35,0.3 GL_CUBE_MASS=0.05,0.05 GL_KP_SCALE=1.5,1.5 GL_OBS_CUBE_NOISE=0.002,0.005"
YAWEVAL="GL_CUBE_YAW_DEG=45 EVAL_YAW_LO=1"
evalone() {  # $1 = checkpoint, $2 = case flags -> "held tilt yaw proxsum proxmin"
  out=$(flock -o $LOCK env $COMMON $YAWEVAL GL_SERVO_ARM=5,5 GL_FINGER_EFFORT=0.177,0.177 NE=576 EVAL_S=12 GL_TABLE_DZ_FIXED=0.0 GL_START_NOISE_DEG=0.75 $2 timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $1 0 2>&1)
  h=$(echo "$out" | grep "^HELD" | awk '{print $NF}')
  t=$(echo "$out" | grep "^SPLIT" | sed -E 's/.*lifting: tilt ([0-9.-]+) yaw ([0-9.-]+).*/\1 \2/')
  p=$(echo "$out" | grep "^PROX_DONE" | sed -E 's/.*\[([0-9.e-]+), ([0-9.e-]+), ([0-9.e-]+)\] sum ([0-9.e-]+).*/\4 \1 \2 \3/' | awk 'NF == 4 { m = $2; if ($3 < m) m = $3; if ($4 < m) m = $4; print $1, m }'); [ -z "$p" ] && p="0 0"
  echo "$h $t $p"
}
evalck() {  # -> "held_nom tilt_nom yaw_nom held_harsh tilt_harsh yaw_harsh proxsum proxmin held_dzm1 held_mu03 held_yaw-35"
  n=$(evalone $1 "GL_MU=0.4,0.35"); [ $(echo $n | wc -w) -lt 5 ] && { sleep 30; n=$(evalone $1 "GL_MU=0.4,0.35"); }
  h=$(evalone $1 "$HARSH"); [ $(echo $h | wc -w) -lt 5 ] && { sleep 30; h=$(evalone $1 "$HARSH"); }
  d=$(evalone $1 "GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=-0.01"); [ $(echo $d | wc -w) -lt 5 ] && { sleep 30; d=$(evalone $1 "GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=-0.01"); }
  f=$(evalone $1 "GL_MU=0.3,0.25"); [ $(echo $f | wc -w) -lt 5 ] && { sleep 30; f=$(evalone $1 "GL_MU=0.3,0.25"); }
  y=$(evalone $1 "GL_MU=0.4,0.35 GL_CUBE_YAW_FIXED_DEG=-35"); [ $(echo $y | wc -w) -lt 5 ] && { sleep 30; y=$(evalone $1 "GL_MU=0.4,0.35 GL_CUBE_YAW_FIXED_DEG=-35"); }
  echo "$(echo $n | awk '{print $1, $2, $3}') $h $(echo $d | awk '{print $1}') $(echo $f | awk '{print $1}') $(echo $y | awk '{print $1}')"
}
score() { echo "$1" | awk '{ if (NF >= 11) { ps = $7 / 3.0; if (ps > 1) ps = 1; pm = $8 / 0.5; if (pm > 1) pm = 1; printf "%.4f", ($1 + $4 + $9 + $10 + $11) / 5.0 - 0.005 * ($2 + $3 + $5 + $6) + 0.025 * ps + 0.05 * pm } }'; }
sweep() {  # 288-env checks of a new best (DZ given: v46_big_cmp.sh sets GL_TABLE_DZ_FIXED=${DZ:-0.0})
  for c in "GL_CUBE_YAW_FIXED_DEG=-35" "GL_CUBE_YAW_FIXED_DEG=45" "GL_CUBE_YAW_DEG=45 GL_KP_SCALE=2.0,2.0" "GL_CUBE_YAW_DEG=45 GL_ACT_DELAY=4" "GL_CUBE_YAW_DEG=45 GL_CUBE_MASS=0.05,0.05"; do
    echo "SWEEP $1 $c $(flock -o $LOCK env DZ=0.0 $c GL_ROBOT=v3 GL_OBS_YAW=1 GL_OBS_Q_TARGET=1 GL_RET_Q_DEG=0,0,7.2,90.0,0,-2.6,0 GL_SERVO_ARM=5,5 GL_FINGER_EFFORT=0.177,0.177 EVAL_YAW_LO=1 GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 $CLOSE GL_GRID_SET=v65f GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=0 ROLL=12 NE=288 bash artifacts/claude_scripts/v46_big_cmp.sh $1 | grep -E '^BIG' | sed -E 's/^BIG [^ ]+ //')"
  done
}
RUN=2026-11-10_06-22-37_v74b_r8; CK=model_25150.pt
# the start under the new conditions, twice, beside round 1 training (training + one eval at a time)
( v1=$(evalck $L/$RUN/$CK); s1=$(score "$v1"); echo "BASE1 $RUN/$CK $v1 -> $s1"
  v2=$(evalck $L/$RUN/$CK); s2=$(score "$v2"); echo "BASE2 $RUN/$CK $v2 -> $s2"
  echo "$(python3 -c "print('%.4f' % ((float('${s1:-0}') + float('${s2:-0}')) / 2))") $RUN/$CK" > $B; echo "BASE_MEAN $(cat $B)" ) &
BPID=$!
for r in $(seq 1 40); do
  tag=v75m_r$r
  echo "== ROUND $r from $RUN/$CK"
  SEED=$((7700 + r)) env $COMMON $TRAIN GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 bash artifacts/claude_scripts/gl_cycle_nr.sh $RUN $CK $tag 200 | grep -E "gl_lifted|HUNG|Traceback" | sed 's/.*gl_lifted:=\([0-9.]*\).*gl_rot_deg:=\([0-9.]*\).*@\([0-9]*\)/TRAIN gl_lifted \1 rot \2 @\3/'
  [ $r -eq 1 ] && wait $BPID
  D=$(ls -d $L/*_$tag | tail -1); rb=-1; rk=""
  for f in $(ls $D/model_*.pt | sort -V | awk 'NR%4==0'); do
    v=$(evalck $f); [ $(echo $v | wc -w) -lt 10 ] && { echo "RETRY_EVAL $f"; sleep 30; v=$(evalck $f); }; echo "EVAL $(basename $D)/$(basename $f) nom(held tilt yaw) harsh(held tilt yaw proxsum proxmin) dzm1 mu03 yaw-35 (held) $v"
    s=$(score "$v"); [ -z "$s" ] && continue
    if python3 -c "import sys; sys.exit(0 if float('$s')>float('$rb') else 1)"; then rb=$s; rk=$(basename $f); fi
  done
  [ -z "$rk" ] && rk=$(ls $D/model_*.pt | sort -V | tail -1 | xargs basename)
  echo "BEST_OF $tag $(basename $D)/$rk $rb"
  if python3 -c "import sys; sys.exit(0 if float('$rb')>float('$(cut -d' ' -f1 $B)') else 1)"; then
    v=$(evalck $D/$rk); s2=$(score "$v"); ra=$(python3 -c "print('%.4f' % ((float('$rb') + float('${s2:-0}')) / 2))")
    echo "RECHECK $(basename $D)/$rk $v -> $s2, mean $ra"
    if python3 -c "import sys; sys.exit(0 if float('$ra')>float('$(cut -d' ' -f1 $B)') else 1)"; then
      echo "$ra $(basename $D)/$rk" > $B; echo "NEW_BEST $ra $(basename $D)/$rk"; sweep $(basename $D)/$rk
    fi
  fi
  RUN=$(basename $D); CK=$(ls $D/model_*.pt | sort -V | tail -1 | xargs basename)  # rounds accumulate
done
echo "DONE best $(cat $B)"
