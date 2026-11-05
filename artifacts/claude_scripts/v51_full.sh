#!/bin/bash
# v51 (user 2026-11-05 "now you train", after the deploy package): sim-to-real robustness beyond v50, from v50_r3 13075.
# 13075's weak spots (288 envs, held to 12 s): drive gains x1.5 88.2% (inside its x0.6-1.5 training range, at the edge),
# everything at once (delay 2, friction 0.35/0.3, 80 g, gains x1.3, camera 2 mm +/-5 mm) 81.6%, friction 0.3 92.7%,
# 100 g 93.8%. A real servo likely sags less than the sim PD (kp 100) whose sag the scripted offsets and the gravity
# hold compensate with the nominal kp, so a stiffer drive leaves the hand off its planned pose.
# Changes vs v50: drive gain range x0.6-1.5 -> x0.6-2.0; checkpoints ranked under a combined case at 576 envs:
# delay 2 steps + friction 0.35/0.3 + cube 80 g + gains x1.5 + camera 2 mm / +/-5 mm (v50 ranked at delay 2 + friction).
# Same score: held - 0.01 x (tilt + yaw) + 0.025 x min(first-joint sum / 3 N, 1) + 0.05 x min(weakest / 0.5 N, 1).
# Rounds of 200 iterations continue from each round's best (new setting); at each new overall best a 288-env sweep
# (nominal, gains x1.5 and x2.0, friction 0.3, 100 g, delay 4) checks that nothing else got worse.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
B=artifacts/claude_scripts/v51_best.txt
COMMON="GL_GRID_SET=v49 GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1"
TRAIN="GL_ACT_DELAY=0,4 GL_MU_RAND=0.25,0.5 GL_KP_SCALE=0.6,2.0 GL_CUBE_MASS=0.04,0.10 GL_OBS_CUBE_NOISE=0.002,0.005 GL_PROX=1 GL_PROX_MIN=2 GL_RET_FREE=1 GL_ROT=30 GL_TABLE_DZ=0.01 GL_HARD_FRAC=0.15 GL_HARD_BOX=-0.08,-0.06,-0.05,0.06 GL_START_NOISE_DEG=0.75 GL_CUBE_BOX=1 GL_RISE=8 GL_LATE=8 GL_XY=50 GL_XY_VEL=20 GL_DRAG=300 GL_EXTRA_COST=0 GL_REWARD_SCALE=0.02 GL_TABLE=3 GL_NO_VCLIP=1 GL_LR=2e-5"
HARSH="GL_ACT_DELAY=2 GL_MU=0.35,0.3 GL_CUBE_MASS=0.08,0.08 GL_KP_SCALE=1.5,1.5 GL_OBS_CUBE_NOISE=0.002,0.005"
RUN=2026-11-05_16-20-11_v50_r3; CK=model_13075.pt
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
    echo "SWEEP $1 $c $(env $c GL_GRID_SET=v49 GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 ROLL=12 NE=288 bash artifacts/claude_scripts/v46_big_cmp.sh $1 | grep -E '^BIG' | sed -E 's/^BIG [^ ]+ //')"
  done
}
v0=$(evalck logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/$RUN/$CK); [ $(echo $v0 | wc -w) -lt 7 ] && { sleep 30; v0=$(evalck logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/$RUN/$CK); }
echo "EVAL start $RUN/$CK held_done_lifted_tilt_yaw_proxdone-sum_proxdone-min $v0"
echo "$(score "$v0") $RUN/$CK" > $B
sweep $RUN/$CK
for r in 1 2 3 4 5 6 7 8; do
  tag=v51_r$r
  echo "== ROUND $r from $RUN/$CK"
  SEED=$((5100 + r)) env $COMMON $TRAIN GL_MU=0.4,0.35 bash artifacts/claude_scripts/gl_cycle_nr.sh $RUN $CK $tag 200 | grep -E "gl_lifted|HUNG|Traceback" | sed 's/.*gl_lifted:=\([0-9.]*\).*gl_rot_deg:=\([0-9.]*\).*@\([0-9]*\)/TRAIN gl_lifted \1 rot \2 @\3/'
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
  RUN=$(basename $D); CK=$rk  # new setting: keep going from this round's best
done
echo "DONE best $(cat $B)"
