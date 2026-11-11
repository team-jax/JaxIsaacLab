#!/bin/bash
# v81 (user 2026-11-11: "match the table position to MuJoCo, no matter what"): the team-jax/IK (JAX_SETUP=real) layout - table in front
# of the robot along base -x (GL_TABLE_TEAM=1: world y 0.17-0.66, x +/-0.355), cube band = the team's reachable spots our side-approach
# grasp covers smoothly (28 of their 41 cells; box dx 0..0.27, dy 0.21..0.31 m + data/team_mask.pt), pre-grasp grids teamf (heading-free
# IK, ik_merge/build_team_grid.py), heading table team_valid.pt. 25550 zero-shot there: held 93.1 %, all three digits 97 % (288 envs).
# Final deploy conditions as v80d/v80f (pre-grasp -0.5 cm floored at -1 cm, L-return shoulder-yaw turn, no tilt cost). From 25550, the
# start scored twice under these evals while round 1 trains. 12 eval spots inside the team mask.
# v80f (2026-11-11): every round at -0.5 cm with the tilt cost went below its start (v80d from 25550: 0.8624, 0.8486; v80e from
# v80c_r1 25625: 0.8639; 25550's start 0.8866, 25625's 0.8759). GL_TILT=30 is the one setting only these runs share with v80c, so it
# is dropped here. From 25550 (start score 0.8866 from v80d, same evals). Else as v80d.
# v80d (user 2026-11-11: "then -0.5 cm"): the 1 cm lower pre-grasp pressed fingers on the table (first 0.5 s of the close 5.2 N
# mean, per-env max ~8 N, up to 20 N; the chain's reach-in end 6-17 N, cube pushed up to 30 mm), so the compromise: evals
# GL_PRE_DZ=-0.005 (floor -0.01 as v80b), training GL_PRE_DZ_RAND=-0.0075,-0.0025 (floor -0.0125). At -0.5 cm with v80c_r1 25625:
# finger 1 not touching 15 % (0 cm: 27 %), all three >= 0.3 N 83 %, table 1.4 N. GL_TILT=30 kept from v80c. From 25550, its two
# scores under these conditions taken while round 1 trains. Else as v80b.
# v80b (2026-11-11): v80 stopped at the start's first score - table 1 cm low + pre-grasp 1 cm lower held 43 % (25550 with the
# table 2 / 3 cm low and no pre-grasp change: 39 / 20 %, the near spots x -12 / -14 stall in the lift at the shoulder pitch,
# shoulder yaw and wrist pitch limits). The pre-grasp's absolute offset is now floored: GL_PRE_DZ_MIN=-0.01 in the evals (1 cm
# lower at the nominal table, unchanged at the 1 cm lower table), -0.015 in training. Else as v80 below.
# v80 (user 2026-11-11: "train so the finger that does not touch touches the cube as much as possible"): finger 1 - the top finger
# at the pre-grasp (the hand's y axis points down, finger 1 link origins 3-4 cm above the cube top) - missed the cube in 26-28 %
# of the held envs at the hand-over (first joint at its 30 deg close cap; caps 45 / 60 deg did not help: it swings over the cube).
# A pre-grasp 1 cm lower took that to 5 % with 25550 unchanged (all three digits >= 0.3 N 73 -> 89 %, held 85 -> 96 %, 288 envs;
# -0.5 cm 11 %, -1.5 cm 4 % / 84 %, -2 cm 51 % held). So, from 25550 (v78d settings):
#  - pre-grasp 1 cm lower: evals GL_PRE_DZ=-0.01, training GL_PRE_DZ_RAND=-0.015,-0.005 (per episode, robust to a height error)
#  - GL_DIGIT_MIN=2: reward on the weakest digit's contact (first + distal, saturating at 0.6 N)
#  - the L-return shoulder-yaw turn GL_RET_YAW=0,45 (2026-11-11 spec, cube front-facing at the hand-over)
#  - score + 0.1 x (share of held envs with all three digits >= 0.3 N at lift_done, nominal case); the start is scored twice
#    under these conditions while round 1 trains
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
CLOSE="GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15"
B=artifacts/claude_scripts/v81_best.txt
LOCK=$SP/isaac_eval.lock
L=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16
export EVAL_CELLS="0.040,0.220;0.140,0.220;0.240,0.220;0.060,0.250;0.160,0.250;0.250,0.250;0.100,0.275;0.180,0.275;0.250,0.275;0.140,0.295;0.200,0.295;0.260,0.295"
COMMON="GL_RETURN_A7=1.0 GL_FINGER_MU=2.0,1.95 GL_CUBE_MU=2.0,1.95 GL_ARM_EFFORT=8,8,8,8,3.92,1.96,0.92,0.177 GL_RL_FORCE=1.2,0.8 GL_DISTAL_SAT=0.6 GL_ROBOT=v3 GL_OBS_YAW=1 GL_OBS_Q_TARGET=1 GL_RET_Q_DEG=0,0,7.2,90.0,0,-2.6,0 GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=0 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 $CLOSE GL_RET_YAW=0,45 GL_PRE_DZ=-0.005 GL_PRE_DZ_MIN=-0.01 GL_TABLE_TEAM=1 GL_GRID_SET=teamf GL_OBS_YAW_THETA=data/team_valid.pt"
TRAIN="GL_FINGER_MU_RAND=1.5,2.5 GL_CUBE_MU_RAND=1.5,2.5 GL_FINGER_EFFORT=0.13,0.22 GL_SERVO_ARM=2,10 GL_ACT_DELAY=0,4 GL_MU_RAND=0.25,0.5 GL_KP_SCALE=0.6,2.0 GL_CUBE_MASS=0.03,0.06 GL_OBS_CUBE_NOISE=0.002,0.005 GL_PROX=2 GL_PROX_MIN=4 GL_PROX_SAT=0.25 GL_RET_FREE=1 GL_ROT=30 GL_TABLE_DZ=0.01 GL_START_NOISE_DEG=0.75 GL_CUBE_BOX=0.0,0.27,0.21,0.31 GL_CUBE_MASK=data/team_mask.pt GL_HARSH_FRAC=0.5 GL_TABLE_DZ_MEAN=0.0 GL_CUBE_YAW_HARD=0.2,30,40,1 GL_RISE=8 GL_LATE=8 GL_XY=50 GL_XY_VEL=20 GL_DRAG=300 GL_EXTRA_COST=0 GL_REWARD_SCALE=0.02 GL_TABLE=3 GL_NO_VCLIP=1 GL_LR=1e-5 GL_ENT=0 GL_DISTAL=2 GL_SLIP=200 GL_PROX_NEED_DISTAL=1 GL_ROT_FROM_LIFTOFF=1 GL_PRE_DZ_RAND=-0.0075,-0.0025 GL_PRE_DZ_MIN=-0.0125 GL_DIGIT_MIN=2"
HARSH="GL_ACT_DELAY=2 GL_MU=0.35,0.3 GL_CUBE_MU=1.5,1.45 GL_FINGER_MU=1.5,1.45 GL_CUBE_MASS=0.05,0.05 GL_KP_SCALE=1.5,1.5 GL_OBS_CUBE_NOISE=0.002,0.005"
YAWEVAL="GL_CUBE_YAW_DEG=45 EVAL_YAW_LO=1"
evalone() {  # $1 = checkpoint, $2 = case flags -> "held tilt yaw proxsum proxmin"
  out=$(flock -o $LOCK env $COMMON $YAWEVAL GL_SERVO_ARM=5,5 GL_FINGER_EFFORT=0.177,0.177 NE=576 EVAL_S=12 GL_TABLE_DZ_FIXED=0.0 GL_START_NOISE_DEG=0.75 $2 timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $1 0 2>&1)
  h=$(echo "$out" | grep "^HELD" | awk '{print $NF}')
  t=$(echo "$out" | grep "^SPLIT" | sed -E 's/.*lifting: tilt ([0-9.-]+) yaw ([0-9.-]+).*/\1 \2/')
  a3=$(echo "$out" | grep "^DIGITS_DONE" | sed -E 's/.*all three >= 0.3 N ([0-9.]+).*/\1/'); [ -z "$a3" ] && a3=0
  p=$(echo "$out" | grep "^PROX_DONE" | sed -E 's/.*\[([0-9.e-]+), ([0-9.e-]+), ([0-9.e-]+)\] sum ([0-9.e-]+).*/\4 \1 \2 \3/' | awk 'NF == 4 { m = $2; if ($3 < m) m = $3; if ($4 < m) m = $4; print $1, m }'); [ -z "$p" ] && p="0 0"
  echo "$h $t $p $a3"
}
evalck() {  # -> "held_nom tilt_nom yaw_nom held_harsh tilt_harsh yaw_harsh proxsum proxmin held_dzm1 held_mu03 held_yaw-35"
  n=$(evalone $1 "GL_MU=0.4,0.35"); [ $(echo $n | wc -w) -lt 5 ] && { sleep 30; n=$(evalone $1 "GL_MU=0.4,0.35"); }
  h=$(evalone $1 "$HARSH"); [ $(echo $h | wc -w) -lt 5 ] && { sleep 30; h=$(evalone $1 "$HARSH"); }
  d=$(evalone $1 "GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=-0.01"); [ $(echo $d | wc -w) -lt 5 ] && { sleep 30; d=$(evalone $1 "GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=-0.01"); }
  f=$(evalone $1 "GL_MU=0.3,0.25 GL_CUBE_MU=1.5,1.45 GL_FINGER_MU=1.5,1.45"); [ $(echo $f | wc -w) -lt 5 ] && { sleep 30; f=$(evalone $1 "GL_MU=0.3,0.25 GL_CUBE_MU=1.5,1.45 GL_FINGER_MU=1.5,1.45"); }
  y=$(evalone $1 "GL_MU=0.4,0.35 GL_CUBE_YAW_FIXED_DEG=-35"); [ $(echo $y | wc -w) -lt 5 ] && { sleep 30; y=$(evalone $1 "GL_MU=0.4,0.35 GL_CUBE_YAW_FIXED_DEG=-35"); }
  echo "$(echo $n | awk '{print $1, $2, $3}') $(echo $h | awk '{print $1, $2, $3, $4, $5}') $(echo $d | awk '{print $1}') $(echo $f | awk '{print $1}') $(echo $y | awk '{print $1}') $(echo $n | awk '{print $6}')"
}
score() { echo "$1" | awk '{ if (NF >= 11) { ps = $7 / 3.0; if (ps > 1) ps = 1; pm = $8 / 0.5; if (pm > 1) pm = 1; a3 = (NF >= 12) ? $12 : 0; printf "%.4f", ($1 + $4 + $9 + $10 + $11) / 5.0 - 0.005 * ($2 + $3 + $5 + $6) + 0.025 * ps + 0.05 * pm + 0.1 * a3 } }'; }
sweep() {  # 288-env checks of a new best (DZ given: v46_big_cmp.sh sets GL_TABLE_DZ_FIXED=${DZ:-0.0})
  for c in "GL_CUBE_YAW_FIXED_DEG=-35" "GL_CUBE_YAW_FIXED_DEG=45" "GL_CUBE_YAW_DEG=45 GL_KP_SCALE=2.0,2.0" "GL_CUBE_YAW_DEG=45 GL_ACT_DELAY=4" "GL_CUBE_YAW_DEG=45 GL_CUBE_MASS=0.05,0.05"; do
    echo "SWEEP $1 $c $(flock -o $LOCK env DZ=0.0 $c GL_RET_YAW=0,45 GL_PRE_DZ=-0.01 GL_PRE_DZ_MIN=-0.01 GL_RETURN_A7=1.0 GL_CUBE_MU=2.0,1.95 GL_FINGER_MU=2.0,1.95 GL_ARM_EFFORT=8,8,8,8,3.92,1.96,0.92,0.177 GL_RL_FORCE=1.2,0.8 GL_DISTAL_SAT=0.6 GL_ROBOT=v3 GL_OBS_YAW=1 GL_OBS_Q_TARGET=1 GL_RET_Q_DEG=0,0,7.2,90.0,0,-2.6,0 GL_SERVO_ARM=5,5 GL_FINGER_EFFORT=0.177,0.177 EVAL_YAW_LO=1 GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 $CLOSE GL_GRID_SET=v65f GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=0 ROLL=12 NE=288 bash artifacts/claude_scripts/v46_big_cmp.sh $1 | grep -E '^BIG' | sed -E 's/^BIG [^ ]+ //')"
  done
}
START=2026-11-11_04-57-42_v78d_r9/model_25550.pt
echo "0 $START" > $B
( v1=$(evalck $L/$START); s1=$(score "$v1"); echo "EVAL_START1 $START $v1 -> $s1"
  v2=$(evalck $L/$START); s2=$(score "$v2"); echo "EVAL_START2 $START $v2 -> $s2"
  ra=$(python3 -c "print('%.4f' % ((float('${s1:-0}') + float('${s2:-0}')) / 2))"); echo "$ra $START" > $B; echo "BASE_MEAN $ra $START" ) &
BPID=$!
for r in $(seq 1 40); do
  tag=v81_r$r
  BEST=$(cut -d' ' -f2 $B); RUN=$(dirname $BEST); CK=$(basename $BEST)  # every round, round 1 included, from the overall best
  echo "== ROUND $r from $RUN/$CK"
  SEED=$((9400 + r)) env $COMMON $TRAIN GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 bash artifacts/claude_scripts/gl_cycle_nr.sh $RUN $CK $tag 200 | grep -E "gl_lifted|HUNG|Traceback" | sed 's/.*gl_lifted:=\([0-9.]*\).*gl_rot_deg:=\([0-9.]*\).*@\([0-9]*\)/TRAIN gl_lifted \1 rot \2 @\3/'
  [ -n "$BPID" ] && { wait $BPID; BPID=""; }
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
      echo "$ra $(basename $D)/$rk" > $B; echo "NEW_BEST $ra $(basename $D)/$rk"  # v81: no front-layout sweep
    fi
  fi
done
echo "DONE best $(cat $B)"
