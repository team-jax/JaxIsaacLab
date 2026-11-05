#!/bin/bash
# v48i: v48h (2.5 cm) with "holding" judged per digit as first + distal pad force (GL_HOLD_DIGIT=1, in the reward's
# hold/hold3 terms that gate the rise rewards, and in lift_done). The distal-only check read deep grasps as not held
# (v48h 12500: lifted 99.1%, lift_done 95.5%), so training kept the force on the distal pads. From v48h_r5 12550.
# v48h: one more depth step, 2.5 cm (GL_REACH_DX=-0.025), first-joint reward back to GL_PROX=1 GL_PROX_MIN=2 (MIN=6 in v48g
# did not stop the first joints letting go). Untrained 11950 at 2.5 cm (288 envs): held 96.9%, tilt 2.0, yaw 4.1, first
# joints at lift_done 0.14 / 0.73 / 0.27 N (2 cm: 0.13 / 0.81 / 0.16), thumb first joint while lifting 0.39 N. From 11950.
# v48g: v48f with the weakest-first-joint reward 2 -> 6 (GL_PROX_MIN=6). v48f r1-r3 from 11950 (GL_PROX_MIN=2) kept held/tilt
# but let the first joints go every round (lift_done sum 1.10 -> 0.80-0.88 N, weakest 0.13 -> 0.04-0.10 N). From 11950.
# v48f: v48e continued from the final pick v48e_r4 11950 with 576-env evals (288-env picks were noise-led: v48e r6 11975
# scored 0.991 at 288 envs but held 97.4% at 576; 11950 held 99.7%, first joints at lift_done 0.13 / 0.81 / 0.16 N).
# v48e: v48d with a balanced first-joint reward (GL_PROX=1 GL_PROX_MIN=2: + 2 x 3 x the weakest pad) and ranked by the
# first joints AT LIFT_DONE (the hand-over): score = held - 0.01 x (tilt + yaw) + 0.025 x min(sum / 1.5 N, 1)
# + 0.05 x min(weakest / 0.2 N, 1). At lift_done v48c 11750 had 0.13 / 0.45 / 0.34 N, v48d 11925 0.17 / 1.04 / 0.00 N.
# From v48c_r1 11750 (roll 12, 2 cm).
# v48d: v48c with the first-joint reward x3 (GL_PROX=3). v48c r1 (GL_PROX=1) let the first joints go again as it trained
# (11650 0.17/0.55/0.42 N -> 11800 0.01/0.53/0.19 N): the term was ~0.33 per step against ~5.7 for the rise, and finger 2
# alone saturated its pad. From v48c_r1 11750 (held 99.7%, tilt 2.0, yaw 4.1, first joints 0.08/0.67/0.40 N).
# v48c: v48b3 (2 cm deeper + first-joint reward) with the pre-grasp hand roll 15 -> 12 deg (GL_ROLL_DEG=12): untrained
# 11625 at 12 deg: first joints 0.22 / 0.89 / 0.45 N (15 deg: 0.05 / 1.11 / 0.35; 10 deg: 0.14 / 0.55 / 0.45 with tilt 3.6),
# lift tilt 2.2, yaw 4.3, held 97.2% (144 envs). All three first joints on the cube. From v48b2_r3 11625.
# v48b3: v48b2 resumed from v48b2_r3 11625 (2 cm: held 99.3%, tilt 2.2, yaw 4.3, first joints 0.05/1.11/0.35 N) after the
# user's pause for the chain render (2026-11-05).
# v48b2: v48b resumed from v48b_r1 11100 (2 cm: held 95.5%, first joints 0.03/1.16/0.35 N) after a GPU hiccup (r1 evals
# empty, r2-r5 hung at startup); an empty eval is retried once.
# v48b: v48 one step deeper, 2 cm (GL_REACH_DX=-0.02), from v48_r3 10925 (1.5 cm: held 100%, first joints 0.04/0.79/0.05 N;
# sideways +/-5 mm and roll 10/20 deg did not bring finger 1 / thumb first joints on; at 2 cm v47_r3 9975 had 0.39/1.28/0.58 N).
# v48 (user: "the first joint should sit fully on the cube"): v47b (hand 1.5 cm deeper) plus a first-joint pad reward
# (GL_PROX=1: per proximal pad min(F / 0.5 N, 1) while the distal grip holds); v47b_r3 10675 had let them go to
# 0.02 / 0.29 / 0.02 N (held 98.3%). Score = held - 0.01 x (lift tilt + lift yaw) + 0.05 x min(first-joint N sum / 1.5, 1).
# v47b: v47 one step deeper, 1.5 cm (GL_REACH_DX=-0.015); v47_r3 9975 untrained there: held 89.2%, first-joint pads
# 0.2-0.36 N (1 cm: ~0.05 N; 2 cm: held 67%, pads 0.4-1.3 N, table yaw 17.5 deg). From the v47 overall best.
# v47 (user 2026-11-04: "grip the cube deeper", "the first joint should sit on the cube"): the pre-grasp hand 1 cm
# deeper (GL_REACH_DX=-0.01, away from the robot). Untrained 9500 there: held 78.5% (0 cm: 98.3%), first-joint pads ~0 N;
# 2 cm: held 35%, first-joint pads 0.3-0.8 N, cube yaw 12 deg. The v35 deep-grasp training collapses (2026-11-04)
# predate the v38 latch-bug fix. Score = held - 0.01 x (lift tilt + lift yaw) (deg), 288-env evals. From v46_r2 9500.
# v46b: v46 where a round that does not beat the overall best is followed by a round from the overall best (v46 r4 from
# r3 9575 peaked at 0.944 < 0.966 and r5 would have resumed from that worse checkpoint). Starts from v46_r3 model_9575.
# v46: v45d (learned grip, GL_RET_FREE) ranked by the hand-over signal instead of the 6.5 s lift: each checkpoint runs
# 12 s (144 envs, friction 0.4) and scores held (lift_done and the cube still up at 12 s) - 0.01 x lift tilt (deg). v45d_r2 model_9350 had done 86.8%
# (v45_r3 9200: 70.1%) with lift 98.6%. Starts from v45d_r2 model_9350.
# v46 also triggers the scripted L return at 4 cm (GL_RET_RISE=0.04, was 5 cm): the IK lift stalled at 4.5-5 cm with the
# elbow at its 105 deg limit (hand orientation held), so those envs never returned (v45d_r2 9350 at x -3 cm: done 44%).
# v45d: v45c with no cube/hand motion costs during the scripted L return (GL_RET_FREE; the policy had learned to stop
# at ~3 cm, under the 5 cm return trigger, to dodge the return's rotation/x-y costs), from v45_r3 model_9200 (100%).
# v45c: v45b with lr 5e-5 -> 2e-5, 200-iteration rounds and evals every 50 iterations (v45 r2 stopped lifting
# at the end of its 300 iterations: 79% -> 6% while the grip stayed). v45b: rounds ranked by success - 0.01 x lift tilt (deg), resumed from r1 model_9000.
# v45 (user 2026-11-04: "option 3, train the grasp"): the policy sets the finger-side and thumb squeeze (actions 7, 9)
# and the distal-joint curl (actions 8, 10; GL_FINGER_RL=1), the touch-then-squeeze force loop stays. Starts from
# v44_r1 model_8725 with those four outputs set to the old fixed grip (6 N each side, straight distal joints).
# v44: v43 with the pre-grasp hand rolled +15 deg (GL_ROLL_DEG; the pinch was tilted, so the cube settled 13 deg
# rolled in the grip as it left the table: +15 -> 5.5 deg, success 96% untrained). v43: v42 with the start pose held by gravity-torque targets (GL_GRAV_COMP) and the IK lift tracking a grasp-point
# target (GL_IK_POS=2): the arm had drifted up 4-11 mm and the policy pressed the cube down against it, rolling it.
# v42 (user 2026-11-04: "the cube angle changing while lifting is annoying, fix it"): the cube tilted 12-14 deg on the
# table while the squeeze built up (the policy pressed the hand down while closing) and the hand pitch added ~7 deg.
# Train in the new layout (robot body front -> table edge 10 cm: GL_TABLE_EDGE=0.225) with the hand pitch action off
# and the cube rotation cost 8 -> 30 per rad; friction 0.4, squeeze 6 N, table +/-1 cm, table-relative cube height,
# L return. Rounds of 300 it from the previous round's best (by lift success); evals of every 4th checkpoint
# (144 envs, friction 0.4, table dz 0) report lift success and cube tilt.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
B=artifacts/claude_scripts/v48i_best.txt
COMMON="GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1"
TRAIN="GL_PROX=1 GL_PROX_MIN=2 GL_RET_FREE=1 GL_ROT=30 GL_TABLE_DZ=0.01 GL_HARD_FRAC=0.15 GL_HARD_BOX=-0.08,-0.06,-0.05,0.06 GL_START_NOISE_DEG=0.75 GL_CUBE_BOX=1 GL_RISE=8 GL_LATE=8 GL_XY=50 GL_XY_VEL=20 GL_DRAG=300 GL_EXTRA_COST=0 GL_REWARD_SCALE=0.02 GL_TABLE=3 GL_NO_VCLIP=1 GL_LR=2e-5"
RUN=2026-11-05_12-18-42_v48h_r5; CK=model_12550.pt  # v45c: from the v45b round-2 best  # v45: v44_r1 model_8725 with the grip outputs (7-10) reset to the scripted grip
evalck() {  # -> "held done lifted tilt_lifting yaw_lifting"
  out=$(env $COMMON GL_MU=0.4,0.35 NE=576 EVAL_S=12 GL_TABLE_DZ_FIXED=0.0 GL_START_NOISE_DEG=0.75 timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $1 0 2>&1)
  h=$(echo "$out" | grep "^HELD" | awk '{print $NF}'); d=$(echo "$out" | grep "^DONE lift" | awk '{print $NF}'); a=$(echo "$out" | grep "^ALL" | awk '{print $NF}')
  t=$(echo "$out" | grep "^SPLIT" | sed -E 's/.*lifting: tilt ([0-9.-]+) yaw ([0-9.-]+).*/\1 \2/')
  p=$(echo "$out" | grep "^PROX_DONE" | sed -E 's/.*\[([0-9.e-]+), ([0-9.e-]+), ([0-9.e-]+)\] sum ([0-9.e-]+).*/\4 \1 \2 \3/' | awk 'NF == 4 { m = $2; if ($3 < m) m = $3; if ($4 < m) m = $4; print $1, m }'); [ -z "$p" ] && p="0 0"
  echo "$h $d $a $t $p"
}
v0="${V0:-0.986 0.988 0.993 2.4 3.6 1.301 0.166}"; echo "EVAL start $RUN/$CK held_done_lifted_tilt_yaw_proxdone-sum_proxdone-min $v0 (576-env eval)"
echo "$(echo $v0 | awk '{ ps = $6 / 1.5; if (ps > 1) ps = 1; pm = $7 / 0.2; if (pm > 1) pm = 1; printf "%.4f", $1 - 0.01 * ($4 + $5) + 0.025 * ps + 0.05 * pm }') $RUN/$CK" > $B
for r in 1 2 3 4 5 6 7 8; do
  tag=v48i_r$r
  echo "== ROUND $r from $RUN/$CK"
  SEED=$((1500 + r)) env $COMMON $TRAIN GL_MU=0.4,0.35 bash artifacts/claude_scripts/gl_cycle_nr.sh $RUN $CK $tag 200 | grep -E "gl_lifted|HUNG|Traceback" | sed 's/.*gl_lifted:=\([0-9.]*\).*gl_rot_deg:=\([0-9.]*\).*@\([0-9]*\)/TRAIN gl_lifted \1 rot \2 @\3/'
  D=$(ls -d logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/*_$tag | tail -1); rb=-1; rk=""
  for f in $(ls $D/model_*.pt | sort -V | awk 'NR%2==0'); do
    v=$(evalck $f); [ $(echo $v | wc -w) -lt 7 ] && { echo "RETRY_EVAL $f"; sleep 30; v=$(evalck $f); }; echo "EVAL $(basename $D)/$(basename $f) held_done_lifted_tilt_yaw_proxdone-sum_proxdone-min $v"
    s=$(echo $v | awk '{ if (NF >= 7) { ps = $6 / 1.5; if (ps > 1) ps = 1; pm = $7 / 0.2; if (pm > 1) pm = 1; printf "%.4f", $1 - 0.01 * ($4 + $5) + 0.025 * ps + 0.05 * pm } }'); [ -z "$s" ] && continue
    if python3 -c "import sys; sys.exit(0 if float('$s')>float('$rb') else 1)"; then rb=$s; rk=$(basename $f); fi
  done
  [ -z "$rk" ] && rk=$(ls $D/model_*.pt | sort -V | tail -1 | xargs basename)
  echo "BEST_OF $tag $(basename $D)/$rk $rb"
  if python3 -c "import sys; sys.exit(0 if float('$rb')>float('$(cut -d' ' -f1 $B)') else 1)"; then echo "$rb $(basename $D)/$rk" > $B; echo "NEW_BEST $rb $(basename $D)/$rk"; fi
  # v46b: continue from the overall best (the file holds "score run/ckpt")
  RUN=$(cut -d' ' -f2 $B | cut -d/ -f1); CK=$(cut -d' ' -f2 $B | cut -d/ -f2)
done
echo "DONE best $(cat $B)"
