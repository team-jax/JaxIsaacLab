#!/bin/bash
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
B=artifacts/claude_scripts/v45d_best.txt
COMMON="GL_FINGER_RL=1 GL_ROLL_DEG=15 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1"
TRAIN="GL_RET_FREE=1 GL_ROT=30 GL_TABLE_DZ=0.01 GL_HARD_FRAC=0.15 GL_HARD_BOX=-0.08,-0.06,-0.05,0.06 GL_START_NOISE_DEG=0.75 GL_CUBE_BOX=1 GL_RISE=8 GL_LATE=8 GL_XY=50 GL_XY_VEL=20 GL_DRAG=300 GL_EXTRA_COST=0 GL_REWARD_SCALE=0.02 GL_TABLE=3 GL_NO_VCLIP=1 GL_LR=2e-5"
RUN=2026-11-05_00-20-48_v45_r3; CK=model_9200.pt  # v45c: from the v45b round-2 best  # v45: v44_r1 model_8725 with the grip outputs (7-10) reset to the scripted grip
evalck() {  # -> "success tilt_on_table tilt_lifting"
  out=$(env $COMMON GL_MU=0.4,0.35 NE=144 GL_TABLE_DZ_FIXED=0.0 GL_START_NOISE_DEG=0.75 timeout -k 20 600 ./isaaclab.sh -p artifacts/claude_scripts/v41_rot_probe2.py $1 0 2>&1)
  a=$(echo "$out" | grep "^ALL" | awk '{print $NF}'); t=$(echo "$out" | grep "^SPLIT" | sed -E 's/.*on_table\(before 1cm\): tilt ([0-9.-]+) .*lifting: tilt ([0-9.-]+) .*/\1 \2/')
  echo "$a $t"
}
v0="1.0 3.6 3.9"; echo "EVAL start $RUN/$CK success_tilt_table_tilt_lift $v0 (v45c r3 eval)"
echo "$(echo $v0 | awk '{ printf "%.4f", $1 - 0.01 * $3 }') $RUN/$CK" > $B
for r in 1 2 3 4 5 6 7 8; do
  tag=v45d_r$r
  echo "== ROUND $r from $RUN/$CK"
  env $COMMON $TRAIN GL_MU=0.4,0.35 bash artifacts/claude_scripts/gl_cycle_nr.sh $RUN $CK $tag 200 | grep -E "gl_lifted|HUNG|Traceback" | sed 's/.*gl_lifted:=\([0-9.]*\).*gl_rot_deg:=\([0-9.]*\).*@\([0-9]*\)/TRAIN gl_lifted \1 rot \2 @\3/'
  D=$(ls -d logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/*_$tag | tail -1); rb=-1; rk=""
  for f in $(ls $D/model_*.pt | sort -V | awk 'NR%2==0'); do
    v=$(evalck $f); echo "EVAL $(basename $D)/$(basename $f) success_tilt_table_tilt_lift $v"
    s=$(echo $v | awk '{ if (NF >= 3) printf "%.4f", $1 - 0.01 * $3 }'); [ -z "$s" ] && continue
    if python3 -c "import sys; sys.exit(0 if float('$s')>float('$rb') else 1)"; then rb=$s; rk=$(basename $f); fi
  done
  [ -z "$rk" ] && rk=$(ls $D/model_*.pt | sort -V | tail -1 | xargs basename)
  echo "BEST_OF $tag $(basename $D)/$rk $rb"
  if python3 -c "import sys; sys.exit(0 if float('$rb')>float('$(cut -d' ' -f1 $B)') else 1)"; then echo "$rb $(basename $D)/$rk" > $B; echo "NEW_BEST $rb $(basename $D)/$rk"; fi
  RUN=$(basename $D); CK=$rk
done
echo "DONE best $(cat $B)"
