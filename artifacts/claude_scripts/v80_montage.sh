#!/bin/bash
# v80 (user 2026-11-11 "영상좀"): as v79_montage.sh under the v80c conditions (pre-grasp 1 cm lower floored at -1 cm, the
# L-return shoulder-yaw turn), from COMMON of v80c_full.sh.
# deploy-style reach montage (as v73_final_video.sh part 1: 9 cube spots x 8 envs, front camera, 3x3) of a grasp-lift checkpoint
# under the v77c conditions (motor torque limits, finger + cube friction 2.0, max grip on the L return, target-angle inputs,
# servo-emulated arm, URDF_1109, shoulder -> table 47.25 cm). The training loop ($2 = its bash PID) is paused (SIGSTOP) while
# nothing trains, the whole montage holds the eval lock (flock -o), then the loop resumes. $1 = checkpoint.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
LOCK=$SP/isaac_eval.lock; C=$1; LOOP=$2
eval "$(grep -E '^(CLOSE|COMMON)=' artifacts/claude_scripts/v80c_full.sh)"
while pgrep -f "python3 scripts/reinforcement_learning/rsl_[r]l/train[.]py" > /dev/null; do sleep 15; done
[ -n "$LOOP" ] && kill -STOP $LOOP && echo "LOOP_PAUSED $LOOP"
sleep 2; pgrep -f "python3 scripts/reinforcement_learning/rsl_[r]l/train[.]py" > /dev/null && { echo "TRAINING_RUNNING abort"; [ -n "$LOOP" ] && kill -CONT $LOOP; exit 1; }
flock -o $LOCK env $COMMON GL_SERVO_ARM=5,5 GL_FINGER_EFFORT=0.177,0.177 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0 \
  RAISE_GRID_SET=v65r REACH_FINAL_GRID_SET=v65f CHAIN_GRAV_COMP=1 CHAIN_GRAV_FULL=1 TEST_ARM_GAIN_SCALE=5 \
  RAISE_POL=nominal RETL=1 STEPS=660 LIFTIK=1 IKP= IKVZ=0.035 IKVXY=0.005 NE=8 \
  SPOTS="0.02,-0.28 0.02,0.0 0.02,0.28 -0.06,-0.20 -0.06,0.0 -0.06,0.20 -0.14,-0.08 -0.14,0.0 -0.14,0.06" TAGSUF=_v80c \
  bash artifacts/claude_scripts/v3_montage.sh $C front 2>&1 | grep -E "lifted|SKIP|MONTAGE|Traceback"
[ -n "$LOOP" ] && kill -CONT $LOOP && echo "LOOP_RESUMED $LOOP"
echo V80_MONTAGE_DONE
