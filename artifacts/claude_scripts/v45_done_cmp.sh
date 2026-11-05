#!/bin/bash
# v45: compare checkpoints on lift success, lift_done (L reached and held) and cube tilt; 144 envs, 8 s, friction 0.4
cd /home/jeongwoo/isaacsim/JaxIsaacLab
L=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16
for f in "$@"; do
  out=$(EVAL_S=${EVAL_S:-12} NE=144 GL_FINGER_RL=1 GL_ROLL_DEG=15 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_MU=0.4,0.35 GL_SQUEEZE=6,0 GL_RETURN_L=1 GL_TABLE_REL=1 GL_TABLE_DZ_FIXED=0.0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_START_NOISE_DEG=0.75 timeout -k 20 700 ./isaaclab.sh -p artifacts/claude_scripts/v45_eval_done.py $L/$f 0 2>&1)
  echo "CMP $f lifted $(echo "$out" | grep '^ALL' | awk '{print $NF}') done $(echo "$out" | grep '^DONE lift' | awk '{print $NF}') $(echo "$out" | grep '^DONE_T' | awk '{print "t_done", $3, "p90", $5}') $(echo "$out" | grep '^SPLIT' | sed -E 's/.*lifting: tilt ([0-9.]+).*/tilt_lift \1/')"
done
