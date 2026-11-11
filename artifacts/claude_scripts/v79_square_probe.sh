#!/bin/bash
# v79 probe (user 2026-11-11: "for MuJoCo, when the cube is turned, guide it to the front in Isaac"): in-hand squareness of a
# checkpoint - the cube's turn in the hand away from square, at lift-off and at lift_done (the MuJoCo hand-over), cube yaw
# U(-45, 45) deg, nominal case of the v78d evals. $1 = checkpoint, $2 = extra env flags (optional)
cd /home/jeongwoo/isaacsim/JaxIsaacLab
eval "$(grep -E '^(COMMON|CLOSE|YAWEVAL|LOCK|SP)=' artifacts/claude_scripts/v78d2_full.sh)"
export EVAL_CELLS="0.02,-0.28;0.02,0.0;0.02,0.28;-0.04,-0.24;-0.04,0.24;-0.08,-0.18;-0.08,0.0;-0.08,0.18;-0.12,-0.14;-0.12,0.12;-0.14,-0.08;-0.14,0.06"
flock -o $LOCK env $COMMON $YAWEVAL GL_SERVO_ARM=5,5 GL_FINGER_EFFORT=0.177,0.177 NE=${NE:-576} EVAL_S=12 GL_TABLE_DZ_FIXED=0.0 GL_START_NOISE_DEG=0.75 GL_MU=0.4,0.35 $2 timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $1 0 2>&1 | grep -E "^(SQUARE|FRONT|PREYAW|HELD|SPLIT|ROT|DONE |ALL|DROP_AT|DROP n|PADS_DONE|DIGITS_DONE|PROX|F1|FINGERQ|INHAND|FRONT_CELL)"
