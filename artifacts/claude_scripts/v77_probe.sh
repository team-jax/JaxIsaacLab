#!/bin/bash
# torque-saturation and failure probe of a checkpoint under the v77b eval conditions (nominal case), via the eval lock
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
eval "$(grep -E '^(CLOSE|COMMON|YAWEVAL)=' artifacts/claude_scripts/v77c_full.sh)"
export EVAL_CELLS="0.02,-0.28;0.02,0.0;0.02,0.28;-0.04,-0.24;-0.04,0.24;-0.08,-0.18;-0.08,0.0;-0.08,0.18;-0.12,-0.14;-0.12,0.12;-0.14,-0.08;-0.14,0.06"
flock -o $SP/isaac_eval.lock env $COMMON $YAWEVAL GL_SERVO_ARM=5,5 GL_FINGER_EFFORT=0.177,0.177 NE=576 EVAL_S=12 GL_TABLE_DZ_FIXED=0.0 GL_START_NOISE_DEG=0.75 GL_MU=0.4,0.35 FINGER_TQ=1 $2 timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $1 0 2>&1 | grep -E "^(HELD|DONE|FAIL|SPLIT|FINGER_TQ|ARM_TQ|ARM_LIMIT|ARM_SAT|FINGER_SAT|THUMB_RET|DROP_AT|PROX|Traceback)|Error"
echo PROBE_DONE
