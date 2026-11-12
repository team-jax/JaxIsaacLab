#!/bin/bash
# deploy check (2026-11-11 night): 25550 under the deploy rules - pre-grasp 0.5 cm lower (floor -1 cm), the same L for all (no shoulder-yaw
# turn), front table layout - the v80d 5-case evals x 2, plus one 576-env probe for finger contact and table force.
eval "$(sed -n "$(grep -n '^cd /home' artifacts/claude_scripts/v80d_full.sh | head -1 | cut -d: -f1),$(grep -n '^score()' artifacts/claude_scripts/v80d_full.sh | cut -d: -f1)p" artifacts/claude_scripts/v80d_full.sh)"
COMMON=${COMMON//GL_RET_YAW=0,45 /}
echo "COMMON has RET_YAW: $(echo $COMMON | grep -c RET_YAW)"
CK=$L/2026-11-11_04-57-42_v78d_r9/model_25550.pt
for rep in 1 2; do e=$(evalck $CK); echo "EVAL$rep 25550 $e -> $(score "$e")"; done
flock -o $LOCK env $COMMON $YAWEVAL GL_SERVO_ARM=5,5 GL_FINGER_EFFORT=0.177,0.177 NE=576 EVAL_S=12 GL_TABLE_DZ_FIXED=0.0 GL_START_NOISE_DEG=0.75 GL_MU=0.4,0.35 timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $CK 0 2>&1 | grep -E "^(HELD|DIGITS|PADS_DONE|RES|SPLIT|DONE_T)" 
echo DEPLOY_CHECK_DONE
