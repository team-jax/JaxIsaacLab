#!/bin/bash
# v81 probe (user 2026-11-11: "match the table to MuJoCo, no matter what"): a checkpoint in the team layout (GL_TABLE_TEAM=1, teamf grids,
# team heading table) under the final deploy conditions (v80d COMMON: pre-grasp -0.5 cm floored at -1 cm, L-return shoulder-yaw turn),
# 12 cube spots inside the team mask. $1 = checkpoint, $2 = extra env flags.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
eval "$(grep -E '^(COMMON|CLOSE|YAWEVAL|LOCK|SP)=' artifacts/claude_scripts/v81b_full.sh)"
TEAM="GL_TABLE_TEAM=1 GL_GRID_SET=teamf GL_OBS_YAW_THETA=data/team_valid.pt"
export EVAL_CELLS="0.040,0.220;0.140,0.220;0.240,0.220;0.060,0.250;0.160,0.250;0.250,0.250;0.100,0.275;0.180,0.275;0.250,0.275;0.140,0.295;0.200,0.295;0.260,0.295"
flock -o $LOCK env $COMMON $TEAM $YAWEVAL GL_SERVO_ARM=5,5 GL_FINGER_EFFORT=0.177,0.177 NE=${NE:-576} EVAL_S=12 GL_TABLE_DZ_FIXED=0.0 GL_START_NOISE_DEG=0.75 GL_MU=0.4,0.35 $2 timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $1 0 2>&1 | grep -E "^(RES|HELD|DONE |ALL|DIGITS|FRONT cube|SPLIT|FAIL|STALL|Z cube|Traceback|.*Error)"
