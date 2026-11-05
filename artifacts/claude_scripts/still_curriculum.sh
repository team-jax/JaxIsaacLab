#!/bin/bash
# v16 hold-still curriculum (arm speed cost per (rad/s)^2, still bonus), 250 iterations each.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
RUN=$1; CK=$2
for P in "2 1" "5 2"; do
  set -- $P; NAME=v16_hold$1
  GL_WRIST_RANGE=0.4 GL_ARM_VEL=$1 GL_STILL=$2 bash artifacts/claude_scripts/gl_cycle.sh $RUN $CK $NAME 250 > $S/$NAME.cycle 2>&1
  D=$(ls -d logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/*_$NAME | tail -1); M=$(ls $D/model_*.pt | sort -V | tail -1)
  LI=$(sed -n 3p $S/$NAME.cycle | grep -oE "gl_lifted=[0-9.]+" | cut -d= -f2)
  echo "$NAME $M lifted=$LI"; sed -n 1,3p $S/$NAME.cycle | grep -oE "(gl_lifted|gl_rise_mm|gl_arm_speed_deg_s|gl_still|gl_spawn_xy_mm|gl_rot_deg)=[-0-9.e]+|@[0-9]+" | tr '\n' ' '; echo
  if [ -z "$LI" ] || awk "BEGIN{exit !($LI < 0.6)}"; then echo "STOP at $NAME"; break; fi
  RUN=$(basename $D); CK=$(basename $M)
done
