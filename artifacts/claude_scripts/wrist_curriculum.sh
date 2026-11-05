#!/bin/bash
# Narrow the v16 wrist pitch range in 250-iteration steps from a run/checkpoint.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
RUN=$1; CK=$2
for W in 0.4 0.3 0.2 0.1; do
  NAME=v16_wr${W/./}
  GL_WRIST_RANGE=$W bash artifacts/claude_scripts/gl_cycle.sh $RUN $CK $NAME 250 > $S/$NAME.cycle 2>&1
  R=$(ls -d logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/*_$NAME | tail -1); M=$(ls $R/model_*.pt | sort -V | tail -1)
  L=$(grep -oE "gl_lifted:=[0-9.]+" $S/$NAME.cycle | tail -1 | cut -d= -f2)
  echo "W=$W $M lifted=$L"; grep -E "gl_lifted" $S/$NAME.cycle | tail -1 | grep -oE "(gl_lifted|gl_spawn_xy_mm|gl_rot_deg|gl_hand_rot_deg|gl_rise_mm):=[-0-9.]+" | tr '\n' ' '; echo
  if [ -z "$L" ] || awk "BEGIN{exit !($L < 0.5)}"; then echo "STOP: lift collapsed at W=$W"; break; fi
  RUN=$(basename $R); CK=$(basename $M)
done
