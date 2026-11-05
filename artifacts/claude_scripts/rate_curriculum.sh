#!/bin/bash
# Pre-grasp arm-target speed-limit curriculum (rad/s) with a stronger
# finger-table force cost, from a run/checkpoint; 250 iterations each.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
RUN=$1; CK=$2
for R in 0.7 0.45; do
  NAME=v12_rate${R/./}
  PRE_RESET_NOISE=1.5 PRE_TARGET_RATE=$R PRE_TABLE_FORCE=0.3 bash artifacts/claude_scripts/pre_cycle.sh $RUN $CK $NAME 250 > $S/$NAME.cycle 2>&1
  D=$(ls -d logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/*_$NAME | tail -1); M=$(ls $D/model_*.pt | sort -V | tail -1)
  SU=$(sed -n 3p $S/$NAME.cycle | grep -oE "pre_success:=[0-9.]+" | cut -d= -f2)
  echo "R=$R $M success=$SU"; sed -n 1,3p $S/$NAME.cycle | grep -oE "(pre_success|pre_touching|pre_cube_shift_m|pre_table_force_n|pre_arm_accel_rms):=[-0-9.e]+" | tr '\n' ' '; echo
  if [ -z "$SU" ] || awk "BEGIN{exit !($SU < 0.6)}"; then echo "STOP at R=$R"; break; fi
  RUN=$(basename $D); CK=$(basename $M)
done
