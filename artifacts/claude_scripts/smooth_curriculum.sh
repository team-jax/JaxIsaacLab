#!/bin/bash
# Pre-grasp target smoothing curriculum from a run/checkpoint: 0.5 -> 0.7 -> 0.85, 250 iterations each.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
RUN=$1; CK=$2
for B in 0.5 0.7 0.85; do
  NAME=v12_sm${B/./}
  PRE_TARGET_SMOOTH=$B bash artifacts/claude_scripts/pre_cycle.sh $RUN $CK $NAME 250 > $S/$NAME.cycle 2>&1
  R=$(ls -d logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/*_$NAME | tail -1); M=$(ls $R/model_*.pt | sort -V | tail -1)
  SU=$(sed -n 3p $S/$NAME.cycle | grep -oE "pre_success:=[0-9.]+" | cut -d= -f2)
  echo "B=$B $M success=$SU"; sed -n 1,3p $S/$NAME.cycle | grep -oE "(pre_success|pre_touching|pre_cube_shift_m|pre_err_m|pre_table_force_n):=[-0-9.e]+" | tr '\n' ' '; echo
  if [ -z "$SU" ] || awk "BEGIN{exit !($SU < 0.6)}"; then echo "STOP at B=$B"; break; fi
  RUN=$(basename $R); CK=$(basename $M)
done
