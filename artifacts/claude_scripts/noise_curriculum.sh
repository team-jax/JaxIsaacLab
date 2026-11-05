#!/bin/bash
# Pre-grasp reset-noise curriculum from a run/checkpoint: +/-1.5 then +/-3 deg, 250 iterations each.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
RUN=$1; CK=$2
for N in 1.5 3; do
  NAME=v12_noise${N/./}
  PRE_RESET_NOISE=$N bash artifacts/claude_scripts/pre_cycle.sh $RUN $CK $NAME 250 > $S/$NAME.cycle 2>&1
  R=$(ls -d logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/*_$NAME | tail -1); M=$(ls $R/model_*.pt | sort -V | tail -1)
  SU=$(sed -n 3p $S/$NAME.cycle | grep -oE "pre_success:=[0-9.]+" | cut -d= -f2)
  echo "N=$N $M success=$SU"; sed -n 1,3p $S/$NAME.cycle | grep -oE "(pre_success|pre_touching|pre_cube_shift_m|pre_table_force_n):=[-0-9.e]+" | tr '\n' ' '; echo
  if [ -z "$SU" ] || awk "BEGIN{exit !($SU < 0.6)}"; then echo "STOP at N=$N"; break; fi
  RUN=$(basename $R); CK=$(basename $M)
done
