#!/bin/bash
cd /home/jeongwoo/isaacsim/JaxIsaacLab
F=source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/v2_front_pregrasp.py
RUN=$1; CK=$2
for rb in 160 165 170; do
  sed -i -E "s/math.radians\(-[0-9.]+\)\)  # RB_CURRICULUM/math.radians(-$rb.0))  # RB_CURRICULUM/" $F
  name=v12_c55_rb$rb; echo "=== rb -$rb from $RUN/$CK"
  if [ $rb = 170 ]; then bash artifacts/claude_scripts/rb170_chain.sh $RUN $CK $name; else bash artifacts/claude_scripts/pre_cycle.sh $RUN $CK $name 250; fi
  R=$(ls -d logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/*_$name | tail -1); RUN=$(basename $R); CK=$(basename $(ls $R/model_*.pt | sort -V | tail -1))
done
echo "=== done $RUN/$CK"
