#!/bin/bash
# usage: height_curriculum.sh LOAD_RUN CHECKPOINT  -- runs offsets 0.03 0.02 0.01 0.00, 250 it each
F=/home/jeongwoo/isaacsim/JaxIsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/v2_front_pregrasp.py
RUN=$1; CK=$2
for off in 0.03 0.02 0.01 0.00; do
  sed -i -E "s/pre_height_offset_m = [0-9.]+  # CURRICULUM/pre_height_offset_m = $off  # CURRICULUM/" $F
  name=v12_c55_h${off/0./}
  echo "=== offset $off from $RUN/$CK -> $name"
  bash /home/jeongwoo/isaacsim/JaxIsaacLab/artifacts/claude_scripts/pre_cycle.sh $RUN $CK $name 250
  R=$(ls -d /home/jeongwoo/isaacsim/JaxIsaacLab/logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/*_$name | tail -1)
  RUN=$(basename $R); CK=$(basename $(ls $R/model_*.pt | sort -V | tail -1))
done
echo "=== done $RUN/$CK"
