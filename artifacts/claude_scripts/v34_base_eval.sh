#!/bin/bash
# v34 baseline: current raise (v25 model_1197) and grasp (v28f_ik model_7075) at table dz -3 / 0 / +3 cm
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
R=logs/rsl_rl/jax_v2_1006_raise_cube_v25/2026-11-01_21-25-47_v25_cube/model_1197.pt
G=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-02_23-47-13_v28f_ik/model_7075.pt
for dz in -0.03 0.0 0.03; do
  timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v34_raise_eval.py $R $dz > $S/v34b_raise_$dz.log 2>&1; grep RAISE_EVAL $S/v34b_raise_$dz.log
  GL_TABLE_DZ_FIXED=$dz GL_LIFT_IK=1 GL_IK_PITCH=0.3 GL_IK_VZ=0.035 GL_RETURN_L=1 GL_WRIST_RANGE=0.4 timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v34_eval.py $G 0 > $S/v34b_grasp_$dz.log 2>&1
  echo "GRASP dz $dz"; grep -E "^Z|^ALL|^RES|Error|Traceback" $S/v34b_grasp_$dz.log
done
