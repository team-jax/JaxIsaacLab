#!/bin/bash
# fair re-eval with GL_HOLD_DIGIT=1 (576 envs), then v48i from v48h_r5 12550 (V0 = its re-eval)
cd /home/jeongwoo/isaacsim/JaxIsaacLab
for c in "2026-11-05_09-19-56_v48e_r4/model_11950.pt -0.02" "2026-11-05_12-06-54_v48h_r4/model_12475.pt -0.025" "2026-11-05_12-18-42_v48h_r5/model_12550.pt -0.025" "2026-11-05_12-18-42_v48h_r5/model_12650.pt -0.025"; do
  set -- $c; echo "## digit-hold $1 reach $2"
  GL_HOLD_DIGIT=1 GL_REACH_DX=$2 ROLL=12 NE=576 bash artifacts/claude_scripts/v46_big_cmp.sh $1
done
L=$(grep -A99 "## digit-hold 2026-11-05_12-18-42_v48h_r5/model_12550.pt" $0.out 2>/dev/null | grep -m1 "^BIG")
echo CHAIN_EVALS_DONE
