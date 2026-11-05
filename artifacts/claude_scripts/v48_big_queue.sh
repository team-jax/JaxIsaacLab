#!/bin/bash
# v48: 576-env evals of the top first-joint candidates, one per training window of the running v48e rounds (two Isaac
# evals at once failed; training + one eval works). Waits for each next round's training to start.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
r=6
for c in "2026-11-05_09-19-56_v48e_r4/model_11950.pt 12" "2026-11-05_08-44-16_v48e_r1/model_11875.pt 12" "2026-11-05_09-19-56_v48e_r4/model_11900.pt 12" "2026-11-05_06-45-23_v48b2_r3/model_11625.pt 15"; do
  set -- $c
  until grep -q "Learning iteration" $SP/v48e_r$r.log 2>/dev/null; do sleep 15; done
  sleep 30
  echo "## $1 roll $2 (during v48e_r$r training)"
  GL_REACH_DX=-0.02 ROLL=$2 NE=576 bash artifacts/claude_scripts/v46_big_cmp.sh $1
  r=$((r + 1))
done
echo QUEUE_DONE
