#!/bin/bash
# v48 wrap-up: after v48e ends, 576-env evals of the remaining candidates (11625 roll 15, v48e overall best roll 12),
# then robustness evals of v48e_r4 11950 (table -1 / +1 cm, friction 0.35/0.3), 288 envs each, all 12 s, 2 cm reach.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
while pgrep -f "v48e_full.sh" >/dev/null; do sleep 20; done; sleep 15
B=$(cut -d' ' -f2 artifacts/claude_scripts/v48e_best.txt)
echo "## v48e overall best: $B"
for c in "2026-11-05_06-45-23_v48b2_r3/model_11625.pt 15" "$B 12"; do
  set -- $c; echo "## 576 $1 roll $2"
  GL_REACH_DX=-0.02 ROLL=$2 NE=576 bash artifacts/claude_scripts/v46_big_cmp.sh $1
done
K=2026-11-05_09-19-56_v48e_r4/model_11950.pt
for c in "DZ=-0.01" "DZ=0.01" "MU=0.35,0.3"; do
  echo "## robust 11950 $c"
  env $c GL_REACH_DX=-0.02 ROLL=12 NE=288 bash artifacts/claude_scripts/v46_big_cmp.sh $K
done
echo FINAL_DONE
