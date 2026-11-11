#!/bin/bash
# v79 (user 2026-11-11, chose option 3): the L return turns the arm about the vertical (shoulder yaw + alpha, alpha = the
# camera-measured cube turn relative to the hand - 2.1 deg, folded to +/-45 deg; GL_RET_YAW=-2.1,45) so the cube reaches the
# hand-over front-facing. No retraining: the v78d 5-case score (+ recheck) of the deployed 25300 and of 25550 with the turn,
# against their scores without it (25300 0.8240, 25550 0.8357).
eval "$(sed -n 52,78p artifacts/claude_scripts/v78d2_full.sh)"
COMMON="$COMMON GL_RET_YAW=-2.1,45"
for ck in 2026-11-11_04-57-42_v78d_r9/model_25550.pt $(cut -d' ' -f2 artifacts/claude_scripts/v78_best.txt); do
  for rep in 1 2; do
    e=$(evalck $L/$ck); echo "EVAL$rep $ck $e -> $(score "$e")"
  done
done
echo V79_CHECK_DONE
