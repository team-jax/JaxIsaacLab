#!/bin/bash
# deploy videos (2026-11-11 night): 25550 + pre-grasp 0.5 cm lower (floor -1 cm), front table layout, the same L for all
cd /home/jeongwoo/isaacsim/JaxIsaacLab
C=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-11_04-57-42_v78d_r9/model_25550.pt
bash artifacts/claude_scripts/final_montage.sh $C
bash artifacts/claude_scripts/final_close.sh $C
echo DEPLOY_VIDEO_DONE
