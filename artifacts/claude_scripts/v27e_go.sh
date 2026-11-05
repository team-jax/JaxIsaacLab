#!/bin/bash
cd /home/jeongwoo/isaacsim/JaxIsaacLab
sed -i "s|^R=.*; C=\$1; CAM=\$2|R=logs/rsl_rl/jax_v2_1006_raise_cube_v25/2026-11-01_21-25-47_v25_cube/model_1197.pt; C=\$1; CAM=\$2|" artifacts/claude_scripts/v27_montage.sh
echo "== MONTAGE v25 raise + v27c 3425"; bash artifacts/claude_scripts/v27_montage.sh logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-02_00-31-59_v27c/model_3425.pt front
echo "== ROUND v27e"; bash artifacts/claude_scripts/v27c_round.sh 2026-11-01_21-25-47_v25_cube model_1197.pt 2026-11-02_00-31-59_v27c model_3425.pt v27e 0 400 0.75
