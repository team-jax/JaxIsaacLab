#!/bin/bash
# deploy videos of v78_r3/model_25300 under the deploy conditions (v77c/v78 eval conditions): 3x3 reach montage (9 spots x 8 envs)
# then the close front view at (-6, 0), 4 envs. Run under systemd-inhibit; nothing else on the GPU.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
bash artifacts/claude_scripts/v77c_montage.sh logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-10_17-34-03_v78_r3/model_25300.pt
bash artifacts/claude_scripts/v77c_video.sh logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-10_17-34-03_v78_r3/model_25300.pt
echo V78_DEPLOY_VIDEO_DONE
