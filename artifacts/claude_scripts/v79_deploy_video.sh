#!/bin/bash
# v79 deploy videos of the deployed 25300 with the L-return shoulder-yaw turn (constant 0: alpha = the camera psi folded; -2.1 gave a
# +2.1 deg mean residual, the cube turning ~2 deg in the hand while gripped) (GL_RET_YAW=0,45): 3x3 reach montage (9 spots x
# 8 envs, cube turns +/-45 deg) then the close front view at (-6, 0), 4 envs. Nothing else on the GPU.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
bash artifacts/claude_scripts/v79_montage.sh logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-10_17-34-03_v78_r3/model_25300.pt
bash artifacts/claude_scripts/v79_video.sh logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-10_17-34-03_v78_r3/model_25300.pt
echo V79_DEPLOY_VIDEO_DONE
