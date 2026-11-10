#!/bin/bash
# after the 2026-11-10 reboot (laptop suspend broke the GPU, Xid 154): the 25150 reach montage under the v77c conditions, then
# v78 training. Launch under systemd-inhibit (no sleep). Output: $SP/v77c_montage.out, $SP/v78.out.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad; mkdir -p $SP
bash artifacts/claude_scripts/v77c_montage.sh logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-10_06-22-37_v74b_r8/model_25150.pt > $SP/v77c_montage.out 2>&1
bash artifacts/claude_scripts/v78_full.sh > $SP/v78.out 2>&1
