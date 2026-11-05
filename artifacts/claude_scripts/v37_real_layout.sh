#!/bin/bash
# v37: chain render (raise v25 model_1197 + grasp 7550, squeeze 6 N) in the real stand layout (GL_LAYOUT_REAL), cameras side and front
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
F=source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/render_front_grasp.py
R=logs/rsl_rl/jax_v2_1006_raise_cube_v25/2026-11-01_21-25-47_v25_cube/model_1197.pt
C=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-04_13-41-42_v34_relret/model_7550.pt
for cam in side front; do
  GL_LAYOUT_REAL=1 GL_SQUEEZE=6,0 GL_TABLE_REL=1 GL_RETURN_L=1 GL_LIFT_IK=1 GL_IK_PITCH=0.3 GL_IK_VZ=0.035 RAISE_WIDE=1 GL_WRIST_RANGE=0.4 timeout -k 20 900 ./isaaclab.sh -p $F --headless --grasp_lift --steps 660 --num_envs 4 --camera $cam --tag real_layout60 --raise_cube=0.0,0.0 --raise_max_s 11 --raise_sag_comp_deg 0,0,0,0,0,0,0,0 --raise_policy $R --checkpoint $C > $S/v37_$cam.log 2>&1
  echo "CAM $cam"; grep -E "^HANDOVER|^CHAIN|Traceback|Error" $S/v37_$cam.log | cut -c1-300
done
