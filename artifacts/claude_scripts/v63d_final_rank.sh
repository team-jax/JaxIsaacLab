#!/bin/bash
# v63d final comparison against the deployed 17650: nominal and combined case at 1152 envs, table -1 cm and fixed turns
# -35 / +45 deg at 576 envs, wide eval grid, grid set v53x (its base table equals v53w).
cd /home/jeongwoo/isaacsim/JaxIsaacLab
L=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16
COMMON="GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v53x GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 EVAL_YAW_LO=1 EVAL_GRID=-0.14,-0.06,0.02;-0.10,-0.02,0.06 EVAL_S=12 GL_START_NOISE_DEG=0.75"
HARSH="GL_ACT_DELAY=2 GL_MU=0.35,0.3 GL_CUBE_MASS=0.05,0.05 GL_KP_SCALE=1.5,1.5 GL_OBS_CUBE_NOISE=0.002,0.005"
for m in 2026-11-07_14-34-23_v63b_r10/model_17650.pt 2026-11-07_17-16-44_v63d_r4/model_18075.pt 2026-11-07_16-48-02_v63d_r2/model_17900.pt; do
  for c in "nominal 1152 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0" "harsh 1152 GL_CUBE_YAW_DEG=45 $HARSH GL_TABLE_DZ_FIXED=0.0" "table-1cm 576 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=-0.01" "yaw-35 576 GL_CUBE_YAW_FIXED_DEG=-35 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0" "yaw+45 576 GL_CUBE_YAW_FIXED_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0"; do
    set -- $c; n=$1; ne=$2; shift 2
    out=$(env $COMMON "$@" NE=$ne timeout -k 20 1500 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $L/$m 0 2>&1)
    echo "RANK $m $n $(echo "$out" | grep '^HELD' | awk '{print $NF}') $(echo "$out" | grep '^SPLIT' | sed -E 's/.*lifting: tilt ([0-9.-]+) yaw ([0-9.-]+).*/tilt \1 yaw \2/')"
    echo "$out" | grep -E "^CELL|Traceback" | sed 's/^/  /'
  done
done
echo RANK_DONE
