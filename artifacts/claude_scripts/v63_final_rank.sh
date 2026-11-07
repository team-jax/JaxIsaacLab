#!/bin/bash
# v63 final pick for the deploy update: the v63 / v63b new bests re-ranked at 1152 envs on the wide eval grid
# (x -14/-6/+2 x y -10/-2/+6 cm, cube turned U(-45, 45) deg), nominal and combined case; same score as the loops.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
L=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16
COMMON="GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v53w GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_CUBE_YAW_DEG=45 EVAL_YAW_LO=1 EVAL_GRID=-0.14,-0.06,0.02;-0.10,-0.02,0.06"
HARSH="GL_ACT_DELAY=2 GL_MU=0.35,0.3 GL_CUBE_MASS=0.05,0.05 GL_KP_SCALE=1.5,1.5 GL_OBS_CUBE_NOISE=0.002,0.005"
for m in 2026-11-07_14-34-23_v63b_r10/model_17650.pt 2026-11-07_13-17-06_v63b_r5/model_17575.pt 2026-11-07_12-14-25_v63b_r1/model_17500.pt 2026-11-07_10-42-18_v63_r3/model_17375.pt; do
  for c in "nominal GL_MU=0.4,0.35" "harsh $HARSH"; do
    set -- $c; n=$1; shift
    out=$(env $COMMON "$@" NE=1152 EVAL_S=12 GL_TABLE_DZ_FIXED=0.0 GL_START_NOISE_DEG=0.75 timeout -k 20 1500 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $L/$m 0 2>&1)
    echo "RANK $m $n $(echo "$out" | grep '^HELD' | awk '{print $NF}') $(echo "$out" | grep '^SPLIT' | sed -E 's/.*lifting: tilt ([0-9.-]+) yaw ([0-9.-]+).*/tilt \1 yaw \2/')"
    echo "$out" | grep -E "^CELL|Traceback" | sed 's/^/  /'
  done
done
echo RANK_DONE
