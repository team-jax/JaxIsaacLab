#!/bin/bash
# v46: re-rank the top checkpoints on 576 envs (64 per cube spot), 12 s, friction 0.4, 4 cm L-return trigger:
# held (lift_done and still up at 12 s), done, lifted, lift tilt; 144-env evals saturated at 100% held.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
L=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16
COMMON="GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=${ROLL:-15} GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1"
for f in "$@"; do
  out=$(env $COMMON GL_MU=${MU:-0.4,0.35} NE=${NE:-576} EVAL_S=12 GL_TABLE_DZ_FIXED=${DZ:-0.0} GL_START_NOISE_DEG=0.75 timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $L/$f 0 2>&1)
  h=$(echo "$out" | grep "^HELD" | awk '{print $NF}'); d=$(echo "$out" | grep "^DONE lift" | awk '{print $NF}'); a=$(echo "$out" | grep "^ALL" | awk '{print $NF}')
  t=$(echo "$out" | grep "^SPLIT" | sed -E 's/.*on_table\(before 1cm\): tilt ([0-9.-]+) yaw ([0-9.-]+) .*lifting: tilt ([0-9.-]+) yaw ([0-9.-]+).*/table_tilt \1 table_yaw \2 lift_tilt \3 lift_yaw \4/')
  dt=$(echo "$out" | grep "^DONE_T" | awk '{print "t_done", $3, "p90", $5}')
  echo "BIG $f held $h done $d lifted $a $t $dt"
  echo "$out" | grep -E "^DROP|^CELL|^PROX|^FINGERQ|Traceback" | sed "s|^|  |"
done
