#!/bin/bash
# v67 (3): the fixed cube turn -35 deg is the weak turn (88.9-90.8% vs 96-97% at -45 / -20 deg). Where (cells) and how wide?
# 18525, 576 envs, 12 cells, nominal friction, fixed turns -30 / -35 / -40 deg, per-cell lines kept. Holds the eval lock.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
LOCK=$SP/isaac_eval.lock
CK=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-07_23-50-53_v66_r4/model_18525.pt
COMMON="GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=0 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_CUBE_YAW_DEG=45 EVAL_YAW_LO=1 NE=576 EVAL_S=12 GL_TABLE_DZ_FIXED=0.0 GL_START_NOISE_DEG=0.75"
export EVAL_CELLS="0.02,-0.28;0.02,0.0;0.02,0.28;-0.04,-0.24;-0.04,0.24;-0.08,-0.18;-0.08,0.0;-0.08,0.18;-0.12,-0.14;-0.12,0.12;-0.14,-0.08;-0.14,0.06"
D="GL_ACT_DELAY=2"; M="GL_MU=0.35,0.3"; W="GL_CUBE_MASS=0.05,0.05"; K="GL_KP_SCALE=1.5,1.5"; N="GL_OBS_CUBE_NOISE=0.002,0.005"
for c in "yaw-30|GL_MU=0.4,0.35 GL_CUBE_YAW_FIXED_DEG=-30" "yaw-35|GL_MU=0.4,0.35 GL_CUBE_YAW_FIXED_DEG=-35" "yaw-40|GL_MU=0.4,0.35 GL_CUBE_YAW_FIXED_DEG=-40"; do
  n=${c%%|*}; fl=${c#*|}
  out=$(flock $LOCK env $COMMON $fl timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $CK 0 2>&1)
  echo "$out" > $SP/v67yaw_$n.log
  echo "YAWC $n held $(echo "$out" | grep '^HELD' | awk '{print $NF}') | +y $(echo "$out" | grep '^CELL' | awk '$3 > 0 {s += $5; k++} END {if (k) printf "%.3f", s / k}') -y $(echo "$out" | grep '^CELL' | awk '$3 < 0 {s += $5; k++} END {if (k) printf "%.3f", s / k}') y0 $(echo "$out" | grep '^CELL' | awk '$3 == 0 {s += $5; k++} END {if (k) printf "%.3f", s / k}')"
done
echo V67_YAW_DONE
