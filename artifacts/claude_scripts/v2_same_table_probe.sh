#!/bin/bash
# v2 at the V3 (URDF) table heights: 22050 on the V2 robot at -5 mm (nominal) and -15 mm (table -1 cm) - robot or table range?; (was: is the shallow V3 grasp (fingertips ~4 mm farther from the cube at the same joint angles: hand 3.5 mm nearer the wrist)
# what costs the hard cases? 22050 on V3 at the real layout, reach-in 0 / -4 / -6 mm deeper (GL_REACH_DX), nominal and table
# -1 cm (-15 mm vs the grids), 576 envs. Holds the eval lock.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
CK=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-08_13-33-41_v71_r6/model_22050.pt
EV="GL_OBS_YAW=1 GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 EVAL_YAW_LO=1 EVAL_S=12 GL_START_NOISE_DEG=0.75 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 NE=576"
export EVAL_CELLS="0.02,-0.28;0.02,0.0;0.02,0.28;-0.04,-0.24;-0.04,0.24;-0.08,-0.18;-0.08,0.0;-0.08,0.18;-0.12,-0.14;-0.12,0.12;-0.14,-0.08;-0.14,0.06"
for rd in 0; do
  for c in "nominal|GL_TABLE_DZ_FIXED=-0.005" "table-1cm|GL_TABLE_DZ_FIXED=-0.015"; do
    n=${c%%|*}; fl=${c#*|}
    out=$(flock $SP/isaac_eval.lock env $EV GL_REACH_DX=$rd $fl timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $CK 0 2>&1)
    echo "V2SAME $n held $(echo "$out" | grep '^HELD' | awk '{print $NF}') $(echo "$out" | grep '^SPLIT' | sed -E 's/.*lifting: tilt ([0-9.-]+) yaw ([0-9.-]+).*/tilt \1 yaw \2/')"
  done
done
echo V2SAME_DONE
