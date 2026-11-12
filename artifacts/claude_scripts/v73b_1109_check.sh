#!/bin/bash
# v73b (user 2026-11-09: "URDF_1109 is really final"): the task URDF now carries the 1109 wrist origin (1 mm; FK equals 1109 with
# only the shoulder roll sign flipped). At the 47.25 cm layout (GL_TABLE_DZ_FIXED=0.0) for v72_r5/model_23025:
# (1) evals 576 envs, spring arm (as trained) and servo emulation x5; (2) chains 9 spots x 16 without video, spring (CHAIN_GRAV_COMP
# as deployed) and servo x5 (TEST_ARM_GAIN_SCALE=5, CHAIN_GRAV_FULL, geometric L) - separates servo effects from the layout.
# flock -o: the Omniverse Hub must not inherit the lock.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
LOCK=$SP/isaac_eval.lock
L=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16; M=$L/2026-11-08_16-21-06_v72_r5/model_23025.pt
CELLS="0.02,-0.28;0.02,0.0;0.02,0.28;-0.04,-0.24;-0.04,0.24;-0.08,-0.18;-0.08,0.0;-0.08,0.18;-0.12,-0.14;-0.12,0.12;-0.14,-0.08;-0.14,0.06"
COMMON="GL_ROBOT=v3 GL_OBS_YAW=1 GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=0 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_CUBE_YAW_DEG=45 EVAL_YAW_LO=1 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0"
QL="GL_RET_Q_DEG=0,0,7.2,90.0,0,-2.6,0"
for c in "1109_spring_nominal|" "1109_servo5_nominal|GL_KP_PART=arm GL_KP_SCALE=5,5 GL_GRAV_COMP_KP=actual $QL"; do
  IFS='|' read n a <<< "$c"
  out=$(flock -o $LOCK env $COMMON EVAL_CELLS="$CELLS" NE=576 EVAL_S=12 GL_START_NOISE_DEG=0.75 $a timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $M 0 2>&1)
  echo "V73B_EVAL $n held $(echo "$out" | grep '^HELD' | awk '{print $NF}') $(echo "$out" | grep '^SPLIT' | sed -E 's/.*lifting: tilt ([0-9.-]+) yaw ([0-9.-]+).*/lift_tilt \1 lift_yaw \2/')"
  echo "$out" | grep -E "^CELL|Traceback" | sed 's/^/  /'
done
SPOTS="0.02,-0.28 0.02,0.0 0.02,0.28 -0.06,-0.20 -0.06,0.0 -0.06,0.20 -0.14,-0.08 -0.14,0.0 -0.14,0.06"
CH="RAISE_GRID_SET=v65r REACH_FINAL_GRID_SET=v65f RAISE_POL=nominal RETL=1 STEPS=660 LIFTIK=1 IKP= IKVZ=0.035 NE=16 CHAIN_NO_VIDEO=1 CHAIN_GRAV_COMP=1"
flock -o $LOCK env $COMMON $CH SPOTS="$SPOTS" TAGSUF=_v73b_spring_1109 bash artifacts/claude_scripts/v3_montage.sh $M front 2>&1 | grep -E "lifted|Traceback" | sed 's/^/V73B_CHAIN spring /'
flock -o $LOCK env $COMMON $CH $QL TEST_ARM_GAIN_SCALE=5 CHAIN_GRAV_FULL=1 SPOTS="$SPOTS" TAGSUF=_v73b_servo5_1109 bash artifacts/claude_scripts/v3_montage.sh $M front 2>&1 | grep -E "lifted|Traceback" | sed 's/^/V73B_CHAIN servo5 /'
echo V73B_DONE
