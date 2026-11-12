#!/bin/bash
# v73 (user 2026-11-09: real AK45 arm in servo mode -> no gravity sag, sag compensation removed; layout shoulder -> table
# 47.25 cm with the table at 0.735 m -> sim table offset 0.0, grids unblended e = 0). Servo emulation for v72_r5/model_23025:
# arm drives x S stiffer with the start-pose hold at the ACTUAL kp (target = pose + tau_g / kp_actual -> the arm sits on the
# geometric pose, like a servo commanded that pose), chain gravity term at full weight from t = 0 (CHAIN_GRAV_FULL), L target
# geometric (GL_RET_Q_DEG = command minus fixed sag). Evals 576 envs over the 12 cells; chain 9 spots x 16 without video.
# Each Isaac run holds the eval lock with flock -o: Isaac starts the Omniverse Hub daemon on its first launch after a reboot, and
# a plain flock let the Hub inherit the lock fd, so the lock stayed held after the eval ended (2026-11-09: 30 min stuck).
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
LOCK=$SP/isaac_eval.lock
L=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16; M=$L/2026-11-08_16-21-06_v72_r5/model_23025.pt
CELLS="0.02,-0.28;0.02,0.0;0.02,0.28;-0.04,-0.24;-0.04,0.24;-0.08,-0.18;-0.08,0.0;-0.08,0.18;-0.12,-0.14;-0.12,0.12;-0.14,-0.08;-0.14,0.06"
COMMON="GL_ROBOT=v3 GL_OBS_YAW=1 GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=0 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_CUBE_YAW_DEG=45 EVAL_YAW_LO=1"
QL="GL_RET_Q_DEG=0,0,7.2,90.0,0,-2.6,0"
servo() { echo "GL_KP_PART=arm GL_KP_SCALE=$1,$1 GL_GRAV_COMP_KP=actual $QL"; }
HARSH="GL_ACT_DELAY=2 GL_MU=0.35,0.3 GL_CUBE_MASS=0.05,0.05 GL_OBS_CUBE_NOISE=0.002,0.005"
for c in "spring_d47.25_harsh|GL_TABLE_DZ_FIXED=0.0 $HARSH GL_KP_SCALE=1.5,1.5" "spring_d48.25_nominal|GL_TABLE_DZ_FIXED=-0.01 GL_MU=0.4,0.35" "servo5_nominal|$(servo 5) GL_TABLE_DZ_FIXED=0.0 GL_MU=0.4,0.35" "servo5_harsh|$(servo 5) GL_TABLE_DZ_FIXED=0.0 $HARSH" "servo10_nominal|$(servo 10) GL_TABLE_DZ_FIXED=0.0 GL_MU=0.4,0.35" "servo5_d48.25|$(servo 5) GL_TABLE_DZ_FIXED=-0.01 GL_MU=0.4,0.35" "servo5_mu0.3|$(servo 5) GL_TABLE_DZ_FIXED=0.0 GL_MU=0.3,0.25"; do
  IFS='|' read n a <<< "$c"
  out=$(flock -o $LOCK env $COMMON EVAL_CELLS="$CELLS" NE=576 EVAL_S=12 GL_START_NOISE_DEG=0.75 $a timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $M 0 2>&1)
  echo "SERVO_EVAL $n held $(echo "$out" | grep '^HELD' | awk '{print $NF}') $(echo "$out" | grep '^SPLIT' | sed -E 's/.*lifting: tilt ([0-9.-]+) yaw ([0-9.-]+).*/lift_tilt \1 lift_yaw \2/')"
  echo "$out" | grep -E "^CELL|Traceback" | sed 's/^/  /'
done
# chain (raise -> reach-in -> grasp-lift -> L -> lift_done), servo emulation x5, no video
flock -o $LOCK env $COMMON GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0 $QL TEST_ARM_GAIN_SCALE=5 CHAIN_GRAV_COMP=1 CHAIN_GRAV_FULL=1 CHAIN_NO_VIDEO=1 \
  RAISE_GRID_SET=v65r REACH_FINAL_GRID_SET=v65f RAISE_POL=nominal RETL=1 STEPS=660 LIFTIK=1 IKP= IKVZ=0.035 NE=16 \
  SPOTS="0.02,-0.28 0.02,0.0 0.02,0.28 -0.06,-0.20 -0.06,0.0 -0.06,0.20 -0.14,-0.08 -0.14,0.0 -0.14,0.06" TAGSUF=_v73_servo5 \
  bash artifacts/claude_scripts/v3_montage.sh $M front 2>&1 | grep -E "lifted|Traceback" | sed 's/^/SERVO_CHAIN /'
echo V73_SERVO_DONE
