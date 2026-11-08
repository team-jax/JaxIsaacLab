#!/bin/bash
# v63 deploy numbers for checkpoint $1 (run/model path under the task log dir): wide-box chain (9 spots x 16 envs, side start
# 0.2 x max(-dy, 0), back-off 2 cm) unless $2 = nochain, then 288-env sweeps on the wide eval grid (cube turned U(-45, 45)
# deg) as for 17100 (drive gains x0.6 / x1.5 / x2.0, fingers only x2, friction 0.3, cube 30 / 50 g, delay 4 steps,
# table -1 / +1 cm), and the original eval grid (x -8/-3/+2, y -5/0/+6 cm) at 576 envs, nominal and combined case.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
CK=$1
BASE="GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v53w GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15"
if [ "$2" != "nochain" ]; then
  (export RAISE_BACK_M=0.02 RAISE_SIDE_K=0.2 GL_CUBE_YAW_DEG=45 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_IK_POS=2 GL_GRAV_COMP=1 GL_TABLE_EDGE=0.225 GL_MU=0.4,0.35 GL_SQUEEZE=6,0 GL_TABLE_DZ_FIXED=0.0 GL_TABLE_REL=1 $BASE; SKIP_DONE=1 RETL=1 STEPS=660 LIFTIK=1 IKP= IKVZ=0.035 XS="-0.14 -0.06 0.02" YS="0.06 -0.02 -0.10" TAGSUF=_c50_yaw45_back2_sidek02_wide bash artifacts/claude_scripts/v27_montage.sh logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/$CK front 2>&1 | grep -E "lifted|SKIP|MONTAGE|Traceback")
  echo "CHAIN_DONE"
fi
for c in "gain_x0.6 GL_KP_SCALE=0.6,0.6" "gain_x1.5 GL_KP_SCALE=1.5,1.5" "gain_x2.0 GL_KP_SCALE=2.0,2.0" "finger_x2 GL_KP_PART=finger GL_KP_SCALE=2.0,2.0" "mu0.3 MU=0.3,0.25" "mass30g GL_CUBE_MASS=0.03,0.03" "mass50g GL_CUBE_MASS=0.05,0.05" "delay4 GL_ACT_DELAY=4" "table-1cm DZ=-0.01" "table+1cm DZ=0.01"; do
  set -- $c; n=$1; shift
  echo "SWEEP $n $(env "$@" $BASE GL_CUBE_YAW_DEG=45 EVAL_YAW_LO=1 EVAL_GRID="-0.14,-0.06,0.02;-0.10,-0.02,0.06" ROLL=12 NE=288 bash artifacts/claude_scripts/v46_big_cmp.sh $CK | grep -E '^BIG' | sed -E 's/^BIG [^ ]+ //')"
done
for c in "old_nominal MU=0.4,0.35" "old_harsh GL_ACT_DELAY=2 MU=0.35,0.3 GL_CUBE_MASS=0.05,0.05 GL_KP_SCALE=1.5,1.5 GL_OBS_CUBE_NOISE=0.002,0.005"; do
  set -- $c; n=$1; shift
  echo "SWEEP $n $(env "$@" $BASE GL_CUBE_YAW_DEG=45 EVAL_YAW_LO=1 EVAL_GRID="-0.08,-0.03,0.02;-0.05,0.0,0.06" ROLL=12 NE=576 bash artifacts/claude_scripts/v46_big_cmp.sh $CK | grep -E '^BIG' | sed -E 's/^BIG [^ ]+ //')"
done
echo FINAL_SWEEP_DONE
