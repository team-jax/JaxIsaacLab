#!/bin/bash
# v69c wrap-up (same as v69_final.sh, for the v69c best): the v69 best (cube-turn input, GL_OBS_YAW=1) against 18525 (no turn input), eight cases at 576 envs over the 12
# cells, each twice (nominal, combined, friction 0.3, table -1 cm, gains x2, fixed -35 / +45 / -30 deg); then per-cell lines of
# the v69 best at -35 / -30 / -40 deg. Waits for the v69 loop to end; holds the eval lock.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
LOCK=$SP/isaac_eval.lock
L=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16
until grep -q "^DONE best" $SP/v69c.out; do sleep 20; done
read _s B69 < artifacts/claude_scripts/v69c_best.txt
CELLS="0.02,-0.28;0.02,0.0;0.02,0.28;-0.04,-0.24;-0.04,0.24;-0.08,-0.18;-0.08,0.0;-0.08,0.18;-0.12,-0.14;-0.12,0.12;-0.14,-0.08;-0.14,0.06"
EV="GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_REACH_DX=0 GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 EVAL_YAW_LO=1 EVAL_S=12 GL_START_NOISE_DEG=0.75 EVAL_CELLS=$CELLS NE=576"
HARSH="GL_ACT_DELAY=2 GL_MU=0.35,0.3 GL_CUBE_MASS=0.05,0.05 GL_KP_SCALE=1.5,1.5 GL_OBS_CUBE_NOISE=0.002,0.005"
for mm in "v69cbest|$B69|GL_OBS_YAW=1" "18525|2026-11-07_23-50-53_v66_r4/model_18525.pt|"; do
  IFS='|' read nm m ox <<< "$mm"
  vals=""
  for c in "nominal GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0" "harsh GL_CUBE_YAW_DEG=45 $HARSH GL_TABLE_DZ_FIXED=0.0" "mu0.3 GL_CUBE_YAW_DEG=45 GL_MU=0.3,0.25 GL_TABLE_DZ_FIXED=0.0" "table-1cm GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=-0.01" "gain_x2 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_KP_SCALE=2.0,2.0 GL_TABLE_DZ_FIXED=0.0" "yaw-35 GL_CUBE_YAW_FIXED_DEG=-35 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0" "yaw+45 GL_CUBE_YAW_FIXED_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0" "yaw-30 GL_CUBE_YAW_FIXED_DEG=-30 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0"; do
    set -- $c; n=$1; shift
    hs=""
    for rep in 1 2; do
      out=$(flock $LOCK env $EV $ox "$@" timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $L/$m 0 2>&1)
      hs="$hs $(echo "$out" | grep '^HELD' | awk '{print $NF}')"
    done
    a=$(echo $hs | awk '{s = 0; for (i = 1; i <= NF; i++) s += $i; printf "%.3f", s / NF}'); vals="$vals $a"
    echo "FIN $nm $n $hs -> $a"
  done
  echo "FIN_MEAN $nm $(echo $vals | awk '{s = 0; for (i = 1; i <= NF; i++) s += $i; printf "%.4f", s / NF}')"
done
for y in -35 -30 -40; do
  out=$(flock $LOCK env $EV GL_OBS_YAW=1 GL_CUBE_YAW_FIXED_DEG=$y GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0 timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $L/$B69 0 2>&1)
  echo "FINCELL $y $(echo "$out" | grep '^CELL' | awk '{printf "(%s,%s)%s ", $2, $3, $5}')"
done
echo V69C_FINAL_DONE
