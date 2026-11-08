#!/bin/bash
# v72 deploy sweep for the V3 final (v72_r5/model_23025, cube-turn input GL_OBS_YAW=1) at the URDF layout (shoulder 1.2125 m,
# table 0.735 m -> table 5 mm below the grids' layout: DZ=-0.005; table -1 / +1 cm cases at -0.015 / +0.005).
# v72b's SWEEP lines ran at table 0.0 (v46_big_cmp.sh sets GL_TABLE_DZ_FIXED=${DZ:-0.0} over the caller's value), so every
# case is redone here with DZ given. 288 envs over the 12 reach-map cells, cube turn +/-45 deg unless fixed. Waits for the
# final video render; holds the eval lock per eval.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
LOCK=$SP/isaac_eval.lock
until grep -q "V3_FINAL_VIDEO_DONE" $SP/v3_final_video.out; do sleep 20; done
M=2026-11-08_16-21-06_v72_r5/model_23025.pt
CELLS="0.02,-0.28;0.02,0.0;0.02,0.28;-0.04,-0.24;-0.04,0.24;-0.08,-0.18;-0.08,0.0;-0.08,0.18;-0.12,-0.14;-0.12,0.12;-0.14,-0.08;-0.14,0.06"
BASE="GL_ROBOT=v3 GL_OBS_YAW=1 GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_REACH_DX=0 GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15"
for c in "yaw0 GL_CUBE_YAW_FIXED_DEG=0" "yaw+20 GL_CUBE_YAW_FIXED_DEG=20" "yaw-20 GL_CUBE_YAW_FIXED_DEG=-20" "yaw+35 GL_CUBE_YAW_FIXED_DEG=35" "yaw-35 GL_CUBE_YAW_FIXED_DEG=-35" "yaw+45 GL_CUBE_YAW_FIXED_DEG=45" "yaw-45 GL_CUBE_YAW_FIXED_DEG=-45" "gain_x0.6 GL_KP_SCALE=0.6,0.6" "gain_x1.5 GL_KP_SCALE=1.5,1.5" "gain_x2.0 GL_KP_SCALE=2.0,2.0" "finger_x2 GL_KP_PART=finger GL_KP_SCALE=2.0,2.0" "mass30g GL_CUBE_MASS=0.03,0.03" "mass50g GL_CUBE_MASS=0.05,0.05" "delay4 GL_ACT_DELAY=4" "table+1cm DZ=0.005"; do
  set -- $c; n=$1; shift
  echo "SWEEP $n $(flock $LOCK env DZ=-0.005 "$@" $BASE GL_CUBE_YAW_DEG=45 EVAL_YAW_LO=1 EVAL_CELLS="$CELLS" ROLL=12 NE=288 bash artifacts/claude_scripts/v46_big_cmp.sh $M | grep -E '^BIG' | sed -E 's/^BIG [^ ]+ //')"
done
echo V72_FINAL_DONE
