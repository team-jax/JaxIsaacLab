#!/bin/bash
# v65 wrap-up after the loop: (1) re-rank the v65 bests (nominal and combined at 1152 envs, table -1 cm and fixed turns -35 /
# +45 deg at 576, 12 cells across the reach map), pick the best mean of the five; (2) chain at 9 spots across the reach map
# (turned approach: RAISE_GRID_SET=v65r, REACH_FINAL_GRID_SET=v65f); (3) 288-env deploy sweeps on the 12 cells.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
L=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16
until grep -q "^DONE best" $SP/v65.out; do sleep 20; done
CELLS="0.02,-0.28;0.02,0.0;0.02,0.28;-0.04,-0.24;-0.04,0.24;-0.08,-0.18;-0.08,0.0;-0.08,0.18;-0.12,-0.14;-0.12,0.12;-0.14,-0.08;-0.14,0.06"
BASE="GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_REACH_DX=0 GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15"
EV="$BASE GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 EVAL_YAW_LO=1 EVAL_S=12 GL_START_NOISE_DEG=0.75 EVAL_CELLS=$CELLS"
HARSH="GL_ACT_DELAY=2 GL_MU=0.35,0.3 GL_CUBE_MASS=0.05,0.05 GL_KP_SCALE=1.5,1.5 GL_OBS_CUBE_NOISE=0.002,0.005"
read _s BEST < artifacts/claude_scripts/v65_best.txt
CANDS=$(printf "%s\n" $BEST 2026-11-07_21-08-24_v65_r7/model_18200.pt 2026-11-07_20-52-49_v65_r6/model_18125.pt 2026-11-07_20-23-36_v65_r4/model_18050.pt | awk '!seen[$0]++')
bestm=""; bests=-1
for m in $CANDS; do
  vals=""
  for c in "nominal 1152 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0" "harsh 1152 GL_CUBE_YAW_DEG=45 $HARSH GL_TABLE_DZ_FIXED=0.0" "table-1cm 576 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=-0.01" "yaw-35 576 GL_CUBE_YAW_FIXED_DEG=-35 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0" "yaw+45 576 GL_CUBE_YAW_FIXED_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0"; do
    set -- $c; n=$1; ne=$2; shift 2
    out=$(env $EV "$@" NE=$ne timeout -k 20 1500 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $L/$m 0 2>&1)
    h=$(echo "$out" | grep '^HELD' | awk '{print $NF}'); vals="$vals ${h:-0}"
    echo "RANK $m $n $h $(echo "$out" | grep '^SPLIT' | sed -E 's/.*lifting: tilt ([0-9.-]+) yaw ([0-9.-]+).*/tilt \1 yaw \2/')"
    [ "$n" = "nominal" ] && echo "$out" | grep -E "^CELL" | sed 's/^/  /'
  done
  s=$(echo $vals | awk '{print ($1 + $2 + $3 + $4 + $5) / 5}'); echo "MEAN5 $m $s"
  if python3 -c "import sys; sys.exit(0 if float('$s') > float('$bests') else 1)"; then bests=$s; bestm=$m; fi
done
echo "PICK $bestm $bests"; echo "$bestm" > $SP/v65_pick.txt
(export $EV GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0 RAISE_GRID_SET=v65r REACH_FINAL_GRID_SET=v65f; SKIP_DONE=1 RETL=1 STEPS=660 LIFTIK=1 IKP= IKVZ=0.035 SPOTS="0.02,-0.28 0.02,0.0 0.02,0.28 -0.06,-0.20 -0.06,0.0 -0.06,0.20 -0.14,-0.08 -0.14,0.0 -0.14,0.06" TAGSUF=_v65_reach bash artifacts/claude_scripts/v65_montage.sh $L/$bestm front 2>&1 | grep -E "lifted|SKIP|MONTAGE|Traceback")
echo "CHAIN_DONE"
for c in "yaw0 GL_CUBE_YAW_FIXED_DEG=0" "yaw+20 GL_CUBE_YAW_FIXED_DEG=20" "yaw-20 GL_CUBE_YAW_FIXED_DEG=-20" "yaw+35 GL_CUBE_YAW_FIXED_DEG=35" "yaw-45 GL_CUBE_YAW_FIXED_DEG=-45" "gain_x0.6 GL_KP_SCALE=0.6,0.6" "gain_x1.5 GL_KP_SCALE=1.5,1.5" "gain_x2.0 GL_KP_SCALE=2.0,2.0" "finger_x2 GL_KP_PART=finger GL_KP_SCALE=2.0,2.0" "mu0.3 MU=0.3,0.25" "mass30g GL_CUBE_MASS=0.03,0.03" "mass50g GL_CUBE_MASS=0.05,0.05" "delay4 GL_ACT_DELAY=4" "table+1cm DZ=0.01"; do
  set -- $c; n=$1; shift
  echo "SWEEP $n $(env "$@" $BASE GL_CUBE_YAW_DEG=45 EVAL_YAW_LO=1 EVAL_CELLS="$CELLS" ROLL=12 NE=288 bash artifacts/claude_scripts/v46_big_cmp.sh $bestm | grep -E '^BIG' | sed -E 's/^BIG [^ ]+ //')"
done
echo V65_FINAL_DONE
