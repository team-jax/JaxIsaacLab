#!/bin/bash
# v66 wrap-up after the loop: (1) compare 18525 / 18625 (v66 r4) / the v66 overall best with the deployed 18125 on seven cases
# at 576 envs (nominal, combined, friction 0.3, table -1 cm, gains x2, fixed -35 / +45 deg), pick the best mean; (2) rendered
# chain at the 9 reach-map spots with the pose-dependent gravity compensation (CHAIN_GRAV_COMP=1, scripted raise); (3) sweeps.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
L=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16
until grep -q "^DONE best" $SP/v66.out; do sleep 20; done
CELLS="0.02,-0.28;0.02,0.0;0.02,0.28;-0.04,-0.24;-0.04,0.24;-0.08,-0.18;-0.08,0.0;-0.08,0.18;-0.12,-0.14;-0.12,0.12;-0.14,-0.08;-0.14,0.06"
BASE="GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_REACH_DX=0 GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15"
EV="$BASE GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 EVAL_YAW_LO=1 EVAL_S=12 GL_START_NOISE_DEG=0.75 EVAL_CELLS=$CELLS"
HARSH="GL_ACT_DELAY=2 GL_MU=0.35,0.3 GL_CUBE_MASS=0.05,0.05 GL_KP_SCALE=1.5,1.5 GL_OBS_CUBE_NOISE=0.002,0.005"
read _s BEST < artifacts/claude_scripts/v66_best.txt
CANDS=$(printf "%s\n" $BEST 2026-11-07_23-50-53_v66_r4/model_18525.pt 2026-11-07_23-50-53_v66_r4/model_18625.pt 2026-11-07_20-52-49_v65_r6/model_18125.pt | awk '!seen[$0]++')
bestm=""; bests=-1
for m in $CANDS; do
  vals=""
  for c in "nominal 576 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0" "harsh 576 GL_CUBE_YAW_DEG=45 $HARSH GL_TABLE_DZ_FIXED=0.0" "mu0.3 576 GL_CUBE_YAW_DEG=45 GL_MU=0.3,0.25 GL_TABLE_DZ_FIXED=0.0" "table-1cm 576 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=-0.01" "gain_x2 576 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_KP_SCALE=2.0,2.0 GL_TABLE_DZ_FIXED=0.0" "yaw-35 576 GL_CUBE_YAW_FIXED_DEG=-35 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0" "yaw+45 576 GL_CUBE_YAW_FIXED_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0"; do
    set -- $c; n=$1; ne=$2; shift 2
    out=$(env $EV "$@" NE=$ne timeout -k 20 1500 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $L/$m 0 2>&1)
    h=$(echo "$out" | grep '^HELD' | awk '{print $NF}'); vals="$vals ${h:-0}"
    echo "RANK $m $n $h $(echo "$out" | grep '^SPLIT' | sed -E 's/.*lifting: tilt ([0-9.-]+) yaw ([0-9.-]+).*/tilt \1 yaw \2/')"
    [ "$n" = "nominal" ] && echo "$out" | grep -E "^CELL" | sed 's/^/  /'
  done
  s=$(echo $vals | awk '{print ($1 + $2 + $3 + $4 + $5 + $6 + $7) / 7}'); echo "MEAN7 $m $s"
  if python3 -c "import sys; sys.exit(0 if float('$s') > float('$bests') else 1)"; then bests=$s; bestm=$m; fi
done
echo "PICK $bestm $bests"; echo "$bestm" > $SP/v66_pick.txt
(export $EV GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0 RAISE_GRID_SET=v65r REACH_FINAL_GRID_SET=v65f CHAIN_GRAV_COMP=1; RAISE_POL=nominal SKIP_DONE=1 RETL=1 STEPS=660 LIFTIK=1 IKP= IKVZ=0.035 SPOTS="0.02,-0.28 0.02,0.0 0.02,0.28 -0.06,-0.20 -0.06,0.0 -0.06,0.20 -0.14,-0.08 -0.14,0.0 -0.14,0.06" TAGSUF=_v66_reach_grav bash artifacts/claude_scripts/v65_montage.sh $L/$bestm front 2>&1 | grep -E "lifted|SKIP|MONTAGE|Traceback")
echo "CHAIN_DONE"
for c in "yaw0 GL_CUBE_YAW_FIXED_DEG=0" "yaw+20 GL_CUBE_YAW_FIXED_DEG=20" "yaw-20 GL_CUBE_YAW_FIXED_DEG=-20" "yaw+35 GL_CUBE_YAW_FIXED_DEG=35" "yaw-45 GL_CUBE_YAW_FIXED_DEG=-45" "gain_x0.6 GL_KP_SCALE=0.6,0.6" "gain_x1.5 GL_KP_SCALE=1.5,1.5" "gain_x2.0 GL_KP_SCALE=2.0,2.0" "finger_x2 GL_KP_PART=finger GL_KP_SCALE=2.0,2.0" "mu0.3 MU=0.3,0.25" "mass30g GL_CUBE_MASS=0.03,0.03" "mass50g GL_CUBE_MASS=0.05,0.05" "delay4 GL_ACT_DELAY=4" "table+1cm DZ=0.01"; do
  set -- $c; n=$1; shift
  echo "SWEEP $n $(env "$@" $BASE GL_CUBE_YAW_DEG=45 EVAL_YAW_LO=1 EVAL_CELLS="$CELLS" ROLL=12 NE=288 bash artifacts/claude_scripts/v46_big_cmp.sh $bestm | grep -E '^BIG' | sed -E 's/^BIG [^ ]+ //')"
done
echo V66_FINAL_DONE
