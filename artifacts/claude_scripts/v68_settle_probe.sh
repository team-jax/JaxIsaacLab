#!/bin/bash
# v68 probe: does a scripted settle in the close (GL_CLOSE_SETTLE="s,n": after all three digits touch, s s with the squeeze capped
# at n N and no lift) close the -35 deg hole without retraining? 18525: (-8, 0) cell at -35 / -30 deg (64 envs, per-env traces),
# then the 12-cell nominal and -35 deg evals (576) with the best variant. Holds the eval lock.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
LOCK=$SP/isaac_eval.lock
CK=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-07_23-50-53_v66_r4/model_18525.pt
COMMON="GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=0 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0 GL_START_NOISE_DEG=0.75 EVAL_S=12"
for st in "0.4,1.0" "0.4,99" "0.8,1.0"; do
  for y in -35 -30; do
    flock $LOCK env $COMMON NE=64 EVAL_CELLS=-0.08,0.0 GL_CLOSE_SETTLE=$st GL_CUBE_YAW_FIXED_DEG=$y timeout -k 20 600 ./isaaclab.sh -p artifacts/claude_scripts/v68_hole_diag.py $CK $SP/settle_${st/,/_}_y$y.pt 2>&1 | grep -E "HOLE_DIAG|Traceback|Error" | sed "s/^/settle $st yaw $y: /" | head -3
  done
done
./_isaac_sim/python.sh artifacts/claude_scripts/v68_hole_ana.py $SP/settle_0.4_1.0_y-35.pt $SP/settle_0.4_99_y-35.pt $SP/settle_0.8_1.0_y-35.pt 2>&1 | grep -E "^==|held n|fail n" | head -20
CELLS="0.02,-0.28;0.02,0.0;0.02,0.28;-0.04,-0.24;-0.04,0.24;-0.08,-0.18;-0.08,0.0;-0.08,0.18;-0.12,-0.14;-0.12,0.12;-0.14,-0.08;-0.14,0.06"
for st in "0.4,1.0" "0.4,99"; do
  for c in "nominal|" "yaw-35|GL_CUBE_YAW_FIXED_DEG=-35"; do
    n=${c%%|*}; fl=${c#*|}
    out=$(flock $LOCK env $COMMON NE=576 EVAL_YAW_LO=1 EVAL_CELLS="$CELLS" GL_CLOSE_SETTLE=$st $fl timeout -k 20 900 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $CK 0 2>&1)
    echo "SETTLE $st $n held $(echo "$out" | grep '^HELD' | awk '{print $NF}') done_t $(echo "$out" | grep '^DONE_T' | awk '{print $3}') | cells $(echo "$out" | grep '^CELL' | awk '{printf "(%s,%s)%s ", $2, $3, $5}')"
  done
done
echo V68_SETTLE_DONE
