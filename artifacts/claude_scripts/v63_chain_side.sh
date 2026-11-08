#!/bin/bash
# v63: once round 8 starts (after round 7's ranking and any sweep), stop v63 (no rendering during training), render the
# wide chain (9 spots x 16 envs) for the overall best with the side start 0.2 x max(-dy, 0) (RAISE_SIDE_K=0.2: 2 cm at y -10, 0 at y >= 0; back-off 2 cm), then
# continue training as v63b (each round from the overall best).
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
L=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16
until grep -q "== ROUND 8" $SP/v63.out; do sleep 10; done
for p in $(ps -eo pid,args | grep "v63_full.sh" | grep -v grep | awk '{print $1}'); do kill $p 2>/dev/null; done; sleep 2
for p in $(ps -eo pid,args | grep -E "v63_r8" | grep -v grep | awk '{print $1}'); do kill $p 2>/dev/null; done
sleep 20
for p in $(ps -eo pid,args | grep -E "v63_r8" | grep -v grep | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
sleep 10
read S0 BEST < artifacts/claude_scripts/v63_best.txt
echo "STOPPED v63 at round 8; best $BEST score $S0"
(export RAISE_BACK_M=0.02 RAISE_SIDE_K=0.2 GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_CUBE_YAW_DEG=45 GL_GRID_SET=v53w GL_TABLE_Z0=-0.0575 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_IK_POS=2 GL_GRAV_COMP=1 GL_TABLE_EDGE=0.225 GL_MU=0.4,0.35 GL_SQUEEZE=6,0 GL_TABLE_DZ_FIXED=0.0 GL_TABLE_REL=1; SKIP_DONE=1 RETL=1 STEPS=660 LIFTIK=1 IKP= IKVZ=0.035 XS="-0.14 -0.06 0.02" YS="0.06 -0.02 -0.10" TAGSUF=_c50_yaw45_back2_sidek02_wide bash artifacts/claude_scripts/v27_montage.sh $L/$BEST front 2>&1 | grep -E "lifted|SKIP|MONTAGE|Traceback")
echo "CHAIN_SIDE_DONE"
bash artifacts/claude_scripts/v63b_full.sh > $SP/v63b.out 2>&1
