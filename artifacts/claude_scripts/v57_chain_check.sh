#!/bin/bash
# v57: after round 1 is ranked, pause training (no rendering during training), render the full chain (hanging arm ->
# v25 raise residual on the v53 path -> reach-in -> grasp-lift policy -> L return) at the 9 cube spots x 16 envs in the
# CAD layout with the v55 close, for the v57 best; then continue training (v57b) from it.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
until grep -q "BEST_OF v57_r1\|DONE best" $SP/v57.out; do sleep 10; done
kill 106713 2>/dev/null; sleep 2
for p in $(ps -eo pid,args | grep -E "v57_r[0-9]" | grep -v grep | awk '{print $1}'); do kill $p 2>/dev/null; done
sleep 20
for p in $(ps -eo pid,args | grep -E "v57_r[0-9]" | grep -v grep | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
sleep 10
read S0 BEST < artifacts/claude_scripts/v57_best.txt
echo "PAUSED v57 after round 1; best $BEST score $S0"
echo "## chain render $BEST"
(export GL_GRID_SET=v53 GL_TABLE_Z0=-0.0575 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_IK_POS=2 GL_GRAV_COMP=1 GL_TABLE_EDGE=0.225 GL_MU=0.4,0.35 GL_SQUEEZE=6,0 GL_TABLE_DZ_FIXED=0.0 GL_TABLE_REL=1; RETL=1 STEPS=660 LIFTIK=1 IKP= IKVZ=0.035 TAGSUF=_cad_v57 bash artifacts/claude_scripts/v27_montage.sh logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/$BEST front)
echo CHAIN_DONE
bash artifacts/claude_scripts/v57b_full.sh
