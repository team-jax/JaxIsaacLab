#!/bin/bash
# chain check of v57b r3 15875 (1152 envs: nominal 97.2%, combined 93.1%) in the CAD layout: after its 288-env sweep,
# pause v57b at its next round ranking (no rendering during training), render the 9 spots x 16, then continue as v57c.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
until grep -q "SWEEP15875_DONE" $SP/sweep15875.out; do sleep 10; done
n0=$(grep -c "BEST_OF v57b_r" $SP/v57_chain.out)
until [ $(grep -c "BEST_OF v57b_r" $SP/v57_chain.out) -gt $n0 ] || grep -q "DONE best" $SP/v57_chain.out; do sleep 10; done
kill 120243 2>/dev/null; sleep 2
for p in $(ps -eo pid,args | grep -E "v57b_r[0-9]" | grep -v grep | awk '{print $1}'); do kill $p 2>/dev/null; done
sleep 20
for p in $(ps -eo pid,args | grep -E "v57b_r[0-9]" | grep -v grep | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
sleep 10
echo "PAUSED v57b after: $(grep 'BEST_OF v57b_r' $SP/v57_chain.out | tail -1)"
echo "## chain render 15875"
(export GL_GRID_SET=v53 GL_TABLE_Z0=-0.0575 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_IK_POS=2 GL_GRAV_COMP=1 GL_TABLE_EDGE=0.225 GL_MU=0.4,0.35 GL_SQUEEZE=6,0 GL_TABLE_DZ_FIXED=0.0 GL_TABLE_REL=1; RETL=1 STEPS=660 LIFTIK=1 IKP= IKVZ=0.035 TAGSUF=_cad_v57c bash artifacts/claude_scripts/v27_montage.sh logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-06_06-07-55_v57b_r3/model_15875.pt front)
echo CHAIN2_DONE
bash artifacts/claude_scripts/v57c_full.sh
