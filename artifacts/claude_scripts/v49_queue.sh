#!/bin/bash
# v49 queue: after the 12750 chain render: friction smoke test -> +/-20 cm grids (base, -3, +3 cm table) -> baseline
# eval of 12750 on the wide box -> v49 training
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
while pgrep -f "v27_montage.sh" >/dev/null; do sleep 15; done; sleep 10
echo "## smoke"
GL_MU_RAND=0.3,0.5 GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_TABLE_REL=1 GL_RETURN_L=1 GL_MU=0.4,0.35 timeout -k 20 400 ./isaaclab.sh -p artifacts/claude_scripts/mu_rand_smoke.py 2>&1 | grep -E "^CFG|^MAT|^GL_MU|AFTER_RESET|SMOKE_OK|Traceback|Error" | head -20
for z in 0.0 -0.03 0.03; do
  n=$(python3 -c "print({'0.0':'v49_base','-0.03':'v49_m3','0.03':'v49_p3'}['$z'])")
  echo "## grid $n"
  GRID_BX=0.115 timeout -k 20 2400 ./isaaclab.sh -p artifacts/claude_scripts/v41_grid_probe.py save 0.20 2 ${n}_raw.pt $z 2>&1 | grep -E "^WP|SAVED|Error|Traceback"
  ./_isaac_sim/python.sh artifacts/claude_scripts/v49_regrid.py ${n}_raw.pt ${n}c.pt 2>&1 | grep -E "SAVED|Error|Traceback"
done
ls -la source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/data/ | grep v49
echo "## baseline 12750 wide"
out=$(GL_GRID_SET=v49 GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 ROLL=12 NE=288 EVAL_GRID="-0.14,-0.06,0.02;-0.10,0.0,0.10" bash artifacts/claude_scripts/v46_big_cmp.sh 2026-11-05_12-45-51_v48i2_r1/model_12750.pt)
echo "$out"
echo "$out" | python3 -c "
import re, sys
s = sys.stdin.read()
b = re.search(r'held ([\d.]+) done ([\d.]+) lifted ([\d.]+) .*lift_tilt ([\d.]+) lift_yaw ([\d.]+)', s)
p = re.search(r'PROX_DONE first_joint_N_at_lift_done \[([\d.]+), ([\d.]+), ([\d.]+)\] sum ([\d.]+)', s)
print(' '.join(b.groups()), p.group(4), min(float(p.group(i)) for i in (1, 2, 3)))
" > artifacts/claude_scripts/v49_v0.txt
echo "V0 $(cat artifacts/claude_scripts/v49_v0.txt)"
bash artifacts/claude_scripts/v49_full.sh > $SP/v49.out 2>&1
