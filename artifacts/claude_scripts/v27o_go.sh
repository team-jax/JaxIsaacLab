#!/bin/bash
cd /home/jeongwoo/isaacsim/JaxIsaacLab
D=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-02_06-36-39_v27n
echo "== EVAL v27n (lr 2e-5, crashed at 5303)"
for m in 4800 4950 5100 5250 5300; do echo -n "model_$m "; GL_LIFT_EXTRA=0.5 GL_START_NOISE_DEG=0.75 GL_WRIST_RANGE=0.4 timeout -k 20 400 ./isaaclab.sh -p artifacts/claude_scripts/v27_eval.py $D/model_$m.pt 0 2>&1 | grep -E "^RES|^ALL" | python3 -c "
import sys,json
o=[];a=''
for l in sys.stdin:
  if l.startswith('RES'): d=json.loads(l[4:]); o.append('(%+.0f,%+.0f)%.2f/%.0f/%.1fN'%(d['dx_cm'],d['dy_cm'],d['lifted'],d['xy_drift_mm'],d['table_N_max']))
  else: a=l.split()[-1]
print(a,' '.join(o))"; done
echo "== ROUND v27o (sideways-joint cost)"
LR=5e-5 EXTRA=0.5 HARD=0.15 HBOX=-0.08,-0.06,-0.05,0.06 XY=50 DRAG=300 XC=10 bash artifacts/claude_scripts/v27c_round.sh 2026-11-01_21-25-47_v25_cube model_1197.pt 2026-11-02_04-17-46_v27j model_4675.pt v27o 0 600 0.75
