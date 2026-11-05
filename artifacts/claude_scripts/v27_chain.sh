#!/bin/bash
# v27: raise (v25 1197 -> wide box) then grasp (v26_rise 2600 -> wide box), then eval sweep
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
echo "== RAISE"; bash artifacts/claude_scripts/raise27_cycle.sh 2026-11-01_21-25-47_v25_cube model_1197.pt v27_wide 300
echo "== GRASP"; GL_CUBE_BOX=1 GL_RISE=8 GL_LATE=8 GL_XY=50 GL_REWARD_SCALE=0.02 GL_TABLE=3 GL_NO_VCLIP=1 GL_LR=5e-5 GL_WRIST_RANGE=0.4 bash artifacts/claude_scripts/gl_cycle_nr.sh 2026-11-01_21-52-07_v26_rise model_2600.pt v27_wide 300
D=$(ls -d logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/*_v27_wide | tail -1)
for m in 2600 2650 2700 2750 2800 2850 2899; do [ -f $D/model_$m.pt ] || continue; echo -n "model_$m "; GL_WRIST_RANGE=0.4 timeout 400 ./isaaclab.sh -p artifacts/claude_scripts/v27_eval.py $D/model_$m.pt 0 2>&1 | grep -E "^RES|^ALL" | python3 -c "
import sys,json
o=[]
for l in sys.stdin:
  if l.startswith('RES'): d=json.loads(l[4:]); o.append('(%+.0f,%+.0f)%.2f/%.0f/%.1fN'%(d['dx_cm'],d['dy_cm'],d['lifted'],d['xy_drift_mm'],d['table_N_max']))
  else: print(l.strip(), end=' ')
print(' '.join(o))"; done
