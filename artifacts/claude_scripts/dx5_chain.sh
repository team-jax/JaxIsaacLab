#!/bin/bash
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad; mkdir -p $S
bash artifacts/claude_scripts/pre_cycle.sh 2026-10-27_08-54-09_v12_c55_h00 model_2789.pt v12_c55_dx5 250
R=$(ls -d logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/*_v12_c55_dx5 | tail -1); M=$(ls $R/model_*.pt | sort -V | tail -1)
F=source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/v2_front_grasp_close.py
python3 - "$M" "$F" <<'P'
import sys; m,f=sys.argv[1],sys.argv[2]; s=open(f).read()
old='    pre_policy_path = "logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/2026-10-27_08-54-09_v12_c55_h00/model_2789.pt"\n'
if m not in s: s=s.replace(old, old+'    # Pre-grasp 1 cm farther back (v12_c55_dx5) for first-joint-only closing.\n    pre_policy_path = "%s"\n' % m)
open(f,'w').write(s)
P
grep -n "pre_policy_path =" $F | tail -1
timeout 400 ./isaaclab.sh -p artifacts/claude_scripts/flat_scan2.py --headless 2>&1 | grep -E "^FS|Error|Traceback" | sed 's/^FS //' > $S/flat_scan3.txt
python3 - $S/flat_scan3.txt <<'P'
import json,collections,statistics as st,sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
g=collections.defaultdict(list)
for r in rows: g[(r['f1'],r['thumb1'])].append(r)
for k,v in g.items():
  ok=sum(all(x>=0.3 for x in r['distal_N']) for r in v)
  print('SCAN',k,'all3',ok,'/4 distal',[round(st.mean(r['distal_N'][i] for r in v),2) for i in range(3)],'prox',[round(st.mean(r['prox_N'][i] for r in v),2) for i in range(3)],'thumb_flat',round(st.mean(r['flat'][2] for r in v),2),'cube',round(st.mean(r['cube_mm'] for r in v),1))
P
