#!/bin/bash
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad; mkdir -p $S
RUN=$1; CK=$2; NAME=$3
bash artifacts/claude_scripts/pre_cycle.sh $RUN $CK $NAME ${ITERS:-250}
R=$(ls -d logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/*_$NAME | tail -1); M=$(ls $R/model_*.pt | sort -V | tail -1)
F=source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/v2_front_grasp_close.py
python3 - "$M" "$F" <<'P'
import sys,re; m,f=sys.argv[1],sys.argv[2]; s=open(f).read()
idx=[x.end() for x in re.finditer(r'    pre_policy_path = "[^"]+"\n', s)][-1]
s=s[:idx]+'    pre_policy_path = "%s"\n' % m + s[idx:]
open(f,'w').write(s)
P
for cam in close_front side; do timeout 400 ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/render_front_grasp.py --headless --close --steps 720 --camera $cam --tag grasp --checkpoint $M 2>&1 | grep -E "FRONT_GRASP_DIR|Error|Traceback" | cut -c1-220; done
python3 -c "
import json;d=json.load(open('$R/videos/front_close_close_front_$(basename $M .pt)_grasp/front_grasp.json'))
for r in d:
  if r['t'] in (5.0,8.0,10.0): print('CLOSE',r['t'],'pad',r['pad_N_env0'],'fing',r['fingers_deg_env0'])"
timeout 300 ./isaaclab.sh -p artifacts/claude_scripts/distal_probe.py --headless --checkpoint $M 2>&1 | grep -E "^DP"
timeout 300 ./isaaclab.sh -p artifacts/claude_scripts/lower_ik_probe.py --headless --dz 0 --rb 0 --pre $M 2>&1 | grep -E "^LZ|Error"
