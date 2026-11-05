#!/bin/bash
# usage: v19_test.sh CHECKPOINT TAG  -- chain render (learned raise -> v16), metrics, contact sheet
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
F=source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/render_front_grasp.py
GL_WRIST_RANGE=0.4 timeout 900 ./isaaclab.sh -p $F --headless --grasp_lift --steps 420 --num_envs 16 --camera side --tag $2 --table_diag \
  --raise_residual --raise_max_s 10 --raise_sag_comp_deg 0,0,0,0,0,0,0,0 --raise_policy $1 \
  --checkpoint logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-10-29_02-09-12_v16_sm3/model_2350.pt > $S/$2.log 2>&1
grep -E "Traceback" -A5 $S/$2.log | head
grep "^RAISE" $S/$2.log | awk 'NR%2==0' | cut -c1-170
grep "^HANDOVER" $S/$2.log | python3 -c "import sys,json; d=json.loads(sys.stdin.read()[9:]); print('HANDOVER t',d['handover_t_s'],d['handover_by'],'err_max',d['err_to_start_deg_max_all_envs'],'keepout_mm',d['raise_keepout_max_mm'],'contact_N',d['raise_hand_contact_max_n'],'cube_mm',d['cube_moved_mm'])"
grep "^CHAIN" $S/$2.log | python3 -c "import sys,json; d=json.loads(sys.stdin.read()[6:]); r=sorted(d['cube_rise_mm_at_end']); print('lifted',d['lifted_4cm_frac'],'rise_med',r[len(r)//2],'dvel',d['max_dvel_deg_s_per_step'])"
D=$(grep FRONT_GRASP_DIR $S/$2.log | cut -d= -f2)
./isaaclab.sh -p - <<PY 2>&1 | tail -1
import imageio.v3 as iio, numpy as np
fr=[f for i,f in enumerate(iio.imiter("$D/rl-video-step-0.mp4")) if i%45==44]
fr=[f[60:420,150:700] for f in fr][:20]
rows=[np.concatenate(fr[r*5:(r+1)*5],1) for r in range(len(fr)//5)]
iio.imwrite("$S/$2.png", np.concatenate(rows,0)[::2,::2]); print("sheet", len(fr))
PY
echo "VIDEO $D/rl-video-step-0.mp4"
