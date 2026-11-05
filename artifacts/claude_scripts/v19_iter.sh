#!/bin/bash
# usage: v19_iter.sh LOAD_RUN CHECKPOINT RUN_NAME ITERS -- train a slice, then video-test its last checkpoint
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
bash artifacts/claude_scripts/raise19_cycle.sh $1 $2 $3 $4 > $S/$3.cycle 2>&1
M=$(tail -1 $S/$3.cycle); echo "CKPT $M"
awk '/Learning iteration/{split($0,a,"iteration ");split(a[2],b,"/");it=b[1]} /raise_ref_err_deg:/{e=$NF} /r_sharp:/{sh=$NF} /raise_settled:/{if(it%50==0) printf "%s ref_err=%s sharp=%s settled=%s\n", it,e,sh,$NF}' $S/$3.log
bash artifacts/claude_scripts/v19_test.sh $M $(basename $(dirname $M))_$(basename $M .pt)
timeout 600 ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/render_raise_l.py --headless --v19 --camera side --tag eval --checkpoint $M 2>&1 | grep "^RAISE_L_RESULT" | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()[15:]); print("EVAL64 success", d["L_success_rate"], "err_per_joint", d["final_err_deg_mean_per_joint"], "contact_envs", d["envs_with_contact_over_0.5n"], "max_contact", max(d["table_or_any_contact_max_n_per_link"].values()))'
