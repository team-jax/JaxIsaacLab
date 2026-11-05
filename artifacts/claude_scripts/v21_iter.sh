#!/bin/bash
# usage: v21_iter.sh LOAD_RUN CHECKPOINT RUN_NAME ITERS -- train a slice, then chain video test
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
bash artifacts/claude_scripts/raise21_cycle.sh $1 $2 $3 $4 > $S/$3.cycle 2>&1
M=$(tail -1 $S/$3.cycle); echo "CKPT $M"
awk '/Learning iteration/{split($0,a,"iteration ");split(a[2],b,"/");it=b[1]} /raise_ref_err_deg:/{e=$NF} /raise_contact_n:/{c=$NF} /r_residual:/{r=$NF} /r_dqd:/{q=$NF} /raise_settled:/{if(it%50==0) printf "%s ref_err=%s contact=%s r_res=%s r_dqd=%s settled=%s\n", it,e,c,r,q,$NF}' $S/$3.log
bash artifacts/claude_scripts/v21_test.sh $M $(basename $(dirname $M))_$(basename $M .pt)
