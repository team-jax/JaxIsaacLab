#!/bin/bash
# usage: raise_cycle.sh LOAD_RUN CHECKPOINT RUN_NAME ITERS  (LOAD_RUN "-" = fresh). v19 learned raise.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad; mkdir -p $S
LOG=$S/$3.log
RES=""; [ "$1" != "-" ] && RES="--resume --load_run $1 --checkpoint $2"
run() { ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-JaxV2-RaiseLearned-Direct-v19 --num_envs 4096 --headless --max_iterations $4 $RES --run_name $3 > $LOG 2>&1 & TP=$!; sleep 90; }
run "$@"
if ! kill -0 $TP 2>/dev/null && grep -q "Segmentation fault" $LOG && ! grep -q "Learning iteration" $LOG; then echo RETRY; sleep 20; run "$@"; fi
while kill -0 $TP 2>/dev/null; do
  age=$(( $(date +%s) - $(stat -c %Y $LOG) ))
  if [ $age -gt 120 ]; then echo "HUNG at $(grep 'Learning iteration' $LOG | tail -1)"; kill $TP; ps -eo pid,args | grep "run_name $3" | grep -v grep | awk '{print $1}' | xargs -r kill -9; break; fi
  sleep 15
done
grep -A5 Traceback $LOG | head
first=$(grep "Learning iteration" $LOG | head -1 | sed -E 's/.*iteration ([0-9]+)\/.*/\1/')
last=$(grep "Learning iteration" $LOG | tail -1 | sed -E 's/.*iteration ([0-9]+)\/.*/\1/')
for it in $((first+50)) $((first+125)) $last; do awk "/Learning iteration $it\//,/ETA/" $LOG | grep -E "raise_|Mean reward" | awk -v it=$it '{printf "%s=%s ", $(NF-1), $NF} END{print "@" it}'; done
R=$(ls -d logs/rsl_rl/jax_v2_1006_raise_learned_v19/*_$3 | tail -1); ls $R/model_*.pt | sort -V | tail -1
