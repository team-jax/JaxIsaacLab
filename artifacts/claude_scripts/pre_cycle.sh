#!/bin/bash
# usage: pre_cycle.sh LOAD_RUN CHECKPOINT RUN_NAME ITERS  (LOAD_RUN "-" = fresh)
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
LOG=$S/$3.log
RES=""; [ "$1" != "-" ] && RES="--resume --load_run $1 --checkpoint $2"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-JaxV2-FrontPregrasp-Direct-v12 --num_envs 7056 --headless --max_iterations $4 $RES --run_name $3 > $LOG 2>&1 &
TP=$!
sleep 90
# Watchdog: a log that stops growing for 120 s means the run hung.
while kill -0 $TP 2>/dev/null; do
  age=$(( $(date +%s) - $(stat -c %Y $LOG) ))
  if [ $age -gt 120 ]; then echo "HUNG (log idle ${age}s) at $(grep 'Learning iteration' $LOG | tail -1)"; pkill -P $TP; kill $TP; ps -eo pid,args | grep "run_name $3" | grep -v grep | awk '{print $1}' | xargs -r kill -9; break; fi
  sleep 15
done
grep -A5 Traceback $LOG | head
first=$(grep "Learning iteration" $LOG | head -1 | sed -E 's/.*iteration ([0-9]+)\/.*/\1/')
last=$(grep "Learning iteration" $LOG | tail -1 | sed -E 's/.*iteration ([0-9]+)\/.*/\1/')
for it in $((first+50)) $((first+125)) $last; do awk "/Learning iteration $it\//,/ETA/" $LOG | grep -E "pre_err_m|pre_height_err_m|pre_arm_accel_rms|pre_plate_low_m|pre_table_force_n|pre_ref_dev_rad2|pre_pad_facing|pre_thumb_side|pre_front_quality|pre_hand_dir_z|pre_touching|pre_success|pre_arm_speed|pre_arm_action_sq|pre_thumb_facing|pre_hand_yaw_deg|pre_wrist_bend|pre_track_err|pre_table_close|pre_descent|pre_cube_shift" | awk -v it=$it '{printf "%s=%s ", $3, $NF} END{print "@" it}'; done
R=$(ls -d logs/rsl_rl/jax_v2_1006_front_pregrasp_v12/*_$3 | tail -1); M=$(ls $R/model_*.pt | sort -V | tail -1); echo $M
for cam in front side; do timeout 400 ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/render_front_grasp.py --headless --pregrasp --steps 720 --camera $cam --checkpoint $M 2>&1 | grep -E "^FGRASP|FRONT_GRASP_DIR|Error" | python3 -c "
import sys,json
for l in sys.stdin:
  if l.startswith('FGRASP '):
    d=json.loads(l[7:])
    if d['t'] in (1.0,3.0,6.0,8.0,10.0): print(d['t'],'q',list(d['q_deg_env0'].values()),'dir',d['hand_dir_env0'],'pad',d['pad_N_env0'])
  else: print(l.strip()[:200])"; done
