#!/bin/bash
# usage: pre_cycle.sh LOAD_RUN CHECKPOINT RUN_NAME ITERS  (LOAD_RUN "-" = fresh)
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
LOG=$S/$3.log
RES=""; [ "$1" != "-" ] && RES="--resume --load_run $1 --checkpoint $2"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-JaxV2-FrontGraspLift-Direct-v16 --num_envs 7056 --headless --max_iterations $4 $RES --run_name $3 > $LOG 2>&1 &
TP=$!
sleep 90
# Isaac Sim sometimes segfaults at startup right after another run; retry once.
if ! kill -0 $TP 2>/dev/null && grep -q "Segmentation fault" $LOG && ! grep -q "Learning iteration" $LOG; then
  echo "RETRY after startup segfault"; sleep 20
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-JaxV2-FrontGraspLift-Direct-v16 --num_envs 7056 --headless --max_iterations $4 $RES --run_name $3 > $LOG 2>&1 &
  TP=$!; sleep 90
fi
# Watchdog: a log that stops growing for 120 s means the run hung.
while kill -0 $TP 2>/dev/null; do
  age=$(( $(date +%s) - $(stat -c %Y $LOG) ))
  if [ $age -gt 120 ]; then echo "HUNG (log idle ${age}s) at $(grep 'Learning iteration' $LOG | tail -1)"; pkill -P $TP; kill $TP; ps -eo pid,args | grep "run_name $3" | grep -v grep | awk '{print $1}' | xargs -r kill -9; break; fi
  sleep 15
done
grep -A5 Traceback $LOG | head
first=$(grep "Learning iteration" $LOG | head -1 | sed -E 's/.*iteration ([0-9]+)\/.*/\1/')
last=$(grep "Learning iteration" $LOG | tail -1 | sed -E 's/.*iteration ([0-9]+)\/.*/\1/')
for it in $((first+50)) $((first+125)) $last; do awk "/Learning iteration $it\//,/ETA/" $LOG | grep -E "pre_err_m|pre_height_err_m|pre_pad_facing|pre_thumb_side|pre_front_quality|pre_hand_dir_z|pre_touching|pre_success|pre_arm_speed|pre_arm_action_sq|pre_thumb_facing|pre_hand_yaw_deg|pre_wrist_bend|pre_track_err|pre_table_close|pre_descent|pre_cube_shift|grasp_both|grasp_thumb_on|grasp_finger_on|grasp_side_n|grasp_close_frac|pre_touch_before_close|lift_rise_m|lift_held|lift_success|lift_clean|lift_hand_rot_deg|lift_slip_mm|lift_xy_mm|lift_rot_deg|lift_hand_rise_m|gl_pads_on|gl_all_distal|gl_table_force_n|gl_spawn_xy_mm|gl_lifted|gl_straight|gl_rise_mm|gl_r_xy_vel|gl_rot_deg|gl_hand_rot_deg|gl_arm_speed_deg_s|gl_still" | awk -v it=$it '{printf "%s=%s ", $3, $NF} END{print "@" it}'; done
R=$(ls -d logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/*_$3 | tail -1); M=$(ls $R/model_*.pt | sort -V | tail -1); echo $M
for cam in close_front side; do timeout 400 ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/render_front_grasp.py --headless --grasp_lift --intro_s 4 --steps 720 --camera $cam --checkpoint $M 2>&1 | grep -E "^FGRASP|FRONT_GRASP_DIR|Error" | python3 -c "
import sys,json
for l in sys.stdin:
  if l.startswith('FGRASP '):
    d=json.loads(l[7:])
    if d['t'] in (4.0,5.5,6.5,8.0,10.0,11.5): print(d['t'],'fing',d['fingers_deg_env0'],'q',list(d['q_deg_env0'].values()),'dir',d['hand_dir_env0'],'pad',d['pad_N_env0'],'rise',d['cube_rise_mm'][0])
  else: print(l.strip()[:200])"; done
