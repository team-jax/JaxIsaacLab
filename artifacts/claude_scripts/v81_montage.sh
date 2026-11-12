#!/bin/bash
# v81b montage of the team-layout chain (CHAIN_TEAM=1, camera "team": from beyond the table), 9 spots x 8 envs, 3x3. $1 = checkpoint,
# $2 = the training loop's bash PID (paused with SIGSTOP while this renders; no rendering during training).
cd /home/jeongwoo/isaacsim/JaxIsaacLab
C=$1; LOOP=$2
while pgrep -f "python3 scripts/reinforcement_learning/rsl_[r]l/train[.]py" > /dev/null; do sleep 15; done
[ -n "$LOOP" ] && kill -STOP $LOOP && echo "LOOP_PAUSED $LOOP"
sleep 2; pgrep -f "python3 scripts/reinforcement_learning/rsl_[r]l/train[.]py" > /dev/null && { echo "TRAINING_RUNNING abort"; [ -n "$LOOP" ] && kill -CONT $LOOP; exit 1; }
SPOTS="${SPOTS:-0.04,0.22 0.10,0.22 0.24,0.22 0.06,0.25 0.16,0.25 0.25,0.25 0.10,0.28 0.18,0.295 0.25,0.295}"
SPOTS="$SPOTS" NOVID= CAM=team TAGSUF=_vid bash artifacts/claude_scripts/v81_chain_check.sh $C
D=$(dirname $C)/videos; M=$(basename $C .pt); ARGS=""
for xy in $SPOTS; do x=${xy%,*}; y=${xy#*,}; lab=$(python3 -c "print('x%+.0fcm,y%+.1fcm'%($x*100,$y*100))")
  ARGS="$ARGS $lab=$(ls -d $D/*_${M}_tc_${x}_${y}_vid | tail -1)/rl-video-step-0.mp4"; done
OUT=$D/montage_team_${M}.mp4
./isaaclab.sh -p artifacts/claude_scripts/montage.py $OUT 3 $ARGS 2>&1 | grep MONTAGE
[ -n "$LOOP" ] && kill -CONT $LOOP && echo "LOOP_RESUMED $LOOP"
echo V81_MONTAGE_DONE
