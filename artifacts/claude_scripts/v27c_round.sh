#!/bin/bash
# usage: v27c_round.sh RAISE_RUN RAISE_CK GRASP_RUN GRASP_CK TAG RAISE_IT GRASP_IT NOISE
# one round: raise retrain -> grasp retrain (start noise) -> grasp eval (9 pos, noisy start) -> chain montage of best
cd /home/jeongwoo/isaacsim/JaxIsaacLab
S=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
if [ "$6" -gt 0 ]; then echo "== RAISE"; bash artifacts/claude_scripts/raise27_cycle.sh $1 $2 $5 $6; fi
RR=$(ls -d logs/rsl_rl/jax_v2_1006_raise_cube_v25/*_$5 2>/dev/null | tail -1); [ -z "$RR" ] && RR=logs/rsl_rl/jax_v2_1006_raise_cube_v25/$1
RM=$(ls $RR/model_*.pt | sort -V | tail -1); echo "RAISE_CK $RM"
echo "== GRASP"; GL_LIFT_EXTRA=${EXTRA:-0} GL_HARD_FRAC=${HARD:-0} GL_HARD_BOX=${HBOX:--0.08,-0.05,-0.05,-0.02} GL_START_NOISE_DEG=$8 GL_CUBE_BOX=1 GL_RISE=8 GL_LATE=8 GL_XY=${XY:-50} GL_XY_VEL=${XYV:-20} GL_DRAG=${DRAG:-0} GL_EXTRA_COST=${XC:-0} GL_REWARD_SCALE=0.02 GL_TABLE=3 GL_NO_VCLIP=1 GL_LR=${LR:-5e-5} GL_WRIST_RANGE=0.4 bash artifacts/claude_scripts/gl_cycle_nr.sh $3 $4 $5 $7
D=$(ls -d logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/*_$5 | tail -1)
best=""; bs=-1
for f in $(ls $D/model_*.pt | sort -V | awk 'NR%3==0'); do
  r=$(GL_LIFT_EXTRA=${EXTRA:-0} GL_START_NOISE_DEG=$8 GL_WRIST_RANGE=0.4 timeout -k 20 400 ./isaaclab.sh -p artifacts/claude_scripts/v27_eval.py $f 0 2>&1 | grep -E "^RES|^ALL" | python3 -c "
import sys,json
o=[];a=''
for l in sys.stdin:
  if l.startswith('RES'): d=json.loads(l[4:]); o.append('(%+.0f,%+.0f)%.2f/%.0f/%.1fN'%(d['dx_cm'],d['dy_cm'],d['lifted'],d['xy_drift_mm'],d['table_N_max']))
  else: a=l.split()[-1]
print(a,' '.join(o))")
  echo "$(basename $f) $r"; s=$(echo $r | cut -d' ' -f1)
  [ -z "$s" ] && continue  # eval failed (e.g. CUDA error)
  if python3 -c "import sys; sys.exit(0 if float('$s')>=float('$bs') else 1)"; then bs=$s; best=$f; fi
done
echo "BEST $best $bs"
sed -i "s|^R=.*; C=\$1; CAM=\$2|R=$RM; C=\$1; CAM=\$2|" artifacts/claude_scripts/v27_montage.sh
bash artifacts/claude_scripts/v27_montage.sh $best front
