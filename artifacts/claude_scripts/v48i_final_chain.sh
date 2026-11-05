#!/bin/bash
# resume after the user's pause: 576-env evals of the rest of v48i2 r1, pick the best (score with caps 3 N / 0.5 N),
# robustness (table -1/+1 cm, friction 0.35) at 288 envs, chain render, then v48i3 training from the best.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
E="GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 ROLL=12"
for ck in 2026-11-05_12-45-51_v48i2_r1/model_12850.pt 2026-11-05_12-45-51_v48i2_r1/model_12900.pt; do
  echo "## eval $ck"; env $E NE=576 bash artifacts/claude_scripts/v46_big_cmp.sh $ck
done
BEST=$(python3 - "$SP/v48i_final.out" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
c = {"2026-11-05_12-45-51_v48i2_r1/model_12750.pt": (0.993, 2.3, 3.9, 4.799, 0.843),
     "2026-11-05_12-45-51_v48i2_r1/model_12800.pt": (0.984, 2.2, 3.8, 5.271, 1.104)}
for blk in s.split("## eval ")[1:]:
    ck = blk.split()[0]
    b = re.search(r"held ([\d.]+) done .*lift_tilt ([\d.]+) lift_yaw ([\d.]+)", blk)
    p = re.search(r"PROX_DONE first_joint_N_at_lift_done \[([\d.]+), ([\d.]+), ([\d.]+)\] sum ([\d.]+)", blk)
    if b and p:
        c[ck] = (float(b.group(1)), float(b.group(2)), float(b.group(3)), float(p.group(4)), min(float(p.group(i)) for i in (1, 2, 3)))
def score(v): return v[0] - 0.01 * (v[1] + v[2]) + 0.025 * min(v[3] / 3.0, 1) + 0.05 * min(v[4] / 0.5, 1)
for k, v in sorted(c.items(), key=lambda kv: -score(kv[1])): print("SCORE %.4f %s %s" % (score(v), k, v), file=sys.stderr)
print(max(c, key=lambda k: score(c[k])))
PY
)
echo "FINAL_PICK $BEST"
for c in "DZ=-0.01" "DZ=0.01" "MU=0.35,0.3"; do
  echo "## robust $c"; env $c $E NE=288 bash artifacts/claude_scripts/v46_big_cmp.sh $BEST
done
echo "## render"
(export GL_HOLD_DIGIT=1 GL_REACH_DX=-0.025 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_IK_POS=2 GL_GRAV_COMP=1 GL_TABLE_EDGE=0.225 GL_MU=0.4,0.35 GL_SQUEEZE=6,0 GL_TABLE_DZ_FIXED=0.0 GL_TABLE_REL=1; RETL=1 STEPS=660 LIFTIK=1 IKP= IKVZ=0.035 TAGSUF=_v48i2_final bash artifacts/claude_scripts/v27_montage.sh logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/$BEST front)
echo RENDER_DONE
RUNB=$(dirname $BEST); CKB=$(basename $BEST)
sed -e "s#^RUN=2026-11-05_12-33-16_v48i_r1; CK=model_12725.pt#RUN=$RUNB; CK=$CKB#" -e 's#B=artifacts/claude_scripts/v48i2_best.txt#B=artifacts/claude_scripts/v48i3_best.txt#' -e 's#tag=v48i2_r$r#tag=v48i3_r$r#' -e 's#SEED=$((1600 + r))#SEED=$((1700 + r))#' artifacts/claude_scripts/v48i2_full.sh > artifacts/claude_scripts/v48i3_full.sh
V0L=$(grep -A6 "## eval $BEST" $SP/v48i_final.out | grep -m1 "^BIG" || true)
echo "v48i3 from $RUNB/$CKB"
bash artifacts/claude_scripts/v48i3_full.sh > $SP/v48i3.out 2>&1
