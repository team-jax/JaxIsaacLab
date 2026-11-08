#!/bin/bash
# v66: during round 8's training, where do the deployed-path failures sit? (1) 18525 nominal and combined evals over the 12
# reach-map cells with the full per-cell output (CELL / FAIL / DROP lines); (2) 256-env headless chains at (+2,-28), where all
# chain failures were drops while lifting (12-50 mm up), with and without the pose-dependent gravity compensation.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
until grep -q "== ROUND 8" $SP/v66.out; do sleep 15; done
etime() { ps -eo etimes,args | grep -v grep | grep "train.py" | grep "run_name v66_r8" | awk 'NR==1 {print $1}'; }
gate() { local e; while true; do e=$(etime); [ -n "$e" ] && [ "$e" -ge 60 ] && [ "$e" -lt $1 ] && return 0; [ -n "$e" ] && [ "$e" -ge $1 ] && return 1; sleep 10; done; }
L=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16
CK=$L/2026-11-07_23-50-53_v66_r4/model_18525.pt
COMMON="GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_REACH_DX=0 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_LIFT_IK=1 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_TABLE_REL=1 GL_RETURN_L=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15"
HARSH="GL_ACT_DELAY=2 GL_MU=0.35,0.3 GL_CUBE_MASS=0.05,0.05 GL_KP_SCALE=1.5,1.5 GL_OBS_CUBE_NOISE=0.002,0.005"
export EVAL_CELLS="0.02,-0.28;0.02,0.0;0.02,0.28;-0.04,-0.24;-0.04,0.24;-0.08,-0.18;-0.08,0.0;-0.08,0.18;-0.12,-0.14;-0.12,0.12;-0.14,-0.08;-0.14,0.06"
for c in nominal harsh; do
  gate 560 || { echo "SKIP eval $c (window passed)"; continue; }
  fl="GL_MU=0.4,0.35"; [ $c = harsh ] && fl="$HARSH"
  env $COMMON GL_CUBE_YAW_DEG=45 EVAL_YAW_LO=1 NE=576 EVAL_S=12 GL_TABLE_DZ_FIXED=0.0 GL_START_NOISE_DEG=0.75 $fl timeout -k 20 400 ./isaaclab.sh -p artifacts/claude_scripts/v46_eval_done.py $CK 0 > $SP/v66cells_$c.log 2>&1
  echo "== $c"; grep -E "^(CELL|FAIL|DROP|HELD|ALL)" $SP/v66cells_$c.log | cut -c1-300
done
F=source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/render_front_grasp.py
C=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-07_20-52-49_v65_r6/model_18125.pt
for g in 1 0; do
  gate 500 || { echo "SKIP chain g$g (window passed)"; continue; }
  tag=v66g256_m28_g$g
  env CHAIN_NO_VIDEO=1 GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_REACH_DX=0 GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_TABLE_REL=1 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0 RAISE_GRID_SET=v65r REACH_FINAL_GRID_SET=v65f CHAIN_GRAV_COMP=$([ $g = 1 ] && echo 1) GL_RETURN_L=1 GL_LIFT_IK=1 GL_IK_VZ=0.035 RAISE_WIDE=1 GL_WRIST_RANGE=0.4 timeout -k 20 400 ./isaaclab.sh -p $F --headless --grasp_lift --num_envs 256 --steps 660 --camera front --tag $tag --raise_cube=0.02,-0.28 --raise_max_s 11 --raise_sag_comp_deg 0,0,0,0,0,0,0,0 --raise_policy nominal --checkpoint $C > $SP/$tag.log 2>&1
  J=$(ls -d $(dirname $C)/videos/*_$tag | tail -1)/chain.json
  echo "G256 (+2,-28) grav=$g $(python3 -c "import json;d=json.load(open('$J'));print('lifted',d['lifted_4cm_frac'],'done',d.get('lift_done_frac'),'pad_N',d['raise_hand_contact_max_n'],'moved',d['cube_moved_mm'])" 2>&1 | tail -1)"
done
echo V66_CELLS_DONE
