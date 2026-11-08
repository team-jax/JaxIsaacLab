#!/bin/bash
# v66: 64-env headless chains at the control spots (+2,-28) and (-6,0) with and without the pose-dependent gravity compensation
# (16-env runs gave 13/16 and 14/16 with it vs 15/16 and 16/16 without: noise or real?), then the outer (+2,+28) both ways.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
T=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/tasks/bks94tw84.output
until grep -q V66_GRAV_DONE $T 2>/dev/null; do sleep 20; done
F=source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/render_front_grasp.py
C=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-07_20-52-49_v65_r6/model_18125.pt
export CHAIN_NO_VIDEO=1 GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_REACH_DX=0 GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_TABLE_REL=1 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0 RAISE_GRID_SET=v65r REACH_FINAL_GRID_SET=v65f
training() { local e; e=$(ps -eo etimes,args | grep -v grep | grep "train.py" | grep "run_name v66_r" | awk 'NR==1 {print $1}'); [ -n "$e" ] && [ "$e" -lt 330 ]; }
for spot in "0.02,-0.28" "-0.06,0.0" "0.02,0.28"; do
  for g in 0 1; do
    until training; do sleep 15; done
    tag=v66g64_${spot/,/_}_g$g
    env CHAIN_GRAV_COMP=$([ $g = 1 ] && echo 1) GL_RETURN_L=1 GL_LIFT_IK=1 GL_IK_VZ=0.035 RAISE_WIDE=1 GL_WRIST_RANGE=0.4 timeout -k 20 600 ./isaaclab.sh -p $F --headless --grasp_lift --num_envs 64 --steps 660 --camera front --tag $tag --raise_cube=$spot --raise_max_s 11 --raise_sag_comp_deg 0,0,0,0,0,0,0,0 --raise_policy nominal --checkpoint $C > $SP/$tag.log 2>&1
    J=$(ls -d $(dirname $C)/videos/*_$tag | tail -1)/chain.json
    echo "G64 $spot grav=$g $(python3 -c "import json;d=json.load(open('$J'));print('lifted',d['lifted_4cm_frac'],'done',d.get('lift_done_frac'),'handover',d['handover_by'],d['handover_t_s'],'pad_N',d['raise_hand_contact_max_n'],'moved',d['cube_moved_mm'])" 2>&1 | tail -1)"
  done
done
echo V66_G64_DONE
