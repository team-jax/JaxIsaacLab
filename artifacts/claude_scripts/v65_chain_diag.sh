#!/bin/bash
# v65: after the wrap-up sweeps, diagnose the chain at the far-|y| spots where the raise path failed (18125): with and
# without the v25 raise residual, with per-link table contact (--table_diag). Single spots, 16 envs, no montage.
cd /home/jeongwoo/isaacsim/JaxIsaacLab
SP=/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad
until grep -q "V65_FINAL_DONE" $SP/v65_final.out; do sleep 20; done
F=source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/render_front_grasp.py
C=logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/$(cat $SP/v65_pick.txt)
R=logs/rsl_rl/jax_v2_1006_raise_cube_v25/2026-11-01_21-25-47_v25_cube/model_1197.pt
export GL_CUBE_SIDE=0.05 GL_CUBE_MASS_NOM=0.04 GL_GRID_SET=v65f GL_REACH_DX=0 GL_TABLE_W=1.2 GL_TABLE_Z0=-0.0575 GL_HOLD_DIGIT=1 GL_CLOSE_GAIN=0.0012 GL_CLOSE_SPEED=0.15 GL_RET_RISE=0.04 GL_FINGER_RL=1 GL_ROLL_DEG=12 GL_GRAV_COMP=1 GL_IK_POS=2 GL_TABLE_EDGE=0.225 GL_SQUEEZE=6,0 GL_TABLE_REL=1 GL_CUBE_YAW_DEG=45 GL_MU=0.4,0.35 GL_TABLE_DZ_FIXED=0.0 RAISE_GRID_SET=v65r REACH_FINAL_GRID_SET=v65f
for spot in "0.02,-0.28" "0.02,0.28" "-0.06,0.20"; do
  for pol in "v25 $R" "nominal nominal"; do
    set -- $pol; tag=diag_${spot/,/_}_$1
    GL_RETURN_L=1 GL_LIFT_IK=1 GL_IK_VZ=0.035 RAISE_WIDE=1 GL_WRIST_RANGE=0.4 timeout -k 20 900 ./isaaclab.sh -p $F --headless --grasp_lift --steps 660 --camera front --tag $tag --raise_cube=$spot --raise_max_s 11 --raise_sag_comp_deg 0,0,0,0,0,0,0,0 --raise_policy $2 --table_diag --checkpoint $C > $SP/$tag.log 2>&1
    J=$(ls -d $(dirname $C)/videos/*_$tag | tail -1)/chain.json
    echo "DIAG $spot raise=$1 $(python3 -c "import json;d=json.load(open('$J'));print('lifted',d['lifted_4cm_frac'],'handover',d['handover_by'],d['handover_t_s'],'pad_N',d['raise_hand_contact_max_n'],'moved',d['cube_moved_mm'],'table_N',d['episode_table_force_max_n'])" 2>&1 | tail -1)"
    grep -E "^RAISE_TABLE_DIAG|^REACH_IN" $SP/$tag.log | cut -c1-300
    grep -E "^RAISE " $SP/$tag.log | tail -2 | sed -E 's/"q_deg_env0": (\[[^]]*\])/q \1/' | cut -c1-200
  done
done
echo V65_DIAG_DONE
