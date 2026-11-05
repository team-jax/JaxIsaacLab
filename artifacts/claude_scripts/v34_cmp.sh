#!/bin/bash
# v34: 144-env (9 spots x 16) deterministic eval with the L return, table dz -3/0/+3, per checkpoint
cd /home/jeongwoo/isaacsim/JaxIsaacLab
for f in "$@"; do
  line="$(basename $(dirname $f))/$(basename $f)"
  for dz in ${DZS:--0.03 0.0 0.03}; do
    a=$(NE=144 GL_RETURN_L=1 GL_TABLE_REL=1 GL_TABLE_DZ_FIXED=$dz GL_LIFT_IK=1 GL_IK_PITCH=0.3 GL_IK_VZ=0.035 GL_IK_VXY=0.005 GL_WRIST_RANGE=0.4 GL_START_NOISE_DEG=0.75 timeout -k 20 600 ./isaaclab.sh -p artifacts/claude_scripts/v34_evalN.py $f 0 2>&1 | grep "^ALL" | awk '{print $NF}')
    line="$line dz$dz=${a:-ERR}"
  done
  echo "CMP $line"
done
