#!/bin/bash
# v36 (user: table height +/-1 cm): chain (raise v25 model_1197 + grasp CK) at the given table dz values, 9 spots each,
# skipping spots already rendered. usage: DZS="-0.01 0.01" v36_chain.sh CK
cd /home/jeongwoo/isaacsim/JaxIsaacLab
for dz in ${DZS:--0.01 0.01}; do
  echo "== dz $dz"
  GL_TABLE_DZ_FIXED=$dz GL_TABLE_REL=1 RETL=1 STEPS=660 LIFTIK=1 IKP=0.3 IKVZ=0.035 TAGSUF=_tdz$dz bash artifacts/claude_scripts/v34_montage_fill.sh $1 front
done
