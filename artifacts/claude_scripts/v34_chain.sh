#!/bin/bash
# v34: full chain (raise v25 model_1197 + grasp CK, IK lift, L return) at table dz -3 / 0 / +3 cm, 9 cube spots each,
# one montage per height. usage: v34_chain.sh CK
cd /home/jeongwoo/isaacsim/JaxIsaacLab
for dz in ${DZS:--0.03 0.0 0.03}; do
  echo "== dz $dz"
  GL_TABLE_DZ_FIXED=$dz GL_TABLE_REL=1 RETL=1 STEPS=660 LIFTIK=1 IKP=0.3 IKVZ=0.035 TAGSUF=_tdz$dz bash artifacts/claude_scripts/v27_montage.sh $1 front
done
