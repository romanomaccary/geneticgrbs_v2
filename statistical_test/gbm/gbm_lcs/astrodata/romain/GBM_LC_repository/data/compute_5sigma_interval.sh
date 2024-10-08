#!/bin/bash

# This program computes the 5-sigma interval given a background-subtracted GRB 
# LC.

thr=5
mrf=128

fermi_ids="fermi_ids.txt"
outfile="5sigma.txt"

while read -r fermi_id; do
    cd "${fermi_id}/LC/"
    lc=`ls -l | awk '{print $9}'`
    echo ${lc}
    crg_find_Tdet_BAT_masklc_fixsigma.exe ${lc} ${thr} ${mrf} > ${outfile}
    cd ..
    cd ..
done < ${fermi_ids}