#!/bin/bash

# This program computes the 7-sigma time interval given a background-subtracted 
# gamma-ray burst light curve.

thr=7   # sigma threshold
mrf=128 # maximum rebin factor

path="../gbm_lcs/astrodata/romain/GBM_LC_repository/data"
# file="fermi_ids.txt"

for fermi_id in `ls -l ${path} | awk '{print $9}'`; do
    dir="${path}/${fermi_id}/LC"
    lc=`ls -l ${dir} | tail -1 | awk '{print $9}'`
    echo ${lc}
    outfile="${dir}/7sigma.txt"
    crg_find_Tdet_BAT_masklc_fixsigma.exe "${dir}/${lc}" "${thr}" "${mrf}" > "${outfile}"
done