#!/bin/bash

# This program makes a tar file containing the list of paths of the 
# background-subtracted light-curves.

declare -a path_list

infile="path_list.txt"

while read -r line; do
    path_list+=("${line}")
done < ${infile}

outfile="lc.tar.gz"

tar -czvf "${outfile}" "${path_list[@]}"