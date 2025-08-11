#!/bin/bash
# bsub_auto.sh
fname="$1"
jobname="${fname%.lsf}"
bsub -J "$jobname" \
     -o "edm_${jobname}_%J.out" \
     -e "edm_${jobname}_%J.err" \
     < "$fname"