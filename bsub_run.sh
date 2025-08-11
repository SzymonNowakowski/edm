#!/bin/bash
# bsub_auto.sh
fname="$1"
jobname="${fname%.lsf}"
bsub -J "$jobname" \
     -o "${jobname}_%J.out" \
     -e "${jobname}_%J.err" \
     < "$fname"