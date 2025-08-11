f="$1"
bsub -J "$(basename "$f" .lsf)" < "$f"