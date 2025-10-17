#!/bin/bash
LINE=$(((($SLURM_ARRAY_TASK_ID - 1)* 4) + $SLURM_PROCID + 1))
ARGS=`sed "${LINE}q;d" arguments.txt`

if [ -n "${ARGS}" ]; then
    python -u classifier.py --mode train $ARGS
    python -u classifier.py --mode test $ARGS
fi
