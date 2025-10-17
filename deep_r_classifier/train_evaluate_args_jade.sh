#!/bin/bash
ARGS=`head -${SLURM_ARRAY_TASK_ID} arguments.txt | tail -1`

python -u classifier.py --mode train $ARGS
python -u classifier.py --mode test $ARGS
