#!/bin/bash
ARGS=`head -${SLURM_ARRAY_TASK_ID} arguments.txt | tail -1`

# Loop through folds
for f in {0..9..1}
do
    python -u classifier.py --mode train_validate --fold $f $ARGS
    python -u classifier.py --mode validate --fold $f $ARGS
done
