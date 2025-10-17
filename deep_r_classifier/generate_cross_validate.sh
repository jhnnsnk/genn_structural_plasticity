#!/bin/bash
for l in 0.01 0.05 0.1 0.5; do
    for f in 0 1 2 3 4 5 6 7; do
        echo "--dataset shd --fold ${f} --record-rewiring --num-epochs 50 --seed 1234 --dataset-threshold 1 --l1-strength ${l} --hidden-size 512 --hidden-recurrent True --hidden-model alif --hidden-input-sparsity 0.05 --hidden-recurrent-sparsity 0.01 --row-padding-prop 5.0 --rewire"
    done
done

