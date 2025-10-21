# GeNN Structural Plasticity

DEEP R classifier and topographic map formation implemented in [GeNN](https://genn-team.github.io).  
This repository contains the code and data to reproduce the figures of the manuscript:

A flexible framework for structural plasticity in GPU-accelerated sparse spiking neural networks  
James C. Knight, Johanna Senk, Thomas Nowotny

If you (re)use the code or otherwise find this resource useful, please respect the [MIT License](https://github.com/jhnnsnk/genn_structural_plasticity/blob/main/LICENSE) agreement and cite the corresponding scientific publication.

## DEEP R classifier

The files are in the directory `deep_r_classifier`. The code was tested on a variety of GPU systems with GeNN 5.2.0, mlGeNN 2.4.0 and Python 3.11.3.

The entry point is `deep_r_classifier/classifier.py` which has is completely controlled via a command line interface. The `--help` argument provides full documentation but for example:

`classifier.py --mode train --dataset dvs_gesture --record-rewiring --num-epochs 50 --seed 1234 --dataset-threshold 1 --l1-strength 0.01 --hidden-size 512 --hidden-recurrent True --hidden-model alif --hidden-input-sparsity 0.01 --hidden-recurrent-sparsity 0.01 --row-padding-prop 2.0 --rewire`

Will train a model with 512 recurrently connected ALIF neurons for 50 epochs on the DVS gesture dataset with standard pre-processing applied.

## Topographic map formation

The model is inspired by [Bamford et al. (2010)](https://dx.doi.org/10.1016/j.neunet.2010.01.005) and [Bogdan et al. (2018)](https://dx.doi.org/10.3389/fnins.2018.00434).

The files are in the directory `topomap`. The code was tested on a single GPU with GeNN 5.3.0 and Python 3.12.3 (numpy 1.26.4 and matplotlib 3.8.4).

The main file is `topomap/run_experiments.py`. Per default a test simulation is run (parameterset `key = 1`).
To reproduce the plots of the manuscript (`key = 2` to `key = 4`), the data in `topomap/data_ms` is used. Set `do_run_model = True` to rerun the simulations.





