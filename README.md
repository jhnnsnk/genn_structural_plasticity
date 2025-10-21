# GeNN Structural Plasticity

DEEP R classifier and topographic map formation implemented in [GeNN](https://genn-team.github.io).  
This repository contains the code and data to reproduce the figures of the manuscript:

A flexible framework for structural plasticity in GPU-accelerated sparse spiking neural networks  
James C. Knight, Johanna Senk, Thomas Nowotny

If you (re)use the code or otherwise find this resource useful, please respect the [MIT License](https://github.com/jhnnsnk/genn_structural_plasticity/blob/main/LICENSE) agreement and cite the corresponding scientific publication.

## DEEP R classifier

## Topographic map formation

The model is inspired by [Bamford et al. (2010)](https://dx.doi.org/10.1016/j.neunet.2010.01.005) and [Bogdan et al. (2018)](https://dx.doi.org/10.3389/fnins.2018.00434).

The files are in the directory `topomap`. The code was tested on a single GPU with GeNN 5.3.0 and Python 3.12.3 (numpy 1.26.4 and matplotlib 3.8.4).

The main file is `topomap/run_experiments.py`. Per default a test simulation is run (parameterset `key = 1`).
To reproduce the plots of the manuscript (`key = 2` to `key = 4`), the data in `topomap/data_ms` is used. Set `do_run_model = True` to rerun the simulations.





