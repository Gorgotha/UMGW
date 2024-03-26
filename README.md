# Multi-Marginal Gromov-Wasserstein Transport and Barycentres

This repository contains the code for the paper 'Multi-Marginal Gromov-Wasserstein Transport and Barycentres'. 
Information and Inference: A Journal of the IMA 12(4), pp. 2753–2781, 2023.
A preprint version is available on [arXiv](https://arxiv.org/abs/2205.06725).

Please cite the paper if you use the code.

## Citation
1. Florian Beier, Robert Beinert, Gabriele Steidl,
   'Multi-Marginal Gromov-Wasserstein Transport and Barycenters',
   Information and Inference: A Journal of the IMA 12(4), pp. 2753–2781, 2023.

## Requirements
The simulations have been performed with Python 3.8.8 and rely on 

* numpy 1.24.2,
* scipy 1.10.0,
* matplotlib 3.7.1,
* pot 0.8.0,
* tqdm 4.62.3,
* anytree 2.8.0,
* torch 1.10.0, 
* scikit-learn 1.0.1,
* opencv-python 4.5.3.56,
* plotly 5.3.1,
* networkx 2.6.3,
* seaborn 0.11.2.

## Experiments
The numerical simulations can be reproduced using the script

* `MGW_complexity_test_euclidean.ipynb` 
    for the bimarginal spade-heart barycentre experiment (Figs 1, 2, 3),

* `MGW_complexity_test_euclidean-domain-multi-marginals.ipynb` 
    for the multimarginal heart barycentre experiment (Figs 4, 5),

* `non-flat-domains-progressive.ipynb` 
    for the progressive interpolation experiment (Fig 6),

* `fused_81_98.ipynb`
    for the fused MNIST experiment (Figs 7, 8),
  
* `fused_camels.ipynb` 
    for the fused camel experiment (Fig 9),

* `particle_transfer.ipynb`
    for the particle transfer experiment (Figs 10, 11).

Parts of the implementation relies on or is built on top of existing implementations from 
[Python Optimal Transport](https://pythonot.github.io/) and
[Unbalanced Gromov-Wasserstein Divergence](https://github.com/thibsej/unbalanced_gromov_wasserstein).
Some of the input data is based on
[The 2D shape structure dataset](https://2dshapesstructure.github.io).

## Contributing
The code is available under a MIT license.
