# DeepRetinotopy

This repository contains all source code necessary to replicate our recent work entitled "Predicting brain function from anatomy 
using geometric deep learning" available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.02.11.934471v2).

## Table of Contents
* [General information](#general-info)
* [Installation and requirements](#installation-and-requirements)
* [Processed data](#processed-data)
* [Figures](#figures)
* [Descriptive statistics](#descriptive-statistics)
* [Generalization](#generalization)
* [License](#license)

## General Information


## Installation and requirements 

Models were generated using [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/). Since this package is under constant updates, we highly recommend that 
you follow the following steps:

- Create a conda environment (or docker container)
- Install pytorch first:
	
	
	conda install pytorch==1.5.0 torchvision cpuonly -c pytorch
	
- Install torch-scatter, torch-sparse, torch-cluster, torch-spline-conv and torch-geometric:
	 
	 
	pip install torch-scatter==1.0.4

	pip install torch-sparse==0.2.2

	pip install torch-cluster==1.1.5

	pip install torch-spline-conv==1.0.4

    pip install torch-geometric==0.3.1


- Install the remaining required packages that are available at requirements.txt: 


    pip install -r requirement.txt
    
- Clone DeepRetinotopy:


    git clone git@github.com:Puckett-Lab/deepRetinotopy.git

Finally, note that in order to generate the left and right hemisphere 
views of the retinotopic maps (as in our manuscript), you have to install the following git repository:

    pip install git+https://github.com/felenitaribeiro/nilearn.git


## Processed data

All processed data are available at https://osf.io/95w4y/.

## Figures

Polar angle and eccentricity maps, as shown in the manuscript (Figure 3-5), were generated using the following scripts:

- ./plots/left(right)_hemi/Eccentricity_maps_L(R)H.py
- ./plots/left(right)_hemi/PA_maps_L(R)H.py

Final model (among the five models that were trained with the final architecture) was selected based on their performance using the 
development dataset. Scripts for this assessment are:

- ./plots/left(right)_hemi/ModelEval_FinalModel_ecc.py
- ./plots/left(right)_hemi/ModelEval_FinalModel_PA.py

## Descriptive statistics

Explained variance and mean error of our models' predictions versus an average map were determined with the following scripts:

- ./plots/left(right)_hemi/ModelEval_explainedVar_ecc.py
- ./plots/left(right)_hemi/ModelEval_explainedVar_PA.py
- ./plots/left(right)_hemi/ModelEval_MeanDeltaTheta_ecc.py
- ./plots/left(right)_hemi/ModelEval_MeanDeltaTheta_PA.py


## Generalization

Scripts to load our model and apply to the test dataset are the following:

- ./ModelEvalGeneralizability_ecc.py
- ./ModelEvalGeneralizability_PA.py

## Citation

Please cite our paper if you use our model or if it was somewhat useful for you :wink:

    @article{Ribeiro2020,
        title = {{Predicting brain function from anatomy using geometric deep learning}},
        author = {Ribeiro, Fernanda L and Bollmann, Steffen and Puckett, Alexander M},
        doi = {10.1101/2020.02.11.934471},
        journal = {bioRxiv},
        url = {http://biorxiv.org/content/early/2020/02/12/2020.02.11.934471.abstract},
        year = {2020}
    }


## Contact
Fernanda L. Ribeiro <[fernanda.ribeiro@uq.edu.au](fernanda.ribeiro@uq.edu.au)>


## License

