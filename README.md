# DeepRetinotopy

This repository contains all Python source code necessary to replicate our recent work entitled "Predicting brain function from anatomy 
using geometric deep learning" available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.02.11.934471v2).

## Table of Contents
* [General information](#general-info)
* [Installation and requirements](#installation-and-requirements)
* [Processed data](#processed-data)
* [Figures](#figures)

## General Information

## Installation and requirements 

Models were generated using Pytorch Geometric. Since this package is under constant updates, we highly recommend that 
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

Final model (among 5 that were trained with the final architecture) was selected based on their performance using the 
development dataset. Scripts for this assessment are:

- ./plots/left(right)_hemi/ModelEval_FinalModel_ecc.py
- ./plots/left(right)_hemi/ModelEval_FinalModel_PA.py
## Descriptive statistics

Explained variance and mean errors of predictions based on our model vs an average map were determined with the following scripts:

- ./plots/left(right)_hemi/ModelEval_explainedVar_ecc.py
- ./plots/left(right)_hemi/ModelEval_explainedVar_PA.py

### Figure 5
./plots/R2Average_plot.py

### Figure 6a
./plots/DeltaThetaVisualCortex_PA.py

### Figure 6b
./plots/left_hemi/ModelEval_featureImportance_PA.py

### Supplementary Figure 1
#### Left Hemisphere
./plots/left_hemi/PA_maps_LH.py

#### Right Hemisphere
./plots/right_hemi/PA_maps_RH.py

### Supplementary Figure 2
#### Left Hemisphere
./plots/left_hemi/Eccentricity_maps_LH.py

#### Right Hemisphere
./plots/right_hemi/Eccentricity_maps_RH.py


### Supplementary Figure 3
#### Left Hemisphere
./plots/left_hemi/PA_maps_LH.py

### Supplementary Figure 4
./plots/left_hemi/DiffNumberLayersErrorPlots_LH.py

## Final model selection

./plots/left(right)_hemi/ModelEval_FinalModel_PA.py

./plots/left(right)_hemi/ModelEval_FinalModel_ecc.py

## Load model and apply to test dataset

./ModelEvalGeneralizability_ecc.py
./ModelEvalGeneralizability_PA.py


## Mean error and standard deviation

./plots/left(right)_hemi/ModelEval_MeanDeltaTheta_ecc.py

./plots/left(right)_hemi/ModelEval_MeanDeltaTheta_PA.py