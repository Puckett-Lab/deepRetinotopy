# DeepRetinotopy

This repository contains all Python codes from our recent work on "Predicting brain function from anatomy using geometric deep learning" available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.02.11.934471v2).

## How to install pytorch and torch_geometric to run locally on cpu

- Create a new conda environment (or docker container)
- Install pytorch first:
	
	conda install pytorch==1.5.0 torchvision cpuonly -c pytorch
- Install torch-scatter, torch-sparse, torch-cluster, tprch-spline-conv and torch-geometric:
	 
	 pip install torch-scatter==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.5.0.html
	 
	 pip install torch-sparse==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.5.0.html
	 
	 pip install torch-cluster==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.5.0.html
	 
	 pip install torch-spline-conv==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.5.0.html
	 
	 pip install torch-geometric

## Packages

- 

## Setting up the dataset for preprocessing




## Figures codes

### Figure 1
./plots/VisualCortexHierarchy_WangPlusFovea.py

### Figure 3

#### Left Hemisphere
./plots/left_hemi/Eccentricity_maps_LH.py
./plots/left_hemi/PA_maps_LH.py

#### Right Hemisphere
./plots/right_hemi/Eccentricity_maps_LH.py
./plots/right_hemi/PA_maps_LH.py

### Figure 4a
./plots/left_hemi/Eccentricity_maps_LH.py
./plots/left_hemi/PA_maps_LH.py

### Figure 4b
./plots/DeltaThetaVisualCortex_PA_subjvsGroup.py

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
./plots/DiffNumberLayersErrorPlots_LH.py

## Final model selection

./plots/left(right)_hemi/ModelEval_FinalModel_PA.py

./plots/left(right)_hemi/ModelEval_FinalModel_ecc.py

## Load model and apply to test dataset

./ModelEvalGeneralizability_ecc.py
./ModelEvalGeneralizability_PA.py


## Mean error and standard deviation

./plots/left(right)_hemi/ModelEval_MeanDeltaTheta_ecc.py

./plots/left(right)_hemi/ModelEval_MeanDeltaTheta_PA.py