# DeepRetinotopy

This repository contains all Python codes from our recent work on "Predicting brain function from anatomy using geometric deep learning".

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

## Figures codes

### Figure 1
VisualCortexHierarchy_WangPlusFovea.py

### Figure 3

#### Left Hemisphere
Eccentricity_maps_LH.py
PA_maps_LH.py

#### Right Hemisphere
Eccentricity_maps_LH.py
PA_maps_LH.py

### Figure 4a
Eccentricity_maps_LH.py
PA_maps_LH.py

### Figure 4b
DeltaThetaVisualCortex_PA_subjvsGroup.py

### Figure 5
R2Average_plot.py

### Figure 6a
DeltaThetaVisualCortex_PA.py

### Figure 6b
ModelEval_featureImportance_PA.py

### Supplementary Figure 1
#### Left Hemisphere
PA_maps_LH.py

#### Right Hemisphere
PA_maps_LH.py

### Supplementary Figure 2
#### Left Hemisphere
Eccentricity_maps_LH.py

#### Right Hemisphere
Eccentricity_maps_LH.py


### Supplementary Figure 3
#### Left Hemisphere
PA_maps_LH.py

### Supplementary Figure 4
DiffNumberLayersErrorPlots_LH.py

## Final model selection

ModelEval_FinalModel_PA.py
ModelEval_FinalModel_ecc.py

## Mean error and standard deviation

ModelEval_MeanDeltaTheta_ecc.py
ModelEval_MeanDeltaTheta_PA.py