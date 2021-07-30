# Manuscript

This folder contains all source code necessary to reproduce all figures and summary statistics of our recent work entitled
 "Predicting the retinotopic organization of human visual cortex from anatomy using geometric deep learning" available on 
 [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.02.11.934471v3).

## Figures

Polar angle and eccentricity maps, as shown in the manuscript (Figure 3-5), were generated using the following scripts:

- ./plots/left(right)_hemi/Eccentricity_maps_L(R)H.py
- ./plots/left(right)_hemi/PA_maps_L(R)H.py

## Descriptive statistics

Results from Table 1 were generated with ./stats/left_hemi/pRFfitsVSmodels.py and were tabulated with 
./stats/left_hemi/pRFfitsVSmodels_Table1.py 

Mean prediction error (polar angle, eccentricity and pRF center) were determined with the following scripts:
- ./stats/left(right)_hemi/ModelEval_MeanDeltaTheta_PA.py
- ./stats/left(right)_hemi/ModelEval_MeanDeltaTheta_ecc.py
- ./stats/left(right)_hemi/ModelEval_DistancepRFcenterNorm.py

Mean vertex-wise explained variance for polar angle and eccentricity models were determined with the following scripts:
- ./stats/left(right)_hemi/ModelEval_explainedVar_VertexWise_ecc.py
- ./stats/left(right)_hemi/ModelEval_explainedVar_VertexWise_PA.py


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

