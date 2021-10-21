# Manuscript

This folder contains all source code necessary to reproduce all figures and summary statistics in our recent work entitled
 "Predicting the retinotopic organization of human visual cortex from anatomy using geometric deep learning" available in [NeuroImage](https://www.sciencedirect.com/science/article/pii/S1053811921008971).

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

Please cite our paper if you used our model or if it was somewhat helpful for you :wink:

    @article{Ribeiro2021,
             author = {Ribeiro, Fernanda L and Bollmann, Steffen and Puckett, Alexander M},
             doi = {https://doi.org/10.1016/j.neuroimage.2021.118624},
             issn = {1053-8119},
             journal = {NeuroImage},
             keywords = {cortical surface, high-resolution fMRI, machine learning, manifold, visual hierarchy,Vision},
             pages = {118624},
             title = {{Predicting the retinotopic organization of human visual cortex from anatomy using geometric deep learning}},
             url = {https://www.sciencedirect.com/science/article/pii/S1053811921008971},
             year = {2021}
            }



