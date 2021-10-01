# Models

This folder contains all source code necessary to train a new model and to generate predictions on the test dataset 
using our pre-trained models available at [Open Science Framework](https://osf.io/95w4y/). 

## Training new models
Scripts for training new models are: 
- ./deepRetinotopy_ecc_LH.py (eccentricity);
- ./deepRetinotopy_PA_LH.py (polar angle).


## Generalization
Scripts for loading our pre-trained models (don't forget to download them from OSF, and to place them 
in ./output), and generating predictions on the test dataset are:
- ./ModelEvalGeneralizability_ecc.py;
- ./ModelEvalGeneralizability_PA.py;
- ./ModelEvalGeneralizability_PA_rotatedROI.py.


## Citation

Please cite our paper if you used our model or if it was somewhat helpful for you :wink:

    @article{Ribeiro2020,
        title = {{Predicting the retinotopic organization of human visual cortex from anatomy using geometric deep learning}},
        author = {Ribeiro, Fernanda L and Bollmann, Steffen and Puckett, Alexander M},
        doi = {10.1101/2020.02.11.934471},
        journal = {bioRxiv},
        url = {http://biorxiv.org/content/early/2020/02/12/2020.02.11.934471.abstract},
        year = {2021}
    }


