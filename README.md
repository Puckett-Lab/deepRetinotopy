# DeepRetinotopy

This repository contains all source code necessary to replicate our recent work entitled "Predicting the retinotopic organization of human visual cortex from anatomy using geometric deep learning" available on 
 [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.02.11.934471v3).

## Table of Contents
* [Installation and requirements](#installation-and-requirements)
* [Manuscript](#manuscript)
* [Models](#models)
* [Retinotopy](#retinotopy)
* [Citation](#citation)
* [Contact](#contact)
* [License](#license)


## Installation and requirements 

Models were generated using [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/). Since this package is under constant updates, we highly recommend that 
you follow the following steps to run our models locally:

- Create a conda environment (or docker container)
- Install torch first:
		
```bash
$ pip install torch==0.4.1    
$ pip install torchvision==0.2.1
```
	
	
- Install torch-scatter, torch-sparse, torch-cluster, torch-spline-conv and torch-geometric:

```bash
$ pip install torch-scatter==1.0.4
$ pip install torch-sparse==0.2.2
$ pip install torch-cluster==1.1.5
$ pip install torch-spline-conv==1.0.4
$ pip install torch-geometric==0.3.1
```

- Install the remaining required packages that are available at requirements.txt: 

```bash
$ pip install -r requirements.txt
```   
    
- Clone DeepRetinotopy:
```bash
$ git clone git@github.com:Puckett-Lab/deepRetinotopy.git
```   

Finally, install the following git repository for plots:
```bash
$ pip install git+https://github.com/felenitaribeiro/nilearn.git
```
    


## Manuscript

This folder contains all source code necessary to reproduce all figures and summary statistics in our manuscript.

## Models

This folder contains all source code necessary to train a new model and to generate predictions on the test dataset 
using our pre-trained models.

## Retinotopy

This folder contains all source code necessary to replicate datasets generation, in addition to functions and labels 
used for figures and models' evaluation. 

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


## Contact
Fernanda L. Ribeiro <[fernanda.ribeiro@uq.edu.au](fernanda.ribeiro@uq.edu.au)>


## License
