#!/bin/bash


cd ./data/raw
unzip S1200_7T_Retinotopy_9Zkk.zip
matlab -nodisplay -nodesktop -r "addpath(genpath('../../'));addpath(genpath('/home/uqfribe1/Desktop/Dataset_ML_RetMap-matlab'));run creating_cifti_files.m"


