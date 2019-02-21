#!/bin/bash

cd ./matlab
wget https://www.artefact.tk/software/matlab/gifti/gifti-1.8.zip
wget ftp://ftp.fieldtriptoolbox.org/pub/fieldtrip/fieldtrip-20190220.zip
unzip gifti-1.8.zip
unzip fieldtrip-20190220.zip

cd ./data/raw
unzip S1200_7T_Retinotopy_9Zkk.zip
matlab -nodisplay -nodesktop -r "addpath(genpath('../../'));run creating_cifti_files.m"


