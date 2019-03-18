#!/bin/bash

mkdir ./matlab/packages &&\
cd ./matlab && \
wget https://www.artefact.tk/software/matlab/gifti/gifti-1.8.zip && \
wget ftp://ftp.fieldtriptoolbox.org/pub/fieldtrip/fieldtrip-20190220.zip && \
unzip gifti-1.8.zip -d packages && \
unzip fieldtrip-20190220.zip -d packages && \
rm gifti-1.8.zip && \
rm fieldtrip-20190220.zip && \


cd ../data/raw && \
mkdir original && \
unzip S1200_7T_Retinotopy_9Zkk.zip 'S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/*' -d original && \
mkdir converted &&\
cd converted &&\
matlab -nodisplay -nodesktop -r "addpath(genpath('../../..'));run data_formatting.m"


