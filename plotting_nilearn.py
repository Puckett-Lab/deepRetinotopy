import numpy as np
from nilearn import plotting
import nibabel as nib
import scipy.io
import os.path as osp
import torch
import sys


path='/home/uqfribe1/PycharmProjects/DEEP-fMRI/data/raw/converted'
curv = scipy.io.loadmat(osp.join(path, 'cifti_curv_all.mat'))['cifti_curv']
eccentricity = scipy.io.loadmat(osp.join(path, 'cifti_eccentricity_all.mat'))['cifti_eccentricity']
polarAngle = scipy.io.loadmat(osp.join(path, 'cifti_polarAngle_all.mat'))['cifti_polarAngle']
R2 = scipy.io.loadmat(osp.join(path, 'cifti_R2_all.mat'))['cifti_R2']

#img=nib.load('/home/uqfribe1/PycharmProjects/pytorch_geometric/my_stuff/data/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.Fit1_Eccentricity_MSMAll.32k_fs_LR.dscalar.nii')
#R2_ni=nib.load('/home/uqfribe1/PycharmProjects/pytorch_geometric/my_stuff/data/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.Fit1_R2_MSMAll.32k_fs_LR.dscalar.nii')

threshold=1
condition=np.isnan(eccentricity['x102311_fit1_eccentricity_msmall'][0][0][0:32492])
condition_R2=np.isnan(R2['x102311_fit1_r2_msmall'][0][0][0:32492])
condition_curv=np.isnan(curv['x102311_curvature'][0][0][0:32492])

R2['x102311_fit1_r2_msmall'][0][0][0:32492][condition_R2]=0
curv['x102311_curvature'][0][0][0:32492][condition_curv]=0

polarAngle['x102311_fit1_polarangle_msmall'][0][0][0:32492][condition]=0
polarAngle['x102311_fit1_polarangle_msmall'][0][0][0:32492][R2['x100610_fit1_r2_msmall'][0][0][0:32492]<2.2]=0
polarAngle['x102311_fit1_polarangle_msmall'][0][0][0:32492][R2['x100610_fit1_r2_msmall'][0][0][0:32492]>2.2]=polarAngle['x100610_fit1_polarangle_msmall'][0][0][0:32492][R2['x100610_fit1_r2_msmall'][0][0][0:32492]>2.2]+threshold


eccentricity['x100610_fit1_eccentricity_msmall'][0][0][0:32492][condition]=0
eccentricity['x100610_fit1_eccentricity_msmall'][0][0][0:32492][R2['x100610_fit1_r2_msmall'][0][0][0:32492]<2.2]=0
eccentricity['x100610_fit1_eccentricity_msmall'][0][0][0:32492][R2['x100610_fit1_r2_msmall'][0][0][0:32492]>2.2]=eccentricity['x100610_fit1_eccentricity_msmall'][0][0][0:32492][R2['x100610_fit1_r2_msmall'][0][0][0:32492]>2.2]+threshold


background=np.reshape(curv['x100610_curvature'][0][0][0:32492],(-1))


plotting.plot_surf(surf_mesh='/home/uqfribe1/PycharmProjects/DEEP-fMRI/data/raw/original/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.L.very_inflated_MSMAll.32k_fs_LR.surf.gii',surf_map=np.reshape(eccentricity['x100610_fit1_eccentricity_msmall'][0][0][0:32492],(1,-1)),bg_map=np.reshape(curv['x100610_curvature'][0][0][0:32492],(1,-1)))
plotting.show()

#for online view
view=plotting.view_surf(surf_mesh='/home/uqfribe1/PycharmProjects/DEEP-fMRI/data/raw/original/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.L.midthickness_MSMAll.32k_fs_LR.surf.gii',surf_map=np.reshape(eccentricity['x100610_fit1_eccentricity_msmall'][0][0][0:32492],(-1)),bg_map=background, threshold=threshold,cmap='gist_rainbow_r',black_bg=True,symmetric_cmap=False,vmax=8+threshold)
view.open_in_browser()

view=plotting.view_surf(surf_mesh='/home/uqfribe1/PycharmProjects/DEEP-fMRI/data/raw/original/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.L.midthickness_MSMAll.32k_fs_LR.surf.gii',surf_map=np.reshape(polarAngle['x100610_fit1_polarangle_msmall'][0][0][0:32492],(-1)),bg_map=background, threshold=threshold,cmap='gist_rainbow_r',black_bg=True,symmetric_cmap=False,vmax=360+threshold)
view.open_in_browser()



#Plotting the outputs from the neural net
a=torch.load('model1_2examples_output_epoch200.pt')
pred=a['Predicted_values'][399].detach()
measured=a['Measured_values'][399].detach()
pred=np.array(pred)+threshold
view=plotting.view_surf(surf_mesh='/home/uqfribe1/PycharmProjects/DEEP-fMRI/data/raw/original/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.L.midthickness_MSMAll.32k_fs_LR.surf.gii',surf_map=np.reshape(pred[0:32492],(-1)),cmap='gist_rainbow_r',black_bg=True,symmetric_cmap=False)
view.open_in_browser()


