import numpy as np
from nilearn import plotting
import scipy.io
import os.path as osp
import torch
from functions.def_ROIs import roi

path='/home/uqfribe1/PycharmProjects/DEEP-fMRI/data/raw/converted'
curv = scipy.io.loadmat(osp.join(path, 'cifti_curv_all.mat'))['cifti_curv']
background=np.reshape(curv['x100610_curvature'][0][0][0:32492],(-1))
#polarangle = scipy.io.loadmat(osp.join(path, 'cifti_polarAngle_all.mat'))['cifti_polarAngle']
#pred=np.reshape(polarangle['x100610_fit1_polarangle_msmall'][0][0][0:32492],(-1))

threshold=1

label_primary_visual_areas = ['V1d', 'V1v', 'V2d', 'V2v', 'V3d', 'V3v']
final_mask_L, final_mask_R, index_L_mask, index_R_mask= roi(label_primary_visual_areas)
pred=np.zeros((32492,1))
measured=np.zeros((32492,1))
R2_thr=np.zeros((32492,1))


a=torch.load('testing.pt',map_location='cpu')
pred[final_mask_L==1]=np.reshape(np.array(a['Predicted_values'][1]),(-1,1))

#R2_thr[final_mask_L==1]=np.reshape(np.array(a['R2'][0]),(-1,1))
#R2_thr=R2_thr<2.2
measured[final_mask_L==1]=np.reshape(np.array(a['Measured_values'][1]),(-1,1))

pred=np.array(pred)+threshold+1
#pred[R2_thr]=0

measured=np.array(measured)+threshold
#measured[R2_thr]=0

view=plotting.view_surf(surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)),'..','data/raw/original/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.L.midthickness_MSMAll.32k_fs_LR.surf.gii'),surf_map=np.reshape(pred[0:32492],(-1)),threshold=threshold,bg_map=background,cmap='gist_rainbow_r',black_bg=True,symmetric_cmap=False,vmax=360)
view.open_in_browser()