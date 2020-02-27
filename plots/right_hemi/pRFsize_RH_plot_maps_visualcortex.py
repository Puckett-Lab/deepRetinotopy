import numpy as np
from nilearn import plotting
import scipy.io
import os.path as osp
import torch
import sys
sys.path.append('..')
from functions.def_ROIs_ROI import roi

path='/home/uqfribe1/PycharmProjects/DEEP-fMRI/data/raw/converted'
curv = scipy.io.loadmat(osp.join(path, 'cifti_curv_all.mat'))['cifti_curv']
background=np.reshape(curv['x100610_curvature'][0][0][32492:],(-1))

threshold=2

nocurv=np.isnan(background)
background[nocurv==1] = 0

label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask= roi(label_primary_visual_areas)
pred=np.zeros((32492,1))
measured=np.zeros((32492,1))
R2_thr=np.zeros((32492,1))



a=torch.load('/home/uqfribe1/PycharmProjects/DEEP-fMRI/plots/right_hemi/model4_nothresh_RH_pRFsize_12layers_smoothL1loss_curvnmyelin_ROI1_k25_batchnorm_dropout010_1_output_epoch200.pt',map_location='cpu')
pred[final_mask_R==1]=np.reshape(np.array(a['Predicted_values'][3]),(-1,1))

#R2_thr[final_mask_L==1]=np.reshape(np.array(a['R2'][0]),(-1,1))
#R2_thr=R2_thr<2.2
measured[final_mask_R==1]=np.reshape(np.array(a['Measured_values'][3]),(-1,1))

pred=np.array(pred)+threshold
#pred[R2_thr]=0

measured=np.array(measured)+threshold
#measured[R2_thr]=0

measured[final_mask_R!=1]=0
pred[final_mask_R!=1]=0
pred[final_mask_R==1][pred[final_mask_R==1]<0]=0+threshold

view=plotting.view_surf(surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)),'../..','data/raw/original/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.R.sphere.32k_fs_LR.surf.gii'),surf_map=np.reshape(pred[0:32492],(-1)),bg_map=background,cmap='gist_rainbow_r',black_bg=True,symmetric_cmap=False,threshold=1,vmax=10)
view.open_in_browser()