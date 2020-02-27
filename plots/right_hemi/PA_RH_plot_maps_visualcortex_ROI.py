import numpy as np


import scipy.io
import os.path as osp
import torch
from functions.def_ROIs_ROI import roi


from nilearn import plotting

path='/home/uqfribe1/PycharmProjects/DEEP-fMRI/data/raw/converted'
curv = scipy.io.loadmat(osp.join(path, 'cifti_curv_all.mat'))['cifti_curv']
background=np.reshape(curv['x680957_curvature'][0][0][32492:],(-1))


threshold=1

nocurv=np.isnan(background)
background[nocurv==1] = 0

label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask= roi(label_primary_visual_areas)
pred=np.zeros((32492,1))
measured=np.zeros((32492,1))
R2_thr=np.zeros((32492,1))


a=torch.load('/home/uqfribe1/PycharmProjects/DEEP-fMRI/testset_results/testset-pred_Model3_PA_RH.pt',map_location='cpu')
pred[final_mask_R==1]=np.reshape(np.array(a['Predicted_values'][8]),(-1,1))
measured[final_mask_R==1]=np.reshape(np.array(a['Measured_values'][8]),(-1,1))


background[background<0]=0
background[background>0]=1

pred=np.array(pred)
minus=pred>180
sum=pred<180
pred[minus]=pred[minus]-180+threshold
pred[sum]=pred[sum]+180+threshold
pred=np.array(pred)
#pred[final_mask_L!=1]=-2
#pred[R2_thr]=0

measured=np.array(measured)
minus=measured>180
sum=measured<180
measured[minus]=measured[minus]-180+threshold
# measured[minus]=np.abs(measured[minus]-360)
measured[sum]=measured[sum]+180+threshold
# measured[sum]=360-measured[sum]
measured=np.array(measured)
#measured[final_mask_L!=1]=-2
#measured[R2_thr]=0

measured[final_mask_R!=1]=0
pred[final_mask_R!=1]=0



# plotting.plot_surf(surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)),'..','data/raw/original/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.L.midthickness_MSMAll.32k_fs_LR.surf.gii'),surf_map=np.reshape(pred[0:32492],(-1)),bg_map=background,cmap='gist_rainbow_r',symmetric_cbar=False,vmax=361,view='medial',avg_method='median',threshold=361,output_file='medial_305_3D.svg')
# plotting.show()



view=plotting.view_surf(surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)),'../..','data/raw/original/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.R.sphere.32k_fs_LR.surf.gii'),surf_map=np.reshape(pred[0:32492],(-1)),bg_map=background,cmap='gist_rainbow',black_bg=False,symmetric_cmap=False,threshold=threshold,vmax=361)
view.open_in_browser()