import numpy as np
from nilearn import plotting
import scipy.io
import os.path as osp
import torch
from functions.def_ROIs_native import roi



path='/home/uqfribe1/PycharmProjects/DEEP-fMRI/data/raw/102311_native'
curv = scipy.io.loadmat(osp.join(path, 'cifti_curv_L_102311_native.mat'))['cifti_curv_L']

threshold=1

label_primary_visual_areas = ['V1d', 'V1v', 'V2d', 'V2v', 'V3d', 'V3v']
final_mask_L, final_mask_R, index_L_mask, index_R_mask= roi(label_primary_visual_areas)
pred=np.zeros((136849,1))



a=torch.load('/home/uqfribe1/PycharmProjects/DEEP-fMRI/testing_native.pt',map_location='cpu')
pred[final_mask_L==1]=10000
print(np.sum(final_mask_L==1))


pred=np.array(pred)
#minus=pred>180
#sum=pred<180
#pred[minus]=pred[minus]-180
#pred[sum]=pred[sum]+180
#pred=pred
#pred[R2_thr]=0

view=plotting.view_surf(surf_mesh='/home/uqfribe1/PycharmProjects/DEEP-fMRI/data/raw/102311_native/102311.L.sphere.native.surf.gii',surf_map=np.reshape(pred,(-1)),cmap='gist_rainbow_r',black_bg=True,symmetric_cmap=False,threshold=0)
view.open_in_browser()