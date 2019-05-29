import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os.path as osp
import torch
from functions.def_ROIs_ROI1 import roi

path='/home/uqfribe1/PycharmProjects/DEEP-fMRI/data/raw/converted'
curv = scipy.io.loadmat(osp.join(path, 'cifti_curv_all.mat'))['cifti_curv']
background=np.reshape(curv['x100610_curvature'][0][0][0:32492],(-1))

threshold=1

label_primary_visual_areas = ['ROI1']
final_mask_L, final_mask_R, index_L_mask, index_R_mask= roi(label_primary_visual_areas)
pred=np.zeros((32492,1))
measured=np.zeros((32492,1))
R2_thr=np.zeros((32492,1))

a=torch.load('/home/uqfribe1/PycharmProjects/DEEP-fMRI/polarAngle/model4_nothresh_rotated_5layers_smoothL1lossR2_contrast_curvnmyelin_ROI1_range10_k25_featmap_output_epoch1000.pt',map_location='cpu')
pred[final_mask_L==1]=np.reshape(np.array(a['Predicted_values'][7]),(-1,1))


R2_thr[final_mask_L==1]=np.reshape(np.array(a['R2'][7]),(-1,1))

R2=R2_thr[final_mask_L==1]

measured[final_mask_L==1]=np.reshape(np.array(a['Measured_values'][7]),(-1,1))

pred=np.array(pred)
minus=pred>180
sum=pred<180
pred[minus]=pred[minus]-180
pred[sum]=pred[sum]+180
pred=pred


measured=np.array(measured)
minus=measured>180
sum=measured<180
measured[minus]=measured[minus]-180
measured[sum]=measured[sum]+180
measured=np.array(measured)



from functions.def_ROIs import roi
V1 = ['V1d','V1v','V2d','V2v','V3d','V3v']
final_mask_L_v1, final_mask_R_v1, index_L_mask_v1, index_R_mask_v1= roi(V1)

final_mask_L_v1=final_mask_L_v1[final_mask_L==1][np.reshape(np.cos(measured[final_mask_L==1]/180*np.pi) > 0,(-1))]



color = ['red' if l==1 else 'blue' for l in final_mask_L_v1]

color=np.array(color)















#plt.scatter(np.sin(measured[final_mask_L==1][R2>2.2][np.cos(measured[final_mask_L==1][R2>2.2]/180*np.pi) > 0] /180*np.pi),np.sin(pred[final_mask_L==1][R2>2.2][np.cos(measured[final_mask_L==1][R2>2.2]/180*np.pi) > 0]/180*np.pi),color=color,marker='.')
#plt.show()
'''
plt.scatter(np.sin(measured[final_mask_L==1][np.cos(measured[final_mask_L==1]/180*np.pi) > 0] /180*np.pi),np.sin(pred[final_mask_L==1][np.cos(measured[final_mask_L==1]/180*np.pi) > 0]/180*np.pi),color=color,marker='.')
plt.show()
'''
plt.scatter(np.sin(measured[final_mask_L==1][R2>60][np.cos(measured[final_mask_L==1][R2>60]/180*np.pi) > 0] /180*np.pi),np.sin(pred[final_mask_L==1][R2>60][np.cos(measured[final_mask_L==1][R2>60]/180*np.pi) > 0]/180*np.pi),color=color[np.reshape(R2[np.cos(measured[final_mask_L==1]/180*np.pi) > 0],(-1))>60],marker='.')
plt.show()

'''view=plotting.view_surf(surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)),'..','data/raw/original/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),surf_map=np.reshape(pred[0:32492],(-1)),bg_map=background,cmap='gist_rainbow_r',black_bg=True,symmetric_cmap=False,vmax=360)
view.open_in_browser()'''