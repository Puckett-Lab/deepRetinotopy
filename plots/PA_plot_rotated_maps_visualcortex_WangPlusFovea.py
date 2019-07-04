import numpy as np
import scipy.io
import os.path as osp
from matplotlib import cm
from functions.def_ROIs import roi
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from nilearn import plotting

path='/home/uqfribe1/PycharmProjects/DEEP-fMRI/data/raw/converted'
curv = scipy.io.loadmat(osp.join(path, 'cifti_curv_all.mat'))['cifti_curv']
background_L=np.reshape(curv['x942658_curvature'][0][0][0:32492],(-1))
background_R=np.reshape(curv['x942658_curvature'][0][0][32492:],(-1))
# background_R[background_R>1]=1 #outlier
threshold=1


nocurv=np.isnan(background_L)
background_L[nocurv==1] = 0

nocurv=np.isnan(background_R)
background_R[nocurv==1] = 0

label_primary_visual_areas = ['V1d', 'V1v','fovea_V1', 'V2d', 'V2v' ,'fovea_V2', 'V3d',  'V3v','fovea_V3','hV4','VO1','VO2','PHC1','PHC2','V3a','V3b','LO1','LO2','TO1','TO2','IPS0','IPS1','IPS2','IPS3','IPS4','IPS5','SPL1']

visual_cortex_L=np.zeros((32492,len(label_primary_visual_areas)))
visual_cortex_R=np.zeros((32492,len(label_primary_visual_areas)))
for i in range(len(label_primary_visual_areas)):
    area=[label_primary_visual_areas[i]]
    final_mask_L, final_mask_R, index_L_mask, index_R_mask= roi(area)
    visual_cortex_L[:,i][final_mask_L==1]=i+threshold
    visual_cortex_R[:, i][final_mask_R == 1] = i + threshold

    if i>8:
        visual_cortex_L[:, i][final_mask_L == 1] = i - 6 + threshold
        visual_cortex_R[:, i][final_mask_R == 1] = i - 6 + threshold

for i in range(3):
    visual_cortex_L[:,i*3+1][visual_cortex_L[:,i*3+1]!=0]=i+1
    visual_cortex_L[:,i*3+2][visual_cortex_L[:,i*3+2]!=0]=i+1
    visual_cortex_R[:, i * 3 + 1][visual_cortex_R[:, i * 3 + 1] != 0] = i + 1
    visual_cortex_R[:, i * 3 + 2][visual_cortex_R[:, i * 3 + 2] != 0] = i + 1


V1_L=np.sum(visual_cortex_L[:,0:3],axis=1)
V1_L[V1_L>1]=1
V2_L=np.sum(visual_cortex_L[:,3:6],axis=1)
V2_L[V2_L>1]=2
V3_L=np.sum(visual_cortex_L[:,6:9],axis=1)
V3_L[V3_L>1]=3

V1_R=np.sum(visual_cortex_R[:,0:3],axis=1)
V1_R[V1_R>1]=1
V2_R=np.sum(visual_cortex_R[:,3:6],axis=1)
V2_R[V2_R>1]=2
V3_R=np.sum(visual_cortex_R[:,6:9],axis=1)
V3_R[V3_R>1]=3



visual_areas_L=np.concatenate((np.reshape(V1_L,(-1,1)),np.reshape(V2_L,(-1,1)),np.reshape(V3_L,(-1,1)),visual_cortex_L[:,9:]),axis=1)
visual_areas_L=np.sum(visual_areas_L,axis=1)
visual_areas_L[V3_L==3]=3
visual_areas_L[V2_L==2]=2
visual_areas_L[V1_L==1]=1

visual_areas_L[visual_areas_L==21]=0
visual_areas_L[visual_areas_L==20]=0
visual_areas_L[visual_areas_L==19]=0
visual_areas_L[visual_areas_L==18]=0
visual_areas_L[visual_areas_L==17]=0

visual_areas_R=np.concatenate((np.reshape(V1_R,(-1,1)),np.reshape(V2_R,(-1,1)),np.reshape(V3_R,(-1,1)),visual_cortex_R[:,9:]),axis=1)
visual_areas_R=np.sum(visual_areas_R,axis=1)
visual_areas_R[V3_R==3]=3
visual_areas_R[V2_R==2]=2
visual_areas_R[V1_R==1]=1

visual_areas_R[visual_areas_R==21]=0
visual_areas_R[visual_areas_R==20]=0
visual_areas_R[visual_areas_R==19]=0
visual_areas_R[visual_areas_R==18]=0
visual_areas_R[visual_areas_R==17]=0

#Color map
top=cm.get_cmap('tab20b',21)
newcolors=top(np.linspace(0,1,21))
fair_pink=np.array([235/255,206/255,229/255,1])
newcolors[-1]=fair_pink
newcmp = ListedColormap(newcolors, name='VisualAreas')


# visual_areas=np.concatenate((np.reshape(visual_cortex[:,2],(-1,1)),np.reshape(visual_cortex[:,5],(-1,1)),np.reshape(visual_cortex[:,8],(-1,1))),axis=1)
# visual_areas=np.sum(visual_areas,axis=1)
#
#

# plotting.plot_surf_roi(surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)),'..','data/raw/original/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),roi_map=np.reshape(visual_areas_L[0:32492],(-1)),hemi='left',bg_map=background_L,cmap=newcmp,symmetric_cbar=False,vmax=21,view='medial',output_file='L_visualareas_sphere.svg')
# plotting.show()
#
# plotting.plot_surf_roi(surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)),'..','data/raw/original/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.R.sphere.32k_fs_LR.surf.gii'),roi_map=np.reshape(visual_areas_R[0:32492],(-1)),hemi='right',bg_map=background_R,cmap=newcmp,symmetric_cbar=False,vmax=21,view='medial',output_file='R_visualareas_sphere.svg')
# plotting.show()
#
# plotting.plot_surf_roi(surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)),'..','data/raw/original/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.L.midthickness_MSMAll.32k_fs_LR.surf.gii'),roi_map=np.reshape(visual_areas_L[0:32492],(-1)),hemi='left',bg_map=background_L,cmap=newcmp,symmetric_cbar=False,vmax=21,view='posterior',output_file='L_visualareas_posterior.svg')
# plotting.show()
#
# plotting.plot_surf_roi(surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)),'..','data/raw/original/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.R.midthickness_MSMAll.32k_fs_LR.surf.gii'),roi_map=np.reshape(visual_areas_R[0:32492],(-1)),hemi='right',bg_map=background_R,cmap=newcmp,symmetric_cbar=False,vmax=21,view='posterior',output_file='R_visualareas_posterior.svg')
# plotting.show()
#



view=plotting.view_surf(surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)),'..','data/raw/original/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.R.sphere.32k_fs_LR.surf.gii'),surf_map=np.reshape(visual_areas_R[0:32492],(-1)),bg_map=background_R,cmap=newcmp,black_bg=True,symmetric_cmap=False,threshold=threshold,vmax=22)
view.open_in_browser()

view=plotting.view_surf(surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)),'..','data/raw/original/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),surf_map=np.reshape(visual_areas_L[0:32492],(-1)),bg_map=background_L,cmap=newcmp,black_bg=True,symmetric_cmap=False,threshold=threshold,vmax=22)
view.open_in_browser()