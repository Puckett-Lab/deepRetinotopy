import os.path as osp
from nilearn import plotting
import torch_geometric.transforms as T
import numpy as np
from functions.def_ROIs_WangParcelsPlusFovea import roi

import scipy.io

import sys
sys.path.append('../..')

from dataset.HCP_3sets_visual_nothr_rotated_ROI1 import Retinotopy
from torch_geometric.data import DataLoader


path='/home/uqfribe1/PycharmProjects/DEEP-fMRI/data/raw/converted'
curv = scipy.io.loadmat(osp.join(path, 'cifti_curv_all.mat'))['cifti_curv']
background=np.reshape(curv['x100610_curvature'][0][0][0:32492],(-1))

threshold=1

nocurv=np.isnan(background)
background[nocurv==1] = 0


label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask= roi(label_primary_visual_areas)
R2_thr=np.zeros((32492,1))



path=osp.join(osp.dirname(osp.realpath(__file__)),'..','data')
pre_transform=T.Compose([T.FaceToEdge()])
train_dataset=Retinotopy(path,'Train', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='polarAngle',myelination=True,hemisphere='Left')
dev_dataset=Retinotopy(path,'Development', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='polarAngle',myelination=True,hemisphere='Left')
test_dataset=Retinotopy(path,'Test', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='polarAngle',myelination=True,hemisphere='Left')
test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False)
train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)
dev_loader=DataLoader(dev_dataset,batch_size=1,shuffle=False)



R2=[]
for data in test_loader:
    R2.append(np.array(data.R2))

R2=np.mean(R2,0)

R2_thr[final_mask_L==1]=np.reshape(R2,(-1,1))+threshold
R2_thr[final_mask_L!=1]=0


view=plotting.view_surf(surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)),'..','data/raw/original/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),bg_map=background,surf_map=np.reshape(R2_thr[0:32492],(-1)),threshold=threshold,cmap='hot',black_bg=False,symmetric_cmap=False,vmax=60+threshold)
view.open_in_browser()