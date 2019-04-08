import os.path as osp
from nilearn import plotting
import torch_geometric.transforms as T
import numpy as np
from functions.def_ROIs import roi

import sys
sys.path.append('../..')

from dataset.test_HCP_3sets_visual_nothr_rotated_myelin import Retinotopy
from torch_geometric.data import DataLoader

label_primary_visual_areas = ['V1d', 'V1v', 'V2d', 'V2v', 'V3d', 'V3v']
final_mask_L, final_mask_R, index_L_mask, index_R_mask= roi(label_primary_visual_areas)
myelin_thr=np.zeros((32492,1))





path=osp.join(osp.dirname(osp.realpath(__file__)),'..','data')
pre_transform=T.Compose([T.FaceToEdge()])
train_dataset=Retinotopy(path,'Train', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='polarAngle',myelination=True)
dev_dataset=Retinotopy(path,'Development', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,myelination=True)
train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)
dev_loader=DataLoader(dev_dataset,batch_size=1,shuffle=False)



myelin=[]
for data in train_loader:
    myelin.append(np.array(data.x))

myelin=np.mean(myelin,0)

myelin_thr[final_mask_L==1]=np.reshape(myelin,(-1,1))


view=plotting.view_surf(surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)),'..','data/raw/original/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),surf_map=np.reshape(myelin_thr[0:32492],(-1)),cmap='gist_rainbow_r',black_bg=True,symmetric_cmap=False)
view.open_in_browser()