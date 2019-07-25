import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np
import sys
import time

sys.path.append('../..')


from dataset.HCP_3sets_visual_nothr_rotated_ROI1 import Retinotopy
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv


path=osp.join(osp.dirname(osp.realpath(__file__)),'data')
pre_transform=T.Compose([T.FaceToEdge()])
train_dataset=Retinotopy(path,'Train', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='polarAngle',myelination=True,hemisphere='Left')
dev_dataset=Retinotopy(path,'Development', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='polarAngle',myelination=True,hemisphere='Left')
train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)
dev_loader=DataLoader(dev_dataset,batch_size=1,shuffle=False)

curv=[]
for data in train_loader:
    curv.append(np.array(data.x).T[0])



#Testing contrast

curv_transform=np.reshape(curv,(-1))
curv_transform=np.sort(curv,axis=None)

upper=0.975*len(curv_transform)
lower=0.025*len(curv_transform)

upper=curv_transform[round(upper)]
lower=curv_transform[round(lower)]
#curv=np.mean(curv,0)



def transform(input,range):
    transform=((input-lower)/(upper-lower))*(range-(-range))+(-range)
    transform[transform>range]=range
    transform[transform<-range]=-range
    return transform



curv=curv[0]

curv_test=transform(curv,10)







curv_thr[final_mask_L==1]=np.reshape(curv_test+1,(-1,1))
curv_thr=curv_thr+1


view=plotting.view_surf(surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)),'..','data/raw/original/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),surf_map=np.reshape(curv_thr[0:32492],(-1)),cmap='gray',black_bg=True,symmetric_cmap=True,vmax=10)
view.open_in_browser()