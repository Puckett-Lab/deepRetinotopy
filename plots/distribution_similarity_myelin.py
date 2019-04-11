import os.path as osp
import torch_geometric.transforms as T
import scipy.stats as sci
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../..')

from dataset.test_HCP_3sets_visual_nothr_rotated_myelin import Retinotopy
from torch_geometric.data import DataLoader




path=osp.join(osp.dirname(osp.realpath(__file__)),'..','data')
pre_transform=T.Compose([T.FaceToEdge()])
train_dataset=Retinotopy(path,'Train', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='polarAngle',myelination=True)
dev_dataset=Retinotopy(path,'Development', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,myelination=True)
train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)
dev_loader=DataLoader(dev_dataset,batch_size=1,shuffle=False)



myelin=[]
for data in train_loader:
    myelin.append(np.array(data.x))



similarity=[]
for i in range(len(myelin)):
    for j in range(len(myelin)):
        similarity.append(sci.pearsonr(myelin[i],myelin[j])[0])

similarity=np.reshape(similarity,(-1))
plt.boxplot(similarity)
plt.show()