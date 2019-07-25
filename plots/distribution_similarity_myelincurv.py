import os.path as osp
import torch_geometric.transforms as T
import scipy.stats as sci
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../..')

from dataset.HCP_3sets_visual_nothr_rotated_ROI1 import Retinotopy
from torch_geometric.data import DataLoader




path=osp.join(osp.dirname(osp.realpath(__file__)),'..','data')
pre_transform=T.Compose([T.FaceToEdge()])
train_dataset=Retinotopy(path,'Train', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='polarAngle',myelination=True,hemisphere='Left')
dev_dataset=Retinotopy(path,'Development', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='polarAngle',myelination=True,hemisphere='Left')
train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)
dev_loader=DataLoader(dev_dataset,batch_size=1,shuffle=False)


PA=[]
for data in train_loader:
    PA.append(np.array(data.y))



similarity_PA=[]
for i in range(len(PA)):
    for j in range(len(PA)):
        if i != j:
            similarity_PA.append(sci.pearsonr(PA[i],PA[j])[0])


myelincurv=[]
for data in train_loader:
    myelincurv.append(np.reshape(np.array(data.x).T,(2,-1)))


myelin=np.zeros((161,3267))
curv=np.zeros((161,3267))
for i in range(161):
    myelin[i]=myelincurv[i][1]
    curv[i]=myelincurv[i][0]

similarity_myelin=[]
for i in range(len(myelin)):
    for j in range(len(myelin)):
        if i != j:
            similarity_myelin.append(sci.pearsonr(myelin[i],myelin[j])[0])

similarity_curv=[]
for i in range(len(myelin)):
    for j in range(len(myelin)):
        if i!=j:
            similarity_curv.append(sci.pearsonr(curv[i],curv[j])[0])




similarity_PA=np.reshape(similarity_PA,(-1))
similarity_myelin=np.reshape(similarity_myelin,(-1))
similarity_curv=np.reshape(similarity_curv,(-1))
print(np.mean(similarity_curv))
print(np.mean(similarity_PA))
print(np.mean(similarity_myelin))
plt.boxplot([similarity_PA,similarity_myelin,similarity_curv],[1,2,3],labels=['PA','Myelin','Curvature'])
plt.show()