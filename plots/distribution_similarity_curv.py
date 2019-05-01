import os.path as osp
import torch_geometric.transforms as T
import numpy as np
import scipy.stats as sci
import matplotlib.pyplot as plt



import sys
sys.path.append('../..')

from dataset.HCP_3sets_visual_nothr_rotated import Retinotopy
from torch_geometric.data import DataLoader


path=osp.join(osp.dirname(osp.realpath(__file__)),'..','data')
pre_transform=T.Compose([T.FaceToEdge()])
train_dataset=Retinotopy(path,'Train', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='polarAngle',myelination=False)
dev_dataset=Retinotopy(path,'Development', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,myelination=False)
train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)
dev_loader=DataLoader(dev_dataset,batch_size=1,shuffle=False)



curv=[]
for data in train_loader:
    curv.append(np.array(data.x))

curv=np.reshape(curv,(-1))
curv=np.sort(curv,axis=None)

upper=0.975*len(curv)
lower=0.025*len(curv)

upper=curv[round(upper)]
lower=curv[round(lower)]
#curv=np.mean(curv,0)

plt.hist(curv,bins=1000)
plt.show()

def transform(input,range):
    transform=((input-lower)/(upper-lower))*(range-(-range))+(-range)
    transform[transform>range]=range
    transform[transform<-range]=-range
    return transform

curv=transform(curv,1)




'''
similarity=[]
for i in range(len(curv)):
    for j in range(len(curv)):
        similarity.append(sci.pearsonr(curv[i],curv[j])[0])

similarity=np.reshape(similarity,(-1))
plt.boxplot(similarity)
plt.show()'''