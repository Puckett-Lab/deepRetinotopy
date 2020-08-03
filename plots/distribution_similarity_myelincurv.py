import os.path as osp
import torch_geometric.transforms as T
import numpy as np
import scipy.stats as sci
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from functions.plusFovea import add_fovea
from functions.def_ROIs_WangParcelsPlusFovea import roi

import sys
sys.path.append('..')

from dataset.HCP_3sets_visual_nothr_rotated_ROI1_notwin import Retinotopy
from torch_geometric.data import DataLoader


path=osp.join(osp.dirname(osp.realpath(__file__)),'../data')
pre_transform=T.Compose([T.FaceToEdge()])
#train_dataset=Retinotopy(path,'Train', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='polarAngle',myelination=True,hemisphere='Left')
#dev_dataset=Retinotopy(path,'Development', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='polarAngle',myelination=True,hemisphere='Left')
test_dataset=Retinotopy(path,'Test', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='polarAngle',myelination=True,hemisphere='Left')

#train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)
#dev_loader=DataLoader(dev_dataset,batch_size=1,shuffle=False)
test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False)

twin_MZ=np.array([[8,7,4,5,6],[10,11,14,17,18]]).T
twin_DZ=np.array([[0,3,1,2,9],[12,13,15,16,19]]).T

prediction=torch.load('/home/uqfribe1/PycharmProjects/DEEP-fMRI/testset_results/testset-pred_Model5_PA_LH_notwin10.pt',map_location='cpu')

#Selecting vertices
#Early visual cortex
label_primary_visual_areas = ['V1d', 'V1v','fovea_V1', 'V2d', 'V2v' ,'fovea_V2', 'V3d',  'V3v','fovea_V3']
V1,V2,V3=add_fovea(label_primary_visual_areas)
primary_visual_areas=np.sum([np.reshape(V1,(-1,1)),np.reshape(V2,(-1,1)),np.reshape(V3,(-1,1))],axis=0)
label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(label_primary_visual_areas)
ROI1 = np.zeros((32492, 1))
ROI1[final_mask_L == 1] = 1
mask = ROI1 + np.reshape(primary_visual_areas, (32492, 1))
mask = mask[ROI1 == 1]



PA_pred=[]
for i in range(len(prediction['Predicted_values'])):
    PA_pred.append(np.array(prediction['Predicted_values'][i])[mask>1])

PA=[]
for data in test_loader:
    PA.append(np.array(data.y)[mask>1])

myelincurv=[]
for data in test_loader:
    myelincurv.append(np.reshape(np.array(data.x)[mask>1].T,(2,-1)))


myelin=np.zeros((20,np.sum(mask>1)))
curv=np.zeros((20,np.sum(mask>1)))
for i in range(20):
    myelin[i]=myelincurv[i][1]
    curv[i]=myelincurv[i][0]



#Creating matrices of correlations
similarity_PA={'MZ':[],'DZ':[],'UN':[]}
similarity_PA_pred={'MZ':[],'DZ':[],'UN':[]}
similarity_curv={'MZ':[],'DZ':[],'UN':[]}
similarity_myelin={'MZ':[],'DZ':[],'UN':[]}


matrix_PA=np.zeros((20,20))
matrix_PA_pred=np.zeros((20,20))
matrix_curv=np.zeros((20,20))
matrix_myelin=np.zeros((20,20))

for i in range(len(PA)):
    for j in range(len(PA)):
        if i!=j:
            matrix_PA[i][j]=sci.pearsonr(PA[i],PA[j])[0]
            matrix_PA_pred[i][j] = sci.pearsonr(PA_pred[i], PA_pred[j])[0]
            matrix_curv[i][j] = sci.pearsonr(curv[i], curv[j])[0]
            matrix_myelin[i][j] = sci.pearsonr(myelin[i], myelin[j])[0]

        else:
            matrix_PA[i][j] = 0 #zero out same ind correlation
            matrix_PA_pred[i][j] = 0
            matrix_curv[i][j] = 0
            matrix_myelin[i][j] = 0



#Selecting MZ, DZ and UN correlations
matrix_PA=np.triu(matrix_PA) #zero out lower triangle
matrix_PA_pred=np.triu(matrix_PA_pred)
matrix_curv=np.triu(matrix_curv)
matrix_myelin=np.triu(matrix_myelin)

for k in range(len(twin_MZ)):
    #MZ twins
    similarity_PA['MZ'].append(matrix_PA[twin_MZ[k][0], twin_MZ[k][1]])
    matrix_PA[twin_MZ[k][0], twin_MZ[k][1]]=0 #zero out
    similarity_PA_pred['MZ'].append(matrix_PA_pred[twin_MZ[k][0], twin_MZ[k][1]])
    matrix_PA_pred[twin_MZ[k][0], twin_MZ[k][1]] = 0  # zero out
    similarity_curv['MZ'].append(matrix_curv[twin_MZ[k][0], twin_MZ[k][1]])
    matrix_curv[twin_MZ[k][0], twin_MZ[k][1]] = 0  # zero out
    similarity_myelin['MZ'].append(matrix_myelin[twin_MZ[k][0], twin_MZ[k][1]])
    matrix_myelin[twin_MZ[k][0], twin_MZ[k][1]] = 0  # zero out

    #DZ twins
    similarity_PA['DZ'].append(matrix_PA[twin_DZ[k][0], twin_DZ[k][1]])
    matrix_PA[twin_DZ[k][0], twin_DZ[k][1]]=0 #zero out
    similarity_PA_pred['DZ'].append(matrix_PA_pred[twin_DZ[k][0], twin_DZ[k][1]])
    matrix_PA_pred[twin_DZ[k][0], twin_DZ[k][1]] = 0  # zero out
    similarity_curv['DZ'].append(matrix_curv[twin_DZ[k][0], twin_DZ[k][1]])
    matrix_curv[twin_DZ[k][0], twin_DZ[k][1]] = 0  # zero out
    similarity_myelin['DZ'].append(matrix_myelin[twin_DZ[k][0], twin_DZ[k][1]])
    matrix_myelin[twin_DZ[k][0], twin_DZ[k][1]]=0 #zero out

#UN individuals
matrix_PA=np.reshape(matrix_PA,(-1))
similarity_PA['UN']=matrix_PA[np.nonzero(matrix_PA)]
matrix_PA_pred=np.reshape(matrix_PA_pred,(-1))
similarity_PA_pred['UN']=matrix_PA_pred[np.nonzero(matrix_PA_pred)]
matrix_curv=np.reshape(matrix_curv,(-1))
similarity_curv['UN']=matrix_curv[np.nonzero(matrix_curv)]
matrix_myelin=np.reshape(matrix_myelin,(-1))
similarity_myelin['UN']=matrix_myelin[np.nonzero(matrix_myelin)]


#figure
features_read=['PA','PA_pred','curv','myelin']
features_title=['Polar Angle - Ground truth','Polar Angle - Predicted','Curvature','Myelin']
fig = plt.figure(figsize=(10,3))
for i in range(len(features_read)):
    print('Mean MZ:'+str(np.mean(vars()['similarity_'+features_read[i]]['MZ']))+', Mean DZ:'+str(np.mean(vars()['similarity_'+features_read[i]]['DZ']))+' Mean UN:'+str(np.mean(vars()['similarity_'+features_read[i]]['UN'])))
    data = np.concatenate([[vars()['similarity_'+features_read[i]]['MZ'], len(vars()['similarity_'+features_read[i]]['MZ'])*['MZ']],
    [vars()['similarity_'+features_read[i]]['DZ'], len(vars()['similarity_'+features_read[i]]['DZ'])*['DZ']],
    [vars()['similarity_'+features_read[i]]['UN'], len(vars()['similarity_'+features_read[i]]['UN'])*['UN']]], axis=1)
    print(data)
    df = pd.DataFrame(columns=['Correlation', 'Zygosity'], data=data.T)
    df['Correlation'] = df['Correlation'].astype(float)

    ax=fig.add_subplot(1,4,i+1)
    # ax = sns.boxplot(y='$\Delta$$\t\Theta$', x='models', order=models, hue='label', data=df, palette=palette,showfliers=False)
    ax = sns.pointplot(y='Correlation', x='Zygosity', hue='Zygosity', data=df,join=False,dodge=True,ci=95,palette='colorblind')
    plt.axhline(np.mean(np.mean(vars()['similarity_'+features_read[i]]['UN'])),linestyle='--',color='gray')
    ax.set_title(features_title[i])
    plt.ylim(-0.1,1)
    legend=plt.legend()

#plt.boxplot([similarity_PA['MZ'],similarity_PA['DZ'],similarity_PA['UN']],[1,2,3],labels=['MZ','DZ','UN'])
plt.show()