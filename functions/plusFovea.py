import numpy as np
from functions.def_ROIs import roi


def add_fovea(list):

    visual_cortex_L=np.zeros((32492,len(list)))
    visual_cortex_R=np.zeros((32492,len(list)))
    for i in range(len(list)):
        area=[list[i]]
        final_mask_L, final_mask_R, index_L_mask, index_R_mask= roi(area)
        visual_cortex_L[:,i][final_mask_L==1]= i +1
        visual_cortex_R[:, i][final_mask_R == 1] = i +1

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


    return V1_L,V2_L,V3_L

def add_fovea_R(list):

    visual_cortex_L=np.zeros((32492,len(list)))
    visual_cortex_R=np.zeros((32492,len(list)))
    for i in range(len(list)):
        area=[list[i]]
        final_mask_L, final_mask_R, index_L_mask, index_R_mask= roi(area)
        visual_cortex_L[:,i][final_mask_L==1]= i +1
        visual_cortex_R[:, i][final_mask_R == 1] = i +1

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


    return V1_R,V2_R,V3_R