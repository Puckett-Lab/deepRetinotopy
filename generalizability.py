import scipy.io
import torch
import os.path as osp
import torch_geometric.transforms as T
import torch.nn.functional as F

import sys
sys.path.append('../..')

from torch_geometric.data import Data
from functions.def_ROIs import roi
from functions.labels import labels
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv



path=osp.join(osp.dirname(osp.realpath(__file__)),'data','raw','converted')
pre_transform=T.Compose([T.FaceToEdge()])
transform=T.Cartesian()



# Selecting only V1,V2 and V3
label_primary_visual_areas = ['V1d', 'V1v', 'V2d', 'V2v', 'V3d', 'V3v']
final_mask_L, final_mask_R, index_L_mask, index_R_mask= roi(label_primary_visual_areas)

faces_R = labels(scipy.io.loadmat(osp.join(path,'tri_faces_R.mat'))['tri_faces_R']-1, index_R_mask)
faces_L = labels(scipy.io.loadmat(osp.join(path, 'tri_faces_L.mat'))['tri_faces_L'] - 1, index_L_mask)

#Defining number of nodes
number_cortical_nodes=int(64984)
number_hemi_nodes=int(number_cortical_nodes/2)


curv = scipy.io.loadmat(osp.join(path,'cifti_curv_all.mat'))['cifti_curv']
pos = torch.tensor(curv['pos'][0][0][0:number_hemi_nodes].reshape((number_hemi_nodes, 3))[final_mask_L == 1], dtype=torch.float)

faces=torch.tensor(faces_L.T,dtype=torch.long)

#Zeros as curvature values
x=torch.zeros(1008,1)
y=torch.zeros(1008,1)

'''
#Random values from a normal distribution with mean 0 and variance 1 as curvatuve values
x=torch.randn(1008,1)
y=torch.zeros(1008,1)
'''

no_curv=Data(x=x,y=y,pos=pos)
no_curv.face=faces

no_curv=pre_transform(no_curv)
no_curv=transform(no_curv)



#Loading the model

no_curv=DataLoader(no_curv,batch_size=1)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=SplineConv(1,8,dim=3,kernel_size=5,norm=False)
        self.conv2=SplineConv(8,16,dim=3,kernel_size=5,norm=False)
        self.conv3=SplineConv(16,16,dim=3,kernel_size=5,norm=False)
        self.conv4=SplineConv(16,8,dim=3,kernel_size=5,norm=False)
        self.conv5 = SplineConv(8, 1, dim=3, kernel_size=5, norm=False)

    def forward(self, data):
        x, edge_index, pseudo=data.x,data.edge_index,data.edge_attr
        x=F.elu(self.conv1(x,edge_index,pseudo))
        x = F.elu(self.conv2(x, edge_index, pseudo))
        x = F.elu(self.conv3(x, edge_index, pseudo))
        x = F.elu(self.conv4(x, edge_index, pseudo))
        x=F.elu(self.conv5(x,edge_index,pseudo)).view(-1)
        return x


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=Net().to(device)
model.load_state_dict(torch.load('/home/uqfribe1/PycharmProjects/DEEP-fMRI/polarAngle/model4_nothresh_rotated_5layers_smoothL1lossR2_norm_visual_curv.pt',map_location='cpu'))

def test():
    MeanAbsError=0
    y=[]
    y_hat=[]
    for data in no_curv:
        pred=model(data.to(device)).detach()
        y_hat.append(pred)
        y.append(data.to(device).y.view(-1))
        MAE = torch.mean(abs(data.to(device).y.view(-1) - pred)).item()
        MeanAbsError += MAE
    test_MAE = MeanAbsError / len(no_curv)
    output = {'Predicted_values': y_hat, 'Measured_values': y, 'MAE': test_MAE}
    return output

evaluation=test()
torch.save({'Predicted_values':evaluation['Predicted_values'],'Measured_values':evaluation['Measured_values']},osp.join(osp.dirname(osp.realpath(__file__)),'testing.pt'))