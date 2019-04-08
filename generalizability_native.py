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



path=osp.join(osp.dirname(osp.realpath(__file__)),'data','raw','102311_native')
pre_transform=T.Compose([T.FaceToEdge()])
transform=T.Cartesian()


faces_R = scipy.io.loadmat(osp.join(path,'tri_faces_R_102311_native.mat'))['tri_faces_R']
faces_L = scipy.io.loadmat(osp.join(path, 'tri_faces_L_102311_native.mat'))['tri_faces_L']-1


curv_L = torch.tensor(scipy.io.loadmat(osp.join(path,'cifti_curv_L_102311_native.mat'))['cifti_curv_L'],dtype=torch.float)
pos = torch.tensor(scipy.io.loadmat(osp.join(path,'mid_pos_L_102311_native.mat'))['mid_pos_L'],dtype=torch.float)

faces=torch.tensor(faces_L.T,dtype=torch.long)



subj_native=Data(x=curv_L,pos=pos)
subj_native.face=faces

subj_native=pre_transform(subj_native)
subj_native=transform(subj_native)

subj_native=[subj_native,subj_native]





#Loading the model

subj_native=DataLoader(subj_native,batch_size=1)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 8, dim=3, kernel_size=3, norm=False)
        self.conv2 = SplineConv(8, 16, dim=3, kernel_size=3, norm=False)
        self.conv3 = SplineConv(16, 32, dim=3, kernel_size=3, norm=False)
        self.conv4 = SplineConv(32, 16, dim=3, kernel_size=3, norm=False)
        self.conv5 = SplineConv(16, 8, dim=3, kernel_size=3, norm=False)
        self.conv6 = SplineConv(8, 1, dim=3, kernel_size=3, norm=False)


    def forward(self, data):
        x, edge_index, pseudo = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv1(x, edge_index, pseudo))
        x = F.elu(self.conv2(x, edge_index, pseudo))
        x = F.elu(self.conv3(x, edge_index, pseudo))
        x = F.elu(self.conv4(x, edge_index, pseudo))
        x = F.elu(self.conv5(x, edge_index, pseudo))
        x = F.elu(self.conv6(x, edge_index, pseudo)).view(-1)
        return x

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=Net().to(device)
model.load_state_dict(torch.load('/home/uqfribe1/PycharmProjects/DEEP-fMRI/polarAngle/model4_5000_nothresh_rotated_6layers_smoothL1_R2_3kernel_v2.pt',map_location='cpu'))

def test():
    MeanAbsError=0
    y=[]
    y_hat=[]
    for data in subj_native:
        pred=model(data.to(device)).detach()
        y_hat.append(pred)
        #y.append(data.to(device).y.view(-1))
        #MAE = torch.mean(abs(data.to(device).y.view(-1) - pred)).item()
        #MeanAbsError += MAE
    #test_MAE = MeanAbsError / len(subj_native)
    output = {'Predicted_values': y_hat}
    return output

evaluation=test()
torch.save({'Predicted_values':evaluation['Predicted_values']},osp.join(osp.dirname(osp.realpath(__file__)),'testing_native.pt'))