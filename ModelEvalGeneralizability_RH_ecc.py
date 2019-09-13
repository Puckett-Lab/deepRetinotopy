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
train_dataset=Retinotopy(path,'Train', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='eccentricity',myelination=True,hemisphere='Right')
dev_dataset=Retinotopy(path,'Development', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='eccentricity',myelination=True,hemisphere='Right')
test_dataset=Retinotopy(path,'Test', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='eccentricity',myelination=True,hemisphere='Right')


train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)
dev_loader=DataLoader(dev_dataset,batch_size=1,shuffle=False)
test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False)




class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = SplineConv(2, 8, dim=3, kernel_size=25, norm=False)
        self.bn1 = torch.nn.BatchNorm1d(8)

        self.conv2 = SplineConv(8, 16, dim=3, kernel_size=25, norm=False)
        self.bn2 = torch.nn.BatchNorm1d(16)

        self.conv3 = SplineConv(16, 32, dim=3, kernel_size=25, norm=False)
        self.bn3 = torch.nn.BatchNorm1d(32)

        self.conv4 = SplineConv(32, 32, dim=3, kernel_size=25, norm=False)
        self.bn4 = torch.nn.BatchNorm1d(32)

        self.conv5 = SplineConv(32, 32, dim=3, kernel_size=25, norm=False)
        self.bn5 = torch.nn.BatchNorm1d(32)

        self.conv6 = SplineConv(32, 32, dim=3, kernel_size=25, norm=False)
        self.bn6 = torch.nn.BatchNorm1d(32)

        self.conv7 = SplineConv(32, 32, dim=3, kernel_size=25, norm=False)
        self.bn7 = torch.nn.BatchNorm1d(32)

        self.conv8 = SplineConv(32, 32, dim=3, kernel_size=25, norm=False)
        self.bn8 = torch.nn.BatchNorm1d(32)

        self.conv9 = SplineConv(32, 32, dim=3, kernel_size=25, norm=False)
        self.bn9 = torch.nn.BatchNorm1d(32)

        self.conv10 = SplineConv(32, 16, dim=3, kernel_size=25, norm=False)
        self.bn10 = torch.nn.BatchNorm1d(16)

        self.conv11 = SplineConv(16, 8, dim=3, kernel_size=25, norm=False)
        self.bn11 = torch.nn.BatchNorm1d(8)

        self.conv12 = SplineConv(8, 1, dim=3, kernel_size=25, norm=False)

    def forward(self, data):
        x, edge_index, pseudo=data.x,data.edge_index,data.edge_attr
        x=F.elu(self.conv1(x,edge_index,pseudo))
        x = self.bn1(x)
        x = F.dropout(x,p=.10,training=self.training)

        x = F.elu(self.conv2(x, edge_index, pseudo))
        x = self.bn2(x)
        x = F.dropout(x,p=.10, training=self.training)

        x = F.elu(self.conv3(x, edge_index, pseudo))
        x = self.bn3(x)
        x = F.dropout(x,p=.10, training=self.training)

        x = F.elu(self.conv4(x, edge_index, pseudo))
        x = self.bn4(x)
        x = F.dropout(x,p=.10, training=self.training)

        x = F.elu(self.conv5(x, edge_index, pseudo))
        x = self.bn5(x)
        x = F.dropout(x,p=.10, training=self.training)

        x = F.elu(self.conv6(x, edge_index, pseudo))
        x = self.bn6(x)
        x = F.dropout(x,p=.10, training=self.training)

        x = F.elu(self.conv7(x, edge_index, pseudo))
        x = self.bn7(x)
        x = F.dropout(x,p=.10, training=self.training)

        x = F.elu(self.conv8(x, edge_index, pseudo))
        x = self.bn8(x)
        x = F.dropout(x,p=.10, training=self.training)

        x = F.elu(self.conv9(x, edge_index, pseudo))
        x = self.bn9(x)
        x = F.dropout(x,p=.10, training=self.training)

        x = F.elu(self.conv10(x, edge_index, pseudo))
        x = self.bn10(x)
        x = F.dropout(x,p=.10, training=self.training)

        x = F.elu(self.conv11(x, edge_index, pseudo))
        x = self.bn11(x)
        x = F.dropout(x,p=.10, training=self.training)

        x=F.elu(self.conv12(x,edge_index,pseudo)).view(-1)
        return x


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=Net().to(device)
model.load_state_dict(torch.load('/home/uqfribe1/PycharmProjects/DEEP-fMRI/model4_nothresh_RH_ecc_12layers_smoothL1loss_curvnmyelin_ROI1_k25_batchnorm_dropout010_4.pt',map_location='cpu'))

def test():
    model.eval()
    MeanAbsError = 0
    y = []
    y_hat = []
    for data in test_loader:
        # Shuffling myelin and curv
        data.x=data.x[torch.randperm(3219)]
        pred = model(data.to(device)).detach()
        y_hat.append(pred)
        y.append(data.to(device).y.view(-1))
        MAE = torch.mean(abs(data.to(device).y.view(-1) - pred)).item()
        MeanAbsError += MAE
    test_MAE = MeanAbsError / len(test_loader)
    output = {'Predicted_values': y_hat, 'Measured_values': y, 'MAE': test_MAE}
    return output


evaluation = test()
torch.save({'Predicted_values': evaluation['Predicted_values'], 'Measured_values': evaluation['Measured_values']},
           osp.join(osp.dirname(osp.realpath(__file__)),'testset_results', 'testset-shuffled-myelincurv_Model4_ecc_RH.pt'))

# def test():
#     model.eval()
#     MeanAbsError=0
#     y=[]
#     y_hat=[]
#     for data in dev_loader:
#         #Shuffling curv
#         data.x.transpose(0,1)[0]=data.x.transpose(0,1)[0][torch.randperm(3267)]
#         pred=model(data.to(device)).detach()
#         y_hat.append(pred)
#         y.append(data.to(device).y.view(-1))
#         MAE = torch.mean(abs(data.to(device).y.view(-1) - pred)).item()
#         MeanAbsError += MAE
#     test_MAE = MeanAbsError / len(dev_loader)
#     output = {'Predicted_values': y_hat, 'Measured_values': y, 'MAE': test_MAE}
#     return output
#
# evaluation=test()
# torch.save({'Predicted_values':evaluation['Predicted_values'],'Measured_values':evaluation['Measured_values']},osp.join(osp.dirname(osp.realpath(__file__)),'testing_RH_shuffled-curv.pt'))


# def test():
#     model.eval()
#     MeanAbsError = 0
#     y = []
#     y_hat = []
#     for data in dev_loader:
#         # Shuffling myelin
#         data.x.transpose(0, 1)[1] = data.x.transpose(0, 1)[1][torch.randperm(3267)]
#         pred = model(data.to(device)).detach()
#         y_hat.append(pred)
#         y.append(data.to(device).y.view(-1))
#         MAE = torch.mean(abs(data.to(device).y.view(-1) - pred)).item()
#         MeanAbsError += MAE
#     test_MAE = MeanAbsError / len(dev_loader)
#     output = {'Predicted_values': y_hat, 'Measured_values': y, 'MAE': test_MAE}
#     return output
#
#
# evaluation = test()
# torch.save({'Predicted_values': evaluation['Predicted_values'], 'Measured_values': evaluation['Measured_values']},
#            osp.join(osp.dirname(osp.realpath(__file__)), 'testing_RH_shuffled-myelin.pt'))



# def test():
#     model.eval()
#     MeanAbsError = 0
#     y = []
#     y_hat = []
#     for data in test_loader:
#         pred = model(data.to(device)).detach()
#         y_hat.append(pred)
#         y.append(data.to(device).y.view(-1))
#         MAE = torch.mean(abs(data.to(device).y.view(-1) - pred)).item()
#         MeanAbsError += MAE
#     test_MAE = MeanAbsError / len(test_loader)
#     output = {'Predicted_values': y_hat, 'Measured_values': y, 'MAE': test_MAE}
#     return output
#
#
# evaluation = test()
# torch.save({'Predicted_values': evaluation['Predicted_values'], 'Measured_values': evaluation['Measured_values']},
#            osp.join(osp.dirname(osp.realpath(__file__)),'testset_results', 'testset-pred_Model4_ecc_RH.pt'))