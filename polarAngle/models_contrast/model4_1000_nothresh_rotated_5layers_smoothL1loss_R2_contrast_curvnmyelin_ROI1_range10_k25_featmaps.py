import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np
import sys

sys.path.append('../..')


from dataset.HCP_3sets_visual_nothr_rotated_ROI1 import Retinotopy
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv



path=osp.join(osp.dirname(osp.realpath(__file__)),'..','..','data')
pre_transform=T.Compose([T.FaceToEdge()])
train_dataset=Retinotopy(path,'Train', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='polarAngle',myelination=True)
dev_dataset=Retinotopy(path,'Development', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='polarAngle',myelination=True)
train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)
dev_loader=DataLoader(dev_dataset,batch_size=1,shuffle=False)

upper_curv=0.36853024
lower_curv=-0.22703196


upper_myelin=1.648841
lower_myelin=1.2585511


def transform(input,range):
    input_T=torch.reshape(input,(-1,2)).transpose(0, 1)

    #Curvature
    transverse_0=((input_T[0]-lower_curv)/(upper_curv-lower_curv))*(range-(-range))+(-range)
    transverse_0[input_T[0]>range]=range
    transverse_0[input_T[0]<-range]=-range

    #Myelin
    transverse_1 = ((input_T[1] - lower_myelin) / (upper_myelin - lower_myelin)) * (range - (-range)) + (-range)
    transverse_1[input_T[1] > range] = range
    transverse_1[input_T[1] < -range] = -range

    transform = torch.cat((torch.reshape(transverse_0,(-1,1)),torch.reshape(transverse_1,(-1,1))),1)

    return transform


class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=SplineConv(2,32,dim=3,kernel_size=25,norm=False)
        self.conv2=SplineConv(32,64,dim=3,kernel_size=25,norm=False)
        self.conv3=SplineConv(64,64,dim=3,kernel_size=25,norm=False)
        self.conv4=SplineConv(64,32,dim=3,kernel_size=25,norm=False)
        self.conv5 = SplineConv(32, 1, dim=3, kernel_size=25, norm=False)

    def forward(self, data):
        x, edge_index, pseudo=transform(data.x,10),data.edge_index,data.edge_attr
        x=F.elu(self.conv1(x,edge_index,pseudo))
        x = F.elu(self.conv2(x, edge_index, pseudo))
        x = F.elu(self.conv3(x, edge_index, pseudo))
        x = F.elu(self.conv4(x, edge_index, pseudo))
        x=F.elu(self.conv5(x,edge_index,pseudo)).view(-1)
        return x

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=Net().to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)


def train(epoch):
    model.train()

    if epoch == 1000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.005

    if epoch == 2000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 3500:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    for data in train_loader:
        data=data.to(device)
        optimizer.zero_grad()

        R2 = data.R2.view(-1)
        threshold = R2.view(-1) > 2.2

        loss=torch.nn.SmoothL1Loss()
        output_loss=loss(R2*model(data),R2*data.y.view(-1))
        output_loss.backward()

        MAE = torch.mean(abs(data.to(device).y.view(-1)[threshold==1] - model(data)[threshold==1])).item()

        optimizer.step()
    return output_loss.detach(), MAE


def test():
    model.eval()
    MeanAbsError =0
    y=[]
    y_hat=[]
    R2_plot=[]
    for data in dev_loader:
        pred = model(data.to(device)).detach()
        y_hat.append(pred)
        y.append(data.to(device).y.view(-1))

        R2 = data.to(device).R2.view(-1)
        R2_plot.append(R2)
        threshold = R2.view(-1) > 2.2

        MAE=torch.mean(abs(data.to(device).y.view(-1)[threshold==1]-pred[threshold==1])).item()
        MeanAbsError += MAE

    test_MAE=MeanAbsError/len(dev_loader)
    output={'Predicted_values':y_hat,'Measured_values':y,'R2':R2_plot,'MAE':test_MAE}
    return output




for epoch in range(1, 1001):
    loss,MAE=train(epoch)
    test_output = test()
    print('Epoch: {:02d}, Train_loss: {:.4f}, Train_MAE: {:.4f}, Test_MAE: {:.4f}'.format(epoch, loss, MAE,test_output['MAE']))
    if epoch%200==0:
        torch.save({'Epoch':epoch,'Predicted_values':test_output['Predicted_values'],'Measured_values':test_output['Measured_values'],'R2':test_output['R2'],'Loss':loss,'Dev_MAE':test_output['MAE']},osp.join(osp.dirname(osp.realpath(__file__)),'..','output','model4_nothresh_rotated_5layers_smoothL1lossR2_contrast_curvnmyelin_ROI1_range10_k25_featmap_output_epoch'+str(epoch)+'.pt'))
    if test_output['MAE']<=10.94: #MeanAbsError from Benson2014
        break


#Saving the model's learned parameter and predicted/y values
torch.save(model.state_dict(),osp.join(osp.dirname(osp.realpath(__file__)),'..','output','model4_nothresh_rotated_5layers_smoothL1lossR2_contrast_curvnmyelin_ROI1_range10_k25_featmap.pt'))
