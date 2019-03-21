import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

import sys
sys.path.append('../..')

from dataset.HCP_3sets_visual_nothr import Retinotopy
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv



path=osp.join(osp.dirname(osp.realpath(__file__)),'..','..','data')
pre_transform=T.Compose([T.FaceToEdge()])
train_dataset=Retinotopy(path,'Train', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='polarAngle',myelination=False)
dev_dataset=Retinotopy(path,'Development', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='polarAngle',myelination=False)
train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
dev_loader=DataLoader(dev_dataset,batch_size=1)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=SplineConv(1,8,dim=3,kernel_size=5,norm=False)
        self.conv2=SplineConv(8,32,dim=3,kernel_size=5,norm=False)
        self.conv3=SplineConv(32,32,dim=3,kernel_size=5,norm=False)
        self.conv4=SplineConv(32,8,dim=3,kernel_size=5,norm=False)
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
optimizer=torch.optim.Adam(model.parameters(),lr=0.1)


def train(epoch):
    model.train()

    if epoch == 600:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.05

    if epoch == 2000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01

    if epoch == 3500:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.005

    for data in train_loader:
        data=data.to(device)
        optimizer.zero_grad()

        R2=data.R2.view(-1)


        loss=torch.nn.MSELoss()
        output_loss=loss(model(data)*R2/data.y.view(-1),data.y.view(-1)*R2/data.y.view(-1))
        output_loss.backward()

        MAE = torch.mean(abs(data.to(device).y.view(-1) - model(data))).item()

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
        MAE=torch.mean(abs(data.to(device).y.view(-1)-pred)).item()
        MeanAbsError += MAE

    test_MAE=MeanAbsError/len(dev_loader)
    output={'Predicted_values':y_hat,'Measured_values':y,'R2':R2_plot,'MAE':test_MAE}
    return output



for epoch in range(1, 5001):
    loss,MAE=train(epoch)
    test_output = test()
    print('Epoch: {:02d}, Train_loss: {:.4f}, Train_MAE: {:.4f}, Test_MAE: {:.4f}'.format(epoch, loss, MAE,test_output['MAE']))
    if epoch%1000==0:
        torch.save({'Epoch':epoch,'Predicted_values':test_output['Predicted_values'],'Measured_values':test_output['Measured_values'],'R2':test_output['R2'],'Loss':loss,'Dev_MAE':test_output['MAE']},osp.join(osp.dirname(osp.realpath(__file__)),'..','output','_model4_nothresh_5layers_lr_feat_loss_output_epoch'+str(epoch)+'.pt'))
    if test_output['MAE']<=10.94: #MeanAbsError from Benson2014
        break


#Saving the model's learned parameter and predicted/y values
torch.save(model.state_dict(),osp.join(osp.dirname(osp.realpath(__file__)),'output','_model4_5000_nothresh_5layers_lr_feat_loss.pt'))