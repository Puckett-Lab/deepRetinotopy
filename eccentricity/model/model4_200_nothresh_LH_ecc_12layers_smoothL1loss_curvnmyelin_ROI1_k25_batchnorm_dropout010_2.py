import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np
import sys
import time

sys.path.append('../..')


from dataset.HCP_3sets_ROI import Retinotopy
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv



path=osp.join(osp.dirname(osp.realpath(__file__)),'..','..','data')
pre_transform=T.Compose([T.FaceToEdge()])
train_dataset=Retinotopy(path,'Train', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='eccentricity',myelination=True,hemisphere='Left')
dev_dataset=Retinotopy(path,'Development', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='eccentricity',myelination=True,hemisphere='Left')
train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)
dev_loader=DataLoader(dev_dataset,batch_size=1,shuffle=False)



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
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)


def train(epoch):
    model.train()

    if epoch == 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.005

    '''if epoch == 2000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 3500:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005'''

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
    MeanAbsError_thr = 0
    y=[]
    y_hat=[]
    R2_plot=[]
    for data in dev_loader:
        pred = model(data.to(device)).detach()
        y_hat.append(pred)
        y.append(data.to(device).y.view(-1))

        R2 = data.R2.view(-1)
        R2_plot.append(R2)
        threshold = R2.view(-1) > 2.2
        threshold2 = R2.view(-1) > 17

        MAE=torch.mean(abs(data.to(device).y.view(-1)[threshold==1]-pred[threshold==1])).item()
        MAE_thr = torch.mean(abs(data.to(device).y.view(-1)[threshold2 == 1] - pred[threshold2 == 1])).item()
        MeanAbsError_thr += MAE_thr
        MeanAbsError += MAE

    test_MAE=MeanAbsError/len(dev_loader)
    test_MAE_thr = MeanAbsError_thr / len(dev_loader)
    output={'Predicted_values':y_hat,'Measured_values':y,'R2':R2_plot,'MAE':test_MAE,'MAE_thr':test_MAE_thr}
    return output

init=time.time()


for epoch in range(1, 201):
    loss,MAE=train(epoch)
    test_output = test()
    print('Epoch: {:02d}, Train_loss: {:.4f}, Train_MAE: {:.4f}, Test_MAE: {:.4f}, Test_MAE_thr: {:.4f}'.format(epoch, loss, MAE,test_output['MAE'],test_output['MAE_thr']))
    if epoch%25==0:
        torch.save({'Epoch':epoch,'Predicted_values':test_output['Predicted_values'],'Measured_values':test_output['Measured_values'],'R2':test_output['R2'],'Loss':loss,'Dev_MAE':test_output['MAE']},osp.join(osp.dirname(osp.realpath(__file__)),'..','output','model4_nothresh_ecc_12layers_smoothL1loss_curvnmyelin_ROI1_k25_batchnorm_dropout010_2_output_epoch'+str(epoch)+'.pt'))
    if test_output['MAE']<=0.3: #MeanAbsError from Benson2014
        break


#Saving the model's learned parameter and predicted/y values
torch.save(model.state_dict(),osp.join(osp.dirname(osp.realpath(__file__)),'..','output','model4_nothresh_ecc_12layers_smoothL1loss_curvnmyelin_ROI1_k25_batchnorm_dropout010_2.pt'))

end=time.time()
time=(end-init)/60
print(str(time)+' minutes')