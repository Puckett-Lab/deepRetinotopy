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



path=osp.join(osp.dirname(osp.realpath(__file__)),'..','..','data')
pre_transform=T.Compose([T.FaceToEdge()])
train_dataset=Retinotopy(path,'Train', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='polarAngle',myelination=True,hemisphere='Left')
dev_dataset=Retinotopy(path,'Development', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='polarAngle',myelination=True,hemisphere='Left')
train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)
dev_loader=DataLoader(dev_dataset,batch_size=1,shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = SplineConv(2, 1, dim=3, kernel_size=25, norm=False)
        self.bn1 = torch.nn.BatchNorm1d(1)

    def forward(self, data):
        x, edge_index, pseudo=data.x,data.edge_index,data.edge_attr

        x=F.elu(self.conv1(x,edge_index,pseudo))
        x = self.bn1(x)
        x = F.dropout(x,p=.10,training=self.training)

        return x.view(-1)

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
        torch.save({'Epoch':epoch,'Predicted_values':test_output['Predicted_values'],'Measured_values':test_output['Measured_values'],'R2':test_output['R2'],'Loss':loss,'Dev_MAE':test_output['MAE']},osp.join(osp.dirname(osp.realpath(__file__)),'..','output','model4_nothresh_rotated_1layers_smoothL1lossR2_curvnmyelin_ROI1_k25_batchnorm_dropout010_2_output_epoch'+str(epoch)+'.pt'))
    if test_output['MAE']<=10.94: #MeanAbsError from Benson2014
        break


#Saving the model's learned parameter and predicted/y values
torch.save(model.state_dict(),osp.join(osp.dirname(osp.realpath(__file__)),'..','output','model4_nothresh_rotated_1layers_smoothL1lossR2_curvnmyelin_ROI1_k25_batchnorm_dropout010_2.pt'))

end=time.time()
time=(end-init)/60
print(str(time)+' minutes')
