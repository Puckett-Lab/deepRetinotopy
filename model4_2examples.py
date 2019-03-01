import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from dataset.HCP_3sets_visual import Retinotopy
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv



path=osp.join(osp.dirname(osp.realpath(__file__)),'data')
pre_transform=T.Compose([T.FaceToEdge()])
train_dataset=Retinotopy(path,'Train', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181)
train_loader=DataLoader(train_dataset[0:2],batch_size=1,shuffle=True)
dev_loader=DataLoader(train_dataset[0:2],batch_size=1)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=SplineConv(1,8,dim=3,kernel_size=5,norm=False)
        self.conv2=SplineConv(8,8,dim=3,kernel_size=5,norm=False)
        self.conv3=SplineConv(8,8,dim=3,kernel_size=5,norm=False)
        self.conv4 = SplineConv(8, 1, dim=3, kernel_size=5, norm=False)

    def forward(self, data):
        x, edge_index, pseudo=data.x,data.edge_index,data.edge_attr
        x=F.elu(self.conv1(x,edge_index,pseudo))
        x = F.elu(self.conv2(x, edge_index, pseudo))
        x = F.elu(self.conv3(x, edge_index, pseudo))
        x=F.elu(self.conv4(x,edge_index,pseudo)).view(-1)-1
        return x

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=Net().to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=0.1)


def train(epoch):
    model.train()

    if epoch == 30:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01

    if epoch == 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001


    for data in train_loader:
        data=data.to(device)
        #print('shape of the data {}'.format(np.shape(data)))
        optimizer.zero_grad()
        loss=torch.nn.MSELoss()
        output_loss=loss(model(data),data.y.view(-1))
        #print(output_loss.item())
        output_loss.backward()
        optimizer.step()
    return output_loss.detach()


def test():
    model.eval()
    MeanAbsError =0
    y=[]
    y_hat=[]
    R2_plot=[]
    for data in dev_loader:
        pred = model(data.to(device)).detach()
        R2=data.to(device).R2
        threshold=R2.view(-1)>2.2
        R2_plot.append(R2)
        y_hat.append(pred)
        y.append(data.to(device).y.view(-1))
        MAE=torch.mean(abs(data.to(device).y.view(-1)[threshold]-pred[threshold])).item()
        MeanAbsError += MAE
        #print('Mean Absolute Error: {:.4f}'.format(MAE))
    test_MAE=MeanAbsError/len(dev_loader)
    output={'Predicted_values':pred,'Measured_values':y,'R2':R2_plot,'MAE':test_MAE}
    return output



for epoch in range(1, 10000):
    loss=train(epoch)
    test_output = test()
    print('Epoch: {:02d}, Train: {:.4f}, Test: {:.4f}'.format(epoch, loss, test_output['MAE']))
    if epoch%1000==0:
        torch.save({'Epoch':epoch,'Predicted_values':test_output['Predicted_values'],'Measured_values':test_output['Measured_values'],'R2':test_output['R2'],'Loss':loss,'Dev_MAE':test_output['MAE']},osp.join(osp.dirname(osp.realpath(__file__)),'output','model41_2examples_output_epoch'+str(epoch)+'.pt'))
    if test_output['MAE']<=10.94: #MeanAbsError from Benson2014
        break


#Saving the model's learned parameter and predicted/y values
torch.save(model.state_dict(),osp.join(osp.dirname(osp.realpath(__file__)),'output','model41_2examples.pt'))
