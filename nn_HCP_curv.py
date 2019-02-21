import os.path as osp
import torch
import torch.nn.functional as F

from dataset.HCP_curv_2sets import Retinotopy
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv
import numpy as np


path=osp.join(osp.dirname(osp.realpath(__file__)), 'data')
print(path)
#path='/home/uqfribe1/PycharmProjects/pytorch_geometric/my_stuff/my_own_data'
pre_transform=T.Compose([T.FaceToEdge(),T.NormalizeFeatures()])
train_dataset=Retinotopy(path,True, transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181)
test_dataset=Retinotopy(path,False,transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181)
train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)
d = train_dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=3, kernel_size=5, norm=False)
        #self.conv2 = SplineConv(32, 64, dim=3, kernel_size=5, norm=False)
        #self.conv3 = SplineConv(64, 32, dim=3, kernel_size=5, norm=False)
        self.conv4 = SplineConv(32, 1, dim=3, kernel_size=5, norm=False)
        #self.conv5 = SplineConv(64, 64, dim=3, kernel_size=5, norm=False)
        #self.conv6 = SplineConv(64, 64, dim=3, kernel_size=5, norm=False)
        #self.fc1 = torch.nn.Linear(64, 256)
        #self.fc2 = torch.nn.Linear(256, d.num_nodes)

    def forward(self, data):
        x, edge_index, pseudo = data.x, data.edge_index, data.edge_attr
        #print('initial shape x {}'.format(np.shape(x)))
        x = F.elu(self.conv1(x, edge_index, pseudo))
        #print('shape after conv1 {}'.format(np.shape(x)))
        #x = F.elu(self.conv2(x, edge_index, pseudo))
        #print('shape after conv2 {}'.format(np.shape(x)))
        #x = F.elu(self.conv3(x, edge_index, pseudo))
        #print(np.shape(x))
        #x = F.elu(self.conv4(x, edge_index, pseudo))
        #x = F.elu(self.conv5(x, edge_index, pseudo))
        #x = F.elu(self.conv6(x, edge_index, pseudo))
        #x = F.elu(self.fc1(x))
        #print(np.shape(x))
        #x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        x = F.elu(self.conv4(x, edge_index, pseudo)).view(-1)
        return x



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()

    if epoch == 61:
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



def test():
    model.eval()


    for data in test_loader:
        pred = model(data.to(device))
        print(pred)
        print(data.to(device).y.view(-1))
        MAE=(1/len(test_dataset))*torch.sum(abs(data.to(device).y.view(-1)-pred))
        #print(MAE)
    return MAE


for epoch in range(1, 100):
    train(epoch)
    test_acc = test()
    print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))