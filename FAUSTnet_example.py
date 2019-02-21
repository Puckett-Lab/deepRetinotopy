import os.path as osp
import torch
import torch.nn.functional as F

from dataset.HCP_3sets import Retinotopy
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv


path=osp.join(osp.dirname(osp.realpath(__file__)), 'HCPdata')
print(path)
pre_transform=T.FaceToEdge()
data=Retinotopy(path,pre_transform=pre_transform,n_examples=181)

data.train_mask=torch.zeros(data.num_nodes,dtype=torch.uint8)
data.train_mask[]



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=3, kernel_size=5, norm=False)
        self.conv2 = SplineConv(32, 64, dim=3, kernel_size=5, norm=False)
        self.conv3 = SplineConv(64, 64, dim=3, kernel_size=5, norm=False)
        self.conv4 = SplineConv(64, 64, dim=3, kernel_size=5, norm=False)
        self.conv5 = SplineConv(64, 64, dim=3, kernel_size=5, norm=False)
        self.conv6 = SplineConv(64, 64, dim=3, kernel_size=5, norm=False)
        self.fc1 = torch.nn.Linear(64, 256)
        self.fc2 = torch.nn.Linear(256, d.num_nodes)

    def forward(self, data):
        x, edge_index, pseudo = data.x, data.edge_index, data.edge_attr
        x = F.elu(norm(self.conv1(x, edge_index, pseudo), edge_index))
        x = F.elu(norm(self.conv2(x, edge_index, pseudo), edge_index))
        x = F.elu(norm(self.conv3(x, edge_index, pseudo), edge_index))
        x = F.elu(norm(self.conv4(x, edge_index, pseudo), edge_index))
        x = F.elu(norm(self.conv5(x, edge_index, pseudo), edge_index))
        x = F.elu(norm(self.conv6(x, edge_index, pseudo), edge_index))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data= Net().to(device),data.to(device)
#target = torch.tensor(d.y, dtype=torch.float, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 61:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    for data in train_loader:
        optimizer.zero_grad()
        F.nll_loss(model(data.to(device)), target).backward()
        optimizer.step()


def test():
    model.eval()
    correct = 0

    for data in test_loader:
        pred = model(data.to(device)).max(1)[1]
        correct += pred.eq(target).sum().item()
    return correct / (len(test_dataset) * d.num_nodes)


for epoch in range(1, 101):
    train(epoch)
    test_acc = test()
    print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))
