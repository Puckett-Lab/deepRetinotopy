import os
import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import sys

sys.path.append('../..')

from Retinotopy.dataset.HCP_3sets_ROI import Retinotopy
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv

path = osp.join(osp.dirname(osp.realpath(__file__)), '../Retinotopy', 'data')
pre_transform = T.Compose([T.FaceToEdge()])

hemisphere = 'Left'  # or 'Right'
# Loading test dataset
test_dataset = Retinotopy(path, 'Test', transform=T.Cartesian(),
                          pre_transform=pre_transform, n_examples=181,
                          prediction='polarAngle', myelination=True,
                          hemisphere=hemisphere)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
        x, edge_index, pseudo = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv1(x, edge_index, pseudo))
        x = self.bn1(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv2(x, edge_index, pseudo))
        x = self.bn2(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv3(x, edge_index, pseudo))
        x = self.bn3(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv4(x, edge_index, pseudo))
        x = self.bn4(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv5(x, edge_index, pseudo))
        x = self.bn5(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv6(x, edge_index, pseudo))
        x = self.bn6(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv7(x, edge_index, pseudo))
        x = self.bn7(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv8(x, edge_index, pseudo))
        x = self.bn8(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv9(x, edge_index, pseudo))
        x = self.bn9(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv10(x, edge_index, pseudo))
        x = self.bn10(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv11(x, edge_index, pseudo))
        x = self.bn11(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv12(x, edge_index, pseudo)).view(-1)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

# First, you need to download our pretrained model at https://osf.io/95w4y/
# Loading trained model
if hemisphere == 'Left':
    model.load_state_dict(torch.load(
        './output/deepRetinotopy_PA_LH_model.pt',
        map_location='cpu'))  # Left hemisphere
else:
    model.load_state_dict(torch.load(
        './output/deepRetinotopy_PA_RH_model.pt',
        map_location='cpu'))  # Right hemisphere

# Create an output folder if it doesn't already exist
if hemisphere == 'Left':
    directory = './../Manuscript/testset_results/left_hemi'
else:
    directory = './../Manuscript/testset_results/right_hemi'
if not osp.exists(directory):
    os.makedirs(directory)

# Naming
if hemisphere == 'Left':
    hemi = 'LH'
else:
    hemi = 'RH'

# Evaluating trained model on the test dataset
def test():
    model.eval()

    y = []
    y_hat = []

    for data in test_loader:
        pred = model(data.to(device)).detach()

        y_hat.append(pred)
        y.append(data.to(device).y.view(-1))

    output = {'Predicted_values': y_hat, 'Measured_values': y}
    return output


evaluation = test()
torch.save({'Predicted_values': evaluation['Predicted_values'],
            'Measured_values': evaluation['Measured_values']},
           osp.join(directory,
                    'testset-pred_deepRetinotopy_PA_' + str(hemi) + '.pt'))


# Evaluating trained model on the test dataset (shuffled curvature/myelin
# values)
def test():
    model.eval()

    y = []
    y_hat = []
    myelin = []
    curv = []
    for data in test_loader:
        # Shuffling myelin and curv
        data.x = data.x[torch.randperm(3267)]

        myelin.append(data.x.transpose(0, 1)[1])
        curv.append(data.x.transpose(0, 1)[0])

        pred = model(data.to(device)).detach()

        y_hat.append(pred)
        y.append(data.to(device).y.view(-1))

    output = {'Predicted_values': y_hat, 'Measured_values': y,
              'Shuffled_myelin': myelin, 'Shuffled_curv': curv}
    return output


evaluation = test()
torch.save({'Predicted_values': evaluation['Predicted_values'],
            'Measured_values': evaluation['Measured_values'],
            'Shuffled_myelin': evaluation['Shuffled_myelin'],
            'Shuffled_curv': evaluation['Shuffled_curv']},
           osp.join(directory,
                    'testset-shuffled-myelincurv_deepRetinotopy_PA_' + str(
                        hemi) + '.pt'))


# Evaluating trained model on the test dataset (constant curvature/myelin
# values)
def test():
    model.eval()

    y = []
    y_hat = []
    for data in test_loader:
        data.x.transpose(0, 1)[0] = 0.027303819
        data.x.transpose(0, 1)[1] = 1.4386905

        pred = model(data.to(device)).detach()

        y_hat.append(pred)
        y.append(data.to(device).y.view(-1))

    output = {'Predicted_values': y_hat, 'Measured_values': y}
    return output


evaluation = test()
torch.save({'Predicted_values': evaluation['Predicted_values'],
            'Measured_values': evaluation['Measured_values']},
           osp.join(directory,
                    'testset-constant_deepRetinotopy_PA_' + str(hemi) + '.pt'))
