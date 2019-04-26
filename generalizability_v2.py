import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T


import sys
sys.path.append('../..')


from dataset.HCP_3sets_visual_nothr_rotated import Retinotopy
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv

def normalize(feature):
    norm=feature-torch.mean(feature)/torch.std(feature)
    return norm


path=osp.join(osp.dirname(osp.realpath(__file__)),'data')
pre_transform=T.Compose([T.FaceToEdge()])
dev_dataset=Retinotopy(path,'Development', transform=T.Cartesian(),pre_transform=pre_transform,n_examples=181,prediction='polarAngle',myelination=False)
dev_loader=DataLoader(dev_dataset,batch_size=1,shuffle=False)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=SplineConv(1,8,dim=3,kernel_size=5,norm=False)
        self.conv2=SplineConv(8,16,dim=3,kernel_size=5,norm=False)
        self.conv3=SplineConv(16,16,dim=3,kernel_size=5,norm=False)
        self.conv4=SplineConv(16,8,dim=3,kernel_size=5,norm=False)
        self.conv5 = SplineConv(8, 1, dim=3, kernel_size=5, norm=False)

    def forward(self, data):
        x, edge_index, pseudo=torch.ones(len(data.x),1),data.edge_index,data.edge_attr
        x=F.elu(self.conv1(x,edge_index,pseudo))
        x = F.elu(self.conv2(x, edge_index, pseudo))
        x = F.elu(self.conv3(x, edge_index, pseudo))
        x = F.elu(self.conv4(x, edge_index, pseudo))
        x=F.elu(self.conv5(x,edge_index,pseudo)).view(-1)
        return x


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=Net().to(device)
model.load_state_dict(torch.load('/home/uqfribe1/PycharmProjects/DEEP-fMRI/polarAngle/model4_5000_nothresh_5layers_smoothL1_R2.pt',map_location='cpu'))

def test():
    MeanAbsError=0
    y=[]
    y_hat=[]
    for data in dev_loader:
        pred=model(data.to(device)).detach()
        y_hat.append(pred)
        y.append(data.to(device).y.view(-1))
        MAE = torch.mean(abs(data.to(device).y.view(-1) - pred)).item()
        MeanAbsError += MAE
    test_MAE = MeanAbsError / len(dev_loader)
    output = {'Predicted_values': y_hat, 'Measured_values': y, 'MAE': test_MAE}
    return output

evaluation=test()
torch.save({'Predicted_values':evaluation['Predicted_values'],'Measured_values':evaluation['Measured_values']},osp.join(osp.dirname(osp.realpath(__file__)),'testing_modelcurv_nonorm_ones.pt'))
/home/uqfribe1/anaconda3/envs/pytorch_geometric/lib/python3.6/importlib/_bootstrap.py:219: ImportWarning: can't resolve package from __spec__ or __package__, falling back on __name__ and __path__
  return f(*args, **kwds)
/home/uqfribe1/anaconda3/envs/pytorch_geometric/lib/python3.6/site-packages/nibabel/gifti/parse_gifti_fast.py:62: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead
  newarr = np.fromstring(zdec, dtype=dt)
/home/uqfribe1/anaconda3/envs/pytorch_geometric/lib/python3.6/site-packages/nibabel/gifti/gifti.py:760: ResourceWarning: unclosed file <_io.BufferedReader name='/home/uqfribe1/PycharmProjects/DEEP-fMRI/plots/../data/raw/original/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'>
  parser.parse(fptr=file_map['image'].get_prepare_fileobj('rb'))
/home/uqfribe1/anaconda3/envs/pytorch_geometric/lib/python3.6/subprocess.py:766: ResourceWarning: subprocess 7621 is still running
  ResourceWarning, source=self)
/home/uqfribe1/anaconda3/envs/pytorch_geometric/lib/python3.6/subprocess.py:766: ResourceWarning: subprocess 7622 is still running
  ResourceWarning, source=self)

Process finished with exit code 0
