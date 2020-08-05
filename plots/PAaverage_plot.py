import os.path as osp
import scipy.io
import sys
import torch_geometric.transforms as T
import numpy as np

sys.path.append('../..')

from functions.def_ROIs_WangParcelsPlusFovea import roi
from nilearn import plotting
from dataset.HCP_3sets_ROI import Retinotopy
from torch_geometric.data import DataLoader


label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)
PA_thr = np.zeros((32492, 1))

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
pre_transform = T.Compose([T.FaceToEdge()])
train_dataset = Retinotopy(path, 'Train', transform=T.Cartesian(),
                           pre_transform=pre_transform, n_examples=181,
                           prediction='polarAngle', myelination=True,
                           hemisphere='Right')
dev_dataset = Retinotopy(path, 'Development', transform=T.Cartesian(),
                         pre_transform=pre_transform, n_examples=181,
                         prediction='polarAngle', myelination=True,
                         hemisphere='Right')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)

PA = []
for data in train_loader:
    PA.append(np.array(data.y))

PA = np.mean(PA, 0)

PA_thr[final_mask_R == 1] = np.reshape(PA, (-1, 1))

PA_thr = np.array(PA_thr)
minus = PA_thr > 180
sum = PA_thr < 180
PA_thr[minus] = PA_thr[minus] - 180
PA_thr[sum] = PA_thr[sum] + 180

view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '..',
                       'data/raw/original/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/S1200_7T_Retinotopy181.R.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(PA_thr[0:32492], (-1)), cmap='gist_rainbow_r',
    black_bg=True, symmetric_cmap=False)
view.open_in_browser()
