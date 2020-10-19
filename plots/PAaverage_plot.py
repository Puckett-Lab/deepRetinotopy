import os.path as osp
import sys
import torch_geometric.transforms as T
import numpy as np

sys.path.append('../..')

from functions.def_ROIs_WangParcelsPlusFovea import roi
from nilearn import plotting
from dataset.HCP_3sets_ROI import Retinotopy
from torch_geometric.data import DataLoader

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')

# Right hemisphere
pre_transform = T.Compose([T.FaceToEdge()])
train_dataset_right = Retinotopy(path, 'Train', transform=T.Cartesian(),
                                 pre_transform=pre_transform, n_examples=181,
                                 prediction='polarAngle', myelination=True,
                                 hemisphere='Right')
train_loader_right = DataLoader(train_dataset_right, batch_size=1,
                                shuffle=True)

label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)
PolarAngle = np.zeros((32492, 1))

# Mean polar angle map
PA = []
for data in train_loader_right:
    PA.append(np.array(data.y))
PA = np.mean(PA, 0)

# Saving the average map
np.savez('./output/AveragePolarAngleMap_RH.npz', list=PA)

# Settings for plot
PolarAngle[final_mask_R == 1] = np.reshape(PA, (-1, 1))
PolarAngle = np.array(PolarAngle)
minus = PolarAngle > 180
sum = PolarAngle < 180
PolarAngle[minus] = PolarAngle[minus] - 180
PolarAngle[sum] = PolarAngle[sum] + 180

view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '..',
                       'data/raw/original/S1200_7T_Retinotopy_9Zkk'
                       '/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k'
                       '/S1200_7T_Retinotopy181.R.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(PolarAngle[0:32492], (-1)), cmap='gist_rainbow_r',
    black_bg=True, symmetric_cmap=False)
view.open_in_browser()


# Left hemisphere
train_dataset_left = Retinotopy(path, 'Train', transform=T.Cartesian(),
                                pre_transform=pre_transform, n_examples=181,
                                prediction='polarAngle', myelination=True,
                                hemisphere='Left')
train_loader_left = DataLoader(train_dataset_left, batch_size=1, shuffle=True)

label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)
PolarAngle = np.zeros((32492, 1))

# Mean polar angle map
PA = []
for data in train_loader_left:
    PA.append(np.array(data.y))
PA = np.mean(PA, 0)

# Saving the average map
np.savez('./output/AveragePolarAngleMap_LH.npz', list=PA)

# Settings for plot
PolarAngle[final_mask_L == 1] = np.reshape(PA, (-1, 1))
PolarAngle = np.array(PolarAngle)
minus = PolarAngle > 180
sum = PolarAngle < 180
PolarAngle[minus] = PolarAngle[minus] - 180
PolarAngle[sum] = PolarAngle[sum] + 180

view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '..',
                       'data/raw/original/S1200_7T_Retinotopy_9Zkk'
                       '/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k'
                       '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(PolarAngle[0:32492], (-1)), cmap='gist_rainbow_r',
    black_bg=True, symmetric_cmap=False)
view.open_in_browser()
