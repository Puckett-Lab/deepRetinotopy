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
pre_transform = T.Compose([T.FaceToEdge()])


# Right hemisphere
train_dataset_right = Retinotopy(path, 'Train', transform=T.Cartesian(),
                           pre_transform=pre_transform, n_examples=181,
                           prediction='eccentricity', myelination=True,
                           hemisphere='Right')
train_loader_right = DataLoader(train_dataset_right, batch_size=1, shuffle=True)

label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)
ecc_thr = np.zeros((32492, 1))

# Mean eccentricity map
ecc = []
for data in train_loader_right:
    ecc.append(np.array(data.y))
ecc = np.mean(ecc, 0)

# Saving the average map
np.savez('./output/AverageEccentricityMap_RH.npz', list=ecc)

# Masking
ecc_thr[final_mask_R == 1] = np.reshape(ecc, (-1, 1))

# Uncomment to visualize the average map
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '..',
                       'data/raw/original/S1200_7T_Retinotopy_9Zkk'
                       '/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k'
                       '/S1200_7T_Retinotopy181.R.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(ecc_thr[0:32492], (-1)), cmap='gist_rainbow_r',
    black_bg=True, symmetric_cmap=False)
view.open_in_browser()



# Left hemisphere
train_dataset_left = Retinotopy(path, 'Train', transform=T.Cartesian(),
                           pre_transform=pre_transform, n_examples=181,
                           prediction='eccentricity', myelination=True,
                           hemisphere='Left')
train_loader_left = DataLoader(train_dataset_left, batch_size=1, shuffle=True)

label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)
ecc_thr = np.zeros((32492, 1))

# Mean eccentricity map
ecc = []
for data in train_loader_left:
    ecc.append(np.array(data.y))
ecc = np.mean(ecc, 0)

# Saving the average map
np.savez('./output/AverageEccentricityMap_LH.npz', list=ecc)

#Masking
ecc_thr[final_mask_L == 1] = np.reshape(ecc, (-1, 1))

view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '..',
                       'data/raw/original/S1200_7T_Retinotopy_9Zkk'
                       '/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k'
                       '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(ecc_thr[0:32492], (-1)), cmap='gist_rainbow_r',
    black_bg=True, symmetric_cmap=False)
view.open_in_browser()