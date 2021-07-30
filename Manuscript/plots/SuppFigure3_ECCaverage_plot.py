import os.path as osp
import sys
import torch_geometric.transforms as T
import numpy as np
import scipy.io

sys.path.append('../..')

from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi
from nilearn import plotting
from Retinotopy.dataset.HCP_3sets_ROI import Retinotopy
from torch_geometric.data import DataLoader

subject_index = 7

hcp_id = ['617748', '191336', '572045', '725751', '198653',
          '601127', '644246', '191841', '680957', '157336']

path_curv = './../../Retinotopy/data/raw/converted'
curv = scipy.io.loadmat(osp.join(path_curv, 'cifti_curv_all.mat'))[
    'cifti_curv']
background = np.reshape(
    curv['x' + hcp_id[subject_index] + '_curvature'][0][0][32492:], (-1))
threshold = 10  # threshold for the curvature map

# Background settings
nocurv = np.isnan(background)
background[nocurv == 1] = 0
background[background < 0] = 0
background[background > 0] = 1

# Right hemisphere
# Loading training data
path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'Retinotopy/data')
pre_transform = T.Compose([T.FaceToEdge()])

train_dataset_right = Retinotopy(path, 'Train', transform=T.Cartesian(),
                                 pre_transform=pre_transform, n_examples=181,
                                 prediction='eccentricity', myelination=True,
                                 hemisphere='Right')
train_loader_right = DataLoader(train_dataset_right, batch_size=1,
                                shuffle=True)

# ROI settings
label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)
ecc_thr = np.zeros((32492, 1))

# Mean eccentricity map
ecc = []
for data in train_loader_right:
    ecc.append(np.array(data.y))
ecc = np.mean(ecc, 0)

# # Saving the average map
# np.savez('./output/AverageEccentricityMap_RH.npz', list=ecc)

# # Mask for prediction errors
# ecc_1to8 = []
# for i in range(len(ecc)):
#     if ecc[i][0] < 1 or ecc[i][0] > 8:
#         ecc_1to8.append(0)
#     else:
#         ecc_1to8.append(ecc[i][0])
# ecc_1to8 = np.reshape(np.array(ecc_1to8),(-1))
# np.savez('./output/MaskEccentricity_above1below8ecc_RH', list = ecc_1to8 > 0)


# Masking
ecc_thr[final_mask_R == 1] = np.reshape(ecc, (-1, 1)) * 10 + threshold
ecc_thr[final_mask_R != 1] = 0

view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../..',
                       'Retinotopy/data/raw/surfaces'
                       '/S1200_7T_Retinotopy181.R.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(ecc_thr[0:32492], (-1)), bg_map=background,
    cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False,
    threshold=threshold, vmax=130)
view.open_in_browser()



# Left hemisphere
background = np.reshape(
    curv['x' + hcp_id[subject_index] + '_curvature'][0][0][0:32492], (-1))

# Background settings
nocurv = np.isnan(background)
background[nocurv == 1] = 0
background[background < 0] = 0
background[background > 0] = 1

# Loading training data
train_dataset_left = Retinotopy(path, 'Train', transform=T.Cartesian(),
                                pre_transform=pre_transform, n_examples=181,
                                prediction='eccentricity', myelination=True,
                                hemisphere='Left')
train_loader_left = DataLoader(train_dataset_left, batch_size=1, shuffle=True)

# ROI settings
label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)
ecc_thr = np.zeros((32492, 1))

# Mean eccentricity map
ecc = []
for data in train_loader_left:
    ecc.append(np.array(data.y))
ecc = np.mean(ecc, 0)

# # Saving the average map
# np.savez('./output/AverageEccentricityMap_LH.npz', list=ecc)

# # Mask for prediction errors
# ecc_1to8 = []
# for i in range(len(ecc)):
#     if ecc[i][0] < 1 or ecc[i][0] > 8:
#         ecc_1to8.append(0)
#     else:
#         ecc_1to8.append(ecc[i][0])
# ecc_1to8 = np.reshape(np.array(ecc_1to8),(-1))
# np.savez('./output/MaskEccentricity_above1below8ecc_LH', list = ecc_1to8 > 0)


# Masking
ecc_thr[final_mask_L == 1] = np.reshape(ecc, (-1, 1)) * 10 + threshold
ecc_thr[final_mask_L != 1] = 0

view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../..',
                       'Retinotopy/data/raw/surfaces'
                       '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(ecc_thr[0:32492], (-1)), bg_map=background,
    cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False,
    threshold=threshold, vmax=130)
view.open_in_browser()