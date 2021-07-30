import os.path as osp
import scipy.io
import sys
import torch_geometric.transforms as T
import numpy as np

sys.path.append('../..')

from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi
from nilearn import plotting
from Retinotopy.dataset.HCP_3sets_ROI import Retinotopy
from torch_geometric.data import DataLoader

path = './../../../Retinotopy/data/raw/converted'
curv = scipy.io.loadmat(osp.join(path, 'cifti_curv_all.mat'))['cifti_curv']
background = np.reshape(curv['x100610_curvature'][0][0][0:32492], (-1))

# Background settings
threshold = 1
nocurv = np.isnan(background)
background[nocurv == 1] = 0

# ROI settings
label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)
R2_thr = np.zeros((32492, 1))

# Loading data - left hemisphere
path = osp.join(osp.dirname(osp.realpath(__file__)), '../../..', 'Retinotopy/data')
pre_transform = T.Compose([T.FaceToEdge()])
test_dataset = Retinotopy(path, 'Test', transform=T.Cartesian(),
                           pre_transform=pre_transform, n_examples=181,
                           prediction='polarAngle', myelination=True,
                           hemisphere='Left')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Average explained variance map
R2 = []
for data in test_loader:
    R2.append(np.array(data.R2))
R2 = np.mean(R2, 0)

# Masking
R2_thr[final_mask_L == 1] = np.reshape(R2, (-1, 1)) + threshold
R2_thr[final_mask_L != 1] = 0

view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
                       'Retinotopy/data/raw/surfaces'
                       '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
    bg_map=background, surf_map=np.reshape(R2_thr[0:32492], (-1)),
    threshold=threshold, cmap='hot', black_bg=False, symmetric_cmap=False,
    vmax=60 + threshold)
view.open_in_browser()
