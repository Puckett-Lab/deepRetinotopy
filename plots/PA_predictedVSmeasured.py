import numpy as np
import scipy.io
import os.path as osp
import torch

from functions.def_ROIs_WangParcelsPlusFovea import roi

path = '/home/uqfribe1/PycharmProjects/DEEP-fMRI/data/raw/converted'
curv = scipy.io.loadmat(osp.join(path, 'cifti_curv_all.mat'))['cifti_curv']
background = np.reshape(curv['x100610_curvature'][0][0][0:32492], (-1))

threshold = 1

label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)
pred = np.zeros((32492, 1))
measured = np.zeros((32492, 1))
R2_thr = np.zeros((32492, 1))

a = torch.load(
    '/home/uqfribe1/PycharmProjects/DEEP-fMRI/polarAngle'
    '/model4_nothresh_rotated_9layers_smoothL1lossR2_curvnmyelin_ROI1_k25_batchnorm_dropout010_output_epoch200.pt',
    map_location='cpu')
pred[final_mask_L == 1] = np.reshape(np.array(a['Predicted_values'][7]),
                                     (-1, 1))

R2_thr[final_mask_L == 1] = np.reshape(np.array(a['R2'][7]), (-1, 1))

R2 = R2_thr[final_mask_L == 1]

measured[final_mask_L == 1] = np.reshape(np.array(a['Measured_values'][7]),
                                         (-1, 1))

pred = np.array(pred)
minus = pred > 180
sum = pred < 180
pred[minus] = pred[minus] - 180
pred[sum] = pred[sum] + 180
pred = pred

measured = np.array(measured)
minus = measured > 180
sum = measured < 180
measured[minus] = measured[minus] - 180
measured[sum] = measured[sum] + 180
measured = np.array(measured)

from functions.def_ROIs_WangParcels import roi

V1 = ['hV4', 'VO1', 'VO2', 'PHC1', 'PHC2', 'V3a', 'V3b', 'LO1', 'LO2', 'TO1',
      'TO2', 'IPS0', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'SPL1']
final_mask_L_v1, final_mask_R_v1, index_L_mask_v1, index_R_mask_v1 = roi(V1)

final_mask_L_v1 = final_mask_L_v1[final_mask_L == 1][
    np.reshape(np.cos(measured[final_mask_L == 1] / 180 * np.pi) > 0, (-1))]

color = ['red' if l == 1 else 'blue' for l in final_mask_L_v1]

color = np.array(color)


print(np.corrcoef(np.reshape(measured[final_mask_L == 1], (-1)),
                  np.reshape(pred[final_mask_L == 1], (-1))))
