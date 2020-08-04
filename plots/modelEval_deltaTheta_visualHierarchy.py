import numpy as np
import matplotlib.pyplot as plt
import torch

from functions.def_ROIs_WangParcels import roi as roi2
from functions.def_ROIs_WangParcelsPlusFovea import roi
from functions.plusFovea import add_fovea


def smallest_angle(x, y):
    difference = []
    dif_1 = np.abs(y - x)
    dif_2 = np.abs(y - x + 2 * np.pi)
    dif_3 = np.abs(y - x - 2 * np.pi)
    for i in range(len(x)):
        difference.append(min(dif_1[i], dif_2[i], dif_3[i]))
    return np.array(difference) * 180 / np.pi


visual_areas = ['hV4', 'VO1', 'VO2', 'PHC1', 'PHC2', 'V3a', 'V3b', 'LO1',
                'LO2', 'TO1', 'TO2', 'IPS0', 'IPS1', 'IPS2', 'IPS3', 'IPS4',
                'IPS5', 'SPL1']

fig = plt.figure(figsize=(12.8, 9.6))
for k in range(len(visual_areas)):

    mean_delta = []
    mean_across = []

    for m in range(12):
        a = torch.load(
            '/home/uqfribe1/PycharmProjects/DEEP-fMRI/polarAngle'
            '/model4_nothresh_rotated_' + str(
                m + 1) +
            'layers_smoothL1lossR2_curvnmyelin_ROI1_k25_batchnorm_dropout010_output_epoch100.pt',
            map_location='cpu')

        # b=torch.load('/home/uqfribe1/PycharmProjects/DEEP-fMRI
        # /testing_shuffled-myelin.pt',map_location='cpu')

        theta_withinsubj = []
        # theta_withinsubj_shuffled=[]
        theta_acrosssubj = []
        theta_acrosssubj_pred = []
        theta_acrosssubj_emp = []

        R2_thr = []

        label_primary_visual_areas = ['ROI']
        final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
            label_primary_visual_areas)
        ROI1 = np.zeros((32492, 1))
        ROI1[final_mask_L == 1] = 1

        final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi2(
            [visual_areas[k]])
        primary_visual_areas = np.zeros((32492, 1))
        primary_visual_areas[final_mask_L == 1] = 1

        mask = ROI1 + primary_visual_areas
        mask = mask[ROI1 == 1]

        # Compute angle between predicted and empirical predictions across subj
        for j in range(len(a['Predicted_values'])):
            theta_across_temp = []
            theta_pred_across_temp = []
            theta_emp_across_temp = []
            R2_thr.append(np.reshape(np.array(a['R2'][j]), (-1)))

            for i in range(len(a['Predicted_values'])):
                # Compute angle between predicted and empirical predictions
                # within subj
                if i == j:
                    # Loading predicted values
                    pred = np.reshape(np.array(a['Predicted_values'][i]),
                                      (-1, 1))
                    # pred_shuffled = np.reshape(np.array(b[
                    # 'Predicted_values'][i]), (-1, 1))

                    measured = np.reshape(np.array(a['Measured_values'][j]),
                                          (-1, 1))

                    # Rescaling polar angles to match the right visual field
                    # (left hemisphere)
                    minus = pred > 180
                    sum = pred < 180
                    pred[minus] = pred[minus] - 180
                    pred[sum] = pred[sum] + 180
                    pred = np.array(pred) * (np.pi / 180)

                    # # Rescaling polar angles to match the right visual
                    # field (left hemisphere)
                    # minus = pred_shuffled > 180
                    # sum = pred_shuffled < 180
                    # pred_shuffled[minus] = pred_shuffled[minus] - 180
                    # pred_shuffled[sum] = pred_shuffled[sum] + 180
                    # pred_shuffled = np.array(pred_shuffled) * (np.pi / 180)

                    minus = measured > 180
                    sum = measured < 180
                    measured[minus] = measured[minus] - 180
                    measured[sum] = measured[sum] + 180
                    measured = np.array(measured) * (np.pi / 180)

                    # Computing delta theta, angle between vector defined
                    # predicted value and empirical value same subj
                    theta = smallest_angle(pred, measured)
                    theta_withinsubj.append(theta)

                    # theta_shuffled = smallest_angle(pred_shuffled, measured)
                    # theta_withinsubj_shuffled.append(theta_shuffled)

                if i != j:
                    # Compute angle between predicted and empirical
                    # predictions across subj
                    # Loading predicted values
                    pred = np.reshape(np.array(a['Predicted_values'][i]),
                                      (-1, 1))
                    pred2 = np.reshape(np.array(a['Predicted_values'][j]),
                                       (-1, 1))
                    measured = np.reshape(np.array(a['Measured_values'][j]),
                                          (-1, 1))
                    measured2 = np.reshape(np.array(a['Measured_values'][i]),
                                           (-1, 1))

                    # Rescaling polar angles to match the right visual field
                    # (left hemisphere)
                    minus = pred > 180
                    sum = pred < 180
                    pred[minus] = pred[minus] - 180
                    pred[sum] = pred[sum] + 180
                    pred = np.array(pred) * (np.pi / 180)

                    minus = pred2 > 180
                    sum = pred2 < 180
                    pred2[minus] = pred2[minus] - 180
                    pred2[sum] = pred2[sum] + 180
                    pred2 = np.array(pred2) * (np.pi / 180)

                    minus = measured > 180
                    sum = measured < 180
                    measured[minus] = measured[minus] - 180
                    measured[sum] = measured[sum] + 180
                    measured = np.array(measured) * (np.pi / 180)

                    minus = measured2 > 180
                    sum = measured2 < 180
                    measured2[minus] = measured2[minus] - 180
                    measured2[sum] = measured2[sum] + 180
                    measured2 = np.array(measured2) * (np.pi / 180)

                    # Computing delta theta, angle between vector defined
                    # predicted i and empirical j map
                    theta = smallest_angle(pred, measured)
                    theta_across_temp.append(theta)

                    # Computing delta theta, angle between vector defined
                    # measured i versus measured j
                    theta_emp = smallest_angle(measured, measured2)
                    theta_emp_across_temp.append(theta_emp)

                    # Computing delta theta, angle between vector defined
                    # pred i versus pred j
                    theta_pred = smallest_angle(pred, pred2)
                    theta_pred_across_temp.append(theta_pred)

            theta_acrosssubj.append(np.mean(theta_across_temp, axis=0))
            theta_acrosssubj_emp.append(np.mean(theta_emp_across_temp, axis=0))
            theta_acrosssubj_pred.append(
                np.mean(theta_pred_across_temp, axis=0))

        R2_thr = np.mean(R2_thr, axis=0)

        mean_theta_acrosssubj = np.mean(np.array(theta_acrosssubj), axis=0)
        mean_theta_withinsubj = np.mean(np.array(theta_withinsubj), axis=0)
        # mean_theta_withinsubj_shuffled=np.mean(np.array(
        # theta_withinsubj_shuffled),axis=0)
        mean_theta_acrosssubj_emp = np.mean(np.array(theta_acrosssubj_emp),
                                            axis=0)
        mean_theta_acrosssubj_pred = np.mean(np.array(theta_acrosssubj_pred),
                                             axis=0)

        mean_delta.append(np.mean(mean_theta_withinsubj[mask > 1]))
        mean_across.append(np.mean(mean_theta_acrosssubj_pred[mask > 1]))

    ax = fig.add_subplot(3, 6, k + 1)
    ax.set_title('' + visual_areas[k])
    ax.scatter(list(range(1, 13)), mean_delta)
    ax.scatter(list(range(1, 13)), mean_across)

plt.show()

# correct for V1-3 with fovea
label_primary_visual_areas = ['V1d', 'V1v', 'fovea_V1', 'V2d', 'V2v',
                              'fovea_V2', 'V3d', 'V3v', 'fovea_V3']
V1, V2, V3 = add_fovea(label_primary_visual_areas)
primary_visual_areas = [V1, V2, V3]
label = ['V1', 'V2', 'V3']

fig = plt.figure()
for k in range(len(primary_visual_areas)):

    mean_delta = []
    mean_across = []

    for m in range(12):
        a = torch.load(
            '/home/uqfribe1/PycharmProjects/DEEP-fMRI/polarAngle'
            '/model4_nothresh_rotated_' + str(
                m + 1) +
            'layers_smoothL1lossR2_curvnmyelin_ROI1_k25_batchnorm_dropout010_output_epoch100.pt',
            map_location='cpu')
        # b=torch.load('/home/uqfribe1/PycharmProjects/DEEP-fMRI
        # /testing_shuffled-myelin.pt',map_location='cpu')

        theta_withinsubj = []
        # theta_withinsubj_shuffled=[]
        theta_acrosssubj = []
        theta_acrosssubj_pred = []
        theta_acrosssubj_emp = []

        R2_thr = []

        label_primary_visual_areas = ['ROI']
        final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
            label_primary_visual_areas)
        ROI1 = np.zeros((32492, 1))
        ROI1[final_mask_L == 1] = 1
        mask = ROI1 + np.reshape(primary_visual_areas[k], (32492, 1))
        mask = mask[ROI1 == 1]

        # Compute angle between predicted and empirical predictions across subj
        for j in range(len(a['Predicted_values'])):
            theta_across_temp = []
            theta_pred_across_temp = []
            theta_emp_across_temp = []
            R2_thr.append(np.reshape(np.array(a['R2'][j]), (-1)))

            for i in range(len(a['Predicted_values'])):
                # Compute angle between predicted and empirical predictions
                # within subj
                if i == j:
                    # Loading predicted values
                    pred = np.reshape(np.array(a['Predicted_values'][i]),
                                      (-1, 1))
                    # pred_shuffled = np.reshape(np.array(b[
                    # 'Predicted_values'][i]), (-1, 1))

                    measured = np.reshape(np.array(a['Measured_values'][j]),
                                          (-1, 1))

                    # Rescaling polar angles to match the right visual field
                    # (left hemisphere)
                    minus = pred > 180
                    sum = pred < 180
                    pred[minus] = pred[minus] - 180
                    pred[sum] = pred[sum] + 180
                    pred = np.array(pred) * (np.pi / 180)

                    # # Rescaling polar angles to match the right visual
                    # field (left hemisphere)
                    # minus = pred_shuffled > 180
                    # sum = pred_shuffled < 180
                    # pred_shuffled[minus] = pred_shuffled[minus] - 180
                    # pred_shuffled[sum] = pred_shuffled[sum] + 180
                    # pred_shuffled = np.array(pred_shuffled) * (np.pi / 180)

                    minus = measured > 180
                    sum = measured < 180
                    measured[minus] = measured[minus] - 180
                    measured[sum] = measured[sum] + 180
                    measured = np.array(measured) * (np.pi / 180)

                    # Computing delta theta, angle between vector defined
                    # predicted value and empirical value same subj
                    theta = smallest_angle(pred, measured)
                    theta_withinsubj.append(theta)

                    # theta_shuffled = smallest_angle(pred_shuffled, measured)
                    # theta_withinsubj_shuffled.append(theta_shuffled)

                if i != j:
                    # Compute angle between predicted and empirical
                    # predictions across subj
                    # Loading predicted values
                    pred = np.reshape(np.array(a['Predicted_values'][i]),
                                      (-1, 1))
                    pred2 = np.reshape(np.array(a['Predicted_values'][j]),
                                       (-1, 1))
                    measured = np.reshape(np.array(a['Measured_values'][j]),
                                          (-1, 1))
                    measured2 = np.reshape(np.array(a['Measured_values'][i]),
                                           (-1, 1))

                    # Rescaling polar angles to match the right visual field
                    # (left hemisphere)
                    minus = pred > 180
                    sum = pred < 180
                    pred[minus] = pred[minus] - 180
                    pred[sum] = pred[sum] + 180
                    pred = np.array(pred) * (np.pi / 180)

                    minus = pred2 > 180
                    sum = pred2 < 180
                    pred2[minus] = pred2[minus] - 180
                    pred2[sum] = pred2[sum] + 180
                    pred2 = np.array(pred2) * (np.pi / 180)

                    minus = measured > 180
                    sum = measured < 180
                    measured[minus] = measured[minus] - 180
                    measured[sum] = measured[sum] + 180
                    measured = np.array(measured) * (np.pi / 180)

                    minus = measured2 > 180
                    sum = measured2 < 180
                    measured2[minus] = measured2[minus] - 180
                    measured2[sum] = measured2[sum] + 180
                    measured2 = np.array(measured2) * (np.pi / 180)

                    # Computing delta theta, angle between vector defined
                    # predicted i and empirical j map
                    theta = smallest_angle(pred, measured)
                    theta_across_temp.append(theta)

                    # Computing delta theta, angle between vector defined
                    # measured i versus measured j
                    theta_emp = smallest_angle(measured, measured2)
                    theta_emp_across_temp.append(theta_emp)

                    # Computing delta theta, angle between vector defined
                    # pred i versus pred j
                    theta_pred = smallest_angle(pred, pred2)
                    theta_pred_across_temp.append(theta_pred)

            theta_acrosssubj.append(np.mean(theta_across_temp, axis=0))
            theta_acrosssubj_emp.append(np.mean(theta_emp_across_temp, axis=0))
            theta_acrosssubj_pred.append(
                np.mean(theta_pred_across_temp, axis=0))

        R2_thr = np.mean(R2_thr, axis=0)

        mean_theta_acrosssubj = np.mean(np.array(theta_acrosssubj), axis=0)
        mean_theta_withinsubj = np.mean(np.array(theta_withinsubj), axis=0)
        # mean_theta_withinsubj_shuffled=np.mean(np.array(
        # theta_withinsubj_shuffled),axis=0)
        mean_theta_acrosssubj_emp = np.mean(np.array(theta_acrosssubj_emp),
                                            axis=0)
        mean_theta_acrosssubj_pred = np.mean(np.array(theta_acrosssubj_pred),
                                             axis=0)

        mean_delta.append(np.mean(mean_theta_withinsubj[mask > 1]))
        mean_across.append(np.mean(mean_theta_acrosssubj_pred[mask > 1]))

    ax = fig.add_subplot(3, 1, k + 1)
    ax.set_title('' + label[k])
    ax.scatter(list(range(1, 13)), mean_delta)
    ax.scatter(list(range(1, 13)), mean_across)

plt.show()

# ratio=mean_theta_acrosssubj_pred/(1+mean_theta_acrosssubj_emp)
#
# fig=sns.violinplot(data=[np.reshape(mean_theta_withinsubj[mask==1],(-1)),np.reshape(mean_theta_acrosssubj[mask==1],(-1)),np.reshape(mean_theta_acrosssubj_pred[mask==1],(-1)),np.reshape(mean_theta_acrosssubj_emp[mask==1],(-1))])
# fig.set_xticklabels(['Pred i vs GT i','Pred i vs GT j','Pred i vs Pred j','GT i vs GT j'])
# plt.ylim(0,180)
# plt.ylabel(r'Mean $\Delta$$\theta$ per node')
# plt.show()


# sns.violinplot(data=[np.reshape(mean_theta_withinsubj[mask==1],(-1)),np.reshape(mean_theta_withinsubj_shuffled[mask==1],(-1))])
# plt.ylim(0,180)
# plt.show()

# sns.violinplot(data=[np.reshape(mean_theta_acrosssubj_pred[R2_thr>0],(-1))])
# plt.ylim(0,180)
# plt.show()
