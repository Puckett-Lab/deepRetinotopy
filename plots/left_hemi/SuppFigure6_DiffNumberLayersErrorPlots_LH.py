import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import pandas as pd

from functions.def_ROIs_WangParcels import roi as roi2
from functions.def_ROIs_WangParcelsPlusFovea import roi
from functions.plusFovea import add_fovea
from functions.least_difference_angles import smallest_angle

clusters = [['hV4'], ['VO1', 'VO2', 'PHC1', 'PHC2'], ['V3a', 'V3b'],
                ['LO1', 'LO2', 'TO1', 'TO2'],
                ['IPS0', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'SPL1']]
number_layers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                 '14', '16', '18', '20']

# ROI settings
visual_hierarchy = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    visual_hierarchy)
ROI1 = np.zeros((32492, 1))
ROI1[final_mask_L == 1] = 1


sns.set_style("whitegrid")
# Higher order visual areas
for k in range(len(clusters)):
    mean_delta = np.zeros((5, 16))
    mean_across = np.zeros((5, 16))

    l = 0
    while l < 5:
        mean_delta_temp = []
        mean_across_temp = []

        for m in range(16):
            if m < 12:
                prediction = torch.load(
                    '/home/uqfribe1/Desktop/Wiener/Project1/July/output'
                    '/model4_nothresh_rotated_' + str(
                        m + 1) +
                    'layers_smoothL1lossR2_curvnmyelin_ROI1_k25_batchnorm_dropout010_' + str(
                        l + 1) + '_output_epoch200.pt', map_location='cpu')

                theta_withinsubj = []
                theta_acrosssubj_pred = []

                # Selecting cluster
                final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi2(
                    clusters[k])
                cluster = np.zeros((32492, 1))
                cluster[final_mask_L == 1] = 1

                mask = ROI1 + cluster
                mask = mask[ROI1 == 1]

                # Compute difference between predicted and empirical maps
                # across subjs
                for j in range(len(prediction['Predicted_values'])):
                    theta_pred_across_temp = []

                    for i in range(len(prediction['Predicted_values'])):
                        # Compute the difference between predicted and
                        # empirical angles
                        # within subj - error
                        if i == j:
                            # Loading predicted values
                            pred = np.reshape(
                                np.array(prediction['Predicted_values'][i]), (-1, 1))
                            measured = np.reshape(np.array(prediction['Measured_values'
                                                             ''][j]), (-1, 1))

                            # Rescaling polar angles to match the correct
                            # visual field (left hemisphere)
                            minus = pred > 180
                            sum = pred < 180
                            pred[minus] = pred[minus] - 180
                            pred[sum] = pred[sum] + 180
                            pred = np.array(pred) * (np.pi / 180)

                            minus = measured > 180
                            sum = measured < 180
                            measured[minus] = measured[minus] - 180
                            measured[sum] = measured[sum] + 180
                            measured = np.array(measured) * (np.pi / 180)

                            # Computing delta theta, difference between
                            # predicted and
                            # empirical angles
                            theta = smallest_angle(pred, measured)
                            theta_withinsubj.append(theta)

                        if i != j:
                            # Compute the difference between predicted maps
                            # across subj - individual variability

                            # Loading predicted values
                            pred = np.reshape(
                                np.array(prediction['Predicted_values'][i]), (-1, 1))
                            pred2 = np.reshape(
                                np.array(prediction['Predicted_values'][j]), (-1, 1))

                            # Rescaling polar angles to match the correct
                            # visual field (left hemisphere)
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

                            # Computing delta theta, difference between
                            # predicted maps
                            theta_pred = smallest_angle(pred, pred2)
                            theta_pred_across_temp.append(theta_pred)

                    theta_acrosssubj_pred.append(
                        np.mean(theta_pred_across_temp, axis=0))

                mean_theta_withinsubj = np.mean(np.array(theta_withinsubj),
                                                axis=0)
                mean_theta_acrosssubj_pred = np.mean(
                    np.array(theta_acrosssubj_pred), axis=0)

                mean_delta_temp.append(
                    np.mean(mean_theta_withinsubj[mask > 1]))
                mean_across_temp.append(
                    np.mean(mean_theta_acrosssubj_pred[mask > 1]))

            else:
                prediction = torch.load(
                    '/home/uqfribe1/Desktop/Wiener/Project1/July/output'
                    '/model4_nothresh_rotated_' + str(
                        12 + (
                                m - 12) * 2) +
                    'layers_smoothL1lossR2_curvnmyelin_ROI1_k25_batchnorm_dropout010_' + str(
                        l + 1) + '_output_epoch200.pt', map_location='cpu')

                theta_withinsubj = []
                theta_acrosssubj_pred = []

                # Selecting cluster
                final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi2(
                    clusters[k])
                cluster = np.zeros((32492, 1))
                cluster[final_mask_L == 1] = 1

                mask = ROI1 + cluster
                mask = mask[ROI1 == 1]

                for j in range(len(prediction['Predicted_values'])):
                    theta_pred_across_temp = []

                    for i in range(len(prediction['Predicted_values'])):
                        if i == j:
                            # Loading predicted values
                            pred = np.reshape(
                                np.array(prediction['Predicted_values'][i]),
                                (-1, 1))
                            measured = np.reshape(
                                np.array(prediction['Measured_values'
                                                    ''][j]), (-1, 1))

                            # Rescaling polar angles to match the correct
                            # visual field (left hemisphere)
                            minus = pred > 180
                            sum = pred < 180
                            pred[minus] = pred[minus] - 180
                            pred[sum] = pred[sum] + 180
                            pred = np.array(pred) * (np.pi / 180)

                            minus = measured > 180
                            sum = measured < 180
                            measured[minus] = measured[minus] - 180
                            measured[sum] = measured[sum] + 180
                            measured = np.array(measured) * (np.pi / 180)

                            # Computing delta theta, difference between
                            # predicted and empirical angles
                            theta = smallest_angle(pred, measured)
                            theta_withinsubj.append(theta)

                        if i != j:
                            # Loading predicted values
                            pred = np.reshape(
                                np.array(prediction['Predicted_values'][i]),
                                (-1, 1))
                            pred2 = np.reshape(
                                np.array(prediction['Predicted_values'][j]),
                                (-1, 1))

                            # Rescaling polar angles to match the correct
                            # visual field (left hemisphere)
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

                            # Computing delta theta, difference between
                            # predicted maps
                            theta_pred = smallest_angle(pred, pred2)
                            theta_pred_across_temp.append(theta_pred)

                    theta_acrosssubj_pred.append(
                        np.mean(theta_pred_across_temp, axis=0))


                mean_theta_withinsubj = np.mean(np.array(theta_withinsubj),
                                                axis=0)
                mean_theta_acrosssubj_pred = np.mean(
                    np.array(theta_acrosssubj_pred), axis=0)

                mean_delta_temp.append(
                    np.mean(mean_theta_withinsubj[mask > 1]))
                mean_across_temp.append(
                    np.mean(mean_theta_acrosssubj_pred[mask > 1]))

        mean_delta[l] = np.array(mean_delta_temp)
        mean_across[l] = np.array(mean_across_temp)
        l += 1

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    data = np.concatenate([[mean_delta[0], number_layers,
                            len(number_layers) * ['Error']],
                           [mean_delta[1], number_layers,
                            len(number_layers) * ['Error']],
                           [mean_delta[2], number_layers,
                            len(number_layers) * ['Error']],
                           [mean_delta[3], number_layers,
                            len(number_layers) * ['Error']],
                           [mean_delta[4], number_layers,
                            len(number_layers) * ['Error']],
                           [mean_across[0], number_layers,
                            len(number_layers) * ['Individual variability']],
                           [mean_across[1], number_layers,
                            len(number_layers) * ['Individual variability']],
                           [mean_across[2], number_layers,
                            len(number_layers) * ['Individual variability']],
                           [mean_across[3], number_layers,
                            len(number_layers) * ['Individual variability']],
                           [mean_across[4], number_layers,
                            len(number_layers) * ['Individual variability']]
                           ], axis=1)
    df = pd.DataFrame(columns=['$\Delta$$\t\Theta$', 'Layers', 'label'],
                      data=data.T)
    df['$\Delta$$\t\Theta$'] = df['$\Delta$$\t\Theta$'].astype(float)
    palette = ['dimgray', 'lightgray']
    ax = sns.boxplot(y='$\Delta$$\t\Theta$', x='Layers', order=number_layers,
                     hue='label', data=df, palette=palette)
    title = ['V4', 'Ventral', 'V3a/b', 'Lateral', 'Parietal']
    ax.set_title(title[k])
    legend = plt.legend()
    legend.remove()
    plt.ylim([0, 70])
    # plt.savefig('./../output/PAdif_cluster' + str(k + 1) + '_LH.svg')
    plt.show()



# Early visual cortex
visual_hierarchy = ['V1d', 'V1v', 'fovea_V1', 'V2d', 'V2v',
                              'fovea_V2', 'V3d', 'V3v', 'fovea_V3']
V1, V2, V3 = add_fovea(visual_hierarchy)
cluster = np.sum(
    [np.reshape(V1, (-1, 1)), np.reshape(V2, (-1, 1)),
     np.reshape(V3, (-1, 1))], axis=0)
label = ['Early visual cortex']

visual_hierarchy = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
                visual_hierarchy)
ROI1 = np.zeros((32492, 1))
ROI1[final_mask_L == 1] = 1
mask = ROI1 + np.reshape(cluster, (32492, 1))
mask = mask[ROI1 == 1]

mean_delta_2 = np.zeros((5, 16))
mean_across_2 = np.zeros((5, 16))


fig = plt.figure()
l = 0
while l < 5:
    mean_delta_temp = []
    mean_across_temp = []

    for m in range(16):
        if m < 12:
            prediction = torch.load(
                '/home/uqfribe1/Desktop/Wiener/Project1/July/output'
                '/model4_nothresh_rotated_' + str(
                    m + 1) +
                'layers_smoothL1lossR2_curvnmyelin_ROI1_k25_batchnorm_dropout010_' + str(
                    l + 1) + '_output_epoch200.pt', map_location='cpu')

            theta_withinsubj = []
            theta_acrosssubj_pred = []

            for j in range(len(prediction['Predicted_values'])):
                theta_pred_across_temp = []

                for i in range(len(prediction['Predicted_values'])):
                    if i == j:
                        # Loading predicted values
                        pred = np.reshape(
                            np.array(prediction['Predicted_values'][i]),
                            (-1, 1))
                        measured = np.reshape(
                            np.array(prediction['Measured_values'
                                                ''][j]), (-1, 1))

                        # Rescaling polar angles to match the correct
                        # visual field (left hemisphere)
                        minus = pred > 180
                        sum = pred < 180
                        pred[minus] = pred[minus] - 180
                        pred[sum] = pred[sum] + 180
                        pred = np.array(pred) * (np.pi / 180)

                        minus = measured > 180
                        sum = measured < 180
                        measured[minus] = measured[minus] - 180
                        measured[sum] = measured[sum] + 180
                        measured = np.array(measured) * (np.pi / 180)

                        # delta theta
                        theta = smallest_angle(pred, measured)
                        theta_withinsubj.append(theta)

                    if i != j:
                        # Loading predicted values
                        pred = np.reshape(
                            np.array(prediction['Predicted_values'][i]),
                            (-1, 1))
                        pred2 = np.reshape(
                            np.array(prediction['Predicted_values'][j]),
                            (-1, 1))

                        # Rescaling polar angles to match the correct
                        # visual field (left hemisphere)
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

                        # delta theta
                        theta_pred = smallest_angle(pred, pred2)
                        theta_pred_across_temp.append(theta_pred)

                theta_acrosssubj_pred.append(
                    np.mean(theta_pred_across_temp, axis=0))

            mean_theta_withinsubj = np.mean(np.array(theta_withinsubj),
                                            axis=0)
            mean_theta_acrosssubj_pred = np.mean(
                np.array(theta_acrosssubj_pred), axis=0)

            mean_delta_temp.append(
                np.mean(mean_theta_withinsubj[mask > 1]))
            mean_across_temp.append(
                np.mean(mean_theta_acrosssubj_pred[mask > 1]))


        else:

            prediction = torch.load(
                '/home/uqfribe1/Desktop/Wiener/Project1/July/output'
                '/model4_nothresh_rotated_' + str(
                    12 + (
                            m - 12) * 2) +
                'layers_smoothL1lossR2_curvnmyelin_ROI1_k25_batchnorm_dropout010_' + str(
                    l + 1) + '_output_epoch200.pt', map_location='cpu')

            theta_withinsubj = []
            theta_acrosssubj_pred = []

            for j in range(len(prediction['Predicted_values'])):
                theta_pred_across_temp = []

                for i in range(len(prediction['Predicted_values'])):
                    if i == j:
                        # Loading predicted values
                        pred = np.reshape(
                            np.array(prediction['Predicted_values'][i]),
                            (-1, 1))
                        measured = np.reshape(
                            np.array(prediction['Measured_values'
                                                ''][j]), (-1, 1))

                        # Rescaling polar angles to match the correct
                        # visual field (left hemisphere)
                        minus = pred > 180
                        sum = pred < 180
                        pred[minus] = pred[minus] - 180
                        pred[sum] = pred[sum] + 180
                        pred = np.array(pred) * (np.pi / 180)

                        minus = measured > 180
                        sum = measured < 180
                        measured[minus] = measured[minus] - 180
                        measured[sum] = measured[sum] + 180
                        measured = np.array(measured) * (np.pi / 180)

                        # delta theta
                        theta = smallest_angle(pred, measured)
                        theta_withinsubj.append(theta)

                    if i != j:
                        # Loading predicted values
                        pred = np.reshape(
                            np.array(prediction['Predicted_values'][i]),
                            (-1, 1))
                        pred2 = np.reshape(
                            np.array(prediction['Predicted_values'][j]),
                            (-1, 1))

                        # Rescaling polar angles to match the correct
                        # visual field (left hemisphere)
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

                        # delta theta
                        theta_pred = smallest_angle(pred, pred2)
                        theta_pred_across_temp.append(theta_pred)

                theta_acrosssubj_pred.append(
                    np.mean(theta_pred_across_temp, axis=0))

            mean_theta_withinsubj = np.mean(np.array(theta_withinsubj),
                                            axis=0)
            mean_theta_acrosssubj_pred = np.mean(
                np.array(theta_acrosssubj_pred), axis=0)

            mean_delta_temp.append(
                np.mean(mean_theta_withinsubj[mask > 1]))
            mean_across_temp.append(
                np.mean(mean_theta_acrosssubj_pred[mask > 1]))

    mean_delta_2[l] = np.array(mean_delta_temp)
    mean_across_2[l] = np.array(mean_across_temp)
    l += 1

ax = fig.add_subplot(1, 1, 1)
data = np.concatenate(
    [[mean_delta_2[0], number_layers, len(number_layers) * ['Error']],
     [mean_delta_2[1], number_layers, len(number_layers) * ['Error']],
     [mean_delta_2[2], number_layers, len(number_layers) * ['Error']],
     [mean_delta_2[3], number_layers, len(number_layers) * ['Error']],
     [mean_delta_2[4], number_layers, len(number_layers) * ['Error']],
     [mean_across_2[0], number_layers,
      len(number_layers) * ['Individual variability']],
     [mean_across_2[1], number_layers,
      len(number_layers) * ['Individual variability']],
     [mean_across_2[2], number_layers,
      len(number_layers) * ['Individual variability']],
     [mean_across_2[3], number_layers,
      len(number_layers) * ['Individual variability']],
     [mean_across_2[4], number_layers,
      len(number_layers) * ['Individual variability']]], axis=1)
df = pd.DataFrame(columns=['$\Delta$$\t\Theta$', 'Layers', 'label'],
                  data=data.T)
df['$\Delta$$\t\Theta$'] = df['$\Delta$$\t\Theta$'].astype(float)
palette = ['dimgray', 'lightgray']
ax = sns.boxplot(y='$\Delta$$\t\Theta$', x='Layers', order=number_layers,
                 hue='label', data=df, palette=palette)
ax.set_title('' + label[0])
plt.legend(loc='upper right')
plt.ylim([0, 70])
# plt.savefig('./../output/PAdif_EarlyVisualArea_LH.svg')
plt.show()
