import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import pandas as pd

from functions.def_ROIs_WangParcels import roi as roi2
from functions.def_ROIs_WangParcelsPlusFovea import roi
from functions.plusFovea import add_fovea
from functions.least_difference_angles import smallest_angle

# Clusters
clusters = [['hV4'], ['VO1', 'VO2', 'PHC1', 'PHC2'], ['V3a', 'V3b'],
                ['LO1', 'LO2', 'TO1', 'TO2'],
                ['IPS0', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'SPL1']]
clusters_title = ['V4', 'Ventral', 'V3a/b', 'Lateral', 'Parietal']

models = ['pred', 'shuffled-myelincurv', 'constant', 'constantZero']
models_name = ['Default', 'Shuffled', 'Constant', 'Zeros']

color = [['darkblue', 'royalblue'], ['steelblue', 'lightskyblue'],
         ['green', 'lightgreen'], ['darkgoldenrod', 'goldenrod'],
         ['saddlebrown', 'chocolate'], ['hotpink', 'pink']]

sns.set_style("whitegrid")
fig = plt.figure(figsize=(25, 7))
for k in range(len(clusters)):
    mean_delta = []
    mean_across = []

    for m in range(len(models)):
        predictions = torch.load(
            '/home/uqfribe1/PycharmProjects/DEEP-fMRI/testset_results'
            '/testset-' +
            models[m] + '_Model3_PA_LH.pt', map_location='cpu')

        theta_withinsubj = []
        theta_acrosssubj_pred = []

        early_visual_cortex = ['ROI']
        final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
            early_visual_cortex)
        ROI1 = np.zeros((32492, 1))
        ROI1[final_mask_L == 1] = 1

        final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi2(
            clusters[k])
        cluster = np.zeros((32492, 1))
        cluster[final_mask_L == 1] = 1

        mask = ROI1 + cluster
        mask = mask[ROI1 == 1]

        # Compute difference between predicted and empirical maps
        # across subjs
        for j in range(len(predictions['Predicted_values'])):
            theta_across_temp = []
            theta_pred_across_temp = []
            theta_emp_across_temp = []

            for i in range(len(predictions['Predicted_values'])):
                # Compute angle between predicted and empirical
                # predictions within subj
                if i == j:
                    # Loading predicted values
                    pred = np.reshape(np.array(predictions['Predicted_values'][i]),
                                      (-1, 1))
                    measured = np.reshape(np.array(predictions['Measured_values'][j]),
                                          (-1, 1))

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
                    # defined pred and empirical maps of the
                    # same subj
                    theta = smallest_angle(pred, measured)
                    theta_withinsubj.append(theta)

                if i != j:
                    # Compute angle difference between predicted
                    # maps

                    # Loading predicted values
                    pred = np.reshape(np.array(predictions['Predicted_values'][i]),
                                      (-1, 1))
                    pred2 = np.reshape(np.array(predictions['Predicted_values'][j]),
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
                    # defined pred i versus pred j
                    theta_pred = smallest_angle(pred, pred2)
                    theta_pred_across_temp.append(theta_pred)

            theta_acrosssubj_pred.append(
                np.mean(theta_pred_across_temp, axis=0))

        mean_theta_withinsubj = np.mean(np.array(theta_withinsubj), axis=0)
        mean_theta_acrosssubj_pred = np.mean(np.array(theta_acrosssubj_pred),
                                             axis=0)

        mean_delta.append(mean_theta_withinsubj[mask > 1])
        mean_across.append(mean_theta_acrosssubj_pred[mask > 1])

    mean_delta = np.reshape(np.array(mean_delta), (4, -1))
    mean_across = np.reshape(np.array(mean_across), (4, -1))

    ax = fig.add_subplot(1, 6, k + 2)
    data = np.concatenate([[mean_delta[0],
                            len(mean_across[0]) * [models_name[0]],
                            len(mean_across[0]) * ['Error'],
                            len(mean_across[0]) * ['Cluster' + str(k + 1)]],
                           [mean_delta[1],
                            len(mean_across[1]) * [models_name[1]],
                            len(mean_across[1]) * ['Error'],
                            len(mean_across[1]) * ['Cluster' + str(k + 1)]],
                           [mean_delta[2],
                            len(mean_across[2]) * [models_name[2]],
                            len(mean_across[2]) * ['Error'],
                            len(mean_across[2]) * ['Cluster' + str(k + 1)]],
                           [mean_delta[3],
                            len(mean_across[3]) * [models_name[3]],
                            len(mean_across[3]) * ['Error'],
                            len(mean_across[3]) * ['Cluster' + str(k + 1)]],
                           [mean_across[0],
                            len(mean_across[0]) * [models_name[0]],
                            len(mean_across[0]) * ['Individual variability'],
                            len(mean_across[0]) * ['Cluster' + str(k + 1)]],
                           [mean_across[1],
                            len(mean_across[1]) * [models_name[1]],
                            len(mean_across[1]) * ['Individual variability'],
                            len(mean_across[1]) * ['Cluster' + str(k + 1)]],
                           [mean_across[2],
                            len(mean_across[2]) * [models_name[2]],
                            len(mean_across[2]) * ['Individual variability'],
                            len(mean_across[2]) * ['Cluster' + str(k + 1)]],
                           [mean_across[3],
                            len(mean_across[3]) * [models_name[3]],
                            len(mean_across[3]) * ['Individual variability'],
                            len(mean_across[3]) * ['Cluster' + str(k + 1)]]],
                          axis=1)
    df = pd.DataFrame(
        columns=['$\Delta$$\t\Theta$', 'Input', 'label', 'cluster'],
        data=data.T)
    df['$\Delta$$\t\Theta$'] = df['$\Delta$$\t\Theta$'].astype(float)
    palette = ['dimgray', 'lightgray']
    ax = sns.pointplot(y='$\Delta$$\t\Theta$', x='Input', order=models_name,
                       hue='label', data=df, palette=color[k + 1],
                       join=False, dodge=True, ci=95)
    ax.set_title(clusters_title[k])
    legend = plt.legend()
    legend.remove()

    plt.ylim([0, 80])
    ax.set_xlabel('')
    # plt.savefig('PAdif_cluster'+str(k+1)+'.svg')



# Primary visual cortex
early_visual_cortex = ['V1d', 'V1v', 'fovea_V1', 'V2d', 'V2v',
                              'fovea_V2', 'V3d', 'V3v', 'fovea_V3']
V1, V2, V3 = add_fovea(early_visual_cortex)
cluster = np.sum(
    [np.reshape(V1, (-1, 1)), np.reshape(V2, (-1, 1)),
     np.reshape(V3, (-1, 1))], axis=0)
label = ['Early visual cortex']

mean_delta_2 = []
mean_across_2 = []

theta_within_subj = []
for m in range(len(models)):
    predictions = torch.load(
        '/home/uqfribe1/PycharmProjects/DEEP-fMRI/testset_results/testset-' +
        models[m] + '_Model3_PA_LH.pt', map_location='cpu')

    theta_withinsubj = []
    theta_acrosssubj = []
    theta_acrosssubj_pred = []
    theta_acrosssubj_emp = []

    visual_hierarchy = ['ROI']
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
        visual_hierarchy)
    ROI1 = np.zeros((32492, 1))
    ROI1[final_mask_L == 1] = 1
    mask = ROI1 + np.reshape(cluster, (32492, 1))
    mask = mask[ROI1 == 1]

    # Compute difference between predicted and empirical maps
    # across subjs
    for j in range(len(predictions['Predicted_values'])):
        theta_across_temp = []
        theta_pred_across_temp = []
        theta_emp_across_temp = []

        for i in range(len(predictions['Predicted_values'])):
            # Compute angle between predicted and empirical
            # predictions within subj
            if i == j:
                # Loading predicted values
                pred = np.reshape(np.array(predictions['Predicted_values'][i]),
                                  (-1, 1))
                measured = np.reshape(
                    np.array(predictions['Measured_values'][j]),
                    (-1, 1))

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
                # defined pred and empirical maps of the
                # same subj
                theta = smallest_angle(pred, measured)
                theta_withinsubj.append(theta)

            if i != j:
                # Compute angle difference between predicted
                # maps

                # Loading predicted values
                pred = np.reshape(np.array(predictions['Predicted_values'][i]),
                                  (-1, 1))
                pred2 = np.reshape(
                    np.array(predictions['Predicted_values'][j]),
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
                # defined pred i versus pred j
                theta_pred = smallest_angle(pred, pred2)
                theta_pred_across_temp.append(theta_pred)

        theta_acrosssubj_pred.append(
            np.mean(theta_pred_across_temp, axis=0))

    mean_theta_withinsubj = np.mean(np.array(theta_withinsubj), axis=0)
    mean_theta_acrosssubj_pred = np.mean(np.array(theta_acrosssubj_pred),
                                         axis=0)

    mean_delta_2.append(mean_theta_withinsubj[mask > 1])
    mean_across_2.append(mean_theta_acrosssubj_pred[mask > 1])

mean_delta_2 = np.reshape(np.array(mean_delta_2), (4, -1))
mean_across_2 = np.reshape(np.array(mean_across_2), (4, -1))

ax = fig.add_subplot(1, 6, 1)
data = np.concatenate([[mean_delta_2[0],
                        len(mean_across_2[0]) * [models_name[0]],
                        len(mean_across_2[0]) * ['Error']],
                       [mean_delta_2[1],
                        len(mean_across_2[1]) * [models_name[1]],
                        len(mean_across_2[1]) * ['Error']],
                       [mean_delta_2[2],
                        len(mean_across_2[2]) * [models_name[2]],
                        len(mean_across_2[2]) * ['Error']],
                       [mean_delta_2[3],
                        len(mean_across_2[3]) * [models_name[3]],
                        len(mean_across_2[3]) * ['Error']],
                       [mean_across_2[0],
                        len(mean_across_2[0]) * [models_name[0]],
                        len(mean_across_2[0]) * ['Individual variability']],
                       [mean_across_2[1],
                        len(mean_across_2[1]) * [models_name[1]],
                        len(mean_across_2[1]) * ['Individual variability']],
                       [mean_across_2[2],
                        len(mean_across_2[2]) * [models_name[2]],
                        len(mean_across_2[2]) * ['Individual variability']],
                       [mean_across_2[3],
                        len(mean_across_2[3]) * [models_name[3]],
                        len(mean_across_2[3]) * ['Individual variability']]],
                      axis=1)
df = pd.DataFrame(columns=['$\Delta$$\t\Theta$', 'Input', 'label'],
                  data=data.T)
df['$\Delta$$\t\Theta$'] = df['$\Delta$$\t\Theta$'].astype(float)
palette = ['dimgray', 'lightgray']


ax = sns.pointplot(y='$\Delta$$\t\Theta$', x='Input', order=models_name,
                   hue='label', data=df, palette=color[0], join=False,
                   dodge=True, ci=95)
ax.set_title('Early visual cortex ')
legend = plt.legend()
ax.set_xlabel('')
plt.ylim([0, 80])
# plt.savefig('ModelEval_AllClusters', format="pdf")
plt.show()
