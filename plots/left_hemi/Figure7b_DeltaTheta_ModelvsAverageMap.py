import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import sys

sys.path.append('../..')


def error_plots(hemisphere, retinotopic_feature):
    """Function to generate error plot.

    Args:
        hemisphere (str): 'LH' or 'RH'.
        retinotopic_feature (str): 'PA' or 'ecc' or 'pRFcenter'.


    Returns:
        Point plot of prediction error from our models' predictions and an
        average-based prediction.
    """

    error_DorsalEarlyVisualCortex_model = np.reshape(np.array(
        np.load('./../../stats/output/ErrorPerParticipant_' + str(
            retinotopic_feature) + '_' + str(
            hemisphere) + '_dorsalV1-3_deepRetinotopy_1-8.npz')['list']),
        (10, -1))
    error_EarlyVisualCortex_model = np.reshape(np.array(
        np.load('./../../stats/output/ErrorPerParticipant_' + str(
            retinotopic_feature) + '_' + str(
            hemisphere) + '_EarlyVisualCortex_deepRetinotopy_1-8.npz')['list']),
        (10, -1))
    error_higherOrder_model = np.reshape(np.array(
        np.load('./../../stats/output/ErrorPerParticipant_' + str(
            retinotopic_feature) + '_' + str(
            hemisphere) + '_WangParcels_deepRetinotopy_1-8.npz')['list']), (10, -1))

    error_DorsalEarlyVisualCortex_average = np.reshape(np.array(
        np.load('./../../stats/output/ErrorPerParticipant_' + str(
            retinotopic_feature) + '_' + str(
            hemisphere) + '_dorsalV1-3_average_1-8.npz')['list']),
        (10, -1))
    error_EarlyVisualCortex_average = np.reshape(np.array(
        np.load('./../../stats/output/ErrorPerParticipant_' + str(
            retinotopic_feature) + '_' + str(
            hemisphere) + '_EarlyVisualCortex_average_1-8.npz')[
            'list']), (10, -1))
    error_higherOrder_average = np.reshape(np.array(
        np.load('./../../stats/output/ErrorPerParticipant_' + str(
            retinotopic_feature) + '_' + str(
            hemisphere) + '_WangParcels_average_1-8.npz')['list']),
        (10, -1))

    # Reformatting data from dorsal early visual cortex
    data_earlyVisualCortexDorsal = np.concatenate([
        [np.mean(error_DorsalEarlyVisualCortex_average, axis=1),
         np.shape(error_DorsalEarlyVisualCortex_average)[0] * [
             'Average Map'],
         np.shape(error_DorsalEarlyVisualCortex_average)[0] * [
             'Early Visual Cortex']
         ],
        [np.mean(error_DorsalEarlyVisualCortex_model, axis=1),
         np.shape(error_DorsalEarlyVisualCortex_model)[0] * ['Model'],
         np.shape(error_DorsalEarlyVisualCortex_model)[0] * [
             'Early Visual Cortex']]],
        axis=1)

    df_0 = pd.DataFrame(
        columns=['$\Delta$$\t\Theta$', 'Prediction', 'Area'],
        data=data_earlyVisualCortexDorsal.T)
    df_0['$\Delta$$\t\Theta$'] = df_0['$\Delta$$\t\Theta$'].astype(float)

    print(
        scipy.stats.ttest_rel(
            np.mean(error_DorsalEarlyVisualCortex_model, axis=1),
            np.mean(error_DorsalEarlyVisualCortex_average,
                    axis=1)))  # Repeated measures t-test

    # Reformatting data from early visual cortex
    data_earlyVisualCortex = np.concatenate([
        [np.mean(error_EarlyVisualCortex_average, axis=1),
         np.shape(error_EarlyVisualCortex_average)[0] * [
             'Average Map'],
         np.shape(error_EarlyVisualCortex_average)[0] * [
             'Early Visual Cortex']
         ],
        [np.mean(error_EarlyVisualCortex_model, axis=1),
         np.shape(error_EarlyVisualCortex_model)[0] * ['Model'],
         np.shape(error_EarlyVisualCortex_model)[0] * [
             'Early Visual Cortex']]],
        axis=1)

    df_1 = pd.DataFrame(
        columns=['$\Delta$$\t\Theta$', 'Prediction', 'Area'],
        data=data_earlyVisualCortex.T)
    df_1['$\Delta$$\t\Theta$'] = df_1['$\Delta$$\t\Theta$'].astype(float)

    print(scipy.stats.ttest_rel(np.mean(error_EarlyVisualCortex_model, axis=1),
                                np.mean(error_EarlyVisualCortex_average,
                                        axis=1)))  # Repeated measures t-test

    # Reformatting data from higher order visual areas
    data_HigherOrder = np.concatenate([
        [np.mean(error_higherOrder_average, axis=1),
         np.shape(error_higherOrder_average)[0] * [
             'Average Map'],
         np.shape(error_higherOrder_average)[0] * [
             'Higher Order Visual Areas']],
        [np.mean(error_higherOrder_model, axis=1),
         np.shape(error_higherOrder_model)[0] * ['Model'],
         np.shape(error_higherOrder_model)[0] * [
             'Higher Order Visual Areas']]
    ],
        axis=1)

    df_2 = pd.DataFrame(
        columns=['$\Delta$$\t\Theta$', 'Prediction', 'Area'],
        data=data_HigherOrder.T)
    df_2['$\Delta$$\t\Theta$'] = df_2['$\Delta$$\t\Theta$'].astype(float)

    print(scipy.stats.ttest_rel(np.mean(error_higherOrder_model, axis=1),
                                np.mean(error_higherOrder_average,
                                        axis=1)))  # Repeated measures t-test

    # Generate the plot
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    title = ['Dorsal V1-3', 'Early visual cortex', 'Higher order visual areas']
    fig = plt.figure(figsize=(10, 5))
    palette = sns.color_palette("Paired")[4:6][::-1]
    for i in range(3):
        fig.add_subplot(1, 3, i + 1)
        ax = sns.swarmplot(x='Prediction', y='$\Delta$$\t\Theta$',
                           data=eval('df_' + str(i)),
                           palette=palette)
        plt.title(title[i])

        # Prediction error for the same participants
        for j in range(10):
            x = [eval('df_' + str(i))['Prediction'][j],
                 eval('df_' + str(i))['Prediction'][j + 10]]
            y = [eval('df_' + str(i))['$\Delta$$\t\Theta$'][j],
                 eval('df_' + str(i))['$\Delta$$\t\Theta$'][j + 10]]
            ax.plot(x, y, color='black', alpha=0.1)
            if retinotopic_feature == 'PA':
                plt.ylim([10, 35])
            if retinotopic_feature == 'ecc':
                plt.ylim([0, 2])
            if retinotopic_feature == 'pRFcenter':
                plt.ylim([0, 1])
            # plt.ylim([15, 45])
    if retinotopic_feature == 'PA':
        plt.ylim([30, 75])
    if retinotopic_feature == 'ecc':
        plt.ylim([1.5, 4])
    if retinotopic_feature == 'pRFcenter':
        plt.ylim([0, 3])

    # plt.savefig('./../output/DeltaTheta_ModelvsAverage.pdf', format="pdf")
    return plt.show()


error_plots('LH', 'PA')
error_plots('LH', 'ecc')
error_plots('LH', 'pRFcenter')
