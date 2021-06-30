import numpy as np
import pandas as pd
import scipy.stats


def summary_results(hemisphere, retinotopic_feature):
    """Function to generate summary results.

    Args:
        hemisphere (str): 'LH' or 'RH'.
        retinotopic_feature (str): 'PA' or 'ecc' or 'pRFcenter'.


    Returns:
        df (pandas data frame): Pandas object with mean (SD) prediction errors
            of deepRetinotopy predictions and average-based prediction.
    """

    error_DorsalEarlyVisualCortex_model = np.reshape(np.array(
        np.load('./output/ErrorPerParticipant_' + str(
            retinotopic_feature) + '_' + str(
            hemisphere) + '_dorsalV1-3_deepRetinotopy_1-8.npz')['list']),
        (10, -1))
    error_EarlyVisualCortex_model = np.reshape(np.array(
        np.load('./output/ErrorPerParticipant_' + str(
            retinotopic_feature) + '_' + str(
            hemisphere) + '_EarlyVisualCortex_deepRetinotopy_1-8.npz')[
            'list']),
        (10, -1))
    error_higherOrder_model = np.reshape(np.array(
        np.load('./output/ErrorPerParticipant_' + str(
            retinotopic_feature) + '_' + str(
            hemisphere) + '_WangParcels_deepRetinotopy_1-8.npz')['list']),
        (10, -1))

    error_DorsalEarlyVisualCortex_average = np.reshape(np.array(
        np.load('./output/ErrorPerParticipant_' + str(
            retinotopic_feature) + '_' + str(
            hemisphere) + '_dorsalV1-3_average_1-8.npz')['list']),
        (10, -1))
    error_EarlyVisualCortex_average = np.reshape(np.array(
        np.load('./output/ErrorPerParticipant_' + str(
            retinotopic_feature) + '_' + str(
            hemisphere) + '_EarlyVisualCortex_average_1-8.npz')[
            'list']), (10, -1))
    error_higherOrder_average = np.reshape(np.array(
        np.load('./output/ErrorPerParticipant_' + str(
            retinotopic_feature) + '_' + str(
            hemisphere) + '_WangParcels_average_1-8.npz')['list']),
        (10, -1))

    # Repeated measures t-test
    # print(
    #     scipy.stats.ttest_rel(
    #         np.mean(error_DorsalEarlyVisualCortex_model, axis=1),
    #         np.mean(error_DorsalEarlyVisualCortex_average,
    #                 axis=1)))
    # print(
    #     scipy.stats.ttest_rel(
    #         np.mean(error_EarlyVisualCortex_model, axis=1),
    #         np.mean(error_EarlyVisualCortex_average,
    #                 axis=1)))
    # print(
    #     scipy.stats.ttest_rel(
    #         np.mean(error_higherOrder_model, axis=1),
    #         np.mean(error_higherOrder_average,
    #                 axis=1)))

    # Reformatting data from dorsal early visual cortex
    data = []
    cluster = ['DorsalEarlyVisualCortex', 'EarlyVisualCortex', 'higherOrder']
    for i in range(len(cluster)):
        data.append([np.mean(
            np.mean(eval('error_' + str(cluster[i]) + '_model'), axis=1)),
            np.std(
                np.mean(eval('error_' + str(cluster[i]) + '_model'),
                        axis=1)),
            str(cluster[i]), 'deep_Retinotopy', hemisphere,
            retinotopic_feature])
        data.append([np.mean(
            np.mean(eval('error_' + str(cluster[i]) + '_average'), axis=1)),
                     np.std(
                         np.mean(eval('error_' + str(cluster[i]) + '_average'),
                                 axis=1)),
                     str(cluster[i]), 'average', hemisphere,
                     retinotopic_feature])

    df = pd.DataFrame(
        columns=['Mean prediction error', 'SD', 'Cluster', 'Model',
                 'Hemisphere',
                 'Retinotopic feature'],
        data=data)
    return df


hemisphere = ['LH', 'RH']
retinotopic_feature = ['PA', 'ecc', 'pRFcenter']

summary_table = pd.DataFrame(
    columns=['Mean prediction error', 'SD', 'Cluster', 'Model', 'Hemisphere',
             'Retinotopic feature'])
for i in range(len(hemisphere)):
    for j in range(len(retinotopic_feature)):
        summary_table = pd.concat(
            [summary_table, summary_results(hemisphere[i],
                                            retinotopic_feature[j])])

print(summary_table)
summary_table.to_excel('./supTable1_data.xlsx')