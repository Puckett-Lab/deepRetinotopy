import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats

# Loading the data
error_DorsalEarlyVisualCortex_model = np.reshape(np.array(
    np.load('../ErrorPerParticipant_dorsalV1-3_model.npz')['list']),
    (10, -1))
error_EarlyVisualCortex_model = np.reshape(np.array(
    np.load('../ErrorPerParticipant_EarlyVisualCortex_model.npz')['list']),
    (10, -1))
error_higherOrder_model = np.reshape(np.array(
    np.load('../ErrorPerParticipant_WangParcels_model.npz')['list']), (10, -1))

error_DorsalEarlyVisualCortex_average = np.reshape(np.array(
    np.load('../ErrorPerParticipant_dorsalV1-3_averageMap.npz')['list']),
    (10, -1))
error_EarlyVisualCortex_average = np.reshape(np.array(
    np.load('../ErrorPerParticipant_EarlyVisualCortex_averageMap.npz')[
        'list']), (10, -1))
error_higherOrder_average = np.reshape(np.array(
    np.load('../ErrorPerParticipant_WangParcels_averageMap.npz')['list']),
    (10, -1))

# Reformatting data from dorsal early visual cortex
data_earlyVisualCortex = np.concatenate([
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
    data=data_earlyVisualCortex.T)
df_0['$\Delta$$\t\Theta$'] = df_0['$\Delta$$\t\Theta$'].astype(float)

print(
    scipy.stats.ttest_rel(np.mean(error_DorsalEarlyVisualCortex_model, axis=1),
                          np.mean(error_DorsalEarlyVisualCortex_average,
                                  axis=1)))

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
                            np.mean(error_EarlyVisualCortex_average, axis=1)))

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
                            np.mean(error_higherOrder_average, axis=1)))

# Generate the plot
sns.set_style("whitegrid", {'grid.linestyle': '--'})
title = ['Dorsal V1-3', 'Early visual cortex', 'Higher order visual areas']
fig = plt.figure(figsize=(10, 5))
for i in range(3):
    fig.add_subplot(1, 3, i + 1)
    ax = sns.swarmplot(x='Prediction', y='$\Delta$$\t\Theta$',
                       data=eval('df_' + str(i)),
                       palette='Paired')
    plt.title(title[i])

    # Prediction error for the same participants
    for j in range(10):
        x = [eval('df_' + str(i))['Prediction'][j],
             eval('df_' + str(i))['Prediction'][j + 10]]
        y = [eval('df_' + str(i))['$\Delta$$\t\Theta$'][j],
             eval('df_' + str(i))['$\Delta$$\t\Theta$'][j + 10]]
        ax.plot(x, y, color='black', alpha=0.1)

        plt.ylim([15, 45])
plt.ylim([30, 75])
plt.savefig('./../output/DeltaTheta_ModelvsAverage.pdf', format="pdf")
plt.show()

# Plotting the difference
# data = pd.DataFrame({'DiffEarly': np.mean(
#     error_EarlyVisualCortex_average,
#     axis=1) - np.mean(error_EarlyVisualCortex_model,
#                       axis=1),
#                      'DiffDorsalEarlyVisualCortex': np.mean(
#                      error_DorsalEarlyVisualCortex_average,
#                                            axis=1) - np.mean(
#                          error_DorsalEarlyVisualCortex_model, axis=1)})
#
# data = pd.melt(data)
# sns.boxplot(x='variable', y='value',
#               data=data,
#               palette='colorblind')
# plt.show()
