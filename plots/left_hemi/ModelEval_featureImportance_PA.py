import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import scipy.io
import os.path as osp
from functions.def_ROIs import roi as roi2
from functions.def_ROIs_ROI import roi
from functions.plusFovea import add_fovea
import pandas as pd


def smallest_angle(x, y):
    difference = []
    dif_1 = np.abs(y - x)
    dif_2 = np.abs(y - x + 2 * np.pi)
    dif_3 = np.abs(y - x - 2 * np.pi)
    for i in range(len(x)):
        difference.append(min(dif_1[i], dif_2[i], dif_3[i]))
    return np.array(difference) * 180 / np.pi


visual_areas = [['hV4'],['VO1','VO2','PHC1','PHC2'],['V3a','V3b'],['LO1','LO2','TO1','TO2'],['IPS0','IPS1','IPS2','IPS3','IPS4','IPS5','SPL1']]
# models=['final-pred','shuffled-myelincurv','shuffled-myelin','shuffled-curv']
models=['pred','shuffled-myelincurv','constant']
models_name=['Default','Shuffled','Constant']


color=[['darkblue','royalblue'],['steelblue','lightskyblue'],['green','lightgreen'],['darkgoldenrod','goldenrod'],['saddlebrown','chocolate'],['hotpink','pink']]

sns.set_style("whitegrid")
fig = plt.figure(figsize=(25, 7))
for k in range(len(visual_areas)):
    mean_delta = []
    mean_across = []


    for m in range(len(models)):
        a=torch.load('/home/uqfribe1/PycharmProjects/DEEP-fMRI/testset_results/testset-'+models[m]+'_Model3_PA_LH.pt',map_location='cpu')

        theta_withinsubj=[]
        theta_acrosssubj=[]
        theta_acrosssubj_pred=[]
        theta_acrosssubj_emp=[]




        label_primary_visual_areas = ['ROI']
        final_mask_L, final_mask_R, index_L_mask, index_R_mask= roi(label_primary_visual_areas)
        ROI1=np.zeros((32492,1))
        ROI1[final_mask_L==1]=1

        final_mask_L, final_mask_R, index_L_mask, index_R_mask= roi2(visual_areas[k])
        primary_visual_areas=np.zeros((32492,1))
        primary_visual_areas[final_mask_L==1]=1

        mask=ROI1+primary_visual_areas
        mask=mask[ROI1==1]


        #Compute angle between predicted and empirical predictions across subj
        for j in range(len(a['Predicted_values'])):
            theta_across_temp=[]
            theta_pred_across_temp = []
            theta_emp_across_temp=[]




            for i in range(len(a['Predicted_values'])):
                # Compute angle between predicted and empirical predictions within subj
                if i==j:


                    #Loading predicted values
                    pred=np.reshape(np.array(a['Predicted_values'][i]),(-1,1))
                    measured=np.reshape(np.array(a['Measured_values'][j]),(-1,1))

                    #Rescaling polar angles to match the right visual field (left hemisphere)
                    minus=pred>180
                    sum=pred<180
                    pred[minus]=pred[minus]-180
                    pred[sum]=pred[sum]+180
                    pred=np.array(pred)*(np.pi/180)

                    minus=measured>180
                    sum=measured<180
                    measured[minus]=measured[minus]-180
                    measured[sum]=measured[sum]+180
                    measured=np.array(measured)*(np.pi/180)



                    #Computing delta theta, angle between vector defined by the predicted value and the one defined by empirical value of the same subj
                    theta = smallest_angle(pred,measured)
                    theta_withinsubj.append(theta)


                if i != j:
                    # Compute angle between predicted and empirical predictions across subj
                    # Loading predicted values
                    pred = np.reshape(np.array(a['Predicted_values'][i]), (-1, 1))
                    pred2= np.reshape(np.array(a['Predicted_values'][j]), (-1, 1))
                    measured = np.reshape(np.array(a['Measured_values'][j]), (-1, 1))
                    measured2 = np.reshape(np.array(a['Measured_values'][i]), (-1, 1))

                    # Rescaling polar angles to match the right visual field (left hemisphere)
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



                    # # Computing delta theta, angle between vector defined predicted i and empirical j map
                    # theta = smallest_angle(pred,measured)
                    # theta_across_temp.append(theta)
                    #
                    # # Computing delta theta, angle between vector defined measured i versus measured j
                    # theta_emp = smallest_angle(measured,measured2)
                    # theta_emp_across_temp.append(theta_emp)

                    # Computing delta theta, angle between vector defined pred i versus pred j
                    theta_pred = smallest_angle(pred,pred2)
                    theta_pred_across_temp.append(theta_pred)


            # theta_acrosssubj.append(np.mean(theta_across_temp,axis=0))
            # theta_acrosssubj_emp.append(np.mean(theta_emp_across_temp, axis=0))
            theta_acrosssubj_pred.append(np.mean(theta_pred_across_temp, axis=0))




        # mean_theta_acrosssubj=np.mean(np.array(theta_acrosssubj),axis=0)
        mean_theta_withinsubj=np.mean(np.array(theta_withinsubj),axis=0)
        # mean_theta_acrosssubj_emp=np.mean(np.array(theta_acrosssubj_emp),axis=0)
        mean_theta_acrosssubj_pred=np.mean(np.array(theta_acrosssubj_pred),axis=0)

        mean_delta.append(mean_theta_withinsubj[mask>1])
        mean_across.append(mean_theta_acrosssubj_pred[mask>1])

    mean_delta=np.reshape(np.array(mean_delta),(3,-1))
    # print('cluster'+str(k))
    # print(scipy.stats.mannwhitneyu(mean_delta[0], mean_delta[1]))
    # print(scipy.stats.mannwhitneyu(mean_delta[0], mean_delta[2]))
    mean_across = np.reshape(np.array(mean_across), (3, -1))


    ax = fig.add_subplot(1, 6, k+2)
    # data = np.concatenate(
    #     [[mean_across[0], len(mean_across[0]) * [models[0]], len(mean_across[0]) * ['Between predicted maps']],
    #      [mean_across[1], len(mean_across[1]) * [models[1]], len(mean_across[1]) * ['Between predicted maps']],
    #      [mean_across[2], len(mean_across[2]) * [models[2]], len(mean_across[2]) * ['Between predicted maps']],
    #      [mean_across[3], len(mean_across[3]) * [models[3]], len(mean_across[3]) * ['Between predicted maps']],
    #      [mean_delta[0], len(mean_across[0]) * [models[0]],
    #       len(mean_across[0]) * ['Between predicted map and ground truth']],
    #      [mean_delta[1], len(mean_across[1]) * [models[1]],
    #       len(mean_across[1]) * ['Between predicted map and ground truth']],
    #      [mean_delta[2], len(mean_across[2]) * [models[2]],
    #       len(mean_across[2]) * ['Between predicted map and ground truth']],
    #      [mean_delta[3], len(mean_across[3]) * [models[3]],
    #       len(mean_across[3]) * ['Between predicted map and ground truth']]], axis=1)
    data = np.concatenate([[mean_delta[0], len(mean_across[0])*[models_name[0]], len(mean_across[0])*['Error'],len(mean_across[0])*['Cluster'+str(k+1)]],
                           [mean_delta[1], len(mean_across[1])*[models_name[1]], len(mean_across[1])*['Error'],len(mean_across[1])*['Cluster'+str(k+1)]],
                           [mean_delta[2], len(mean_across[2])*[models_name[2]], len(mean_across[2])*['Error'],len(mean_across[2])*['Cluster'+str(k+1)]],
                           [mean_across[0], len(mean_across[0])*[models_name[0]], len(mean_across[0])*['Individual variability'],len(mean_across[0])*['Cluster'+str(k+1)]],
                           [mean_across[1], len(mean_across[1])*[models_name[1]], len(mean_across[1])*['Individual variability'],len(mean_across[1])*['Cluster'+str(k+1)]],
                           [mean_across[2], len(mean_across[2])*[models_name[2]], len(mean_across[2])*['Individual variability'],len(mean_across[2])*['Cluster'+str(k+1)]]], axis=1)
    df = pd.DataFrame(columns=['$\Delta$$\t\Theta$', 'Input', 'label','cluster'], data=data.T)
    df['$\Delta$$\t\Theta$'] = df['$\Delta$$\t\Theta$'].astype(float)
    palette = ['dimgray', 'lightgray']
    # ax = sns.boxplot(y='$\Delta$$\t\Theta$', x='models', order=models_name, hue='label', data=df, palette=palette,showfliers=False)
    ax = sns.pointplot(y='$\Delta$$\t\Theta$', x='Input', order=models_name, hue='label', data=df, palette=color[k+1],
                       join=False, dodge=True, ci=95)
    ax.set_title('Cluster ' + str(k+1))
    legend=plt.legend()
    legend.remove()

    plt.ylim([0,80])
    #x = sns.swarmplot(data=mean_delta,color='gray')
    ax.set_xlabel('')
    # plt.savefig('PAdif_cluster'+str(k+1)+'.svg')







#Primary visual cortex
label_primary_visual_areas = ['V1d', 'V1v','fovea_V1', 'V2d', 'V2v' ,'fovea_V2', 'V3d',  'V3v','fovea_V3']
V1,V2,V3=add_fovea(label_primary_visual_areas)
primary_visual_areas=np.sum([np.reshape(V1,(-1,1)),np.reshape(V2,(-1,1)),np.reshape(V3,(-1,1))],axis=0)
label=['Early visual cortex']


mean_delta_2=[]
mean_across_2=[]

for m in range(len(models)):
    a=torch.load('/home/uqfribe1/PycharmProjects/DEEP-fMRI/testset_results/testset-'+models[m]+'_Model3_PA_LH.pt',map_location='cpu')

    theta_withinsubj=[]
    theta_acrosssubj=[]
    theta_acrosssubj_pred=[]
    theta_acrosssubj_emp=[]

    label_primary_visual_areas = ['ROI']
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(label_primary_visual_areas)
    ROI1 = np.zeros((32492, 1))
    ROI1[final_mask_L == 1] = 1
    mask = ROI1 + np.reshape(primary_visual_areas, (32492, 1))
    mask = mask[ROI1 == 1]


    #Compute angle between predicted and empirical predictions across subj
    for j in range(len(a['Predicted_values'])):
        theta_across_temp=[]
        theta_pred_across_temp = []
        theta_emp_across_temp=[]




        for i in range(len(a['Predicted_values'])):
            # Compute angle between predicted and empirical predictions within subj
            if i==j:


                #Loading predicted values
                pred=np.reshape(np.array(a['Predicted_values'][i]),(-1,1))
                measured=np.reshape(np.array(a['Measured_values'][j]),(-1,1))

                #Rescaling polar angles to match the right visual field (left hemisphere)
                minus=pred>180
                sum=pred<180
                pred[minus]=pred[minus]-180
                pred[sum]=pred[sum]+180
                pred=np.array(pred)*(np.pi/180)

                minus=measured>180
                sum=measured<180
                measured[minus]=measured[minus]-180
                measured[sum]=measured[sum]+180
                measured=np.array(measured)*(np.pi/180)



                #Computing delta theta, angle between vector defined predicted value and empirical value same subj
                theta = smallest_angle(pred,measured)
                theta_withinsubj.append(theta)


            if i != j:
                # Compute angle between predicted and empirical predictions across subj
                # Loading predicted values
                pred = np.reshape(np.array(a['Predicted_values'][i]), (-1, 1))
                pred2= np.reshape(np.array(a['Predicted_values'][j]), (-1, 1))
                measured = np.reshape(np.array(a['Measured_values'][j]), (-1, 1))
                measured2 = np.reshape(np.array(a['Measured_values'][i]), (-1, 1))

                # Rescaling polar angles to match the right visual field (left hemisphere)
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



                # # Computing delta theta, angle between vector defined predicted i and empirical j map
                # theta = smallest_angle(pred,measured)
                # theta_across_temp.append(theta)
                #
                # # Computing delta theta, angle between vector defined measured i versus measured j
                # theta_emp = smallest_angle(measured,measured2)
                # theta_emp_across_temp.append(theta_emp)

                # Computing delta theta, angle between vector defined pred i versus pred j
                theta_pred = smallest_angle(pred,pred2)
                theta_pred_across_temp.append(theta_pred)


        # theta_acrosssubj.append(np.mean(theta_across_temp,axis=0))
        # theta_acrosssubj_emp.append(np.mean(theta_emp_across_temp, axis=0))
        theta_acrosssubj_pred.append(np.mean(theta_pred_across_temp, axis=0))




    # mean_theta_acrosssubj=np.mean(np.array(theta_acrosssubj),axis=0)
    mean_theta_withinsubj=np.mean(np.array(theta_withinsubj),axis=0)
    # mean_theta_acrosssubj_emp=np.mean(np.array(theta_acrosssubj_emp),axis=0)
    mean_theta_acrosssubj_pred=np.mean(np.array(theta_acrosssubj_pred),axis=0)

    mean_delta_2.append(mean_theta_withinsubj[mask>1])
    mean_across_2.append(mean_theta_acrosssubj_pred[mask>1])

mean_delta_2=np.reshape(np.array(mean_delta_2),(3,-1))
mean_across_2 = np.reshape(np.array(mean_across_2), (3, -1))
# print(scipy.stats.mannwhitneyu(mean_delta_2[0], mean_delta_2[1]))
# print(scipy.stats.mannwhitneyu(mean_delta_2[0], mean_delta_2[2]))
# fig = plt.figure()
ax = fig.add_subplot(1, 6, 1)
# data = np.concatenate([[mean_across_2[0], len(mean_across_2[0])*[models[0]], len(mean_across_2[0])*['Between predicted maps']],
#                        [mean_across_2[1], len(mean_across_2[1])*[models[1]], len(mean_across_2[1])*['Between predicted maps']],
#                        [mean_across_2[2], len(mean_across_2[2])*[models[2]], len(mean_across_2[2])*['Between predicted maps']],
#                        [mean_across_2[3], len(mean_across_2[3])*[models[3]], len(mean_across_2[3])*['Between predicted maps']],
#                        [mean_delta_2[0], len(mean_across_2[0])*[models[0]], len(mean_across_2[0])*['Between predicted map and ground truth']],
#                        [mean_delta_2[1], len(mean_across_2[1])*[models[1]], len(mean_across_2[1])*['Between predicted map and ground truth']],
#                        [mean_delta_2[2], len(mean_across_2[2])*[models[2]], len(mean_across_2[2])*['Between predicted map and ground truth']],
#                        [mean_delta_2[3], len(mean_across_2[3])*[models[3]], len(mean_across_2[3])*['Between predicted map and ground truth']]], axis=1)
data = np.concatenate([[mean_delta_2[0], len(mean_across_2[0])*[models_name[0]], len(mean_across_2[0])*['Error']],
                       [mean_delta_2[1], len(mean_across_2[1])*[models_name[1]], len(mean_across_2[1])*['Error']],
                       [mean_delta_2[2], len(mean_across_2[2])*[models_name[2]], len(mean_across_2[2])*['Error']],
                       [mean_across_2[0], len(mean_across_2[0])*[models_name[0]], len(mean_across_2[0])*['Individual variability']],
                       [mean_across_2[1], len(mean_across_2[1])*[models_name[1]], len(mean_across_2[1])*['Individual variability']],
                       [mean_across_2[2], len(mean_across_2[2])*[models_name[2]], len(mean_across_2[2])*['Individual variability']]], axis=1)
df = pd.DataFrame(columns=['$\Delta$$\t\Theta$', 'Input', 'label'], data=data.T)
df['$\Delta$$\t\Theta$'] = df['$\Delta$$\t\Theta$'].astype(float)
palette = ['dimgray', 'lightgray']

# ax = sns.boxplot(y='$\Delta$$\t\Theta$', x='models', order=models, hue='label', data=df, palette=palette,showfliers=False)
ax = sns.pointplot(y='$\Delta$$\t\Theta$', x='Input', order=models_name, hue='label', data=df, palette=color[0],join=False,dodge=True,ci=95)
ax.set_title('Early visual cortex ')
legend=plt.legend()
# legend.remove()
ax.set_xlabel('')
plt.ylim([0,80])




plt.savefig('ModelEval_AllClusters.svg', format="svg")
plt.show()
