import numpy as np
import os.path as osp

def roi(list_of_labels):

    #Defining number of nodes
    number_cortical_nodes=int(64984)
    number_hemi_nodes=int(number_cortical_nodes/2)

    list_primary_visual_areas = np.zeros([6, 64984])
    for i in range(len(list_of_labels)):
        with open(osp.join(osp.dirname(osp.realpath(__file__)),'..','labels/VisualAreasLabels_Wang2015',
                           list_of_labels[i] + '_label.txt')) as fp:
            list_primary_visual_areas[i] = fp.read().split("\n")[0:64984]

    final_mask_L = np.sum(list_primary_visual_areas, axis=0)[0:number_hemi_nodes]
    final_mask_R = np.sum(list_primary_visual_areas, axis=0)[number_hemi_nodes:number_cortical_nodes]


    index_L_mask = [i for i, j in enumerate(final_mask_L) if j == 1]
    index_R_mask = [i for i, j in enumerate(final_mask_R) if j == 1]

    return final_mask_L,final_mask_R,index_L_mask,index_R_mask
