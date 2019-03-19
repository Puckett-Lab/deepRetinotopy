import nibabel as nib
import os.path as osp


path=osp.join(osp.dirname(osp.realpath(__file__)),'data','S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k')


path='/home/uqfribe1/PycharmProjects/pytorch_geometric/HCP_gCNN/data/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k'

R2=nib.load(osp.join(path,'S1200_7T_Retinotopy181.Fit1_R2_MSMAll.32k_fs_LR.dscalar.nii'))
curv=nib.load(osp.join(path,'S1200_7T_Retinotopy181.curvature_MSMAll.32k_fs_LR.dscalar.nii'))
polarAngle=nib.load(osp.join(path,'S1200_7T_Retinotopy181.Fit1_PolarAngle_MSMAll.32k_fs_LR.dscalar.nii'))
eccentricity = nib.load(osp.join(path,'S1200_7T_Retinotopy181.Fit1_Eccentricity_MSMAll.32k_fs_LR.dscalar.nii'))



surface_L= nib.load(osp.join(path,'S1200_7T_Retinotopy181.L.midthickness_MSMAll.32k_fs_LR.surf.gii'))
faces_L = surface_L.darrays[1].data
pos_L= surface_L.darrays[0].data


surface_R= nib.load(osp.join(path,'S1200_7T_Retinotopy181.R.midthickness_MSMAll.32k_fs_LR.surf.gii'))
faces_R= surface_R.darrays[1].data
pos_R=surface_R.darrays[0].data







'''
import nibabel as nib





path='/home/uqfribe1/PycharmProjects/pytorch_geometric/my_stuff/data/S1200_7T_Retinotopy_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k'

c = nibabel.load(osp.join(path,'S1200_7T_Retinotopy181.Fit1_Eccentricity_MSMAll.32k_fs_LR.dscalar.nii'))
s = nibabel.load(osp.join(path,'S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii')

brain_model_cortex_left = [brain_model for brain_model in c.header.get_index_map(1).brain_models if brain_model.brain_structure == 'CIFTI_STRUCTURE_CORTEX_LEFT'][0]
vertex_indices = list(brain_model_cortex_left.vertex_indices)
grayordinates_surface = s.darrays[0].data
grayordinates_cortex_left = [(brain_model_cortex_left.index_offset + i, grayordinates_surface[v]) for i, v in enumerate(vertex_indices)]
topo_surface = s.darrays[1].data
topo_cortex_left = [(brain_model_cortex_left.index_offset + i, topo_surface[v]) for i, v in enumerate(vertex_indices)]'''

