clear all

%Connectivity of triangular faces
mid_gifti_L=gifti('S1200_7T_Retinotopy181.L.midthickness_MSMAll.32k_fs_LR.surf.gii');
mid_gifti_R=gifti('S1200_7T_Retinotopy181.R.midthickness_MSMAll.32k_fs_LR.surf.gii');

tri_faces_L=mid_gifti_L.faces;
save('tri_faces_L','tri_faces_L')
tri_faces_R=mid_gifti_R.faces;
save('tri_faces_R','tri_faces_R');
mid_pos_L=mid_gifti_L.vertices;
save('mid_pos_L','mid_pos_L');
mid_pos_R=mid_gifti_R.vertices;
save('mid_pos_R','mid_pos_R');

%Loading cifti files, and getting measures for each cortical vertex, where
%the brain structure labels are so that 1=Left; 2=Right; >2=subcortical structures

%Eccentricity values
cifti_eccentricity=ft_read_cifti('S1200_7T_Retinotopy181.All.Fit1_Eccentricity_MSMAll.32k_fs_LR.dscalar.nii');
%eccentricity_All=cifti_eccentricity.fit1_eccentricity_msmall(cifti_eccentricity.brainstructure<=2);
save('cifti_eccentricity_all','cifti_eccentricity')

%Curvature values
cifti_curv=ft_read_cifti('S1200_7T_Retinotopy181.All.curvature_MSMAll.32k_fs_LR.dscalar.nii');
%curvature_All=cifti_curv.curvature_msmall;
save('cifti_curv_all','cifti_curv')

%R2 fit values
cifti_R2=ft_read_cifti('S1200_7T_Retinotopy181.All.Fit1_R2_MSMAll.32k_fs_LR.dscalar.nii');
%R2_All=cifti_R2.fit1_r2_msmall(cifti_R2.brainstructure<=2);
save('cifti_R2_all','cifti_R2')

%Polar Angle values
cifti_polarAngle=ft_read_cifti('S1200_7T_Retinotopy181.All.Fit1_PolarAngle_MSMAll.32k_fs_LR.dscalar.nii');
%polarAngle_All=cifti_polarAngle.fit1_polarangle_msmall(cifti_polarAngle.brainstructure<=2);
save('cifti_polarAngle_all','cifti_polarAngle')

