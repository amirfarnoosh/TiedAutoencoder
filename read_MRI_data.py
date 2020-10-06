# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:17:44 2019

@author: Amir
"""

import os
import numpy as np
import nibabel as nib
from glob import glob
#import matplotlib.pyplot as plt


def get_patches(data, patch_dim):
    
    d1, d2, d3 = patch_dim
    patches = [data[d1*i:d1*(i+1), d2*j:d2*(j+1), d3*k:d3*(k+1)] \
               for i in range(data.shape[0]//d1) \
               for j in range(data.shape[1]//d2) \
               for k in range(data.shape[2]//d3) \
               if np.any(data[d1*i:d1*(i+1), d2*j:d2*(j+1), d3*k:d3*(k+1)])]
    
    return np.asarray(patches) 
    
        

data_path = 'S:/Users/Amir/fMRI/Data/'
data_list = os.listdir(data_path)
data_folders = [data_list[i] for i in range(len(data_list)) if os.path.isdir(os.path.join(data_path,data_list[i]))]
patch_dim = (5,5,5)

MRI_patches = np.array([]).reshape((0,patch_dim[0],patch_dim[1],patch_dim[2]))
for i in range(len(data_folders)):
    data_dir = os.path.join(data_path, data_folders[i], 'anat')
    files_dir = glob(data_dir + '/*.gz')
    for j in range(len(files_dir)):
        img = nib.load(files_dir[j])
        data = img.get_fdata()
        patches = get_patches(data, patch_dim)
        MRI_patches = np.concatenate((MRI_patches,patches),axis=0)
        

save_path = 'S:/Users/Amir/fMRI/'
np.savez_compressed(save_path + 'MRI_patches', MRI_patches)



"""
import numpy as np

data = np.ones((32, 32, 15, 100), dtype=np.int16)
img = nib.Nifti1Image(data, np.eye(4))

img.to_filename(os.path.join('build','test4d.nii.gz'))

or

http://nipy.org/nibabel/coordinate_systems.html

http://nipy.org/nibabel/coordinate_systems.html#naming-reference-spaces

from nibabel.affines import apply_affine
apply_affine(epi_img.affine, epi_vox_center)

"""