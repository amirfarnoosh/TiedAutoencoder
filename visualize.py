# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 16:58:21 2019

@author: Amir
"""

import os
import numpy as np
import nibabel as nib
from scipy import ndimage
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
plt.subplots_adjust(top = 0.99, bottom=0.1, hspace=0.1, wspace=0.1)

# mixed approach
class MixedApproachTiedAutoEncoder(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.encoder = nn.Linear(inp, out, bias=True)
        self.bias_param = nn.Parameter(torch.randn(inp))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        encoded_feats = self.sigmoid(self.encoder(input))
        reconstructed_output = F.linear(encoded_feats, self.encoder.weight.t(), bias = self.bias_param)
        return self.encoder.weight, encoded_feats, reconstructed_output, self.encoder.bias, self.bias_param

def my_sigmoid(x):
  return 1 / (1 + np.exp(-x))


def ROI_extractor(x, ROI_dim, Crop):
    d0, d1, d2 = ROI_dim
    x = x[Crop[0]:x.shape[0]-Crop[0],\
          Crop[1]:x.shape[1]-Crop[1],\
          Crop[2]:x.shape[2]-Crop[2]] 
    idxs = ()
    for i in range(x.shape[0]//20 + 1):
        if d0*(i+1)<=x.shape[0]:
            l_max = d0*(i+1)
        elif d0*i==x.shape[0]:
            continue
        else:
            l_max = x.shape[0]
            
        for j in range(x.shape[1]//20 + 1):
            if d1*(j+1)<=x.shape[1]:
                m_max = d1*(j+1)
            elif d1*j==x.shape[1]:
                continue
            else:
                m_max = x.shape[1]
                
            for k in range(x.shape[2]//20 + 1):
                if d2*(k+1)<=x.shape[2]:
                    n_max = d2*(k+1)
                elif d2*k==x.shape[2]:
                    continue
                else:
                    n_max = x.shape[2]
                    
                idx = [[l+Crop[0],m+Crop[1],n+Crop[2]]\
                       for l in range(d0*i,l_max)\
                       for m in range(d1*j,m_max)\
                       for n in range(d2*k,n_max)\
                       if np.any(data[l,m,n,:])]
                if len(idx)!=0:
                    idxs += (np.asarray(idx),)
    return idxs

def connectivity_matrix(x, ROIs):
    ROI_sigs = \
    [np.mean(x[ROIs[i][:,0],ROIs[i][:,1],ROIs[i][:,2],:], axis = 0)\
     for i in range(len(ROIs))]
    ROI_sigs = np.asarray(ROI_sigs)
    
    return np.absolute(np.corrcoef(ROI_sigs))
                    
inp = 125*5
out = 150*5
s = 0.05
batch_size = 10000
epoch_num = 40
hyp_values = [(1,2,30*5), (0,2,50), (4,2,150*5)]
out = hyp_values[2][2]
beta = hyp_values[2][0] * batch_size
gamma = hyp_values[2][1] * (batch_size/(inp*out))
    
save_PATH = './ckpt_files/fMRI_1'
export_PATH = './convolved/fMRI_1'
if not os.path.exists(export_PATH):
    os.makedirs(export_PATH)

fig_save_PATH = './results/images/fMRI_1'

PATH_AutoEnc = save_PATH + '/Auto_Enc_s%0.4f_b%0.4f_g%0.4f_epoch%d_out%d' % (s, beta, gamma, epoch_num, out)

tied_module_mixed = MixedApproachTiedAutoEncoder(inp, out)
tied_module_mixed.load_state_dict(torch.load(PATH_AutoEnc))
weight = tied_module_mixed.encoder.weight.detach().numpy()
bias = tied_module_mixed.encoder.bias.detach().numpy()

#file_dir = r'S:\Users\Amir\fMRI\Data\sub-control01\func\sub-control01_task-music_run-1_bold.nii.gz'
file_dir = './Data/sub-control01_task-music_run-1_bold.nii.gz'

img = nib.load(file_dir)
data = img.get_fdata()
data = data/378

ROIs = ROI_extractor(data, ROI_dim = [15,20,15], Crop = [10,0,0])

num_plots = 10
idx_rand = np.random.randint(0,out, num_plots)
plt.figure(figsize=(20,30))
n_row = np.sqrt(num_plots+1)
n_col = np.ceil((num_plots+1)/n_row)
plt.subplot(n_row,n_col,1)
plt.imshow(connectivity_matrix(data, ROIs), vmin=0, vmax =1)
plt.title('Original Conn_Mat')

plt_cnt = 2
#for i in range(out):
for i in idx_rand:

    kernel = weight[i].reshape(5,5,5,5)
    new_data = my_sigmoid(ndimage.convolve(data,kernel) + bias[i])
    new_data_max = ndimage.maximum_filter(new_data, size=(5,5,5,5))[::5,::5,::5,::5]
    new_data_nifti = nib.Nifti1Image(new_data, np.eye(4))
    new_data_max_nifti = nib.Nifti1Image(new_data_max, np.eye(4))
    new_data_nifti.to_filename(export_PATH + '/sub-control01_beta%.4f_gamma%.4f_out%d_filter%.3d.nii.gz' %(beta,gamma,out,i+1))
    new_data_max_nifti.to_filename(export_PATH + '/sub-control01_beta%.4f_gamma%.4f_out%d_filter%.3d_maxpooled.nii.gz' %(beta,gamma,out,i+1))
    #if i in idx_rand:
    plt.subplot(n_row,n_col,plt_cnt)
    plt.imshow(connectivity_matrix(new_data, ROIs), vmin=0, vmax =1)
    plt.title('Conn_Mat%d' %(i+1))
    plt_cnt += 1
                
plt.savefig(fig_save_PATH + '/Conn_Mats_out%d.png' %out)