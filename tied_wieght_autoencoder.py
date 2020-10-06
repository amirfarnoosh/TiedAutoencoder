# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 17:11:14 2019

@author: Amir
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import os
import time
import matplotlib
matplotlib.rcParams.update({'font.size': 25})
import matplotlib.pyplot as plt 
from glob import glob
plt.subplots_adjust(top = 0.99, bottom=0.1, hspace=0.1, wspace=0.1)
#left=None, bottom=None, right=None, top=None, wspace=None, hspace=None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(' Processor is %s' % (device))

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
    
 
## https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, root_dir):
        'Initialization'
        self.list_IDs = list_IDs
        self.root_dir = root_dir

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = np.load(os.path.join(self.root_dir, ID))['arr_0'].reshape(-1)
        #normalize
        X = X / 378  #std()
        X = torch.FloatTensor(X)

        return X


MSELoss = nn.MSELoss(size_average = False, reduce = True)

def KLLoss(encoded_feats, s):
    
    s_hat = torch.mean(encoded_feats, dim = 0) 
    KL = s * torch.log(s / (s_hat + 1e-6) ) + (1-s) * torch.log((1-s) / (1-s_hat + 1e-6))          
    return torch.sum(KL) 

def TiedAutoEncoderLoss(weight, encoded_feats, reconstructed_output, input):
    
    L2Loss = MSELoss(reconstructed_output, input)
    sparsity_loss = KLLoss(encoded_feats, s)
    overfit_loss =  torch.sum(torch.pow(weight, 2))
    
    return L2Loss + beta * sparsity_loss + gamma * overfit_loss, L2Loss, sparsity_loss, overfit_loss    


def train_autoencoder():
        
    PATH_AutoEnc = save_PATH + '/Auto_Enc_s%0.4f_b%0.4f_g%0.4f_epoch%d_out%d' % (s, beta, gamma, epoch_num, out)
    
    tied_module_mixed = MixedApproachTiedAutoEncoder(inp, out)

    optim_mixed = optim.Adam(tied_module_mixed.parameters(), lr=1e-2)

    # common input
    ###input = torch.rand(100, 125)

    loss_value_concat = []
    L2Loss_value_concat = []
    sparsity_loss_value_concat = []
    overfit_loss_value_concat = []

    if Restore == False:
        
        print("Training...")
        
        for i in range(epoch_num):
            time_start = time.time()
            loss_value = 0.0
            L2Loss_value = 0.0
            sparsity_loss_value = 0.0
            overfit_loss_value = 0.0
            
            for j in range(len(sub_indx)):
                training_set = np.load(os.path.join(root_dir, 'fMRI_patches_sub%.2d.npz' %(j+1)))['arr_0']
                #normalize
                training_set = training_set / 378  #std()
                training_set = torch.FloatTensor(training_set)
                train_loader = data.DataLoader(training_set, **params)
                for batch_indx, batch_data in enumerate(train_loader):
                # update AutoEncoder
                    batch_data = Variable(batch_data)
                    data_ae = batch_data.to(device)
    
                    optim_mixed.zero_grad()
                
                    # get output from both modules	
                    weight, encoded_feats, reconstructed_output, encoder_bias, decoder_bias = tied_module_mixed.forward(data_ae)
                    
                    # back propagation
                    AutoEncoder_loss, L2Loss, sparsity_loss, overfit_loss = TiedAutoEncoderLoss(weight, encoded_feats, reconstructed_output, data_ae)
                    AutoEncoder_loss.backward()
    
                    optim_mixed.step()
        
                    loss_value += AutoEncoder_loss.data[0] 
                    L2Loss_value += L2Loss.item()
                    sparsity_loss_value += sparsity_loss.item()
                    overfit_loss_value += overfit_loss.item()
    
            time_end = time.time()
            print('elapsed time (min) : %0.1f' % ((time_end-time_start)/60))
            print('====> Epoch: %d Obj_Loss : %0.8f | L2_Loss : %0.8f | Sparsity_Loss : %0.8f | Overfit_Loss : %0.8f'\
                  % ((i + 1), loss_value / len(train_loader.dataset),\
                     L2Loss_value / len(train_loader.dataset),\
                     sparsity_loss_value / len(train_loader.dataset),\
                     overfit_loss_value / len(train_loader.dataset)))
    
            torch.save(tied_module_mixed.state_dict(), PATH_AutoEnc)
            
            
            ###### loss plots #####
            
            loss_value_concat.append(loss_value / len(train_loader.dataset))
            L2Loss_value_concat.append(L2Loss_value / len(train_loader.dataset))
            sparsity_loss_value_concat.append(sparsity_loss_value / len(train_loader.dataset))
            overfit_loss_value_concat.append(overfit_loss_value / len(train_loader.dataset))
            
            All_loss_temp = (loss_value_concat, L2Loss_value_concat, sparsity_loss_value_concat, overfit_loss_value_concat)
            
            
            if i > 0:
                plt.close('all')
                plt.figure(figsize=(20,30))
                ylabels = ['Total Loss', 'L2 Loss', 'Sparsity Loss', 'Overfit Loss']
                for j in range(4):            
                    plt.subplot(2,2,j+1)
                    plt.ylabel(ylabels[j])
                    
                    for k in range(len(All_loss[j])):
                        
                        plt.plot(All_loss[j][k][1:])
                        
                    plt.plot(All_loss_temp[j][1:])
                    plt.legend(legends)
                    
                plt.pause(1)
                plt.savefig(fig_save_PATH + '/loss.png')
    
    sub_power = []
    err_power = []
    feats_selected = []
     
    if Restore:
        tied_module_mixed.load_state_dict(torch.load(PATH_AutoEnc)) #, map_location=lambda storage, loc: storage))
    
    test_set = np.load(os.path.join(root_dir, 'fMRI_patches_sub%.2d.npz' %(1)))['arr_0']
    test_set = test_set[rand_num]
    test_set = test_set / 378  #std()
    test_set = torch.FloatTensor(test_set)
    test_loader = data.DataLoader(test_set, **params)
    for batch_indx, batch_data in enumerate(test_loader):   
        _, encoded_feats, _, _, _ = tied_module_mixed.forward(batch_data)
        feats_selected = encoded_feats.detach().numpy()
        
    for j in range(len(sub_indx)):
        sub_power_value = 0
        sub_err_value = 0
        init_indx = 0
        tests_set = np.load(os.path.join(root_dir, 'fMRI_patches_sub%.2d.npz' %(j+1)))['arr_0']
        #normalize
        tests_set = tests_set / 378  #std()
        tests_set = torch.FloatTensor(tests_set)
        for k in range(len(sub_indx[j])):
            
            test_set = tests_set[init_indx:init_indx+sub_indx[j][k]]
            test_loader = data.DataLoader(test_set, **params)
            sub_run_err = 0
            sub_run_power = 0
            cnt = 0
            for batch_indx, batch_data in enumerate(test_loader):        
                _, _, reconstructed_output, _, _ = tied_module_mixed.forward(batch_data)
                sub_run_power += torch.mean(torch.pow(batch_data,2)).item()
                sub_run_err += torch.mean(torch.pow(batch_data-reconstructed_output,2)).item()
                cnt += 1
            sub_power_value += np.sqrt(sub_run_power/cnt) 
            sub_err_value += np.sqrt(sub_run_err/cnt)
            init_indx = init_indx + sub_indx[j][k]
            
        sub_power.append(sub_power_value/len(sub_indx[j]))
        err_power.append(sub_err_value/len(sub_indx[j]))
        
          
    return loss_value_concat, L2Loss_value_concat, sparsity_loss_value_concat, overfit_loss_value_concat, sub_power, err_power, feats_selected
    

def fMRI_reconst(fmdata, patch_dim):
    
    fdata = np.zeros(np.shape(fmdata)) 
    d1, d2, d3, d4 = patch_dim
    tied_module_mixed = MixedApproachTiedAutoEncoder(inp, out)
    PATH_AutoEnc = save_PATH + '/Auto_Enc_s%0.4f_b%0.4f_g%0.4f_epoch%d_out%d' % (s, beta, gamma, epoch_num, out)
    tied_module_mixed.load_state_dict(torch.load(PATH_AutoEnc))
    for i in range(fmdata.shape[0]//d1):
        for j in range(fmdata.shape[1]//d2):
            for k in range(fmdata.shape[2]//d3):
                for l in range(fmdata.shape[3]//d4):
                    if np.any(fmdata[d1*i:d1*(i+1), d2*j:d2*(j+1), d3*k:d3*(k+1), d4*l:d4*(l+1)].detach().numpy()):
                        patch = fmdata[d1*i:d1*(i+1), d2*j:d2*(j+1), d3*k:d3*(k+1), d4*l:d4*(l+1)].reshape(-1,d1*d2*d3*d4)
                        _, _, reconstructed_output, _, _ = tied_module_mixed.forward(patch)                
                        fdata[d1*i:d1*(i+1), d2*j:d2*(j+1), d3*k:d3*(k+1), d4*l:d4*(l+1)] = reconstructed_output.detach().numpy().reshape(d1,d2,d3,d4)        
    
    mid_slice_reconst = fdata[:,:, int(z/2), int(w/2)]
    
    return mid_slice_reconst
  
""" ############
Code Begins Here
"""   
    
if __name__ == '__main__':   
    
    inp = 125*5
    out = 150*5
    s = 0.05
    batch_size = 10000
    epoch_num = 40
    Restore = False
    
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 1}
    
    root_dir = 'S:/Users/Amir/fMRI/Data_Patches/'
    gamma = 7 * (batch_size/(inp*out))
    
    # load  Dataset
    train_set = np.load(os.path.join(root_dir, 'fMRI_patches_sub%.2d.npz' %(1)))['arr_0']
    
    data_list = os.listdir(root_dir)
    data_folders = [data_list[i] for i in range(len(data_list)) if os.path.isdir(os.path.join(root_dir,data_list[i]))]
    sub_indx = []
    for i in range(len(data_folders)):
        data_dir = os.path.join(root_dir, data_folders[i], 'func')
        files_dir = os.listdir(data_dir)
        run_indx = ()
        for j in range(len(files_dir)):
            run_indx += (len(glob(os.path.join(data_dir,files_dir[j],'*.npz'))),)
        
        sub_indx.append(run_indx)
    
    # normalize dataset if needed (TODO)
    
    nPatches = train_set.shape[0]
    
    #loss initialization
    loss_value_tuple = ()
    L2Loss_value_tuple = ()
    sparsity_loss_value_tuple = ()
    overfit_loss_value_tuple = ()
    sub_power_tuple = ()
    err_power_tuple = ()
    feats_selected_tuple = ()
    All_loss = ((),(),(),())
    
    #Path parameters
    save_PATH = './ckpt_files/fMRI_1'
    if not os.path.exists(save_PATH):
        os.makedirs(save_PATH)
        
    fig_save_PATH = './results/images/fMRI_1'
    if not os.path.exists(fig_save_PATH):
        os.makedirs(fig_save_PATH)

    #Hyperparameters values (beta,gamma,out)
    hyp_values = [(1,2,30*5), (0,2,50), (4,2,150*5)]
    #plot parameters
    legends = []
    for i in range(len(hyp_values)):
        legends.append(r'$\beta$ %.1f, $\gamma$ %.1f, $out$ %d' %(hyp_values[i][0], hyp_values[i][1], hyp_values[i][2]))
        
    rand_num = np.random.randint(0, nPatches, 100)
    
    for i in range(len(hyp_values)):
        
        out = hyp_values[i][2]
        beta = hyp_values[i][0] * batch_size
        gamma = hyp_values[i][1] * (batch_size/(inp*out))
        
        loss_value_concat, L2Loss_value_concat, sparsity_loss_value_concat, overfit_loss_value_concat, sub_power, err_power, feats_selected\
        = train_autoencoder()
        
        loss_value_tuple += (loss_value_concat,)
        L2Loss_value_tuple += (L2Loss_value_concat,)
        sparsity_loss_value_tuple += (sparsity_loss_value_concat,)
        overfit_loss_value_tuple += (overfit_loss_value_concat,)
        sub_power_tuple += (sub_power,)
        err_power_tuple +=(err_power,)
        feats_selected_tuple += (feats_selected,)
        
        All_loss= (loss_value_tuple, L2Loss_value_tuple, sparsity_loss_value_tuple, overfit_loss_value_tuple)
        

    sub_power_mean = np.mean(sub_power_tuple, axis = 1)
    sub_power_std = np.std(sub_power_tuple, axis = 1)
    err_power_mean = np.mean(err_power_tuple, axis = 1)
    err_power_std = np.std(err_power_tuple, axis = 1)
        
    exp_num = np.arange(1,len(hyp_values)+1 ,1)
    
    plt.close('all')
    plt.figure(figsize=(20,30))
    plt.bar(exp_num,sub_power_mean, width=0.8, yerr=sub_power_std)
    plt.bar(exp_num,err_power_mean, width=0.6, yerr=err_power_std)
    
    plt.xticks(exp_num, legends)
    plt.legend(['Image Power', 'Error Power'])
    plt.savefig(fig_save_PATH + '/reconst_err.png')
    

    plt.figure(figsize=(20,30))
    n_row = np.sqrt(len(hyp_values)).astype('int')
    n_col = np.ceil(len(hyp_values)/n_row).astype('int')
    for i in range(len(hyp_values)):
        plt.subplot(n_row,n_col,i+1)
        plt.imshow(feats_selected_tuple[i], vmin = 0, vmax = 1)
        plt.title(legends[i])
        plt.xlabel('Features')
        plt.ylabel('Data')
    
    plt.savefig(fig_save_PATH + '/feature_maps.png')
    
    if Restore == False:
        np.savez_compressed(fig_save_PATH + '/data.npz', legends=legends, hyp_values=hyp_values, All_loss = All_loss, sub_power_tuple = sub_power_tuple,\
                 err_power_tuple = err_power_tuple)
        
    
    ########## sample image plots ########
    
    import nibabel as nib
    import copy
    
    patch_dim = (5,5,5,5)
    file_dir = 'S:/Users/Amir/fMRI/Data/sub-control01/func/sub-control01_task-music_run-1_bold.nii.gz'
    img = nib.load(file_dir)
    fmdata = img.get_fdata()
    _, _, z, w = fmdata.shape
    
    mid_slice = copy.deepcopy(fmdata[:,:,int(z/2), int(w/2)])

    fmdata = fmdata / fmdata.std()
    fmdata =  torch.FloatTensor(fmdata)
    
    plt.figure(figsize=(20,30))
    n_row = np.sqrt(len(hyp_values)+1).astype('int')
    n_col = np.ceil((len(hyp_values)+1)/n_row).astype('int')
    plt.subplot(n_row,n_col,1)
    plt.imshow(mid_slice)
    plt.title('mid slice')

    for i in range(len(hyp_values)):
        
        out = hyp_values[i][2]
        beta = hyp_values[i][0] * batch_size
        gamma = hyp_values[i][1] * (batch_size/(inp*out))
        
        mid_slice_reconst = fMRI_reconst(fmdata, patch_dim)
        plt.subplot(n_row,n_col,i+2)
        #mid_slice_reconst = mid_slice_reconst - mid_slice_reconst.min()
        #mid_slice_reconst = mid_slice_reconst / mid_slice_reconst.max() * bins[3]
        plt.imshow(mid_slice_reconst)
        plt.title('Reconst. ' + legends[i])
        
    plt.savefig(fig_save_PATH + '/reconst_sample.png')
        
        
    if Restore == True:
        
        loss_data = np.load(fig_save_PATH + '/data.npz')
        #legends=loss_data('legends')
        All_loss = loss_data['All_loss']
        plt.figure(figsize=(20,30))
        ylabels = ['Total Loss', 'L2 Loss', 'Sparsity Loss', 'Overfit Loss']
        for j in range(4):            
            plt.subplot(2,2,j+1)
            plt.ylabel(ylabels[j])
            for k in range(len(All_loss[j])):
                
                plt.plot(All_loss[j][k][1:])
                        
            plt.legend(legends)
                  
        plt.savefig(fig_save_PATH + '/loss_restored.png')
    
        
    