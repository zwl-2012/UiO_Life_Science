
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt
from utilities3 import *


import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io
import os

from config_IFNO import Config, load_datasets, normalize_sets


"""A collection of functions for IFNO"""


################################################################
#################### 4d fourier layers
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))


    ########## Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y,z,t ), (in_channel, out_channel, x,y,z,t) -> (batch, out_channel, x,y,z,t)
        #return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
        return torch.einsum("bixy,ioxy->boxy", input, weights)


    def forward(self, x):
        batchsize = x.shape[0] 
        ########## Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-2,-1])

        ########## Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:] = self.compl_mul2d(x_ft[:, :, :self.modes1, -self.modes2:], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:] = self.compl_mul2d(x_ft[:, :, -self.modes1:, -self.modes2:], self.weights4)

        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)))
        return x




class FNO2d(nn.Module):
    #def __init__(self, modes1, modes2, width, nlayer, T_in, var, T_out): 
    def __init__(self, setup):
        super(FNO2d, self).__init__()

        self.modes1 = setup.modes
        self.width = setup.width
        self.T_in = setup.T_in
        self.var_in = setup.var_in
        self.T_out = setup.T_out
        self.var_out = setup.var_out
        
        self.nlayer = setup.nlayer
        
        self.convlayer = nn.ModuleList([SpectralConv2d(self.width, self.width, 
                self.modes1, self.modes1).to(device) for i in range(1)])
        
        self.w = nn.ModuleList([nn.Conv2d(self.width, self.width, 1).to(device) for i in range(1)])
        
        self.enc = nn.Conv2d(self.var_in*self.T_in, self.width, 1) # encoder
        self.dec = nn.Conv2d(self.width, self.var_out*self.T_out, 1) # decoder
        

    def forward(self, x):  
        batchsize, size_x, size_y, var_in, T_in = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        coef = 1./self.nlayer
        
        ########## Reconstruct ##########
        x = x.reshape(batchsize, size_x, size_y, var_in*T_in)
        x = x.permute(0,3,1,2)
        
        ########## Predict ##########
        x = self.enc(x) # Encoder # [BS,width,32,32,32]
        x = torch.tanh(x)

        for i in range(self.nlayer):
            x1 = self.convlayer[0](x) #[BS,width,32,32,32]
            x2 = self.w[0](x)  #[BS,width,32,32,32]
            x = torch.tanh(x1+x2)*coef + x   #[BS,width,32,32,32]
    
        x = self.dec(x) # Decoder   #[BS,width,32,32,32]---->#[BS,var*T_out,32,32,32]
        x = x.permute(0,2,3,1) #[BS,32,32,32,,var*T_out]
        x = x.reshape(batchsize, size_x, size_y, self.var_out, self.T_out)
        return x








################################################################
#################### Configs ##############################
################################################################
def tune_model(setup, train_years, val_years, test_years, folder_path, folder_save_MSE, 
               epochs_tuning=60):
    #################### Device ####################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #################### Setup ####################
    epochs_tuning = epochs_tuning

    ################### Define training, validation and test years ####################
    
    #################### Load data ####################
    ##### dir_path the folder path for both input and output datasets

    ##### Make Dataloaders
    train_loader, val_loader, test_loader = load_datasets(folder_path, train_years, 
                                                val_years, test_years, setup.batch_size)

    model = FNO2d(setup)

    optimizer = torch.optim.Adam(model.parameters(), lr=setup.learning_rate, weight_decay=setup.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=setup.scheduler_step, gamma=setup.scheduler_gamma)
    myloss = torch.nn.MSELoss()
    train_pre = []; val_pre = []  # might want to store the losses for each epoch

    epoch_min = 0 # find the epoch that yields the lowest validation loss
    curr_val_loss_sum = 1000

    for ep in range(epochs_tuning):
        #################### Training ####################
        print(f"Training epoch {ep+1}")
        t1 = default_timer()
        model.train()
        train_loss_sum = 0.0
        for xx, yy in train_loader:
            bs = xx.shape[0] # batchsize
            xx = xx.to(device) # input ([batchsize, 32, 64, 1, 1])
            yy = yy.to(device) # target ([batchsize, 64, 128, 1])
            pre = model(xx)  # predicted ([batchsize, 32, 64, 1, 1])
            l_pred = myloss(pre.reshape(bs, -1), yy.reshape(bs, -1))    #[BS, 64, 128, 1, T_out]
            loss = setup.alpha * l_pred
            train_loss_sum += l_pred.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_loss_sum = train_loss_sum / len(train_loader)
        train_pre.append(train_loss_sum)


        #################### Validation (predict only) ####################
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad(): 
            for xx, yy in val_loader:
                loss = 0
                bs = xx.shape[0] # batchsize
                xx = xx.to(device) # input ([batchsize, 32, 64, 1, 1])
                yy = yy.to(device) # target ([batchsize, 64, 128, 1])
                pre = model(xx)  # predicted ([batchsize, 32, 64, 1, 1])
                loss = myloss(pre.reshape(bs, -1), yy.reshape(bs, -1))    #[BS, 64, 128, 1, T_out]
                val_loss_sum += loss.item()
            val_loss_sum = val_loss_sum / len(val_loader)
            val_pre.append(val_loss_sum)

        t2 = default_timer()
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)

        if ep == 0:
            print("Epoch | ", "Time | ", "[Train Pred MSE] | ", "[Val Pred MSE] | ",
                "[Allocated MB] | ", "[Reserved MB] | ")

        print(f"{ep} | {t2-t1:.2f} | {train_loss_sum:.6f} | {val_loss_sum:.6f} |" \
              f"{allocated_memory:.2f} | {reserved_memory:.2f}")

        if val_loss_sum < curr_val_loss_sum: 
            curr_val_loss_sum = val_loss_sum
            epoch_min = ep+1

    overview = f"\n---------- OPTIMAL HYPERPARAMETERS FOR EACH CONFIG ----------\n" \
        f"{setup.__str__()}" \
        f"Optimal epoch = {epoch_min} | MSELoss = {curr_val_loss_sum:.6f}\n"
    print(overview)
    

    ########## Save the loss for training and validation
    #MSE_save = np.dstack((train_pre, val_pre)).squeeze()
    #np.savetxt(f"{folder_save_MSE}loss_train_val_m{setup.modes}_w{setup.width}_n{setup.nlayer}_s{setup.scheduler_step}.dat", 
    #           MSE_save, fmt="%16.7f")
    #print("Loss data file saved")

    print("\n-----------------------------\nOPTIMAL DONE\n------------------------------")
