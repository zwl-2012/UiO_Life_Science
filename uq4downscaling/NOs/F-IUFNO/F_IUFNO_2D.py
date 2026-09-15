
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

from einops import rearrange
from feedforward import FeedForward
from linear import WNLinear
from config_F_IUFNO import Config, load_datasets, normalize_sets

"""A collection of functions for F-IUFNO"""


################################################################
########## 4d fourier layers
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm,
                 n_ff_layers, layer_norm, use_fork, dropout):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.use_fork = use_fork
        self.fourier_weight = fourier_weight
        
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes1, modes2]:
                weight = torch.FloatTensor(in_channels, out_channels, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)
           
        if use_fork:
            self.forecast_ff = forecast_ff
            if not self.forecast_ff:
                self.forecast_ff = FeedForward(
                    out_channels, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                out_channels, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)


    def forward(self, x):
        x = self.forward_fourier(x)
        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None
        return b, f


    def forward_fourier(self, x):
        x = rearrange(x, 'b s1 s2 i -> b i s1 s2')
        B, I, S1, S2 = x.shape

        ########## Dimesion Y ##########
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        out_ft = x_fty.new_zeros(B, I, S1, S2 // 2 + 1)
        out_ft[:, :, :, :self.modes2] = torch.einsum("bixy,ioy->boxy",
            x_fty[:, :, :, :self.modes2], torch.view_as_complex(self.fourier_weight[1]))
        xy = torch.fft.irfft(out_ft, n=S2, dim=-1, norm='ortho')



        ########## Dimesion X ##########
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        out_ft = x_ftx.new_zeros(B, I, S1 // 2 + 1, S2)
        out_ft[:, :, :self.modes1, :] = torch.einsum("bixy,iox->boxy",
            x_ftx[:, :, :self.modes1, :], torch.view_as_complex(self.fourier_weight[0]))
        xx = torch.fft.irfft(out_ft, n=S1, dim=-2, norm='ortho')
        

        ########## Combining Dimensions ##########
        x = xx + xy
        x = rearrange(x, 'b i s1 s2 -> b s1 s2 i')
        return x




############################################################
#################### U-net ##############################
############################################################
class U_net(nn.Module):  
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate): #width,width,3,0
        super(U_net, self).__init__()
        self.input_channels = input_channels
        self.conv1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate) #28,28,3,0
        self.conv2 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        self.conv3 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv3_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        
        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels*2, output_channels)
        self.deconv0 = self.deconv(input_channels*2, output_channels)
    
        self.output_layer = self.output(input_channels*2, output_channels, 
                                         kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)


    def forward(self, x):        #[BS,width,32,32,32]        
        batchsize, width = x.shape[0], x.shape[1]
        out_conv1 = self.conv1(x)  #[BS,width,16,16,16]
        out_conv2 = self.conv2_1(self.conv2(out_conv1)) #[BS,width,8,8,8]
        out_conv3 = self.conv3_1(self.conv3(out_conv2)) #[BS,width,4,4,4]

        out_deconv2 = self.deconv2(out_conv3)  #[BS,width,8,8,8]
        concat2 = torch.cat((out_conv2, out_deconv2), 1)  #[BS,2*width,8,8,8]
        out_deconv1 = self.deconv1(concat2)  #[BS,width,16,16,16]
        concat1 = torch.cat((out_conv1, out_deconv1), 1)  #[BS,2*width,16,16,16]
        out_deconv0 = self.deconv0(concat1)   #[BS,width,32,32,32]  
        concat0 = torch.cat((x, out_deconv0), 1)   #[BS,2*width,32,32,32]  
        out = self.output_layer(concat0) #[BS,width,32,32,32]  
        return out   


    def conv(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias = False),
            nn.LeakyReLU(0.1, inplace=True),  #x>0, is x; x<0 is 0.1x
            nn.Dropout(dropout_rate))

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4,
                                stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

    def output(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                          stride=stride, padding=(kernel_size - 1) // 2)




##########################################################################
#################### F-IUFNO ########################################
##########################################################################
class FNO2d(nn.Module):
    #def __init__(self, modes1, modes2, width, nlayer, T_in, var_in, T_out, var_out, 
    #             share_weight, factor, ff_weight_norm, n_ff_layers, layer_norm): 
    def __init__(self, setup):
        super(FNO2d, self).__init__()

        self.modes1 = setup.modes
        self.width = setup.width
        self.T_in = setup.T_in
        self.var_in = setup.var_in
        self.T_out = setup.T_out
        self.var_out = setup.var_out
        self.fourier_weight = None 
        if setup.share_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [self.modes1, self.modes1]:
                weight = torch.FloatTensor(setup.width, setup.width, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        self.spectral_layers = nn.ModuleList([])  
        self.nlayer = setup.nlayer
        
        self.convlayer = nn.ModuleList([SpectralConv2d(self.width, self.width, self.modes1, self.modes1, 
                forecast_ff=None, backcast_ff=None, fourier_weight=self.fourier_weight,
                factor=setup.factor, ff_weight_norm=setup.ff_weight_norm, n_ff_layers=setup.n_ff_layers,
                layer_norm=setup.layer_norm, use_fork=False, dropout=0.0).to(device) for i in range(1)])
        
        self.u = nn.ModuleList([U_net(self.width, self.width, 3, 0).to(device) for i in range(1)])
        
        self.enc = nn.Conv2d(self.var_in*self.T_in, self.width, 1) # encoder
        self.dec_pred = nn.Conv2d(self.width, self.var_out*self.T_out, 1) # decoder
        

    def forward(self, x):   #[2, 32, 32, 32, 3, 5] 
        batchsize, size_x, size_y, var_in, T_in = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        coef = 1./self.nlayer
        
        ##### Reconstruct #####
        x = x.reshape(batchsize, size_x, size_y, var_in*T_in)
        x = x.permute(0,3,1,2)

        ##### Predict #####
        x = self.enc(x) # Encoder # [BS,width,32,32,32]
        x = torch.tanh(x)

        for i in range(self.nlayer):
            x = x.permute([0,2,3,1])
            b, _ = self.convlayer[0](x) #[BS,width,32,32,32]
            x = x.permute([0,3,1,2])
            b = b.permute([0,3,1,2])
            #x1 = x + b
            x3 = self.u[0](x-b) #[BS,width,32,32,32]
            x = torch.tanh(b+x3)*coef + x   #[BS,width,32,32,32]
    
        x = self.dec_pred(x) # Decoder   #[BS,width,32,32,32]---->#[BS,var*T_out,32,32,32]
        x = x.permute(0,2,3,1) #[BS,32,32,var*T_out]
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
