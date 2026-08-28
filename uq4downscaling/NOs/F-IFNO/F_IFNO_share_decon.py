# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 03:33:23 2021

@author: admin
"""
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

torch.manual_seed(0)
np.random.seed(0)

# os.chdir(r'D:\BaiduNetdiskDownload\zhijie_recently_code\U-FNO_3D-tunning')
################################################################
# 4d fourier layers
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm,
                 n_ff_layers, layer_norm, use_fork, dropout):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3
        self.use_fork = use_fork
        self.fourier_weight = fourier_weight
        
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes1, modes2, modes3]:
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
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        #x = x.permute([0,2,3,4,1])
        x = self.forward_fourier(x)

        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None
        #x = x.permute([0,4,1,2,3])
        #b = b.permute([0,4,1,2,3])

        #if f is None:
            #print("forecast_ff returned None")

        return b, f

    def forward_fourier(self, x):
        x = rearrange(x, 'b s1 s2 s3 i -> b i s1 s2 s3')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        B, I, S1, S2, S3 = x.shape
        print("Printing forward_fourier: ", x.shape) # torch.Size([8, 3, 32, 30, 5])

        # # # Dimesion Z # # #
        x_ftz = torch.fft.rfft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_ftz.new_zeros(B, I, S1, S2, S3 // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]
        #print("x_ftz shape:", x_ftz.shape)
        #print("fourier_weight[2] shape:", self.fourier_weight[2].shape)


        out_ft[:, :, :, :, :self.modes3] = torch.einsum(
            "bixyz,ioz->boxyz",
            x_ftz[:, :, :, :, :self.modes3],
            torch.view_as_complex(self.fourier_weight[2]))

        xz = torch.fft.irfft(out_ft, n=S3, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion Y # # #
        x_fty = torch.fft.rfft(x, dim=-2, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_fty.new_zeros(B, I, S1, S2 // 2 + 1, S3)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        out_ft[:, :, :, :self.modes2, :] = torch.einsum(
            "bixyz,ioy->boxyz",
            x_fty[:, :, :, :self.modes2, :],
            torch.view_as_complex(self.fourier_weight[1]))

        xy = torch.fft.irfft(out_ft, n=S2, dim=-2, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-3, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, S1 // 2 + 1, S2, S3)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        out_ft[:, :, :self.modes1, :, :] = torch.einsum(
            "bixyz,iox->boxyz",
            x_ftx[:, :, :self.modes1, :, :],
            torch.view_as_complex(self.fourier_weight[0]))

        xx = torch.fft.irfft(out_ft, n=S1, dim=-3, norm='ortho')
        #print("xx",xx.shape)
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xx + xy + xz

        x = rearrange(x, 'b i s1 s2 s3 -> b s1 s2 s3 i')
        #print("x",x.shape)
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x

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
            nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias = False),
            nn.LeakyReLU(0.1, inplace=True),  #x>0, is x; x<0 is 0.1x
            nn.Dropout(dropout_rate)
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(input_channels, output_channels, kernel_size=4,
                                stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size,
                          stride=stride, padding=(kernel_size - 1) // 2)


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, nlayer, T_in, var, T_out, share_weight, factor, ff_weight_norm, n_ff_layers, layer_norm): # width相当于输入输出通道
        super(FNO3d, self).__init__()

        """
        input: the solution of the first 5 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, z=64, dim=3, c=5+3)
        output: the solution of the next  timestep
        output shape: (batchsize, x=64, y=64, z=64, dim=3, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.T_in = T_in
        self.var = var
        self.T_out = T_out
        #self.in_proj = WNLinear(input_dim, self.width, wnorm=ff_weight_norm)
        self.fourier_weight = None 
        if share_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes1, modes2, modes3]:
                weight = torch.FloatTensor(width, width, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        self.spectral_layers = nn.ModuleList([])  
        #self.out = nn.Sequential(
            #WNLinear(self.width, 128, wnorm=ff_weight_norm),
            #WNLinear(128, output_dim, wnorm=ff_weight_norm))        
        # self.fc0 = nn.Linear(T_in*var+3, self.width)  
        self.nlayer = nlayer

        """
        self.convlayer = nn.ModuleList([SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3, 
                                                       forecast_ff=None,
                                                       backcast_ff=None,
                                                       fourier_weight=self.fourier_weight,
                                                       factor=factor,
                                                       ff_weight_norm=ff_weight_norm,
                                                       n_ff_layers=n_ff_layers,
                                                       layer_norm=layer_norm,
                                                       use_fork=False,
                                                       dropout=0.0).cuda() for i in range(1)])
        """

        self.convlayer = nn.ModuleList([SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3, 
                                                               forecast_ff=None,
                                                               backcast_ff=None,
                                                               fourier_weight=self.fourier_weight,
                                                               factor=factor,
                                                               ff_weight_norm=ff_weight_norm,
                                                               n_ff_layers=n_ff_layers,
                                                               layer_norm=layer_norm,
                                                               use_fork=False,
                                                               dropout=0.0).to(device) for i in range(1)])
        
        #self.w = nn.ModuleList([nn.Conv3d(self.width, self.width, 1).cuda() for i in range(1)])
        #self.u = nn.ModuleList([U_net(self.width, self.width, 3, 0).cuda() for i in range(1)])
        
        self.enc = nn.Conv3d(var*T_in, width, 1)
        # self.dec_rec = nn.Conv3d(width, var*T_in, 1)
        self.dec = nn.Conv3d(width, var*T_out, 1)
        
        # self.fc1 = nn.Linear(self.width, 512)
        # self.fc2 = nn.Linear(512, 1)

    def forward(self, x):   #[2, 32, 30, 5, 1, 1] 
        batchsize, size_x, size_y, size_z, var, T_in = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.shape[5]  
        coef = 1./self.nlayer

        print(batchsize, size_x, size_y, size_z, var, T_in)

        # grid = self.get_grid(batchsize, size_x, size_y, size_z, x.device) #torch.Size([2, 32, 32, 32, 3])
        # x = torch.cat((x, grid), dim=-1) 
        # x = self.fc0(x)   #经过后输出[bs,32,32,32,3,width]
        # x = x.permute(0, 5, 1, 2, 3) #(2,28,32,32,32,3)
        
        # Reconstruct
        x = x.reshape(batchsize, size_x, size_y, size_z, var*T_in).permute([0,4,1,2,3]) # [BS,32,32,32,var*T_in]-->[BS,var*T_in,32,32,32]
        x_reconstruct = self.enc(x)  # [BS,width,32,32,32]
        x_reconstruct = torch.tanh(x_reconstruct)  # [BS,width,32,32,32]
        x_reconstruct = self.dec(x_reconstruct)  # [BS,var*T_in,32,32,32]
        x_reconstruct = x_reconstruct.permute([0,2,3,4,1]) # [BS,32,32,32,var*T_in]
        x_reconstruct = x_reconstruct.reshape(batchsize, size_x, size_y, size_z, var, T_in)  # [BS,32,32,32,var,T_in]
        
        #predict
        x = self.enc(x) # Encoder # [BS,width,32,32,32]
        x = torch.tanh(x)
        # x_w = x    # used for part5 convolution # [BS,width,32,32,32]

        for i in range(self.nlayer):
            # x1 = self.convlayer[i](x)
            # x2 = self.w[i](x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_w)
            # x3 = self.u[i](x-x1)
            x = x.permute([0,2,3,4,1])
            b, _ = self.convlayer[0](x) #[BS,width,32,32,32]
            x = x.permute([0,4,1,2,3])
            b = b.permute([0,4,1,2,3])
            #x1 = x + b
            #x2 = self.w[0](x)  #[BS,width,32,32,32]
            #x3 = self.u[0](x-x1) #[BS,width,32,32,32]
            x = torch.tanh(b)*coef+x
        # x1 = self.convlayer[self.nlayer-1](x)
        # x2 = self.w[self.nlayer-1](x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_w)
        # x3 = self.u[self.nlayer-1](x-x1)
        #------nlayer-1
        # x1 = self.convlayer[0](x)
        # x2 = self.w[0](x)
        # x3 = self.u[0](x-x1)
        # x = (x1+x2+x3)*coef + x
      
        # x = x.permute(0, 2, 3, 4, 5, 1) #torch.Size([2, 32, 32, 32, 3, 28])
        # x = self.fc1(x)
        # x = F.gelu(x)
        # x = self.fc2(x)
        # return x  #torch.Size([2, 32, 32, 32, 3, 1])
    
        x = self.dec(x) # Decoder   #[BS,width,32,32,32]---->#[BS,var*T_out,32,32,32]
        x = x.permute(0,2,3,4,1) #[BS,32,32,32,,var*T_out]
        x = x.reshape(batchsize, size_x, size_y, size_z, var, T_out)
        return x, x_reconstruct   # beta_loss_[BS,32,32,32,var,T_out], alpha_loss_[BS,32,32,32,var,T_in]

    # def get_grid(self, batchsize, size_x, size_y, size_z, device ): #[bs,32,32,32,3]
    #     gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    #     gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
    #     gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    #     gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
    #     gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
    #     gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        # return torch.cat((gridx, gridy, gridz), dim=-1).to(device) #

# input size should be [bs,64,64,64,3,5]
################################################################
# modes = 10
# width = 30
# nlayer = 10
# T_in = 1
# var = 3
# T_out = 1
# device = torch.device('cuda:1')
# model = FNO3d(modes, modes, modes, width, nlayer, T_in, var, T_out).to(device)  #模型放到GPU上
# print(count_params(model)) 
# #(batchsize, x=32, y=32, z=32, c=3, t=5) c is 3 channel 
# x = torch.rand(2,32,32,32,var,T_in).to(device) #input 12 step, output 1 step
# print('input tensor size = ', x.shape)
# pred = model(x) 
# print(len(pred),pred[0].shape,pred[1].shape)



########## configs
################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda")
print(device)
#-------------------------------------------------------------------------------需要调节的参数
#tunning3
modes = 2
width = 3 #3n
nlayer = 4
#----------------data config
T_in = 1       # input step
T_out = 1       # output step
var = 3         # controls the number of channels for the expected input
                # if each grid cell is a vector of size 3 -> var = 3 channels
                # if each grid cell is a scalar -> var = 1 channels
#-----------------training setting
batch_size = 8
epochs = 2
learning_rate = 0.005
weight_decay_value = 1e-4
scheduler_step = 5
scheduler_gamma = 0.5  # 衰减率
#-------------loss efficient
alpha = 1
beta = 0
#------------Factorization
share_weight = False
factor = 4
ff_weight_norm = True
n_ff_layers = 2
layer_norm = False
#网络层数
#显卡内存有没有爆掉cd C:\Program Files\NVIDIA Corporation\NVSMI，nvidia-smi
#保存模型的文件名
#---------------------------------------------------------------------------------------------

#print(f"modes: {modes}\nwidth: {width}\nnlayer: {nlayer}\nepochs: {epochs}\nlearning_rate: {learning_rate}\n"
#      f"scheduler_step: {scheduler_step}\nscheduler_gamma: {scheduler_gamma}\nalpha: {alpha}\nbeta: {beta}")

################################################################
########## load data
#------------------------------------------------上面4行运行一次后保存出数据，下次直接加载数据
vor_data = np.load('/home/zliu2/UiO_Life_Science/uq4downscaling/Datasets/uq4turbu_data/uq4turbu_mini_2sim_20steps/vel_2sim_20step_32x30x5_vec3.npy')
vor_data = torch.from_numpy(vor_data) 

#print(type(vor_data)) # <class 'torch.Tensor'>
#print(vor_data.shape) #torch.Size([2, 20, 32, 30, 5, 3])


#print(vor_data.shape) #torch.Size([2, 20, 32, 30, 1])

input_list = []
output_list = []


#print(vor_data.shape[0]) # 2
#print(vor_data.shape[1]-T_in) # 20-1=19


for j in range(vor_data.shape[0]):
    for i in range(vor_data.shape[1]-T_in):
        # print(i)
        input_list.append(vor_data[j,i:i+T_in,...])
        output_6m5 = (vor_data[j, i+T_in, ...]) # 
        output_list.append(output_6m5) 
     
input_set = torch.stack(input_list) 
#print(input_set.shape) # torch.Size([38, 1, 32, 30, 5, 3])
#print(input_set.shape) # torch.Size([38, 1, 32, 30, 1])

output_set = torch.stack(output_list) 
#print(output_set.shape) # torch.Size([38, 32, 30, 5, 3])
#print(output_set.shape) # torch.Size([38, 32, 30, 1])

"""[simulations, len of a single state, x dim, y dim, z dim, vector] -> 
[total number of states, x dim, y dim, z dim, vector, len of a single state]
Note that the total number of states = number of simulations * timesteps each simulation

[total number of states, len of single state, x dim, y dim, scalar] -> 
[total number of states, x dim, y dim, scalar, len of a single state] ->  [38, 32, 30, 1, 1]
"""
input_set = input_set.permute(0,2,3,4,5,1) 
#print(input_set.shape) # torch.Size([38, 32, 30, 5, 3, 1])
#print(input_set.shape) # torch.Size([38, 32, 30, 1, 1])



full_set = torch.utils.data.TensorDataset(input_set, output_set) # len(full_set) = 38
train_dataset, test_dataset = torch.utils.data.random_split(full_set, [int(0.8*len(full_set)), 
                                                                        len(full_set)-int(0.8*len(full_set))])

#print(type(train_dataset)); print(type(test_dataset)) # <class 'torch.utils.data.dataset.Subset'>
#print(len(train_dataset), len(test_dataset)) # 30, 8





######### Data loader

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#print(type(train_loader)) # <class 'torch.utils.data.dataloader.DataLoader'>
#print(len(train_loader), len(test_loader)) # len(train_dataset) // len(batch_size) = len(train_loader)






################################################################
# training and evaluation
################################################################
model = FNO3d(modes, modes, modes, width, nlayer, T_in, var, T_out, share_weight, factor, ff_weight_norm, n_ff_layers, layer_norm).to(device)
#print(count_params(model)) # modes=2, width=3 234


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_value)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


train_pre = []
train_recon = []
test_pre = []
test_recon = []


myloss = LpLoss()  # 定义loss function, need utilize3 file
# myloss =torch.nn.MSELoss()
# myloss = torch.nn.MSELoss(reduction='mean')


for ep in range(epochs):
    print(f"Training epoch {ep+1}")
    model.train()
    t1 = default_timer()
    train_recons_full = 0
    train_pred_full = 0
    for xx, yy in train_loader:
        l_recons = 0
        bs = xx.shape[0]
        xx = xx.to(device)
        yy = yy.to(device)
        pre, im_re = model(xx)  # prediction, reconstruction 

        print("Model prediction made")
        l_recons = myloss(im_re.reshape(bs, -1), xx.reshape(bs, -1)) #[BS, 32, 32, 32, 3 T_in]
        l_pred = myloss(pre.reshape(bs, -1), yy.reshape(bs, -1))    #[BS, 64, 64, 64, 3, T_out]
        loss = alpha*l_pred + beta*l_recons  # alpha and beta 可以调整参数

        train_pred_full += l_pred.item()
        train_recons_full += l_recons.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    train_pred_full = train_pred_full / len(train_loader)
    train_recons_full = train_recons_full / len(train_loader)
    train_pre.append(train_pred_full)
    train_recon.append(train_recons_full)


    test_pred_full = 0
    test_recons_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            bs = xx.shape[0]
            xx = xx.to(device)
            yy = yy.to(device)
            pre,im_re = model(xx)

            l_recons = myloss(im_re.reshape(bs, -1), xx.reshape(bs, -1)) 
            l_pred = myloss(pre.reshape(bs, -1), yy.reshape(bs, -1))    

            test_pred_full += l_pred.item()
            test_recons_full += l_recons.item()

        test_pred_full = test_pred_full / len(test_loader)
        test_recons_full = test_recons_full / len(test_loader)
        test_pre.append(test_pred_full)
        test_recon.append(test_recons_full)

    t2 = default_timer()
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # 转换为MB
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)  # 转换为MB

    if ep == 0:
        print("Epoch,","Time,","[Train Recons MSE],","[Train Pred MSE],","[Test Recons MSE],","[Test Pred MSE]","[Allocated Memory]","[Reserved Memory]")
    print(ep, "%.2f"%(t2-t1), "%.6f"%(train_recons_full), "%.6f"%(train_pred_full), 
          "%.6f"%(test_recons_full), "%.6f"%(test_pred_full), "%.2f"%(allocated_memory), "%.2f"%(reserved_memory))
    #torch.save(model.state_dict(), f'new_m12w90n40g45fac4nff2_{ep+1}ep1e8w.pth')  # 注意修改保存模型格式






#MSE_save = np.dstack((train_pre, train_recon,test_pre,test_recon)).squeeze()
#np.savetxt('./loss_data_new.dat', MSE_save, fmt="%16.7f")


#torch.save(model.state_dict(), 'm12w90n40g45_30ep1e8w.pth')  # 注意修改保存模型格式
