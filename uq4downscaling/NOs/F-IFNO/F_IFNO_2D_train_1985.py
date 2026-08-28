
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

import xarray as xr

# From the neuraloperator repo, if necessary
#from src.data import load_resolution_pairs, make_split_indices, compute_training_stats, make_dataloaders
#from src.config import Config




################################################################
# 4d fourier layers
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, forecast_ff, backcast_ff, fourier_weight, 
                 factor, ff_weight_norm, n_ff_layers, layer_norm, use_fork, dropout):
        super(SpectralConv2d, self).__init__()

        """2D Fourier layer. It does FFT, linear transform, and Inverse FFT."""

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
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        #x = x.permute([0,2,3,4,1])
        #print(x.shape) # torch.Size([8, 32, 30, 3])
        x = self.forward_fourier(x)

        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None
        #x = x.permute([0,4,1,2,3])
        #b = b.permute([0,4,1,2,3])

        #if f is None:
            #print("forecast_ff returned None")

        return b, f

    def forward_fourier(self, x):
        #print("Before:", x.shape) # torch.Size([8, 32, 30, 3])
        x = rearrange(x, 'b s1 s2 i -> b i s1 s2')
        # x.shape == [batch_size, in_dim, grid_size, grid_size, grid_size] // [batch_size, in_dim, grid_size, grid_size]

        
        B, I, S1, S2 = x.shape #  8 3 32 64
        #print("After:", x.shape) # torch.Size([8, 3, 32, 30]) # the in_dim in this case is equal to width
        #print("SpectralConv2d forward fourier:", B, I, S1, S2)


        # # # Dimesion Y # # #    ####### SHOULD dim=-1 ??? #######
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_fty.new_zeros(B, I, S1, S2 // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        out_ft[:, :, :, :self.modes2] = torch.einsum(
            "bixy,ioy->boxy",
            x_fty[:, :, :, :self.modes2],
            torch.view_as_complex(self.fourier_weight[1]))

        xy = torch.fft.irfft(out_ft, n=S2, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]



        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, S1 // 2 + 1, S2)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        out_ft[:, :, :self.modes1, :] = torch.einsum(
            "bixy,iox->boxy",
            x_ftx[:, :, :self.modes1, :],
            torch.view_as_complex(self.fourier_weight[0]))

        xx = torch.fft.irfft(out_ft, n=S1, dim=-2, norm='ortho')
        #print("xx",xx.shape)
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xx + xy

        x = rearrange(x, 'b i s1 s2 -> b s1 s2 i')
        #print("x",x.shape)
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x



class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, nlayer, T_in, var, T_out, share_weight, factor, ff_weight_norm, n_ff_layers, layer_norm): # width
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width  # same as hidden channels in this case
        self.T_in = T_in
        self.var = var
        self.T_out = T_out
        #self.in_proj = WNLinear(input_dim, self.width, wnorm=ff_weight_norm)
        self.fourier_weight = None 
        if share_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes1, modes2]:
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


        self.convlayer = nn.ModuleList([SpectralConv2d(self.width, self.width, self.modes1, self.modes2, 
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
        
        self.enc = nn.Conv2d(var*T_in, width, 1)

        self.dec = nn.Conv2d(width, var*T_out, 1)
        
        # self.fc1 = nn.Linear(self.width, 512)
        # self.fc2 = nn.Linear(512, 1)

    def forward(self, x):   #[2, 32, 30, 5, 1, 1] 
        batchsize, size_x, size_y, var, T_in = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]  
        coef = 1./self.nlayer
        #print("FNO2D forward:", batchsize, size_x, size_y, var, T_in) #FNO2D forward: 8 32 64 1 1
        
        # grid = self.get_grid(batchsize, size_x, size_y, size_z, x.device) #torch.Size([2, 32, 32, 32, 3])
        # x = torch.cat((x, grid), dim=-1) 
        # x = self.fc0(x)   #经过后输出[bs,32,32,32,3,width]
        # x = x.permute(0, 5, 1, 2, 3) #(2,28,32,32,32,3)
        
        # Reconstruct
        #x = x.reshape(batchsize, size_x, size_y, size_z, var*T_in).permute([0,4,1,2,3]) # [BS,32,30,5,var*T_in]-->[BS,var*T_in,32,30,5]
        x = x.reshape(batchsize, size_x, size_y, var*T_in).permute([0,3,1,2]) # [BS,32,30,var*T_in]-->[BS,var*T_in,32,30]
        x_reconstruct = self.enc(x)  # [BS,width,32,30,5] // [BS,width,32,30]
        x_reconstruct = torch.tanh(x_reconstruct)  # [BS,width,32,30,5] // [BS,width,32,30]
        x_reconstruct = self.dec(x_reconstruct)  # [BS,var*T_in,32,30,5] // [BS,var*T_in,32,30]
        x_reconstruct = x_reconstruct.permute([0,2,3,1]) # [BS,32,30,5,var*T_in] // [BS,32,30,var*T_in]
        x_reconstruct = x_reconstruct.reshape(batchsize, size_x, size_y, var, T_in)  # [BS,32,30,5,var,T_in] // [BS,32,30,var,T_in]
        
        #predict
        x = self.enc(x) # Encoder # [BS,width,32,30,5] // [BS,width,32,30]
        x = torch.tanh(x)
        # x_w = x    # used for part5 convolution # [BS,width,32,30,5] // [BS,width,32,30]

        for i in range(self.nlayer):
            # x1 = self.convlayer[i](x)
            # x2 = self.w[i](x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_w)
            # x3 = self.u[i](x-x1)
            x = x.permute([0,2,3,1])
            b, _ = self.convlayer[0](x) #[BS,width,32,30,5] // [BS,width,32,30]
            x = x.permute([0,3,1,2])
            b = b.permute([0,3,1,2])
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


        '''This is the part that reshapes from 32x64 to 64x128, optional to use(?)'''
        #x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


        x = self.dec(x) # Decoder   #[BS,width,32,30,5]---->#[BS,var*T_out,32,30,5] // [BS,width,32,30]-->[BS,var*T_out,32,30]
        x = x.permute(0,2,3,1) #[BS,32,30,5,var*T_out] // [BS,32,30,var*T_out]

        #x_out, y_out = x.shape[1], x.shape[2]
        x = x.reshape(batchsize, size_x, size_y, var, T_out)
        return x, x_reconstruct   # beta_loss_[BS,32,30,5,var,T_out], alpha_loss_[BS,32,30,5,var,T_in]

    # def get_grid(self, batchsize, size_x, size_y, size_z, device ): #[bs,32,32,32,3]
    #     gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    #     gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
    #     gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    #     gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
    #     gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
    #     gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        # return torch.cat((gridx, gridy, gridz), dim=-1).to(device) #


################################################################
# modes = 10
# width = 30
# nlayer = 10
# T_in = 1
# var = 3
# T_out = 1
# device = torch.device('cuda:1')
# model = FNO2d(modes, modes, modes, width, nlayer, T_in, var, T_out).to(device)  
# print(count_params(model)) 
# #(batchsize, x=32, y=32, z=32, c=3, t=5) c is 3 channel 
# x = torch.rand(2,32,32,32,var,T_in).to(device) #input 12 step, output 1 step
# print('input tensor size = ', x.shape)
# pred = model(x) 
# print(len(pred),pred[0].shape,pred[1].shape)



########## configs
################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------------------------------------------------------------------------
#tunning3
modes = 10
width = 10 # 3n
nlayer = 10
#----------------data config
T_in = 1       # input step
T_out = 1       # output step
var = 1         # controls the number of channels for the expected input
                # if each grid cell is a vector of size 3 -> var = 3 channels
                # if each grid cell is a scalar -> var = 1 channels
#-----------------training setting
batch_size = 5
epochs = 20
learning_rate = 0.0005
weight_decay_value = 1e-5
scheduler_step = 5
scheduler_gamma = 0.5  # 
#-------------loss efficient
alpha = 1
beta = 0
#------------Factorization
share_weight = False
factor = 4
ff_weight_norm = True
n_ff_layers = 2
layer_norm = False
#---------------------------------------------------------------------------------------------

print(f"modes: {modes}\nwidth: {width}\nnlayer: {nlayer}\nepochs: {epochs}\nlearning_rate: {learning_rate}\n"
      f"scheduler_step: {scheduler_step}\nscheduler_gamma: {scheduler_gamma}\nalpha: {alpha}\nbeta: {beta}")

################################################################
########## load data
#------------------------------------------------

path_lr_100 = "/home/zliu2/UiO_Life_Science/uq4downscaling/Datasets/Weatherbench/2m_temp_both_res_1985/2m_temperature_low_res_1985_1000.nc"
path_hr_100 = "/home/zliu2/UiO_Life_Science/uq4downscaling/Datasets/Weatherbench/2m_temp_both_res_1985/2m_temperature_high_res_1985_1000.nc"

torch.manual_seed(0)
np.random.seed(0)

input_list = []
output_list = []

input_ds = xr.open_dataset(path_lr_100)
input_ds = input_ds["t2m"].values
input_ds = torch.from_numpy(input_ds) # ([1000, 32, 64])
input_ds = input_ds[:, np.newaxis, ...] # ([1000, 1, 32, 64]) -> [batch, channels, x dim, y dim]

'''Interpolate input from 32x64 to 64x128'''
input_ds = F.interpolate(input_ds, scale_factor=2, mode="bilinear", align_corners=False) # [1000, 1, 64, 128]

input_ds = input_ds[..., np.newaxis] # ([1000, 1, 64, 128, 1])

output_ds = xr.open_dataset(path_hr_100)
output_ds = output_ds["t2m"].values
output_ds = torch.from_numpy(output_ds) # ([1000, 64, 128])
output_ds = output_ds[..., np.newaxis] # ([1000, 64, 128, 1])

for i in range(len(input_ds)):
    input_list.append(input_ds[i, ...])
    output_list.append(output_ds[i, ...])

input_set = torch.stack(input_list)  # ([1000, 1, 64, 128, 1])
output_set = torch.stack(output_list) # ([1000, 64, 128, 1])

'''[simulations, len of a single state, x dim, y dim, z dim, vector] -> 
[total number of states, x dim, y dim, z dim, vector, len of a single state]
Note that the total number of states = number of simulations * timesteps each simulation

[total number of states, len of single state, x dim, y dim, scalar] -> 
[total number of states, x dim, y dim, scalar, len of a single state] ->  [38, 32, 30, 1, 1]
'''

input_set = input_set.permute(0,2,3,4,1) # ([1000, 64, 128, 1, 1])


### Normalization
n_samples = len(input_set)
train_fraction = 0.8
generator = torch.Generator().manual_seed(0)
indices = torch.randperm(n_samples, generator=generator)
n_train = int(train_fraction * n_samples)
train_idx = indices[:n_train]
test_idx = indices[n_train:]

x_train = input_set[train_idx]
y_train = output_set[train_idx]
x_test = input_set[test_idx]
y_test = output_set[test_idx]

x_mean = x_train.mean()
x_std = x_train.std()
y_mean = y_train.mean()
y_std = y_train.std()

x_train_norm = (x_train - x_mean) / x_std
y_train_norm = (y_train - y_mean) / y_std
x_test_norm = (x_test - x_mean) / x_std
y_test_norm = (y_test - y_mean) / y_std

train_dataset = torch.utils.data.TensorDataset(x_train_norm, y_train_norm)
test_dataset = torch.utils.data.TensorDataset(x_test_norm, y_test_norm)


######### Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
#print(type(train_loader)) # torch.utils.data.dataloader.DataLoader
#print(len(train_loader), len(test_loader)) # len(train_dataset) // batch_size = len(train_loader)



if __name__ == "__main__":
    ################################################################
    # training and evaluation
    ################################################################
    model = FNO2d(modes, modes, width, nlayer, T_in, var, T_out, share_weight, factor, ff_weight_norm, n_ff_layers, layer_norm).to(device)
    #print(count_params(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_value)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    train_pre = []; train_recon = []
    test_pre = []; test_recon = []


    #myloss = LpLoss()  # loss function, need utilize3 file
    myloss = torch.nn.MSELoss()
    #myloss = torch.nn.MSELoss(reduction='mean')


    save_path = "/home/zliu2/UiO_Life_Science/uq4downscaling/NOs/F-IFNO/weights/t2m_1985_1000_days/"

    for ep in range(epochs):
        print(f"Training epoch {ep+1}")
        model.train()
        t1 = default_timer()
        train_recons_full = 0
        train_pred_full = 0
        for xx, yy in train_loader:
            l_recons = 0
            bs = xx.shape[0] # batchsize
            xx = xx.to(device) # ([batchsize, 32, 64, 1, 1])
            yy = yy.to(device) # ([batchsize, 64, 128, 1])
            pre, im_re = model(xx)  # ([batchsize, 32, 64, 1, 1]),  ([batchsize, 32, 64, 1, 1])

            #print("Model prediction made")
            l_recons = myloss(im_re.reshape(bs, -1), xx.reshape(bs, -1)) #[BS, 64, 128, 1, T_in]
            l_pred = myloss(pre.reshape(bs, -1), yy.reshape(bs, -1))    #[BS, 64, 128, 1, T_out]
            loss = alpha*l_pred + beta*l_recons 

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
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
        reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert to MB

        if ep == 0:
            print("Epoch,","Time,","[Train Recons MSE],","[Train Pred MSE],","[Test Recons MSE],","[Test Pred MSE]","[Allocated Memory]","[Reserved Memory]")
        print(ep, "%.2f"%(t2-t1), "%.6f"%(train_recons_full), "%.6f"%(train_pred_full), 
            "%.6f"%(test_recons_full), "%.6f"%(test_pred_full), "%.2f"%(allocated_memory), "%.2f"%(reserved_memory))
        torch.save(model.state_dict(), f'{save_path}F_IFNO_t2m_1985_1000_ep{ep+1}.pth')  


    #MSE_save = np.dstack((train_pre, train_recon,test_pre,test_recon)).squeeze()
    #np.savetxt('./loss_data_new.dat', MSE_save, fmt="%16.7f")

    #torch.save(model.state_dict(), 'm12w90n40g45_30ep1e8w.pth')  # 
