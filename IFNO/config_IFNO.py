"""config.py file contains several files for the model project: 
Config: a collection of the initial setups
-----------------------------------------------------------------------------
load_datasets: takes a dataset directory path, training, validation and test years
    and makes dataloaders by opening the files, sorting and normalizing
-----------------------------------------------------------------------------
normalize_sets: normalize datasets using the mean and std derived from the training set
"""


import torch; import numpy as np
import torch.nn as nn; import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from utilities3 import *
import operator
from functools import reduce; from functools import partial
from timeit import default_timer
import scipy.io
import os
from einops import rearrange

import xarray as xr
from dataclasses import dataclass



@dataclass
class Config:
    ########## IFNO
    """ 
    modes: int
        Number of Fourier coefficients in the spectral transformation 
        (using FFT on the image domain)
        Just like Fourier series, the more coefficients, the more accurate
        more features get extracted/retained
    -------------------------------------------------------------------------
    width: int
        Number of hidden channels 
    -------------------------------------------------------------------------
    nlayer: int
    -------------------------------------------------------------------------
    T_in: int
        How many input steps ahead
    -------------------------------------------------------------------------
    T_out: int
        How many output steps ahead
    -------------------------------------------------------------------------
    var_in: int
        Number of input variables
    -------------------------------------------------------------------------
    var_out: int
        Number of output variables
    -------------------------------------------------------------------------
    """
    modes: int = 12
    width: int = 90
    nlayer: int = 40

    T_in: int = 1 
    T_out: int = 1
    var_in: int = 3
    var_out: int = 1


    ##### Hyper-parameters
    """ 
    epochs: int
        Number of model training sessions to update the weights
    -------------------------------------------------------------------------
    batch_size: int
        Number of samples before each weight update
    -------------------------------------------------------------------------
    learning_rate: float
    -------------------------------------------------------------------------
    weight_decay: float
    -------------------------------------------------------------------------
    scheduler_step: int
        if scheduler_step=5 --> learning_rate updated every 5 epochs
    -------------------------------------------------------------------------
    scheduler_gamma: float
        if scheduler_gamma=0.5 --> learning_rate updated with
        learning_rate = learning_rate * 0.5  for each learning_rate update
    -------------------------------------------------------------------------
    """

    epochs: int = 40
    batch_size: int = 5
    learning_rate: float = 0.0005 # 0.0005
    weight_decay: float = 1e-8
    scheduler_step: int = 5
    scheduler_gamma: float = 0.5
    alpha: float = 1.0


    ##### Mean and std from training set
    mean_std_lst: list = None


    ##### Store the relevant variables
    var_in_set = {"t2m": f"2m_temperature_5.625deg/2m_temperature_year_5.625deg.nc", 
            "u10": f"10m_u_component_of_wind_5.625deg/10m_u_component_of_wind_year_5.625deg.nc", 
            "v10": f"10m_v_component_of_wind_5.625deg/10m_v_component_of_wind_year_5.625deg.nc"}

    var_out_set = {"t2m": f"2m_temperature_2.8125deg/2m_temperature_year_2.8125deg.nc"}



    def __str__(self):
        overview = f"++++++++++ F-IFNO ++++++++++\n" \
            f"Modes={self.modes} | width={self.width} | nlayer={self.nlayer}\n" \
            f"Input variables = {self.var_in} | Output variables = {self.var_out}\n" \
            f"Epochs={self.epochs} | Batch size={self.batch_size}\n" \
            f"Learning rate={self.learning_rate} | Weight decay={self.weight_decay}\n" \
            f"Scheduler step={self.scheduler_step} | Gamma={self.scheduler_gamma}\n" \
            f"++++++++++++++++++++++++++++++\n"
        return overview



################################################################################
########## Load dataset --> make dataloader for training set ##########
########## Load only training set ###########
################################################################################
def load_dataset_train(dir_path, data_years, batch_size, var_in_set, var_out_set):
    """ INPUTS
    dir_path : str
        Path of folder/directory where the datasets are contained
    ------------------------------------------------------------------------
    data_years : list[int] --> int
        List of selected years 
    ------------------------------------------------------------------------
    batch_size : int
        Batch size for training the model
    ------------------------------------------------------------------------
    var_in_set : dict[str] --> str
        A dictionary where key = variable_name, 
        value = dataset filename with corresponding variable
    ------------------------------------------------------------------------
    var_out_set : dict[str] --> str
        A dictionary where key = variable_name,
        value = dataset filename with corresponding variable
    ------------------------------------------------------------------------


    OUTPUTS
    train_loader : Dataloader
    ------------------------------------------------------------------------
    mean_std_lst : list
        Contains the mean and std for training inputs and output
    ------------------------------------------------------------------------"""

    
    #################### load selected dataset ####################
    for year in data_years:
        ########## input set ##########
        in_lst = []
        
        for variable in var_in_set:
            curr_in_lst = []
            curr_path_in_file = var_in_set[variable].replace("year", f"{year}")
            curr_path_in = f"{dir_path}{curr_path_in_file}"
            print(f"\nOpen file: {curr_path_in}\n")
            curr_in_ds = xr.open_dataset(curr_path_in)
            curr_in_ds = curr_in_ds[variable].values
            curr_in_ds = torch.from_numpy(curr_in_ds) # ([8760, 32, 64])
            curr_in_ds = curr_in_ds[:, np.newaxis, ...] # ([8760, 1, 32, 64])

            ##### Interpolate input from 32x64 to 64x128
            curr_in_ds = F.interpolate(curr_in_ds, scale_factor=2, mode="bilinear", 
                                       align_corners=False) # [8760, 1, 64, 128]
            curr_in_ds = curr_in_ds[..., np.newaxis] # ([8760, 1, 64, 128, 1])

            for i in range(len(curr_in_ds)):
                curr_in_lst.append(curr_in_ds[i, ...])
            curr_in_set = torch.stack(curr_in_lst)  # ([8760, 1, 64, 128, 1])
            in_lst.append(curr_in_set)
        in_set = torch.cat(in_lst, dim=1) # ([8760, var_in, 64, 128, 1])
        in_set = in_set.permute(0,2,3,1,4) # ([8760, 64, 128, var_in, 1])


    for year in data_years:
        ########## output set ##########
        out_lst = []; out_var = ""
        for key in var_out_set: out_var = key 
        curr_path_out_file = var_out_set[out_var].replace("year", f"{year}")
        curr_path_out = f"{dir_path}{curr_path_out_file}"
        print(f"\nOpen file: {curr_path_out}\n")
        curr_out_ds = xr.open_dataset(curr_path_out)
        curr_out_ds = curr_out_ds[out_var].values
        curr_out_ds = torch.from_numpy(curr_out_ds) # ([8760, 64, 128])
        curr_out_ds = curr_out_ds[..., np.newaxis] # ([8760, 64, 128, 1])
        for i in range(len(curr_in_ds)):
            out_lst.append(curr_out_ds[i, ...])
        out_set = torch.stack(out_lst) # ([8760, 64, 128, 1])


    #################### Normalization ####################
    ########## mean and std from training set
    in_train_mean = in_set.mean(dim=(0,1,2,4), keepdim=True)  # ([1, 1, 1, var_in, 1])
    in_train_std = in_set.std(dim=(0,1,2,4), keepdim=True)    # ([1, 1, 1, var_in, 1])
    out_train_mean = out_set.mean()   # since we only have one target variable
    out_train_std = out_set.std()     # since we only have one target variable

    mean_std_lst = [in_train_mean, in_train_std, out_train_mean, out_train_std]
    in_train_norm, out_train_norm = normalize_sets(in_set, out_set, 
            mean_std_lst[0], mean_std_lst[1], mean_std_lst[2], mean_std_lst[3])

    
    #################### Make dataloaders ####################
    train_dataset = torch.utils.data.TensorDataset(in_train_norm, out_train_norm)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size, shuffle=True)

    return train_loader, mean_std_lst





################################################################################
########## Load dataset --> make dataloader for NON-training set ##########
########## Load only NON-training set ###########
################################################################################
def load_dataset(dir_path, data_years, batch_size, var_in_set, var_out_set, mean_std_lst,
                 shuffle_set=True):
    """ INPUTS
    dir_path : str
        Path of folder/directory where the datasets are contained
    ------------------------------------------------------------------------
    data_years : list[int] --> int
        List of selected years 
    ------------------------------------------------------------------------
    batch_size : int
        Batch size for training the model
    ------------------------------------------------------------------------
    var_in_set : dict[str] --> str
        A dictionary where key = variable_name, 
        value = dataset filename with corresponding variable
    ------------------------------------------------------------------------
    var_out_set : dict[str] --> str
        A dictionary where key = variable_name,
        value = dataset filename with corresponding variable
    ------------------------------------------------------------------------
    mean_std_lst : list
        Contains the mean and std from training inputs and output
    ------------------------------------------------------------------------


    OUTPUTS
    loader : Dataloader
    ------------------------------------------------------------------------"""

    
    #################### load selected dataset ####################
    for year in data_years:
        ########## input set ##########
        in_lst = []
        
        for variable in var_in_set:
            curr_in_lst = []
            curr_path_in_file = var_in_set[variable].replace("year", f"{year}")
            curr_path_in = f"{dir_path}{curr_path_in_file}"
            print(f"\nOpen file: {curr_path_in}\n")
            curr_in_ds = xr.open_dataset(curr_path_in)
            curr_in_ds = curr_in_ds[variable].values
            curr_in_ds = torch.from_numpy(curr_in_ds) # ([8760, 32, 64])
            curr_in_ds = curr_in_ds[:, np.newaxis, ...] # ([8760, 1, 32, 64])

            ##### Interpolate input from 32x64 to 64x128
            curr_in_ds = F.interpolate(curr_in_ds, scale_factor=2, mode="bilinear", 
                                       align_corners=False) # [8760, 1, 64, 128]
            curr_in_ds = curr_in_ds[..., np.newaxis] # ([8760, 1, 64, 128, 1])

            for i in range(len(curr_in_ds)):
                curr_in_lst.append(curr_in_ds[i, ...])
            curr_in_set = torch.stack(curr_in_lst)  # ([8760, 1, 64, 128, 1])
            in_lst.append(curr_in_set)
        in_set = torch.cat(in_lst, dim=1) # ([8760, var_in, 64, 128, 1])
        in_set = in_set.permute(0,2,3,1,4) # ([8760, 64, 128, var_in, 1])


    for year in data_years:
        ########## output set ##########
        out_lst = []; out_var = ""
        for key in var_out_set: out_var = key 
        curr_path_out_file = var_out_set[out_var].replace("year", f"{year}")
        curr_path_out = f"{dir_path}{curr_path_out_file}"
        print(f"\nOpen file: {curr_path_out}\n")
        curr_out_ds = xr.open_dataset(curr_path_out)
        curr_out_ds = curr_out_ds[out_var].values
        curr_out_ds = torch.from_numpy(curr_out_ds) # ([8760, 64, 128])
        curr_out_ds = curr_out_ds[..., np.newaxis] # ([8760, 64, 128, 1])
        for i in range(len(curr_in_ds)):
            out_lst.append(curr_out_ds[i, ...])
        out_set = torch.stack(out_lst) # ([8760, 64, 128, 1])


    #################### Normalization ####################
    in_norm, out_norm = normalize_sets(in_set, out_set, 
            mean_std_lst[0], mean_std_lst[1], mean_std_lst[2], mean_std_lst[3])

    
    #################### Make dataloader ####################
    dataset = torch.utils.data.TensorDataset(in_norm, out_norm)
    loader = torch.utils.data.DataLoader(dataset=dataset, 
                                         batch_size=batch_size, shuffle=shuffle_set)

    return loader

















def load_datasets(dir_path, train_years, val_years, test_years, batch_size):
    #################### Make training, validation and test sets
    in_vars_lst = ["t2m", "u10", "v10"]
    #out_vars_lst = ["t2m"]

    ############################################################
    #################### Training set ####################
    for year in train_years:
        ########## training inputs ##########
        in_train_lst = []
        in_vars = {"t2m":f"2m_temperature_{year}_5.625deg.nc", 
                    "u10":f"10m_u_component_of_wind_{year}_5.625deg.nc", 
                    "v10":f"10m_v_component_of_wind_{year}_5.625deg.nc"}
        
        for variable in in_vars_lst:
            curr_in_lst = []
            curr_path_in = f"{dir_path}{in_vars[variable]}"
            curr_in_ds = xr.open_dataset(curr_path_in)
            curr_in_ds = curr_in_ds[variable].values
            curr_in_ds = torch.from_numpy(curr_in_ds) # ([8760, 32, 64])
            curr_in_ds = curr_in_ds[:, np.newaxis, ...] # ([8760, 1, 32, 64])

            ##### Interpolate input from 32x64 to 64x128
            curr_in_ds = F.interpolate(curr_in_ds, scale_factor=2, mode="bilinear", 
                                       align_corners=False) # [8760, 1, 64, 128]
            curr_in_ds = curr_in_ds[..., np.newaxis] # ([8760, 1, 64, 128, 1])

            for i in range(len(curr_in_ds)):
                curr_in_lst.append(curr_in_ds[i, ...])
            curr_in_set = torch.stack(curr_in_lst)  # ([8760, 1, 64, 128, 1])
            in_train_lst.append(curr_in_set)
        in_train_set = torch.cat(in_train_lst, dim=1) # ([8760, var_in, 64, 128, 1])
        in_train_set = in_train_set.permute(0,2,3,1,4) # ([8760, 64, 128, var_in, 1])
        

    for year in train_years:
        ########## training outputs ##########
        out_train_lst = []
        out_vars = {"t2m":f"2m_temperature_{year}_2.8125deg.nc"}
        curr_path_out = f"{dir_path}{out_vars['t2m']}"
        curr_out_ds = xr.open_dataset(curr_path_out)
        curr_out_ds = curr_out_ds["t2m"].values
        curr_out_ds = torch.from_numpy(curr_out_ds) # ([8760, 64, 128])
        curr_out_ds = curr_out_ds[..., np.newaxis] # ([8760, 64, 128, 1])
        for i in range(len(curr_in_ds)):
            out_train_lst.append(curr_out_ds[i, ...])
        out_train_set = torch.stack(out_train_lst) # ([8760, 64, 128, 1])


    
    ############################################################
    #################### Validation set ####################
    for year in val_years:
        ########## validation inputs ##########
        in_val_lst = []
        in_vars = {"t2m":f"2m_temperature_{year}_5.625deg.nc", 
                    "u10":f"10m_u_component_of_wind_{year}_5.625deg.nc", 
                    "v10":f"10m_v_component_of_wind_{year}_5.625deg.nc"}
        
        for variable in in_vars_lst:
            curr_in_lst = []
            curr_path_in = f"{dir_path}{in_vars[variable]}"
            curr_in_ds = xr.open_dataset(curr_path_in)
            curr_in_ds = curr_in_ds[variable].values
            curr_in_ds = torch.from_numpy(curr_in_ds) # ([8760, 32, 64])
            curr_in_ds = curr_in_ds[:, np.newaxis, ...] # ([8760, 1, 32, 64])

            ##### Interpolate input from 32x64 to 64x128
            curr_in_ds = F.interpolate(curr_in_ds, scale_factor=2, mode="bilinear", 
                                       align_corners=False) # [8760, 1, 64, 128]
            curr_in_ds = curr_in_ds[..., np.newaxis] # ([8760, 1, 64, 128, 1])

            for i in range(len(curr_in_ds)):
                curr_in_lst.append(curr_in_ds[i, ...])
            curr_in_set = torch.stack(curr_in_lst)  # ([8760, 1, 64, 128, 1])
            in_val_lst.append(curr_in_set)
        in_val_set = torch.cat(in_val_lst, dim=1) # ([8760, var_in, 64, 128, 1])
        in_val_set = in_val_set.permute(0,2,3,1,4) # ([8760, 64, 128, var_in, 1])

    for year in val_years:
        ########## validation outputs ##########
        out_val_lst = []
        out_vars = {"t2m":f"2m_temperature_{year}_2.8125deg.nc"}
        curr_path_out = f"{dir_path}{out_vars['t2m']}"
        curr_out_ds = xr.open_dataset(curr_path_out)
        curr_out_ds = curr_out_ds["t2m"].values
        curr_out_ds = torch.from_numpy(curr_out_ds) # ([8760, 64, 128])
        curr_out_ds = curr_out_ds[..., np.newaxis] # ([8760, 64, 128, 1])
        for i in range(len(curr_in_ds)):
            out_val_lst.append(curr_out_ds[i, ...])
        out_val_set = torch.stack(out_val_lst) # ([8760, 64, 128, 1])


    ############################################################
    #################### Testing set ####################
    for year in test_years:
        ########## testing inputs ##########
        in_test_lst = []
        in_vars = {"t2m":f"2m_temperature_{year}_5.625deg.nc", 
                    "u10":f"10m_u_component_of_wind_{year}_5.625deg.nc", 
                    "v10":f"10m_v_component_of_wind_{year}_5.625deg.nc"}
        
        for variable in in_vars_lst:
            curr_in_lst = []
            curr_path_in = f"{dir_path}{in_vars[variable]}"
            curr_in_ds = xr.open_dataset(curr_path_in)
            curr_in_ds = curr_in_ds[variable].values
            curr_in_ds = torch.from_numpy(curr_in_ds) # ([8760, 32, 64])
            curr_in_ds = curr_in_ds[:, np.newaxis, ...] # ([8760, 1, 32, 64])

            ##### Interpolate input from 32x64 to 64x128
            curr_in_ds = F.interpolate(curr_in_ds, scale_factor=2, mode="bilinear", 
                                       align_corners=False) # [8760, 1, 64, 128]
            curr_in_ds = curr_in_ds[..., np.newaxis] # ([8760, 1, 64, 128, 1])

            for i in range(len(curr_in_ds)):
                curr_in_lst.append(curr_in_ds[i, ...])
            curr_in_set = torch.stack(curr_in_lst)  # ([8760, 1, 64, 128, 1])
            in_test_lst.append(curr_in_set)
        in_test_set = torch.cat(in_test_lst, dim=1) # ([8760, var_in, 64, 128, 1])
        in_test_set = in_test_set.permute(0,2,3,1,4) # ([8760, 64, 128, var_in, 1])

    for year in test_years:
        ########## testing outputs ##########
        out_test_lst = []
        out_vars = {"t2m":f"2m_temperature_{year}_2.8125deg.nc"}
        curr_path_out = f"{dir_path}{out_vars['t2m']}"
        curr_out_ds = xr.open_dataset(curr_path_out)
        curr_out_ds = curr_out_ds["t2m"].values
        curr_out_ds = torch.from_numpy(curr_out_ds) # ([8760, 64, 128])
        curr_out_ds = curr_out_ds[..., np.newaxis] # ([8760, 64, 128, 1])
        for i in range(len(curr_in_ds)):
            out_test_lst.append(curr_out_ds[i, ...])
        out_test_set = torch.stack(out_test_lst) # ([8760, 64, 128, 1])



    ############################################################
    #################### Normalization ####################
    ########## mean and std from training set
    in_train_mean = in_train_set.mean(dim=(0,1,2,4), keepdim=True) # ([1, 1, 1, var_in, 1])
    in_train_std = in_train_set.std(dim=(0,1,2,4), keepdim=True) # ([1, 1, 1, var_in, 1])
    out_train_mean = out_train_set.mean() # since var_out=1
    out_train_std = out_train_set.std()

    in_train_norm, out_train_norm = normalize_sets(in_train_set, out_train_set, 
                        in_train_mean, in_train_std, out_train_mean, out_train_std)

    in_val_norm, out_val_norm = normalize_sets(in_val_set, out_val_set, 
                        in_train_mean, in_train_std, out_train_mean, out_train_std)

    in_test_norm, out_test_norm = normalize_sets(in_test_set, out_test_set, 
                        in_train_mean, in_train_std, out_train_mean, out_train_std)


    ############################################################
    #################### Make dataloaders ####################
    train_dataset = torch.utils.data.TensorDataset(in_train_norm, out_train_norm)
    val_dataset = torch.utils.data.TensorDataset(in_val_norm, out_val_norm)
    test_dataset = torch.utils.data.TensorDataset(in_test_norm, out_test_norm)

    ########## dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader





def normalize_sets(inputs, outputs, input_mean, input_std, output_mean, output_std):
    inputs_norm = (inputs - input_mean) / input_std
    outputs_norm = (outputs - output_mean) / output_std
    return inputs_norm, outputs_norm



########## Load original 32x64 t2m data
def original_data_load(dir_path, years, variable="t2m"):
    ds_all = []
    for year in years:
        dir_file = dir_path + f"2m_temperature_5.625deg/2m_temperature_{year}_5.625deg.nc"
        curr_in_ds = xr.open_dataset(dir_file)
        curr_in_ds = curr_in_ds[variable].values
        curr_in_ds = torch.from_numpy(curr_in_ds) # ([8760, 32, 64])
        curr_in_ds = curr_in_ds[:, np.newaxis, ...] # ([8760, 1, 32, 64])
        ds_all.append(curr_in_ds)
    ds_all = torch.cat(ds_all, dim=0)
    ds_all = ds_all.numpy()
    return ds_all

