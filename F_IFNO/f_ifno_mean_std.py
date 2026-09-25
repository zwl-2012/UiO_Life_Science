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

from config_F_IFNO import Config, load_dataset, load_dataset_train, mean_std_train
from F_IFNO_2D import FNO2d, predict_model, plot_predicted_sample



if __name__=="__main__":
    print("\nF-IFNO predict model\n")

    
    ########## Initialize model
    setup_pred = Config()


    #################### Find mean and std from training set
    folder_path = "/home/zliu2/life_science/Weatherbench/"
    train_years = [1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006]
    trained_mean_std = mean_std_train(folder_path, train_years, setup_pred.var_in_set, setup_pred.var_out_set)
    #setup_pred.mean_std_lst = trained_mean_std


    ##### Analyze the mean and std from training set
    print(type(trained_mean_std))
    print(type(trained_mean_std[0]))
    print(type(trained_mean_std[2]))
    print(trained_mean_std)

    with open("mean_std_lst.pth", "wb") as f: torch.save(trained_mean_std, f); print("File saved")
    

    """
    mean_std_load = torch.load("mean_std_lst.pth", map_location="cpu", weights_only=True)
    print(type(mean_std_load))
    print(type(mean_std_load[0]))
    print(type(mean_std_load[2]))
    print(mean_std_load)
    """
