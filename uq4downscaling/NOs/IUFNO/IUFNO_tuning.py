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
#from feedforward import FeedForward
#from linear import WNLinear
import xarray as xr

from config_IUFNO import Config, load_datasets, normalize_sets
from IUFNO_2D import SpectralConv2d, FNO2d, tune_model



"""Tuning hyper-parameters of F-IFNO
Hyper-parameters that are fixed: '
    T_in, T_out: Fixed as 1
    -----------------------------------------------------------
    var_in: Fixed
    -----------------------------------------------------------'
    var_out: Fixed
    -----------------------------------------------------------
    epochs: Sets to 40. Pick the best performing ones
    -----------------------------------------------------------
    batch_size: Sets to 5, since the paper chose it
    -----------------------------------------------------------
    scheduler_step: Sets to 5, since the paper chose it
    -----------------------------------------------------------
    scheduler_gamma: Sets to 0.5, since the paper chose it
    -----------------------------------------------------------
    factor: Sets to 4 due to the paper, or set to 1 for 
        experiment relevance
    -----------------------------------------------------------
    weight_decay: Fixed
    -----------------------------------------------------------
    learning_rate : Fixed
    -----------------------------------------------------------
    scheduler_step : Fixed
    -----------------------------------------------------------
Hyper-parameters to tune: 
    modes
    -----------------------------------------------------------
    width
    -----------------------------------------------------------
    nlayer
    -----------------------------------------------------------
"""

############################################################
#################### Tuning values ####################
#modes_lst = [10, 12, 16, 20, 30]
#width_lst = [10, 20, 40, 60, 80, 90]
#nlayer_lst = [20, 40, 60, 80]



modes_lst = [2]
width_lst = [2]
sched_step_lst = [5]


train_years = [1990, 1991, 2006]
val_years = [1994]
test_years = [2000]




folder_path = "/home/zliu2/UiO_Life_Science/uq4downscaling/Datasets/Weatherbench/mini_sample_1/"
folder_save_MSE = ""
epochs_tuning = 2


if __name__ == "__main__":
    print("IFNO tuning")

    for m in modes_lst:
        for w in width_lst:
            for s in sched_step_lst:
                curr_setup = Config()
                curr_setup.modes=m; curr_setup.width=w; curr_setup.scheduler_step=s
                tune_model(curr_setup, train_years, val_years, test_years,
                           folder_path, folder_save_MSE, epochs_tuning=epochs_tuning)
    print("\n-------------------- TUNING DONE --------------------\n")