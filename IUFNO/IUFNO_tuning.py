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

from config_IUFNO import Config, normalize_sets, load_dataset_train, load_dataset
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

#modes_lst = [12, 14]
#width_lst = [90, 100]
#train_years = [1979, 1990]
#al_years = [1994]

train_years = [1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006]
val_years = [2007,2008,2009,2010,2011,2012,2013,2014]

if __name__ == "__main__":
    print("IUFNO tuning")
    folder_path = "/home/zliu2/life_science/Weatherbench/"
    folder_save_MSE = ""
    curr_setup = Config()
    curr_setup.modes, curr_setup.width, epochs_tuning = 12, 90, 50
    save_folder_weight = f"/home/zliu2/life_science/IUFNO/weights/" \
                        f"m{curr_setup.modes}_w{curr_setup.width}_n40/"

    try:
        tune_model(curr_setup, train_years, val_years, folder_path, 
                save_folder_weight, folder_save_MSE, epochs_tuning=epochs_tuning)
    except Exception as exc:
        print(f"TUNING ERROR: {type(exc).__name__}: {exc}", flush=True)
        raise

    print("\n-------------------- TUNING DONE --------------------\n")
