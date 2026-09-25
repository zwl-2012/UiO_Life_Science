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

from config_F_IUFNO import Config, load_dataset, load_dataset_train
from F_IUFNO_2D import FNO2d, predict_model, plot_predicted_sample



if __name__=="__main__":
    print("\nF-IUFNO predict model\n")

    #################### years, setup parameters, folder for the weights 
    test_years = [2015,2016,2017,2018]
    test_year_pred = [test_years[1]]   # 2016


    ########## Initialize model
    setup_pred = Config()


    #################### Find mean and std from training set
    folder_path = "/home/zliu2/life_science/Weatherbench/"

    trained_mean_std = torch.load("mean_std_lst.pth", map_location="cpu", weights_only=True)
    setup_pred.mean_std_lst = trained_mean_std


    #################### Make testloader and prediction
    folder_path_pred = "/home/zliu2/life_science/Weatherbench/prediction_set/"
    folder_weight_path = "/home/zliu2/life_science/F-IUFNO/weights/m12_w90_n40/"
    test_loader = load_dataset(folder_path_pred, test_year_pred, setup_pred.batch_size, 
        setup_pred.var_in_set, setup_pred.var_out_set, setup_pred.mean_std_lst, shuffle_set=False)

    chosen_epoch = 6

    try: filename_weight = f"F_IUFNO_t2m_{chosen_epoch}.pth"
    except: raise FileNotFoundError(f"Chosen epoch doesn't exist")


    #################### Make model predictions and targets ####################
    originals, predictions, targets = predict_model(setup_pred, test_year_pred, folder_path_pred, 
        folder_weight_path, filename_weight, shuffle_set=False, is_testloader=test_loader)


    #################### Plot a predicted vs target sample ####################
    sample_id = 1
    plot_predicted_sample(originals, predictions, targets, sample_id, chosen_epoch)
    
    
