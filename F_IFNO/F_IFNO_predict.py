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

from config_F_IFNO import Config, load_dataset, load_dataset_train
from F_IFNO_2D import FNO2d, predict_model, plot_predicted_sample



if __name__=="__main__":
    print("\nF-IFNO predict model\n")

    #################### years, setup parameters, folder for the weights 
    test_years = [2015,2016,2017,2018]
    test_year_pred = [test_years[1]]   # 2016


    ########## Initialize model
    setup_pred = Config()


    #################### Find mean and std from training set
    folder_path = "/home/zliu2/life_science/Weatherbench/"
    #train_years = [1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006]
    #trained_mean_std = mean_std_train(folder_path, train_years, setup_pred.var_in_set, setup_pred.var_out_set)

    trained_mean_std = torch.load("mean_std_lst.pth", map_location="cpu", weights_only=True)
    setup_pred.mean_std_lst = trained_mean_std


    #################### Make testloader and prediction
    folder_path_pred = "/home/zliu2/life_science/Weatherbench/prediction_set/"
    folder_weight_path = "/home/zliu2/life_science/F-IFNO/weights/m12_w90_n40/"
    test_loader = load_dataset(folder_path_pred, test_year_pred, setup_pred.batch_size, 
        setup_pred.var_in_set, setup_pred.var_out_set, setup_pred.mean_std_lst, shuffle_set=False)

    chosen_epoch = 43

    try: filename_weight = f"F_IFNO_t2m_{chosen_epoch}.pth"
    except: raise FileNotFoundError(f"Chosen epoch doesn't exist")

    
    #################### Make model predictions and targets ####################
    originals, predictions, targets = predict_model(setup_pred, test_year_pred, folder_path_pred, 
        folder_weight_path, filename_weight, shuffle_set=False, is_testloader=test_loader)


    #################### Plot a predicted vs target sample ####################
    sample_id = 1
    plot_predicted_sample(originals, predictions, targets, sample_id, chosen_epoch)
    
    


    """
    path = ("/fp/projects01/ec35/ec-zhenwl/F-IFNO/weights/small_test/F_IFNO_t2m_1.pth")
    weights = torch.load(path, map_location="cpu")
    print("Checkpoint loaded successfully.")
    print("Number of state entries:", len(weights))
    """

