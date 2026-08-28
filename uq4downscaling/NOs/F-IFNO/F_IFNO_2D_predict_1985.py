
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

from F_IFNO_share_decon_2D_downscale import FNO3d, test_loader, y_mean, y_std


########## configs
################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------------------------------------------------------------------------
#tunning3
modes = 10
width = 10 #3n
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

################################################################
######## 
################################################################
model_predict = FNO3d(modes, modes, width, nlayer, T_in, var, T_out, share_weight, factor, ff_weight_norm, n_ff_layers, layer_norm).to(device)

weight_pth = f"/home/zliu2/UiO_Life_Science/uq4downscaling/NOs/F-IFNO/weights/t2m_1985_1000_days/F_IFNO_t2m_1985_1000_ep{epochs}.pth"
checkpoint = torch.load(weight_pth, map_location=device)
#print(checkpoint.keys())
model_predict.load_state_dict(checkpoint)
model_predict.eval()


predictions_all = []; targets_all = []

with torch.no_grad():
    for pre, tar in test_loader:
        """test_loader contains normalized values"""
        pre = pre.to(device) # torch.Size([8, 64, 128, 1, 1])
        tar = tar.to(device) # torch.Size([8, 64, 128, 1])

        predicted, im_re = model_predict(pre)

        ## Denormalize original data
        predicted_original = predicted * y_std + y_mean
        target_original = tar * y_std + y_mean

        predictions_all.append(predicted_original)
        targets_all.append(target_original)


### Merge
predictions_all = torch.cat(predictions_all, dim=0) # ([20, 64, 128, 1, 1])
predictions_all = predictions_all.permute(0,4,1,2,3) # [20, 1, 64, 128, 1]
targets_all = torch.cat(targets_all, dim=0) # ([20, 64, 128, 1])
predictions_all = predictions_all.numpy() # (20, 1, 64, 128, 1)
targets_all = targets_all.numpy() # (20, 64, 128, 1)
predictions_all = predictions_all.squeeze(-1) # (20, 1, 64, 128)
targets_all = targets_all.squeeze(-1) # (20, 64, 128)
targets_all = targets_all[:, np.newaxis, ...] # (20, 1, 64, 128)


### Plot the prediction vs target
sample_id = 10

predict_sample = predictions_all[sample_id, 0]
target_sample = targets_all[sample_id, 0]


fig, axes = plt.subplots(1,2,figsize=(15,4))

im_0 = axes[0].imshow(predict_sample, cmap="coolwarm")
axes[0].set_title(f"Prediction | epochs={epochs}")
fig.colorbar(im_0, ax=axes[0])

im_1 = axes[1].imshow(target_sample, cmap="coolwarm")
axes[1].set_title("Target")
fig.colorbar(im_1, ax=axes[1])
plt.tight_layout(); plt.show()
