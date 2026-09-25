
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utilities3 import *
import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io
import os

from config_IFNO import Config, load_dataset_train, load_dataset, original_data_load


"""A collection of functions for IFNO"""


################################################################
#################### 4d fourier layers
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))


    ########## Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y,z,t ), (in_channel, out_channel, x,y,z,t) -> (batch, out_channel, x,y,z,t)
        #return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
        return torch.einsum("bixy,ioxy->boxy", input, weights)


    def forward(self, x):
        batchsize = x.shape[0] 
        ########## Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-2,-1])

        ########## Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:] = self.compl_mul2d(x_ft[:, :, :self.modes1, -self.modes2:], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:] = self.compl_mul2d(x_ft[:, :, -self.modes1:, -self.modes2:], self.weights4)

        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)))
        return x




class FNO2d(nn.Module):
    #def __init__(self, modes1, modes2, width, nlayer, T_in, var, T_out): 
    def __init__(self, setup):
        super(FNO2d, self).__init__()

        self.modes1 = setup.modes
        self.width = setup.width
        self.T_in = setup.T_in
        self.var_in = setup.var_in
        self.T_out = setup.T_out
        self.var_out = setup.var_out
        
        self.nlayer = setup.nlayer
        
        self.convlayer = nn.ModuleList([SpectralConv2d(self.width, self.width, 
                self.modes1, self.modes1).to(device) for i in range(1)])
        
        self.w = nn.ModuleList([nn.Conv2d(self.width, self.width, 1).to(device) for i in range(1)])
        
        self.enc = nn.Conv2d(self.var_in*self.T_in, self.width, 1) # encoder
        self.dec = nn.Conv2d(self.width, self.var_out*self.T_out, 1) # decoder
        

    def forward(self, x):  
        batchsize, size_x, size_y, var_in, T_in = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        coef = 1./self.nlayer
        
        ########## Reconstruct ##########
        x = x.reshape(batchsize, size_x, size_y, var_in*T_in)
        x = x.permute(0,3,1,2)
        
        ########## Predict ##########
        x = self.enc(x) # Encoder # [BS,width,32,32,32]
        x = torch.tanh(x)

        for i in range(self.nlayer):
            x1 = self.convlayer[0](x) #[BS,width,32,32,32]
            x2 = self.w[0](x)  #[BS,width,32,32,32]
            x = torch.tanh(x1+x2)*coef + x   #[BS,width,32,32,32]
    
        x = self.dec(x) # Decoder   #[BS,width,32,32,32]---->#[BS,var*T_out,32,32,32]
        x = x.permute(0,2,3,1) #[BS,32,32,32,,var*T_out]
        x = x.reshape(batchsize, size_x, size_y, self.var_out, self.T_out)
        return x








################################################################
#################### Tune model ##############################
################################################################
def tune_model(setup, train_years, val_years, folder_path, save_folder_weight, folder_save_MSE, 
               epochs_tuning=60):
    #################### Device ####################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #################### Setup ####################
    epochs_tuning = epochs_tuning

    ################### Define training, validation and test years ####################
    
    #################### Load data ####################
    ##### dir_path the folder path for both input and output datasets

    ##### Make Dataloaders
    train_loader, mean_std_lst = load_dataset_train(folder_path, train_years, setup.batch_size,
                                        setup.var_in_set, setup.var_out_set)
    setup.mean_std_lst = mean_std_lst

    val_loader = load_dataset(folder_path, val_years, setup.batch_size, 
                setup.var_in_set, setup.var_out_set, setup.mean_std_lst)

    model = FNO2d(setup); model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=setup.learning_rate, weight_decay=setup.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=setup.scheduler_step, gamma=setup.scheduler_gamma)
    myloss = torch.nn.MSELoss()
    train_pre = []; val_pre = []  # might want to store the losses for each epoch

    epoch_min = 0 # find the epoch that yields the lowest validation loss
    curr_val_loss_sum = 1000

    count = 0       # keeps track of how long the training has gone without improvements
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

        ##### Only save the ones that reduces loss every epoch
        if val_loss_sum < curr_val_loss_sum: 
            curr_val_loss_sum = val_loss_sum
            epoch_min = ep+1
            #################### Save weights for each epoch ##############################
            this_path = f'{save_folder_weight}IFNO_t2m_{ep+1}.pth'
            print(f"Saving: {this_path}")
            with open(this_path, "wb") as f: torch.save(model.state_dict(), f)
            count = 0
        count += 1
        if count >= 20: # If 20 epochs have elapsed without improvement, finish training
            print(f"No further improvements made from epochs={epoch_min} to epochs={ep+1}\n")
            break
    
    overview = f"\n---------- OPTIMAL HYPERPARAMETERS FOR CONFIG ----------\n" \
        f"{setup.__str__()}" \
        f"Optimal epoch = {epoch_min} | MSELoss = {curr_val_loss_sum:.6f}\n"
    print(overview)
    

    ########## Save the loss for training and validation
    #MSE_save = np.dstack((train_pre, val_pre)).squeeze()
    #np.savetxt(f"{folder_save_MSE}loss_train_val_m{setup.modes}_w{setup.width}_n{setup.nlayer}_s{setup.scheduler_step}.dat", 
    #           MSE_save, fmt="%16.7f")
    #print("Loss data file saved")

    print("\n-----------------------------\nOPTIMAL DONE\n------------------------------")











################################################################################
#################### model train ########################################
################################################################################
def train_model(setup, train_years, folder_path, save_folder_weight):
    #################### Device ####################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    overview = f"\n---------- HYPERPARAMETERS ----------\n" \
            f"{setup.__str__()}"; print(f"{overview}\n")
    

    #################### Load data ####################
    ########## dir_path the folder path for both input and output datasets
    ########## Make Dataloader
    train_loader, mean_std_lst = load_dataset_train(folder_path, train_years, setup.batch_size,
                                      setup.var_in_set, setup.var_out_set)


    setup.mean_std_lst = mean_std_lst


    #################### Initialize model ####################
    model = FNO2d(setup); model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=setup.learning_rate, 
                                 weight_decay=setup.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=setup.scheduler_step, 
                                                gamma=setup.scheduler_gamma)
    myloss = torch.nn.MSELoss()
    train_pre = []
    for ep in range(setup.epochs):
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


            """try:
                pre = model(xx)
            except Exception as exc:
                import sys
                import traceback

                print(
                    f"\nMODEL ERROR: {type(exc).__name__}: {exc}",
                        file=sys.stderr, flush=True)
                print(
                    f"Input: shape={xx.shape}, dtype={xx.dtype}, device={xx.device}",
                        file=sys.stderr, flush=True)
                traceback.print_exc()
                raise"""



            l_pred = myloss(pre.reshape(bs, -1), yy.reshape(bs, -1))    #[BS, 64, 128, 1, T_out]
            loss = setup.alpha * l_pred
            train_loss_sum += l_pred.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_loss_sum = train_loss_sum / len(train_loader)
        train_pre.append(train_loss_sum)

        t2 = default_timer()
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)


        #################### Show progress for each epoch ####################
        if ep == 0:
            print("Epoch | ", "Time | ", "[Train Pred MSE] | ",
                  "[Allocated MB] | ", "[Reserved MB] | ")
        print(f"{ep} | {t2-t1:.2f} | {train_loss_sum:.6f} |" \
              f"{allocated_memory:.2f} | {reserved_memory:.2f}")


        #################### Save weights for each epoch ##############################
        this_path = f'{save_folder_weight}F_IFNO_t2m_{ep+1}.pth'
        with open(this_path, "wb") as f: torch.save(model.state_dict(), f)
        


        #from pathlib import Path
        """
        import io
        import os

        # Serialize the actual model in memory first
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)

        size = buffer.tell()
        print(f"Checkpoint size: {size / 1024**2:.2f} MiB", flush=True)

        # Use a unique filename for this Slurm job
        job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
        path = f"{save_folder_weight}F_IFNO_t2m_{ep+1}.pth"

        print(f"Saving to: {path}", flush=True)

        try:
            with open(path, "wb") as f:
                view = buffer.getbuffer()
                try:
                    f.write(view)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    view.release()
        except OSError as exc:
            print(f"SAVE ERROR: {exc!r}", flush=True)
            raise
        finally:
            buffer.close()

        print("Checkpoint saved successfully.", flush=True)"""



    print("\n----------------------------\nTRAINING DONE\n-----------------------------\n")















################################################################################
#################### model predict ########################################
################################################################################
def predict_model(setup, test_years, folder_path, folder_weight_path, filename_weight,
                  shuffle_set=True, is_testloader=None):
    if setup.mean_std_lst is None: raise TypeError("Mean/std list doesn't exist")

    overview = f"\n---------- HYPERPARAMETERS ----------\n" \
                f"{setup.__str__()}"; print(f"{overview}\n")
    

    #################### Make test loader ####################
    if is_testloader is None:           ## if testloader dont exist -> make testloader
        test_loader = load_dataset(folder_path, test_years, setup.batch_size, 
            setup.var_in_set, setup.var_out_set, setup.mean_std_lst, shuffle_set=shuffle_set)
    else:  ## if testloader exist -> use is_testloader as testloader
        test_loader = is_testloader


    #################### Initialize model with trained weights ####################
    model = FNO2d(setup)
    file_weight_path = f"{folder_weight_path}{filename_weight}"
    checkpoint = torch.load(file_weight_path, map_location=device)
    model.load_state_dict(checkpoint); model.eval()


    #################### Prediction ####################
    predictions_all = []; targets_all = []
    with torch.no_grad(): 
        for xx, yy in test_loader:
            xx = xx.to(device) # input ([batchsize, 32, 64, 3, 1])
            yy = yy.to(device) # target ([batchsize, 64, 128, 1])
            pre = model(xx)  # predicted ([batchsize, 32, 64, 1, 1])

            ##### Denormalize original data
            predicted_original = pre * setup.mean_std_lst[3] + setup.mean_std_lst[2]
            target_original = yy * setup.mean_std_lst[3] + setup.mean_std_lst[2]

            ##### Gather the results
            predictions_all.append(predicted_original)
            targets_all.append(target_original)

    ##### Merge (for plotting purposes)
    original_all = original_data_load(folder_path, test_years)
    

    predictions_all = torch.cat(predictions_all, dim=0) # ([ts, 64, 128, 1, 1])
    predictions_all = predictions_all.permute(0,4,1,2,3) # ([ts, 1, 64, 128, 1])
    predictions_all = predictions_all.numpy() # (ts, 1, 64, 128, 1)
    predictions_all = predictions_all.squeeze(-1) # (ts, 1, 64, 128)

    targets_all = torch.cat(targets_all, dim=0) # ([ts, 64, 128, 1])
    targets_all = targets_all.permute(0,3,1,2) # ([ts, 1, 64, 128])
    targets_all = targets_all.numpy() # (ts, 1, 64, 128)

    print("\n--------------------------\nPREDICTIONS DONE\n---------------------------\n")
    return original_all, predictions_all, targets_all



################################################################################
#################### Plot predicted vs target for a given sample ####################
################################################################################
def plot_predicted_sample(originals, predictions, targets, sample_id, epoch):
    ##### Need to rotate the image 180 degrees
    original_sample = originals[sample_id, 0]
    original_sample = np.rot90(original_sample, k=2) # rotate image 180 degrees to fix 


    bicubic_sample = original_sample.copy()
    bicubic_sample = bicubic_sample[np.newaxis, np.newaxis, ...] # [1, 1, 32, 64]
    bicubic_sample = torch.from_numpy(bicubic_sample)
    bicubic_sample = F.interpolate(bicubic_sample, scale_factor=2, mode="bicubic", 
                        align_corners=False) # ([1, 1, 64, 128])
    bicubic_sample = bicubic_sample.numpy() 
    bicubic_sample = np.squeeze(bicubic_sample, axis=(0,1))


    predict_sample = predictions[sample_id, 0]
    predict_sample = np.rot90(predict_sample, k=2) # rotate image 180 degrees to fix issue

    target_sample = targets[sample_id, 0]
    target_sample = np.rot90(target_sample, k=2) # rotate image 180 degrees to fix issue

    sample_images = [original_sample, bicubic_sample, predict_sample, target_sample]
    sample_titles = ["Original", "Bicubic", f"prediction | ep={epoch}", "Target"]


    ##### Configure plotting setup
    fig, axes = plt.subplots(2,2,figsize=(16,8))

    for ax, image, title in zip(axes.flat, sample_images, sample_titles):
        im = ax.imshow(image, cmap="coolwarm")
        ax.set_title(title)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax)

    #axes[1, 1].set_visible(False)

    plt.tight_layout(); plt.show()
    print("\n--------------------------\nPLOTTING DONE\n---------------------------\n")

