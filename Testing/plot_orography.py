import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
#from torchvision import transforms

import matplotlib.pyplot as plt

import xarray as xr




def plot_lsm(file_path_open):
    lsm_ds = xr.open_dataset(file_path_open)
    lsm_ds = lsm_ds["lsm"].values   # (64,128)
    lsm = np.rot90(lsm_ds, k=2)
    plt.imshow(lsm, cmap="Greys")
    plt.tight_layout(); plt.show()




if __name__=="__main__":
    print("PLotting orography (static field)")

    folder_path = "/home/zliu2/life_science/Weatherbench/"
    file_path = folder_path + "orography/orography_2.8125deg.nc"
    var_out = {"orography/lsm_2.8125deg.nc":"lsm"}


    """
    ds = xr.open_dataset(file_path)
    print(list(ds.variables))
    #print(ds["lsm"].attrs) # lsm = land-sea-mask
    #print(ds["orography"].attrs) # = physical height

    ds_var = ds["lsm"].data
    ds_lat = ds["lat"].data
    ds_lon = ds["lon"].data


    ##### convert into xarrays
    ds_out = xr.DataArray(ds_var, dims=["lat", "lon"], 
            coords={"lat": ds_lat, "lon": ds_lon}, name="lsm")


    ##### Save new datasets
    save_new_path = folder_path + "orography/lsm_2.8125deg.nc"
    ds_out.to_netcdf(save_new_path)
    print(f"Datasets saved:{"lsm"}")
    """



    ##### Open dataset lsm
    file_path_lsm = folder_path + "orography/lsm_2.8125deg.nc"

    plot_lsm(file_path_lsm)
    

