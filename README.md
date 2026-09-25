# UiO Life Science
-----
UiO Life Science 2026 summer research project

Uncertainty quantification for weather downscaling 

-----

#################### UPDATES ####################

F-IFNO, F-IUFNO, IFNO, IUFNO models modified for 2D downscaling tasks 

Main goal is to downscale t2m temperature using multiple variable inputs, i.e. downscale t2m LR to t2m HR. \
LR grid = 32x64 (5.625deg) \
HR grid = 64x128 (2.8125deg)

Predictions from each tuned model is made. These can be found in every models folder **outputs** \
The folder includes the following: 
* A plot containing; the original LR sample, a benchmark (bicubic interpolation), the predicted downscaled sample and the target sample
* A folder for the weights

-----

#################### PENDING ####################
* UQ methods

-----

#################### ADDITIONAL NOTES ####################
* The weights are stored for F-IFNO and F-IUFNO, but not IFNO and IUFNO due to their size
* In each model, there is a stored file for the mean and std from the training set (*f_ifno_mean_std.py* and *mean_std_lst.pth*)
* Weatherbench dataset references can be found there
