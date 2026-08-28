# F-IFNO

The original source code: https://github.com/cc0429/UQ4Turbu/tree/main/NOs/F-IFNO/F_IFNO_share_decon.py
The original code seems to take a dataset where each input-output pair is such that the input is turbulence (vector-field)
at time t and output is turbulence at time t+1. And its worth noting that the input and output have the same resolution. 


## Idea for the modified F-IFNO code 
F_IFNO_share_decon_2D_downscale.py

The original code have been reduced from 3D to 2D. To do so the z-dimension have been removed, 
and dataset have also been modified to not include a z-dimension. 
These modifications include changes to the encoder, decoder using nn.Conv2D() instead of nn.Conv3D(). 
This means that if the original dataset was [50 x 600 x 32 x 32 x 32 x 3] --> [50 x 600 x 32 x 32 x 3]
In order to not mess with the original code too much due to its dataset constrictions, even though the temperature (t2m) is a scalar field, 
a wrapper was made around the scalar, converting from scalar to a "vector" with 1 element. 

As for the task going from time-series prediction to downscaling, the original, but 2D-modified code didn't accept 
an input-output pair of mismatching resolutions,  which is a problem. 
The current "solution" to this (although still under development) is to interpolate the input resolution onto the output resolution 
so that it satisfies the same-resolution condition for the input-output pair. 
This means that the low resolution first have to be converted into high resolution before using it as input for the model. 
Since the low resolution inputs have rough features, the choice of the interpolation technique should reflect that. 
Hence the interpolation choice is Bilinear (although other methods like Linear can also be considered), and not Bicubic. 
