import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy as sp
import snntorch.surrogate as sr

from torchaudio import transforms
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

def kernel_based_split_vectorized3d_row_based_arhs(input,kernel_size=5, stride=5):
    '''
    Function used to flatten the input image across 2 orientations, vertically and horizontally. First the column major block is placed followed by the row major

    Arguments
    ---------
    -input (torch.tensor)
        Input image of shape `(B x 784)`
    - kernel_size (int)
        The kernel size used to form the blocks
    - stride (int)
        Stride of each kernel

    Returns
    -------
        - output (torch.tensor)
            The flattened image of shape `(B, 2 * num_blocks * (kernel_size * kernel_size))` where `num_blocks = (input.size(1) // kernel_size) * (input.size(2) // kernel_size)`
    
    '''
    num_blocks = (input.size(1) // kernel_size) * (input.size(2) // kernel_size)
    
    # Create output tensor with appropriate shape
    output = torch.zeros(input.size(0), 2 * num_blocks * (kernel_size * kernel_size))

    
    idx = 0  # track position in output for each kernel

    for h in range(input.size(1) // kernel_size):
        for w in range(input.size(2) // kernel_size):
            # Calculate the start and end positions of the block
            vert_start = h * stride
            vert_end = vert_start + kernel_size
            horiz_start = w * stride
            horiz_end = horiz_start + kernel_size

            # Extract the block
            block = input[:,vert_start:vert_end, horiz_start:horiz_end]
            
            # Flatten in column-major order
            column_major = torch.transpose(block,1,2).flatten(start_dim=1,end_dim=2)
            # Flatten in row-major order
            row_major = torch.flatten(block,start_dim=1,end_dim=2)
            # Place flattened block into the output tensor
            first_block_idx = idx
            first_block_idx_end = idx + column_major.size(1)

            output[:, first_block_idx:first_block_idx_end] = column_major
            output[:,first_block_idx_end:first_block_idx_end + column_major.size(1)] = row_major
            idx += 2*column_major.size(1)

    return output

def prepare_image(images_flattened : torch.tensor, Symbol_Rate : float):
    """
    Upsamples a flattened image using signal resampling.


    Parameters
    ----------
    images_flattened : torch.Tensor
        A 2D tensor representing the input images in flattened form, 
        with shape `(B, num_pixels)`, where `B` is the batch size.
    
    Symbol_Rate : float
        The pixel rate used for upsampling, measured in symbols per second.

    Returns
    -------
    input_signal_resampled : np.ndarray
        The resampled image tensor
    
    dt : float
        The time step size after upsampling.
    
    sps : int
        Samples per symbol used for upsampling (fixed at 2).
    
    Nt : int
        Total number of time-domain samples after upsampling.
    
    t : np.ndarray
        Time vector for the resampled signal, with length `Nt`.
    
    f : np.ndarray
        Frequency vector for the resampled signal, corresponding to the Fourier domain.
    
    Notes
    -----
    - The function assumes a fixed upsampling factor of `sps = 2`.
    - The upsampling is performed using `scipy.signal.resample`, which applies Fourier-based interpolation.
    - The frequency axis `f` is computed based on the Nyquist theorem.
    """
    sps = 2
    Nt = images_flattened.shape[1]*sps
    dt=1/Symbol_Rate/sps; 
   
    
    Tw = Nt * dt

    t = np.arange(start=0,stop=Tw,step=dt)

    dw=1/dt/Nt*2*np.pi;
    wmax=1/dt*np.pi;
    wmin=-1/dt*np.pi;
    w= np.arange(wmin,wmax,dw)
    f=w/2/np.pi;


    input_signal_resampled = sp.signal.resample(images_flattened, int(images_flattened.shape[1]* sps), axis=1)

    return input_signal_resampled,  dt, sps, Nt, t, f

