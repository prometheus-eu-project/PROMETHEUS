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

from abc import ABC


class BaseFilter(ABC, nn.Module):
    """
    Abstract Base Filter class that defines the fundamental properties and computations shared by all filter types.

    Attributes
    ----------
        symbol_rate (float): The symbol rate in symbols per second (default is 100).
        sps (int): Samples per symbol (default is 2).
        Nt (int): Total number of time samples, computed as `input_features * sps`.
        dt (float): Time step duration, calculated as `1 / (symbol_rate * sps)`.
        num_filters (int): Number of filters used in the processing.
        Tw (float): Total time window, computed as `Nt * dt`.
        t (numpy.ndarray): Time vector from 0 to `Tw` with a step of `dt`.
        dw (float): Frequency step size, calculated as `2π / (dt * Nt)`.
        wmax (float): Maximum angular frequency, computed as `π / dt`.
        wmin (float): Minimum angular frequency, computed as `-π / dt`.
        w (numpy.ndarray): Frequency range from `wmin` to `wmax` with step `dw`.
        f (numpy.ndarray): Frequency values in Hz, derived from `w / (2π)`.
        device (torch.device): Device specification (set to "cuda").
        ds_ratio (int): Downsampling ratio.
        each_filter_output (int): Number of output features per filter, computed as `input_features / ds_ratio`.
        CR (float): Compression ratio, given by `input_features / (num_filters * each_filter_output)`.
        downsampling (torch.nn.Module): Resampling transformation from `Nt` to `each_filter_output`.
        layer1 (torch.nn.Linear): Fully connected layer that maps `n_filters * each_filter_output` to `output_features`.
        pooling (bool): Flag indicating whether pooling is applied.
        photodiode_order (int): Order of the photodiode filter (set if pooling is enabled).
        photodiode_f0 (float): Center frequency of the photodiode filter (set if pooling is enabled).
        photodiode_bw (float): Bandwidth of the photodiode filter (set if pooling is enabled).

    Parameters
    ----
        input_features (int): Number of input features.
        output_features (int): Number of output features.
        n_filters (int): Number of filters.
        pooling (bool): Whether pooling is applied.
        ds_ratio (int): Downsampling ratio.
    """
    def __init__(self, input_features,output_features, n_filters,pooling, ds_ratio):
        super().__init__()
        self.symbol_rate = 100
        self.sps = 2
        self.Nt = input_features*self.sps # 784 * sps 
        self.dt=1/self.symbol_rate /self.sps; 
        
        self.num_filters = n_filters

        self.Tw = self.Nt * self.dt

        self.t = np.arange(start=0,stop=self.Tw,step=self.dt)

        self.dw=1/self.dt/self.Nt*2*np.pi;
        self.wmax=1/self.dt*np.pi;
        self.wmin=-1/self.dt*np.pi;
        self.w = np.arange(self.wmin,self.wmax, self.dw)
        self.f = self.w/2/np.pi;
        
        self.device = torch.device("cuda")

        self.ds_ratio = ds_ratio
        self.each_filter_output = int(input_features / self.ds_ratio)
        
        self.CR = input_features / (self.num_filters * self.each_filter_output)
        
        self.downsampling = transforms.Resample(self.Nt , self.each_filter_output)

        
        self.layer1 = nn.Linear(n_filters*self.each_filter_output, output_features)
        self.pooling = pooling
        if pooling:
            self.photodiode_order = 4
            self.photodiode_f0 = 0.0
            self.photodiode_bw = 5.0
    
    @classmethod
    def forward(self,x):
        """
        Forward Pass Function, all members must define it since we are implementing a torch Module in order to allow for the backpropagation
        """
        
        pass


    def _butterworth(self, x, order,bw,f0):
        """
        Creates a Butterworth filter transfer function and applies it to the input signal 
        in the frequency domain.

        The Butterworth filter transfer function is given by:

            H(f) = 1 / (1 + 1j* (( (f - f0) / BW) ** (N)))

        where:
            - H(f) is the transfer function.
            - f0 is the central frequency.
            - BW is the bandwidth.
            - N is the order of the filter.
            - f represents the frequency components.

        Parameters
        ----------
        x : array-like
            Input signal in the time domain.
        order : int
            Order of the Butterworth filter (N).
        bw : float
            Bandwidth of the filter (BW).
        f0 : float
            Central frequency of the filter (f0).

        Returns
        -------
        y : array-like
            Filtered signal in the time domain.
        f : array-like
            Calculated frequency components
        """
        j = torch.arange(-(self.Nt / 2), ((self.Nt  / 2) ))	
        
        f = j * (1 / self.Nt );
        f = (f / self.dt).to(torch.device("cuda")); 
        
        H = 1 / (1 + 1j * ((f - f0) / (bw))**order)

        xf = torch.fft.fftshift(torch.fft.fft(x,dim=1), dim=1)
        #xf = torch.fft.fftshift(np.fft.fft(x, axis=1) , axes=1);
        yf = torch.multiply(H, xf);
        y = torch.fft.ifft(torch.fft.fftshift(yf,dim = 1), dim = 1);

        return y,f
        
    
    



    
    
    
