from .BaseFilter import BaseFilter
import torch.nn as nn
import torch
from torchaudio import transforms


import matplotlib.pyplot as plt
import torch.nn.functional as F




class TrainableTransferFunctionComplex(BaseFilter):
    '''
    Trainable Coefficients Complex Transfer Function derived from BaseFilter.

    This class extends `BaseFilter` to implement a custom data-driven transfer function with both real and imaginary coefficients.
    
    Attributes
    ---------
        - N (int): Number of trainable coefficients (points)
        - coeffs (nn.ParameterList): Parameter list containing the complex coefficients
            - The real part is between `[0,1]`
            - The imaginary part is between `-[-π, π]
        
        - cache_i : Contains the transfer function in the frequency domain of each filter ( `i = 1 ... self.num_filters`)

        - upsampling_filter (torchaudio.transforms.Resample): Resampling Transform used to upsample the interpolated transfer function points

    Parameters
    ----------
        - input_features (int): Number of input features.
        - n_filters (int) : Number of filters
        - N (int) : Number of trainable points
        - output_features (int): Number of output features.
        - pooling (boolean) : Whether pooling is used or not
        - ds_ratio (int) : Downsampling ratio
    
    '''
    def __init__(self, input_features, n_filters,N, output_features, pooling=True,ds_ratio=7):
        super().__init__(input_features,output_features,n_filters,pooling,ds_ratio)


        self.N = N
        self.coeffs = nn.ParameterList([
            nn.Parameter((torch.rand(self.N) + 1j * (2 * torch.rand(self.N) - 1) * torch.pi), requires_grad=True) for _ in range(self.num_filters)
            
        ])
        
        for i in range(self.num_filters):
            self.register_buffer(f"cache_{i}", torch.zeros(self.Nt, dtype=torch.complex64, device=torch.device("cpu")))        
    
        self.upsampling_filter = transforms.Resample(2*(N*2) , self.Nt)


    def __filter(self,coefficient_buffer,xf):
        '''
            Frequency-Domain Filtering Function

            This function applies filtering in the frequency domain by constructing a symmetric transfer function and performing 
            convolution in the Fourier domain.

            Steps:
            1. **Create a Local Buffer**: 
            - Initializes a buffer of size `2 * self.N` filled with zeros.
            
            2. **Interpolate Real and Imaginary Parts**:
            - Uses bicubic interpolation to estimate intermediate values between the sampled points.

            3. **Populate the Buffer**:
            - The first `self.N` points in the buffer are set to the interpolated values.
            - The remaining points are kept as zero.

            4. **Ensure Fourier Transform Symmetry**:
            - The buffer is flipped and concatenated with itself, forming a final buffer of size `2 * (N * 2)`.
            - This step enforces:
                (a) Symmetric real part in the Fourier domain.
                (b) Antisymmetric imaginary part in the Fourier domain.
            
            5. **Upsample to Match Input Samples**:
            - Ensures that the final filter response matches the input signal's resolution.

            6. **Construct Transfer Function**:
            - Combines the real and imaginary components to form the complex transfer function: 
                \[
                H(f) = \text{real}(f) + j \cdot \text{imaginary}(f)
                \]

            7. **Apply Frequency-Domain Filtering**:
            - Multiplies the input signal by the transfer function in the frequency domain.

            8. **Return to Time Domain**:
            - Uses Inverse Fast Fourier Transform (IFFT) to transform the filtered signal back to the time domain.

            9. **Optional: Apply Photodiode Response**:
            - If pooling is enabled, a 4th-order Butterworth filter (centered at 0 GHz) is applied in the frequency domain.

            10. **Downsample the Output**:
            - Before returning, the filtered signal and the transfer function are moved to the CPU.
            - The output is then downsampled to `self.each_filter_output`.

            Arguments
            ---------
                - coefficient_buffer (nn.Parameter) 
                    The **complex** trainable coefficients of the filter
                - xf (torch.tensor)
                    Input in the **frequency** domain

            Returns
            -------
                - y_clone (torch.tensor)
                    Transfer Function in the GPU
                - filtered_output (torch.tensor)
                    Filtered output in the time domain

        '''
        buffer_real = coefficient_buffer.view(1,1, coefficient_buffer.shape[0],1).real # Fetch the real trainable coefficients
        buffer_imag = coefficient_buffer.view(1,1, coefficient_buffer.shape[0],1).imag # Fetch the imaginary trainable coefficites
        
        # create local buffers one for real one for imaginary
        local_buffer_real = torch.zeros(2*self.N)
        local_buffer_imag = torch.zeros(2*self.N)


        coeff_ups = F.interpolate(buffer_real, size=(self.N, 1), mode='bicubic', align_corners=True) # interpolate between the points
        coeff_ups = coeff_ups.view(-1)  
        local_buffer_real[:self.N] = coeff_ups # set the first N points to the interpolated ones
        flipped = torch.flip(local_buffer_real, dims=[0]) # Flip the tensor
        local_buffer_real = torch.cat((flipped, local_buffer_real))  # concatenate both tensors to create the real part 
        

        # The same procedure for the imaginary part
        coeff_ups_imag = F.interpolate(buffer_imag, size=(self.N, 1), mode='bicubic', align_corners=True)
        coeff_ups_imag = coeff_ups_imag.view(-1)  
        local_buffer_imag[:self.N] = coeff_ups_imag
        flipped = torch.flip(local_buffer_imag, dims=[0])
        local_buffer_imag = torch.cat((-flipped, local_buffer_imag))  # concatenate with -flipped to create the antisymmetric effect required by Fourier
        

        # Perform upsampling
        y_clone_real = self.upsampling_filter(local_buffer_real.to(self.device))
        y_clone_imag = self.upsampling_filter(local_buffer_imag.to(self.device))


        # Create the transfer function by adding the real + j * imaginary part
        y_clone = y_clone_real + 1j*y_clone_imag

        # Multiply with the fourier transform of the input
        yf = torch.multiply( y_clone, xf)
        
        # IFFT to return to the time domain
        filter_output = torch.fft.ifft(torch.fft.fftshift(yf,dim = 1), dim = 1)
        
        if self.pooling:
            filter_output,_ = self._butterworth(torch.abs(filter_output)**2, self.photodiode_order,self.photodiode_bw,self.photodiode_f0)
        else:
            filter_output = torch.abs(filter_output)**2
            
        filter_output = F.interpolate( filter_output.real.unsqueeze(1), size=self.each_filter_output, mode='linear', align_corners=False).squeeze(1)

        
        return y_clone, filter_output, y_clone.detach().cpu()


    def plot_filter_transfer_function(self, filter_idx, logarithmic=True):
        """
        Plots the transfer function of the specified filter.

        Parameters
        ----------
        filter_idx : int
            Index of the filter to plot.
        logarithmic : bool, optional
            If True, plots the magnitude in dB. Otherwise, plots the linear scale.
        
        The function plots:
        - Magnitude response (dB or linear scale)
        - Phase response (lag) in radians
        """
        cache = getattr(self, f"cache_{filter_idx}")
        cache = cache.to(torch.device("cpu"))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        # Plot magnitude response in the first subplot
        color = 'tab:blue'
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Magnitude", color=color)
        if logarithmic:
            ax1.plot(self.f, 10 * torch.log10(torch.abs(cache) ** 2), label=f"Filter {filter_idx} (Magnitude)", color=color)
            ax1.set_ylabel("Magnitude (dB)", color=color)
        else:
            ax1.plot(self.f, torch.abs(cache), label=f"Filter {filter_idx} (Magnitude)", color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend()

        # Plot phase response in the second subplot
        color = 'tab:red'
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (radians)", color=color)
        ax2.plot(self.f, torch.angle(cache), label=f"Filter {filter_idx} (Phase)", color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend()

        # Adjust layout to avoid overlap
        fig.tight_layout()
        plt.show()
            
    def plot_filters(self, n_filters, logarithmic= True):
        '''
        Plots the magnitude and phase transfer functions for a range of filters.

        Parameters
        ----------
        n_filters : int
            The number of filters to plot. If `n_filters == -1`, all filters are plotted.
        logarithmic : bool, optional
            If True, the magnitude is plotted in dB. Otherwise, the magnitude is plotted on a linear scale. Default is True.

        This function will iterate through the filters from index 1 to `n_filters` (or all filters if `n_filters == -1`) and
        generate plots of both the magnitude and phase (lag) responses for each filter by calling `plot_filter_transfer_function`.
        '''
        end = n_filters
        
        if n_filters == -1:
            end = self.num_filters
        
        for cache_idx in range(1,end+1):
            self.plot_filter_transfer_function(cache_idx,logarithmic)


    def forward(self, x):
        
        self.filterouts = []
        xf = torch.fft.fftshift(torch.fft.fft(x,dim=1), dim=1) # FFT on the input to calculate the frequency domain
        for f_idx in range(0,self.num_filters):
            coeff = self.coeffs[f_idx] # fetch the coefficients
            _, filterout, cache = self.__filter(coeff, xf) # use them to filter the input signal
            self.filterouts.append(filterout) # append the output to the buffer
            getattr(self, f"cache_{f_idx}").data.copy_(cache) # Copy the values into the cache, updating it on each iteration

                            
        out = torch.hstack(self.filterouts)     # stack the outputs
        return self.layer1(out) # Use them as input to the linear classifier layer
    


class TrainableTransferFunctionReal(BaseFilter):
    '''
    Trainable Coefficients Real Transfer Function derived from BaseFilter.

    This class extends `BaseFilter` to implement a custom data-driven transfer function with only real coefficients and operates the same way as `TrainableTransferFunctionComplex`, refer to this documentation
    for more details.
    
    Attributes
    ---------
        - N (int): Number of trainable coefficients (points)
        - coeffs (nn.ParameterList): Parameter list containing only the real coefficients  between `[0,1]`
        
        - cache_i : Contains the transfer function in the frequency domain of each filter ( `i = 1 ... self.num_filters`)

        - upsampling_filter (torchaudio.transforms.Resample): Resampling Transform used to upsample the interpolated transfer function points

    Parameters
    ----------
        - input_features (int): Number of input features.
        - n_filters (int) : Number of filters
        - N (int) : Number of trainable points
        - output_features (int): Number of output features.
        - pooling (boolean) : Whether pooling is used or not
        - ds_ratio (int) : Downsampling ratio
    
    
    '''
    def __init__(self, input_features, n_filters,N, output_features, pooling=True,ds_ratio=7):
        super().__init__(input_features,output_features,n_filters,pooling,ds_ratio)

        self.N = N
        self.coeffs = nn.ParameterList([nn.Parameter(torch.rand(self.N), requires_grad=True) for _ in range(self.num_filters)])
        
        for i in range(self.num_filters):
            self.register_buffer(f"cache_{i}", torch.zeros(self.Nt))      
        self.upsampling_filter = transforms.Resample(2*(N*2) , self.Nt)


    def plot_filter_transfer_function(self, filter_idx, logarithmic = True):
        cache = getattr(self, f"cache_{filter_idx}")
        if logarithmic:
            plt.plot(self.f, 10*torch.log10(torch.abs(cache)**2), label=f"filter_{filter_idx}")
        else:
            plt.plot(self.f, cache,label=f"filter_{filter_idx}" )
        plt.legend()
            
    def plot_filters(self, n_filters, logarithmic= True):
        end = n_filters
        
        if n_filters == -1:
            end = self.num_filters
        
        for cache_idx in range(1,end+1):
            self.plot_filter_transfer_function(cache_idx,logarithmic)

        
    def __filter(self,coefficient_buffer,xf):
        buffer1 = coefficient_buffer.view(1,1, coefficient_buffer.shape[0],1)
        local_buffer = torch.zeros(2*self.N)
        coeff_ups = F.interpolate(buffer1, size=(self.N, 1), mode='bicubic', align_corners=True)
        coeff_ups = coeff_ups.view(-1)  # Keep it in the computation graph
        local_buffer[:self.N] = coeff_ups
        flipped = torch.flip(local_buffer, dims=[0])
        local_buffer = torch.cat((flipped, local_buffer))  # Keep coeff in the graph

        y_clone = self.upsampling_filter(local_buffer.to(self.device))
        
        yf = torch.multiply( y_clone, xf)
        
        filter_output = torch.fft.ifft(torch.fft.fftshift(yf,dim = 1), dim = 1)
        
        if self.pooling:
            filter_output,_ = self._butterworth(torch.abs(filter_output)**2, self.photodiode_order,self.photodiode_bw,self.photodiode_f0)
        else:
            filter_output = torch.abs(filter_output)**2
            
        filter_output = self.downsampling(filter_output.real)
        return y_clone, filter_output, y_clone.detach().cpu()
        
    def forward(self, x):
        
        self.filterouts = []
        xf = torch.fft.fftshift(torch.fft.fft(x,dim=1), dim=1)
        for f_idx in range(0,self.num_filters):
            coeff = self.coeffs[f_idx]
            _, filterout, cache = self.__filter(coeff, xf)
            self.filterouts.append(filterout)
            getattr(self, f"cache_{f_idx}").data.copy_(cache) # Updates registered buffer

                            
        out = torch.hstack(self.filterouts)    
        return self.layer1(out)