from .BaseFilter import BaseFilter
import torch.nn as nn
import torch
import torch.autograd.profiler as profiler
import torch.nn.functional as F
class TrainableFrequency(BaseFilter):
    """
    Trainable Central Frequency and Bandwidth Butterworth Filter class derived from BaseFilter.

    This class extends `BaseFilter` to implement a set of Butterworth filters with trainable or fixed central frequencies 
    and bandwidths. Each filter's parameters are specified via a configuration dictionary.

    Attributes
    -----------
        - frequencies (nn.ParameterDict): Parameter dictionary containg the central frequency of each Butterworth filter, trainable if specified.
        - bws (nn.ParameterDict): Parameter dictionary containing the bandwidth of each Butterworth filter, trainable if specified.

    Parameters
    ----
        - input_features (int): Number of input features.
        - output_features (int): Number of output features.
        - config (dict): Dictionary containing filter configurations. Each filter has a central frequency (`f`) and 
                    bandwidth (`bw`), with options to set them as trainable.
                    Example format:
                    ```
                    config = {
                        "filter_0": {
                            "f": {"value": 3.0, "trainable": False},
                            "bw": {"value": 7.0, "trainable": False}
                        },
                        "filter_1": {
                            "f": {"value": 10.0, "trainable": False},
                            "bw": {"value": 7.0, "trainable": False}
                        },
                        ...,
                        "filter_n": {
                            "f": {"value": 45.0, "trainable": False},
                            "bw": {"value": 7.0, "trainable": False}
                        }
                    }
                    ```
        - pooling (bool): Whether pooling is applied.
        - ds_ratio (int): Downsampling ratio.
    """
     
    def __init__(self,input_features,output_features, config, pooling = False, ds_ratio=7):
        super().__init__(input_features,output_features,len(config.keys()),pooling, ds_ratio)
        
        self.frequencies = nn.ParameterDict({
                key: nn.Parameter(torch.tensor(value["f"]["value"], dtype=torch.float32), requires_grad=value["f"]["trainable"])
                for key, value in config.items()
        })
        
        self.bws = nn.ParameterDict({
                key: nn.Parameter(torch.tensor(value["bw"]["value"], dtype=torch.float32),requires_grad=value["bw"]["trainable"])
                for key, value in config.items()
        })
        
        
    
    def forward(self,x):
        """
        Overloaded forward pass function that applies frequency-domain filtering using trainable Butterworth filters.

        For each filter, the central frequency and bandwidth are retrieved, and the input signal is filtered in the 
        frequency domain. If `pooling=True`, a photodiode is simulated as a 4th-order Butterworth filter centered at 0 GHz, 
        and the signal is filtered again by calling the `_butterworth` protected method.

        After filtering, each output is downsampled to match `self.each_filter_output` using interpolation 
        (`self.downsampling` is slow for relatively small `ds_ratio`). Finally, the filtered outputs are flattened 
        and passed through a fully connected (FC) layer for classification.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor with shape `(B, Nt)`, where:
            - `B` is the batch size.
            - `Nt` is the total number of time samples.

        Returns
        -------
        torch.Tensor
            The processed output tensor after filtering, downsampling and FC layer
        """
        filter_outputs = []
       
        for f_idx in range(0,self.num_filters):
            bw = self.bws[f"filter_{f_idx}"]
            f0 = self.frequencies[f"filter_{f_idx}"]
            y,_ = self._butterworth(x,1, bw, f0)
            
            if self.pooling:
                output,_ = self._butterworth(torch.abs(y)**2,self.photodiode_order,self.photodiode_bw,self.photodiode_f0)
            else:
                output,_ = torch.abs(y)**2
            
            
            downsampled = F.interpolate( output.real.unsqueeze(1), size=self.each_filter_output, mode='linear', align_corners=False).squeeze(1)
            #downsampled = self.downsampling(output.real)
            
            
            filter_outputs.append(downsampled)
        
        out = torch.hstack(filter_outputs)
        return self.layer1(out)