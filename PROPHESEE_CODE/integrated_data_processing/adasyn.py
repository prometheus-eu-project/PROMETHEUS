from snntorch._neurons import LIF
import torch
import torch.nn as nn
import sys
import snntorch.spikegen as spikegen
import snntorch as snn
class AdaSyn(LIF):  
    """
    Adaptive Synaptic First Order Leaky Integrate and Fire (LIF) neuron.

    This class extends the base `LIF` class from the `snnTorch` module to implement an adaptive synaptic 
    version of the Synaptic Leaky Integrate and Fire (LIF) neuron. The class adds additional parameters such as `alpha` 
    and `beta` for controlling the synaptic dynamics, along with the ability to adaptively learn the threshold
    during training. 

    Args:
        alpha (float): The alpha parameter controlling the synaptic dynamics.
        beta (float): The leakage factor controlling the rate of membrane potential decay.
        beta_scale (float, optional): A scaling factor for `beta`. Default is 1.0.
        threshold (float, optional): The membrane potential threshold at which the neuron spikes. Default is 1.0.
        threshold_decay (float, optional): The factor by which the threshold decays after each spike. Default is 0.9.
        spike_grad (float or None, optional): The gradient for spikes. Default is None.
        surrogate_disable (bool, optional): Whether to disable surrogate gradient computation. Default is False.
        init_hidden (bool, optional): Whether to initialize hidden states. Default is False.
        inhibition (bool, optional): Whether to apply inhibition. Default is False.
        learn_alpha (bool, optional): Whether to learn the `alpha` value. Default is False.
        learn_beta (bool, optional): Whether to learn the `beta` value. Default is False.
        reset_mechanism (str, optional): Mechanism to reset the membrane potential after a spike. Default is "subtract".
        state_quant (bool, optional): Whether to use state quantization. Default is False.
        learn_threshold_decay (bool, optional): Whether to learn the threshold decay. Default is False.
        learn_scale (bool, optional): Whether to learn the scaling factor. Default is False.
        output (bool, optional): Whether to output the spiking neurons. Default is False.
        reset_delay (bool, optional): Whether to introduce a delay in the reset process. Default is True.
        device (torch.device, optional): The device on which the neuron model should run. Default is `torch.device("cuda")`.
        base_threshold (float, optional): The base value for the threshold. Default is 1.0.

    Attributes:
        threshold_init (float): The initial threshold value for the neuron.
        reset_delay (bool): Whether to introduce a delay in the reset process.
        device (torch.device): The device on which the neuron model should run.
        base_thr (float): The base threshold value used for comparison during spiking.
        
    Methods:
        __init__(self, ...): Initializes the Adaptive Synaptic LIF neuron with the given parameters.
        _alpha_register_buffer(self, alpha, learn_alpha): Registers the alpha parameter and whether it should be learned.
        _beta_register_buffer(self, beta, learn_beta): Registers the beta parameter and whether it should be learned.
        _threshold_register_buffer(self, threshold_decay, learn_threshold_decay): Registers the threshold decay parameter.
        _scaling_register_buffer(self, beta_scale, learn_scale): Registers the scaling factor for `beta`.
        _init_membrane(self): Initializes the membrane potential and related variables.



    """  
    def __init__(
        self,
        alpha,
        beta,
        beta_scale=1.0,
        threshold=1.0,
        threshold_decay=0.9,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_alpha=False,
        learn_beta=False,
        reset_mechanism="subtract",
        state_quant=False,
        learn_threshold_decay=False,
        learn_scale=False,
        output=False,
        reset_delay=True,
        device = torch.device("cuda"),
        base_threshold=1.0
    ):
        super().__init__(
            beta,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_beta,
            False,
            reset_mechanism,
            state_quant,
            output,
        )
        
        self._alpha_register_buffer(alpha, learn_alpha)
        self._beta_register_buffer(beta,learn_beta)
        self._threshold_register_buffer(threshold_decay, learn_threshold_decay)
        self.threshold_init = threshold
        self.reset_delay = reset_delay
        self.device = device
        self.base_thr = base_threshold
        self._init_membrane()
        
        self._scaling_register_buffer(beta_scale,learn_scale)
        
        
    def forward(self, input_,mem,syn,threshold):
        if not syn == None:
            self.synaptic_current = syn

        if not mem == None:
            self.membrane = mem
            
        if not threshold == None:
            self.threshold = threshold
            
        if not self.synaptic_current.shape == input_.shape:
            self.synaptic_current = torch.zeros_like(input_, device=self.synaptic_current.device)

        if not self.membrane.shape == input_.shape:
            self.membrane = torch.zeros_like(input_, device=self.membrane.device)
        
        
        
        #in every iteration, it checks if mem > threshold
        #returns a reset signal detached from graph 
        self.reset = self.mem_reset(self.membrane).to(self.membrane.device)
        
        
        self.threshold = self.base_thr + self.threshold_decay*self.threshold + self.beta_scale*self.reset
        
        self.synaptic_current = (self.synaptic_current * self.alpha + input_)
        
        self.membrane = self.beta * self.membrane + self.synaptic_current - self.threshold*self.reset  # Eq (1)
        
        spk = self.fire(self.membrane)
        
        return spk, self.membrane, self.synaptic_current, self.threshold
    
    def fire(self, mem):
 
        mem_shift = mem - self.threshold
        spk = self.spike_grad(mem_shift)

        return spk
    
    
    def mem_reset(self, mem):
        """Generates detached reset signal if mem > threshold.
        Returns reset."""
        mem_shift = mem - self.threshold
        reset = self.spike_grad(mem_shift).clone().detach()

        return reset
    
    def reset_mem(self):
        self.membrane = torch.zeros(1).to(self.device)
        self.synaptic_current = torch.zeros(1).to(self.device)
        self.threshold = torch.tensor(self.threshold_init).to(self.device)
        return self.synaptic_current, self.membrane, self.threshold       
        
        
    def _init_membrane(self):
        membrane = torch.zeros(1).to(self.device)
        synaptic_current = torch.zeros(1).to(self.device)
        threshold = torch.tensor(self.threshold_init).to(self.device)
        
        self.register_buffer("membrane", membrane)
        self.register_buffer("synaptic_current", synaptic_current)
        self.register_buffer("threshold", threshold)
        
    
    def _scaling_register_buffer(self, beta_scale, learn_beta_scale):
        if not isinstance(beta_scale, torch.Tensor):
            beta_scale = torch.as_tensor(beta_scale)
        if learn_beta_scale:
            self.beta_scale = nn.Parameter(beta_scale)
        else:
            self.register_buffer("beta_scale", beta_scale)
        
    def _threshold_register_buffer(self, threshold, learn_threshold):
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.as_tensor(threshold)
        if learn_threshold:
            self.threshold_decay = nn.Parameter(threshold)
        else:
            self.register_buffer("threshold_decay", threshold)

    def _beta_register_buffer(self,beta,learn_beta):
        if not isinstance(beta, torch.Tensor):
            beta = torch.as_tensor(beta)
        if learn_beta:
            self.beta = nn.Parameter(beta)
        else:
            self.register_buffer("beta", beta)
            
    def _alpha_register_buffer(self, alpha, learn_alpha):
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.as_tensor(alpha)
        if learn_alpha:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("alpha", alpha)
  