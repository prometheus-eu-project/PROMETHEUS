from snntorch._neurons import SpikingNeuron
import torch
import torch.nn as nn
import sys
import snntorch.spikegen as spikegen
import snntorch as snn

class SGRU(SpikingNeuron):
    """
    Spiking Gated Recurrent Unit (SGRU) neuron model.

    This class extends the base `SpikingNeuron` class to implement a spiking version of the Gated Recurrent Unit (GRU) 
    used in recurrent neural networks (RNNs). The model utilizes the GRU mechanism to capture temporal dependencies while 
    incorporating spiking dynamics, making it suitable for spiking neural networks (SNNs).

    Args:
        input_size (int): The number of input features to the GRU cell.
        hidden_size (int): The number of hidden units in the GRU cell.
        bias (bool, optional): Whether to include a bias term in the GRU cell. Default is `True`.
        threshold (float, optional): The membrane potential threshold. Default is `1.0`.
        spike_grad (float or None, optional): Gradient for spikes. Default is `None`.
        surrogate_disable (bool, optional): Whether to disable the surrogate gradient computation. Default is `False`.
        init_hidden (bool, optional): Whether to initialize hidden states. Default is `False`.
        inhibition (bool, optional): Whether to apply inhibition to the neuron. Default is `False`.
        learn_threshold (bool, optional): Whether to learn the threshold parameter during training. Default is `False`.
        reset_mechanism (str, optional): Mechanism for resetting the membrane potential after a spike. Default is `"subtract"`.
        state_quant (bool, optional): Whether to apply state quantization. Default is `False`.
        output (bool, optional): Whether to output the spiking neurons. Default is `False`.

    Attributes:
        input_size (int): The number of input features to the GRU cell.
        hidden_size (int): The number of hidden units in the GRU cell.
        bias (bool): Whether to include a bias term in the GRU cell.
        gru_cell (nn.GRUCell): The GRU cell used to compute the hidden states.
        
    Methods:
        __init__(self, ...): Initializes the SGRU neuron with the given parameters.
        _init_membrane(self): Initializes the membrane potential and synaptic current for the spiking neuron.

    """
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
    ):
        super().__init__(threshold=threshold, 
                         spike_grad=spike_grad, surrogate_disable=surrogate_disable, init_hidden=init_hidden, inhibition=inhibition, learn_threshold=learn_threshold, reset_mechanism=reset_mechanism, state_quant=state_quant, output=output)
        self._init_membrane() # init membrane potential and synaptic current
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.gru_cell = nn.GRUCell(input_size, hidden_size, bias=bias)
        
    
    def forward(self, input_,mem=None):
        
        if not mem == None:
            self.mem = mem

        if self.init_hidden and (not mem == None ):
            raise TypeError(
                "`mem` or `syn` should not be passed as an argument while `init_hidden=True`"
            )

        size = input_.size()
        correct_shape = (size[0], self.hidden_size)

       
        if not self.mem.shape == input_.shape:
            self.mem = torch.zeros(correct_shape, device=self.mem.device)
            
        self.reset = self.mem_reset(self.mem)
        
        self.mem = self._base_sub(input_)
        
        self.spk = self.fire(self.mem)
        
        return self.spk, self.mem
        

    def fire(self, mem):
 
        mem_shift = mem - self.threshold
        spk = self.spike_grad(mem_shift)

        return spk
    
    def _base_sub(self, input_):
    
        mem = self.gru_cell(input_, self.mem)
        mem -= self.reset * self.threshold
        return  mem
    
    def mem_reset(self, mem):
        """Generates detached reset signal if mem > threshold.
        Returns reset."""
        mem_shift = mem - self.threshold
        reset = self.spike_grad(mem_shift).clone().detach()

        return reset
        
    def _init_membrane(self):
        
        mem = torch.zeros(0)

        self.register_buffer("mem", mem, False)
        
        
    def reset_mem(self):
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        return self.mem

    def init_sgru(self):
        """Deprecated, use :class:`SLSTM.reset_mem` instead"""
        return self.reset_mem()
