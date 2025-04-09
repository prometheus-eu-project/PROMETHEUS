from snntorch._neurons import LIF
import torch
import torch.nn as nn
import sys
import snntorch.spikegen as spikegen
import snntorch as snn
class CustomSynaptic(LIF):
    """
    Custom Synaptic Model with Leaky Integrate-and-Fire (LIF) neuron dynamics.

    This class extends the base `LIF` class to incorporate custom synaptic dynamics for spiking neurons. 
    It allows for customization of key parameters, including synaptic weights and the threshold. This neuron 
    model can be used in spiking neural networks (SNNs) to capture more complex behaviors and interactions between neurons.

    Args:
        alpha (float): A parameter that may scale or control synaptic dynamics in the neuron.
        beta (float): The base synaptic time constant, controlling the membrane decay rate.
        threshold (float, optional): The membrane potential threshold for spiking. Default is `1.0`.
        spike_grad (float or None, optional): Gradient used for spikes. Default is `None`.
        surrogate_disable (bool, optional): Whether to disable the surrogate gradient. Default is `False`.
        init_hidden (bool, optional): Whether to initialize the hidden state. Default is `False`.
        inhibition (bool, optional): Whether to apply inhibition in the neuron model. Default is `False`.
        learn_alpha (bool, optional): Whether to allow the learning of the alpha parameter. Default is `False`.
        learn_beta (bool, optional): Whether to allow the learning of the beta parameter. Default is `False`.
        reset_mechanism (str, optional): The method used for resetting the membrane potential after a spike. Default is `"subtract"`.
        state_quant (bool, optional): Whether to apply quantization to the neuron state. Default is `False`.
        output (bool, optional): Whether to generate an output from the neuron. Default is `False`.
        reset_delay (bool, optional): Whether to include a reset delay after a spike. Default is `True`.
        device (torch.device, optional): The device on which the neuron model is to run. Default is `"cuda"`.
        learn_threshold (bool, optional): Whether to allow the learning of the threshold. Default is `True`.

    Attributes:
        alpha (float): A parameter controlling the synaptic dynamics.
        beta (float): The base synaptic time constant, controlling the decay of the membrane potential.
        threshold (float): The membrane potential threshold for spiking.
        spike_grad (float or None): Gradient for spikes.
        surrogate_disable (bool): Whether to disable surrogate gradients.
        init_hidden (bool): Whether to initialize the hidden state.
        inhibition (bool): Whether to apply inhibition to the neuron.
        learn_alpha (bool): Whether to learn the alpha parameter.
        learn_beta (bool): Whether to learn the beta parameter.
        reset_mechanism (str): The reset mechanism for the membrane potential.
        state_quant (bool): Whether to apply quantization to the neuron state.
        output (bool): Whether to produce the output spikes.
        reset_delay (bool): Whether to introduce a delay before resetting the neuron state.
        device (torch.device): The device to run the model on.
        learn_threshold (bool): Whether to allow learning of the threshold.

    Methods:
        __init__(self, ...): Initializes the CustomSynaptic neuron with the provided parameters.

    """
    def __init__(
        self,
        alpha,
        beta,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_alpha=False,
        learn_beta=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        reset_delay=True,
        device = torch.device("cuda"),
        learn_threshold = True
    ):
        super().__init__(
            beta,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
        )
        
        self._alpha_register_buffer(alpha, learn_alpha)
        self._beta_register_buffer(beta,learn_beta)
        self.reset_delay = reset_delay
        self.device = device
        
        self._init_membrane()

        
        
    def forward(self, input_,mem,syn):
        if not syn == None:
            self.synaptic_current = syn

        if not mem == None:
            self.membrane = mem
            
        
            
        if not self.synaptic_current.shape == input_.shape:
            self.synaptic_current = torch.zeros_like(input_, device=self.synaptic_current.device)

        if not self.membrane.shape == input_.shape:
            self.membrane = torch.zeros_like(input_, device=self.membrane.device)
        
        
        self.reset = self.mem_reset(self.membrane).to(self.membrane.device)

        
        self.synaptic_current = (self.synaptic_current * self.alpha + input_)
        
        self.membrane = self.beta * self.membrane + self.synaptic_current - self.threshold*self.reset  # Eq (1)
        
        spk = self.fire(self.membrane)
        
        return spk, self.membrane, self.synaptic_current
    
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
        
        return self.synaptic_current, self.membrane
        
        
    def _init_membrane(self):
        membrane = torch.zeros(1).to(self.device)
        synaptic_current = torch.zeros(1).to(self.device)
        
        self.register_buffer("membrane", membrane)
        self.register_buffer("synaptic_current", synaptic_current)
        
    
    
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

if __name__ == "main":
    import matplotlib.pyplot as plt
    import snntorch.spikegen as sgen
    import snntorch as snn
    import torch 
    import torch.nn as nn

    import sys
    import os
    num_steps = 100

    w = 0.3
    #
    spk_period = torch.cat((torch.ones(1)*w, torch.zeros(9)), 0)
    spk_in = spk_period.repeat(20)

    alpha = 0.9
    beta = 0.8


    slstm = CustomSynaptic(alpha,beta,threshold=1.0,learn_alpha=True,learn_beta=True,learn_threshold = True)
    syn, mem = slstm.reset_mem()
    mem_rec = []
    syn_thr = []
    spk_rec = []
    for timestep in range(num_steps):
        
        spk, mem, syn = slstm(spk_in[timestep].to(torch.device("cuda")), mem, syn)
        mem_rec.append(mem.detach().cpu())
        syn_thr.append(syn.detach().cpu())
        spk_rec.append(spk.detach().cpu())


    spk = torch.stack(spk_rec)
    syn = torch.stack(syn_thr)
    mem = torch.stack(mem_rec)


    # Create a figure and plot the input and membrane potential with the desired size
    fig, ax = plt.subplots(2,1,figsize=(6, 7))

    # Plot the rate-coded vector (input) and membrane potential
    ax[0].plot(syn, label="Input: Synaptic Current", color='b')
    ax[0].plot(mem, label="Membrane Potential", color='r')




    ax[1].plot(syn.squeeze(1), label="Synaptic Current", color='b')
    ax[1].plot(spk_in.squeeze(1), label="Input", color='g')
    ax[1].set_ylabel("Amplitude")
    ax[1].set_title("Synaptic Current")
    ax[1].set_xlabel("Time (s)")
    ax[1].legend()

    # Adjust layout to prevent clipping
    plt.tight_layout()

    #plt.savefig("/Users/georgiosmoustakas/Downloads/moust6.png", dpi=300, bbox_inches='tight')


    plt.show()