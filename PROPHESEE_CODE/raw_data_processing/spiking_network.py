import snntorch as snn
from snntorch import surrogate
import torch.nn as nn
import torch
import tqdm
      


class Network(nn.Module):
    """
    A spiking neural network (SNN) model that utilizes two Leaky Integrate-and-Fire (LIF) neurons and a fully connected layer. 
    The first Leaky neuron processes the input spikes, while the second Leaky neuron processes the output weighted spikes from the first LIF.

    The network processes inputs through a series of layers, accumulating the spikes and membrane potentials across timesteps,
    and returns the final output spikes and membrane potentials.

    Attributes:
        spiking_input (snn.Leaky): The first Leaky neuron responsible for processing the input spikes.
        fullyConnected (nn.Linear): A fully connected linear layer with 100 input features and 3 output features.
        spiking_output (snn.Leaky): The second Leaky neuron responsible for processing the output from the fully connected layer.
    
    Methods:
        __init__: Initializes the network with a spiking input neuron, a fully connected layer, and a spiking output neuron.
        forward: Defines the forward pass, processing the input through the spiking neurons and fully connected layer.
    """
    def __init__(self):
        super().__init__()
        self.spiking_input = snn.Leaky(beta=0.95,learn_beta=True,learn_threshold=True)
        self.fullyConnected = nn.Linear(100 , 3)
        self.spiking_output = snn.Leaky(beta=0.95,learn_beta=True,learn_threshold=True)
        
        

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, input_features).
                The input data is expected to be a sequence of spikes across time for each feature.
        
        Returns:
            torch.Tensor: A tensor containing the spikes at each timestep, with shape (batch_size, sequence_length, output_features).
            torch.Tensor: A tensor containing the membrane potentials at each timestep, with shape (batch_size, sequence_length, output_features).
        """
        batch_size = x.shape[0]  # Get the batch size (e.g., 128 in this case)
        spikes_output = []  # List to accumulate output spikes for the batch
        membranes_out = []  # List to accumulate output membrane potentials
        
        # Initialize membrane potentials for input and output neurons
        row_membrane = self.spiking_input.init_leaky()
        row_spikes = []
        row_membranes = []
        
        second_mem = self.spiking_output.init_leaky()
        
        # Process each timestep (row) of the input sequence
        for row in range(x.shape[1]):
            row_data = x[:, row, :]  # Extract data for the current timestep
            row_spike, row_membrane = self.spiking_input(row_data, row_membrane)  # Get spikes and membrane potential from the input layer
            
            row_spikes.append(row_spike)  # Accumulate spikes
            row_membranes.append(row_membrane)  # Accumulate membrane potentials
            
            fc1_output = self.fullyConnected(row_spike)  # Pass the spikes through the fully connected layer
            
            spk2, second_mem = self.spiking_output(fc1_output, second_mem)  # Get the spikes from the output neuron
            spikes_output.append(spk2)  # Accumulate output spikes
            membranes_out.append(second_mem)  # Accumulate output membrane potentials
        
        # Return the output spikes and membrane potentials as tensors
        return torch.stack(spikes_output), torch.stack(membranes_out)
