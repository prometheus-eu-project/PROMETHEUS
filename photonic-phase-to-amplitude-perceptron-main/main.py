import numpy as np
import matplotlib.pyplot as plt
from electronics import digital_to_analog
from problem_loader import *
from photonic_functions import *

# Set random seed for reproducibility
np.random.seed(888)

# ---------------------------- PARAMETERS ---------------------------- #

# Masking settings
mask_samples = 200  
points_per_symbol = 25  
mask_sampling_frequency = 25e9  # Defines theta  
de_synchronization_parameter = 0  # Desynchronization of mask time and delay  

# Encoding method
encoding_method = 'ZOH'  # Choose 'RZ' or 'ZOH'  

# Hardware constraints
awg_bandwidth = 100e9  
photodiode_bandwidth = 0.75 * mask_sampling_frequency  

# Signal processing parameters
taps = 1  
power = 20e-3  
scale = 1  # Input scale  
modulation_type = 'AM'  # Choose 'AM' for amplitude, 'PM' for phase modulation  

# Reservoir computing settings
linear_reg = False  # True for linear reservoir, False for MRR  
enable_noise = 1  
output_port = 'Drop'  # Options: 'Drop', 'Through', 'Differential'  

# Feedback loop settings
loop_type = "T2D"  # Options: 'T2T' (Through/Through loop), 'T2D' (Through/Drop loop), 'Fano'  
radius = 6.75e-6  
freq_detuning = -5e9  
radius_vectors = [[radius]]  
coupling_vectors = [[0.01, 0.01]]  
feedback_strengths = [0.9]  
ring_phases = [[compute_ring_phase_shift(freq_detuning, radius)]]  
loop_phase = 0.55 * 2 * np.pi  

# Visualization flags
show_transfer_function = 1  
show_electric_modulation = 0  
show_signal_filter = 0  
show_non_linear_function = 1  

# ---------------------------- CODE EXECUTION ---------------------------- #

# Generate NARMA10 dataset for training and testing
task = SeriesGenerator(n_training=2000, n_testing=1000)
task.narma10()  

# Create masked input signal
analog_input, t_delay, sample_freq = create_masked_input(
    task, mask_samples, points_per_symbol, mask_sampling_frequency,
    de_synchronization_parameter, encoding_method
)

# Determine processing method (Linear vs. MRR Reservoir)
if linear_reg:
    reservoir_output = analog_input  # Directly use input signal
else:
    # Convert digital input to analog optical field
    input_field = digital_to_analog(
        analog_input, awg_bandwidth, sample_freq, power, scale, 
        modulation_type, plot=show_electric_modulation
    )

    # Initialize photonic reservoir
    reservoir = ReservoirLayer(
        nodes=1, t_delay=t_delay, loop_type=loop_type, loop_phase=loop_phase,
        sample_freq=sample_freq, signal_length=len(input_field),
        bandwidth=photodiode_bandwidth
    )

    # Generate and update reservoir parameters
    parameters = reservoir.generate_nodes(radius_vectors, coupling_vectors, feedback_strengths, ring_phases)
    reservoir.update_parameters(parameter_vector=parameters, plot=show_transfer_function)

    # Propagate input through the reservoir
    reservoir_output = reservoir.propagate(
        input_field, output_port, enable_noise, plot=show_signal_filter
    )

# Post-processing and feature extraction
x = post_processing(reservoir_output, task, taps, points_per_symbol, plot=show_non_linear_function)

# Train and evaluate the model
nmse_train, nmse_test = feedforward_equalizer(x, task)

# Display performance metrics
print("The training NMSE is %.4E \n" % nmse_train)
print("The testing NMSE is %.4E \n" % nmse_test)

# Evaluate memory capacity
memory_capacity(x, task)
