import numpy as np
import matplotlib.pyplot as plt
from electronics import modulated_field
from problem_loader import*
from photonic_functions import*

" Parameters"
mask_samples = 25
n_symbols = 4000
points_per_symbol = 50
mask_sampling_frequency = 20e9
AWG_bandwidth = 35e9
Photodiode_bandwidth = 35e9
taps = 10
power = 10e-3
scale = 1
modulation_type = 'PM'
linear_reg = False

R = 100e-6
df = -25e9
radius_vectors = [[100e-6]]
coupling_vectors = [[0.4, 0.4]]
feedback_strengths = [0.5]
ring_phases = [[compute_ring_phase_shift(df, R)]]
loop_phase = 0*2*np.pi
loop_type = "T2T"

show_transfer_function = 0
show_electric_modulation = 0
show_signal_filter = 0
show_non_linear_function = 0

k_range = np.arange(0.1, 0.4 + 0.1, 0.1) #np.array([0.9])
loop_phase_range = np.linspace(0, 2*np.pi, 20)
detuning_range = np.linspace(0, 40e9, 20)
tap_range = [3] #np.linspace(1, 10, 10)
feedback_range = [0.5] #np.arange(0.3, 1 + 0.1, 0.1)


X, Y = SeriesGenerator().SantaFe()
analog_input, frequencies, Tdelay, signal_bandwidth, data_starting_point, X_expanded = \
    create_masked_input(X, n_symbols, mask_samples, points_per_symbol, theta= 1/mask_sampling_frequency)

for f in range(0, len(feedback_range)):
    feedback_strengths = [feedback_range[f]]
    for t in range(0, len(tap_range)):

        taps = int(tap_range[t])

        for k in range(0, len(k_range)):
            coupling_vectors = [[k_range[k], k_range[k]]]

            NMSE_mat = np.zeros((len(loop_phase_range), len(detuning_range)))

            for d in range(0, len(detuning_range)):

                ring_phases = [[compute_ring_phase_shift(detuning_range[d], R)]]

                for  l in range(0, len(loop_phase_range)):

                    loop_phase = loop_phase_range[l]

                    field = modulated_field(analog_input, AWG_bandwidth, frequencies, power, scale, modulation_type, plot=show_electric_modulation)

                    if linear_reg==True:
                        reservoir_output = np.zeros((1, len(analog_input)))
                        reservoir_output[0, :] = analog_input
                    else:
                        reservoir = reservoir_layer(nodes=1, Tdelay=Tdelay, loop_type=loop_type,  loop_phase=loop_phase,\
                            simulation_bandwidth=frequencies, bandwidth=Photodiode_bandwidth)
                        parameters = reservoir.generate_nodes(radius_vectors, coupling_vectors, feedback_strengths,\
                            ring_phases)

                        input_field = np.zeros(((len(radius_vectors)), len(field))) + 0*1j
                        input_field[0, :] = field + 0*1j

                        reservoir.update_parameters(parameter_vector=parameters, plot=show_transfer_function)
                        reservoir_output = reservoir.propagate(input_field, plot=show_signal_filter)


                    NMSE_train, NMSE_test, Y_prediction = feedforward_equalizer(reservoir_output, X_expanded, Y, taps, \
                        n_symbols, points_per_symbol, data_starting_point, plot=show_transfer_function)

                    NMSE_mat[l, d] = NMSE_test
                
            filename = "SantaFe\SantaFe_Through_MRR_k_%.1f_kf_%.2f_taps_%dpower_%.2fmW.txt"%(k_range[k], feedback_strengths[0], taps, power*1e3)

            with open(filename, 'w') as file_object:
                
                for l in range(0, len(loop_phase_range)):
                    for d in range(0, len(detuning_range)):
                        file_object.write("%.4E\t%.4f\t%.4E\n"%(detuning_range[d], loop_phase_range[l], NMSE_mat[l,d]))