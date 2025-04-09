from cmath import inf
from tokenize import Number
import numpy as np
from photonic_functions import run_the_laser
from digital import *
from scipy import signal
import matplotlib.pyplot as plt

def butterworth(x, sample_freq, cut_freq, n):
    """
    Applies a Butterworth filter to the input signal in the frequency domain.
    
    Parameters:
    x (numpy array): Input signal.
    sample_freq (float): Sampling frequency.
    cut_freq (float): Cutoff frequency.
    n (int): Filter order.
    
    Returns:
    numpy array: Filtered signal.
    """
    x_spectrum = np.fft.fftshift(np.fft.fft(x))
    nyquist_freq = sample_freq / 2
    f = np.linspace(-nyquist_freq, nyquist_freq, len(x))
    
    H = 1 / (1 + 1j * (f / cut_freq) ** n)
    y_spectrum = H * x_spectrum
    y = np.fft.ifft(np.fft.ifftshift(y_spectrum))
    
    return np.abs(y)

def quantization_noise(x, precision=8):
    """
    Introduces quantization noise to the input signal.
    
    Parameters:
    x (numpy array): Input signal.
    precision (int): Number of bits for quantization.
    
    Returns:
    numpy array: Quantized signal.
    """
    x_min, x_max = np.min(x), np.max(x)
    dx = (x_max - x_min) / 2 ** precision
    k = np.round((x - x_min) / dx)
    x_quantized = x_min + k * dx
    
    return x_quantized

def digital_to_analog(data_in, awg_bandwidth, sample_freq, power, scale, modulation_type, plot, precision=64):
    """
    Converts digital data to an analog optical field signal.
    
    Parameters:
    data_in (numpy array): Input digital signal.
    awg_bandwidth (float): Arbitrary waveform generator bandwidth.
    sample_freq (float): Sampling frequency.
    power (float): Signal power.
    scale (float): Scaling factor.
    modulation_type (str): 'AM' for amplitude modulation, other for phase modulation.
    plot (bool): Whether to plot the signal.
    precision (int): Quantization precision.
    
    Returns:
    numpy array: Optical field signal.
    """
    if modulation_type == 'AM':
        amplitude = np.sqrt(power * scale * data_in)
        amplitude = quantization_noise(amplitude, precision)
        amplitude = butterworth(amplitude, sample_freq, awg_bandwidth, n=4)
        phase = 0
    
        if plot:
            sample_period = 1 / sample_freq
            time = np.arange(0, len(data_in) * sample_period, sample_period)
            plt.plot(time[0:5000] * 1e9, 1e3 * amplitude[0:5000] ** 2)
            plt.xlabel('Time (ns)')
            plt.ylabel('Power (mW)')
            plt.grid()
            plt.show()
    else:
        data_in = -1 + 2 * data_in  # Rescale from [0, 1] to [-1, 1]
        phase = 2 * np.pi * scale * data_in
        phase = quantization_noise(phase, precision)
        phase = butterworth(phase, sample_freq, awg_bandwidth, n=4)
        amplitude = np.sqrt(power)
    
        if plot:
            sample_period = 1 / sample_freq
            time = np.arange(0, len(data_in) * sample_period, sample_period)
            plt.plot(time[0:5000], phase[0:5000])
            plt.xlabel('Time (s)')
            plt.ylabel('Phase (rad)')
            plt.grid()
            plt.show()
    
    field = amplitude * np.exp(1j * phase)
    
    return field

def optical_information_encoding(information, symbol_rate, central_wavelength, simulation_resolution, encoding_type='Phase', run=True):
    """
    Encodes optical information onto a laser field.
    
    Parameters:
    information (numpy array): Input signal.
    symbol_rate (float): Symbol rate.
    central_wavelength (float): Laser central wavelength.
    simulation_resolution (int): Resolution of the simulation.
    encoding_type (str): 'Phase' for phase encoding, other for amplitude encoding.
    run (bool): Whether to run the laser simulation.
    
    Returns:
    tuple: (encoded field, simulation bandwidth, encoding limits)
    """
    P_in_dBm = 6
    samples_per_symbol = simulation_resolution
    dt = 1.0 / (samples_per_symbol * symbol_rate)
    laser_relaxation_time = 20e-9
    silence_time = 20e-9
    pre_points = int(laser_relaxation_time / dt)
    after_points = int(silence_time / dt)
    number_of_samples = len(information) + pre_points + after_points
    
    f_max = 1 / (2 * dt)
    f_min = -1 / (2 * dt)
    df = (f_max - f_min) / number_of_samples
    
    central_frequency = 3e8 / central_wavelength
    simulation_bandwidth = central_frequency + np.arange(f_min, f_max, df)
    
    if run:
        laser_field = run_the_laser(wavelength=central_wavelength, n_samples=number_of_samples, bias_current=17e-3, dt=dt, phase_noise=False)
    else:
        laser_field = np.ones(number_of_samples) + 0 * 1j
    
    P_input = 1e-3 * 10 ** (P_in_dBm / 10)
    field = laser_field * np.sqrt(P_input) / np.sqrt(np.mean(np.abs(laser_field) ** 2))
    
    limit1 = pre_points
    limit2 = pre_points + len(information)
    
    cut_frequency = symbol_rate
    df = simulation_bandwidth[1] - simulation_bandwidth[0]
    sampling_frequency = df * number_of_samples
    normalized_frequency = cut_frequency / (sampling_frequency / 2.0)
    b, a = signal.butter(4, normalized_frequency, 'low')
    information = signal.filtfilt(b, a, information)
    
    if encoding_type == 'Phase':
        field[limit1:limit2] *= np.exp(1j * information)
    else:
        field[limit1:limit2] *= np.sqrt(information)
    
    return field, simulation_bandwidth, [limit1, limit2]
