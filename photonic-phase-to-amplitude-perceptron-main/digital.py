import numpy as np

def sample_and_mask(detected_signal, virtual_nodes, samples_per_symbol, n_symbols, simulation_resolution, dt, theta):
    """
    Samples and masks the detected signal using a random mask for processing.
    
    Parameters:
    detected_signal (numpy array): The input signal to be processed.
    virtual_nodes (int): Number of virtual nodes for masking.
    samples_per_symbol (int): Number of samples per symbol.
    n_symbols (int): Total number of symbols.
    simulation_resolution (int): Resolution for simulation.
    dt (float): Time step for sampling.
    theta (float): Parameter for mapping signal to phase.
    
    Returns:
    numpy array: The phase-modulated signal after masking.
    """
    # Downsample the detected signal
    downsampled_signal = detected_signal[0:-1:samples_per_symbol + 1]
    
    # Normalize the signal between 0 and 1
    signal_normalized = (downsampled_signal - np.min(downsampled_signal)) / (np.max(downsampled_signal) - np.min(downsampled_signal))
    
    # Create a random mask matrix
    mask = np.random.randn(virtual_nodes, samples_per_symbol)
    mask_length = int(n_symbols * virtual_nodes)
    masked_signal = np.zeros(mask_length)
    
    # Apply the mask to the signal
    for i in range(0, n_symbols):
        vector = signal_normalized[i * samples_per_symbol:(i + 1) * samples_per_symbol]
        masked_signal[i * virtual_nodes: (i + 1) * virtual_nodes] = np.dot(mask, vector)
    
    # Expand the masked signal in time domain
    message = np.zeros(n_symbols * virtual_nodes * simulation_resolution)
    points_per_mask_symbol = int(theta / dt)
    
    for i in range(0, len(masked_signal)):
        message[i * points_per_mask_symbol: (i + 1) * points_per_mask_symbol] = masked_signal[i]
    
    # Map the message to phase values between -π and π
    phase_message = -np.pi + 2 * np.pi * message
    
    return phase_message

def bit_error_rate(y, y_ideal):
    """
    Computes the Bit Error Rate (BER) between two signals.
    
    Parameters:
    y (numpy array): The received signal.
    y_ideal (numpy array): The ideal reference signal.
    
    Returns:
    float: The computed bit error rate.
    """
    difference = np.equal(y, y_ideal)  # Compare the received and ideal signals
    error = 1 - difference.astype(int)  # Compute error (1 if different, 0 if same)
    ber = np.mean(error)  # Calculate mean error rate
    
    return ber

def pammod(y, M):
    """
    Performs Pulse Amplitude Modulation (PAM) on input signal.
    
    Parameters:
    y (numpy array): Input signal.
    M (int): Modulation order.
    
    Returns:
    numpy array: Modulated signal.
    """
    d = M / 2  # Compute scaling factor
    y = y * d  # Scale input values
    y = y - (M - 1)  # Adjust levels to PAM range
    
    return y

def pamdemod(y, M):
    """
    Performs Pulse Amplitude Demodulation (PAM) on input signal.
    
    Parameters:
    y (numpy array): Modulated signal.
    M (int): Modulation order.
    
    Returns:
    numpy array: Demodulated signal.
    """
    # Clip values that exceed PAM range
    cut_down = np.where(y < -(M - 1))
    cut_up = np.where(y > (M - 1))
    
    if len(cut_up[0]) != 0:
        y[cut_up[0]] = 3
    if len(cut_down[0]) != 0:
        y[cut_down[0]] = -3
    
    # Compute quantization step
    d = (np.max(y) - np.min(y)) / (M - 1)
    y -= np.min(y)
    
    # Normalize and round to nearest integer symbol
    result = np.round(y / (d + 1e-38))
    
    return result.astype(int)

def butterworth(x, samples, pulse_duration, N, fc, f0):
    """
    Applies a Butterworth filter to the input signal in the frequency domain.
    
    Parameters:
    x (numpy array): Input signal.
    samples (int): Number of samples.
    pulse_duration (float): Duration of the pulse.
    N (int): Order of the Butterworth filter.
    fc (float): Cutoff frequency.
    f0 (float): Center frequency.
    
    Returns:
    numpy array: Filtered signal.
    """
    dt = pulse_duration / samples  # Compute time step
    r = np.arange(-samples / 2, samples / 2, 1)  # Frequency range for FFT
    f = r / samples  # Normalize frequency range
    f = f / dt  # Convert to actual frequency
    
    # Compute Butterworth filter transfer function
    Hpar = 1 + 1j * (((f - f0) / fc) ** N)
    H = Hpar ** (-1)
    
    # Apply filter in frequency domain
    x = np.fft.fft(x)
    x = np.fft.fftshift(x)
    x = H * x
    x = np.fft.ifftshift(x)
    x = np.fft.ifft(x)
    
    return x
