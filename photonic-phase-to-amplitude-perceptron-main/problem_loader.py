import numpy as np
import matplotlib.pyplot as plt
import scipy.io as loader  # For loading MATLAB files
from scipy.signal import resample  # For signal resampling
from sklearn.linear_model import Ridge  # For ridge regression


class SeriesGenerator:
    """Class for generating time series data for reservoir computing tasks."""
    
    def __init__(self, n_training, n_testing, n_ignore=1000) -> None:
        """
        Initialize series generator.
        
        Args:
            n_training: Number of training samples
            n_testing: Number of testing samples
            n_ignore: Number of initial samples to ignore (transient)
        """
        self.n_training = n_training
        self.n_testing = n_testing
        self.n_ignore = n_ignore
        self.x = []  # Input data
        self.x_expanded = []  # Masked input data
        self.y = []  # Target data
        self.data_starting_point = 0  # Starting index after transient

    def narma10(self):
        """Generate NARMA-10 time series (nonlinear autoregressive moving average)."""
        n_total = self.n_testing + self.n_training + self.n_ignore

        # Generate random input signal
        x = 0.5 * np.random.rand(n_total)
        y = np.zeros(len(x))  # Initialize output

        # NARMA-10 equation implementation
        j = np.linspace(0, 9, 10).astype(int)  # Delay indices
        for i in range(9, len(y) - 1):
            y[i + 1] = 0.3 * y[i] + 0.05 * y[i] * np.sum(y[i - j]) + 1.5 * x[i - 9] * x[i] + 0.1

        self.x, self.y = x, y

    def santa_fe(self):
        """Load Santa Fe time series (laser intensity data)."""
        data = loader.loadmat('santa.mat')['santa']  # Load from MATLAB file

        n_total = self.n_testing + self.n_training + self.n_ignore

        # Create input-output pairs (current value predicts next value)
        x = data[0:n_total]
        y = data[1:n_total + 1]

        self.x, self.y = x, y


def rz_pulse(time, peak, duty_cycle):
    """
    Generate a return-to-zero (RZ) pulse.
    
    Args:
        time: Time array
        peak: Pulse amplitude
        duty_cycle: Pulse duty cycle (0-1)
        
    Returns:
        RZ pulse waveform
    """
    L_pulse = int(len(time) * duty_cycle)  # Pulse duration
    L_silence = int(len(time) * (1 - duty_cycle))  # Zero duration

    pulse = np.zeros(len(time))  # Initialize pulse

    # Center pulse in the time window
    start = int(0.5 * L_silence)
    end = start + L_pulse

    pulse[start:end] = peak  # Set pulse amplitude

    return pulse


def create_masked_input(task, mask_samples, points_per_symbol, mask_sampling_frequency, de_synchronization_parameter=0,
                        encoding_method='ZOH'):
    """
    Create masked input signal for reservoir computing.
    
    Args:
        task: SeriesGenerator object with input data
        mask_samples: Number of mask samples per input symbol
        points_per_symbol: Number of time points per symbol
        mask_sampling_frequency: Mask sampling frequency
        de_synchronization_parameter: Delay parameter
        encoding_method: 'ZOH' (zero-order hold) or 'RZ' (return-to-zero)
        
    Returns:
        analog_input: Generated analog signal
        T_delay: Effective delay time
        fs: Sampling frequency
    """
    # Calculate timing parameters
    theta = 1 / mask_sampling_frequency
    dt = theta / points_per_symbol  # Simulation time step
    T_mask = theta * mask_samples  # Mask symbol duration
    fs = 1 / dt  # Sampling frequency
    
    # Set starting point after transient
    task.data_starting_point = task.n_ignore * mask_samples - 1

    """ Generate the masked input """
    T_delay = T_mask * (mask_samples + de_synchronization_parameter) / mask_samples

    # Create random mask
    mask = np.random.rand(mask_samples)

    # Expand input with mask
    x_expanded = np.zeros(len(task.x) * mask_samples)
    for i in range(0, len(task.x)):
        x_expanded[i * mask_samples:(i + 1) * mask_samples] = task.x[i] * mask

    task.x_expanded = x_expanded
    
    """ Create the analog signal """
    x_expanded /= np.max(np.abs(x_expanded))  # Normalize signal
    analog_input = np.zeros(len(x_expanded) * points_per_symbol)

    if encoding_method == 'RZ':
        # Generate RZ pulses
        for i in range(0, len(x_expanded)):
            pulse_time = np.arange(0, points_per_symbol, 1)
            analog_input[i * points_per_symbol:(i + 1) * points_per_symbol] = \
                rz_pulse(pulse_time, peak=x_expanded[i], duty_cycle=0.5)
    else:
        # Zero-order hold (constant value between samples)
        for i in range(0, len(x_expanded)):
            analog_input[i * points_per_symbol:(i + 1) * points_per_symbol] = x_expanded[i]

    return analog_input, T_delay, fs


def post_processing(signal, task, taps, points_per_symbol, plot):
    """
    Process reservoir output for training/validation.
    
    Args:
        signal: Reservoir output signal
        task: SeriesGenerator object
        taps: Number of temporal taps to consider
        points_per_symbol: Points per input symbol
        plot: Whether to plot input-output relationship
        
    Returns:
        out: Processed feature matrix
    """
    n_symbols = task.n_training + task.n_testing

    masked_samples = int(task.data_starting_point / (task.n_ignore - 1))
    
    # Handle single-node case
    if len(np.shape(signal)) == 1:
        temp_signal = signal + 0
        signal = np.zeros((1, len(signal)))
        signal[0, :] = temp_signal

    nodes, _ = np.shape(signal)
    out = np.zeros((n_symbols, taps * masked_samples * nodes))

    """ Create training and testing datasets """
    start = int(points_per_symbol/2)  # Sample at center of symbol
    sampled_data = signal[:, start::points_per_symbol]  # Downsample
    
    # Normalize data
    X = (sampled_data - np.mean(sampled_data)) / np.std(sampled_data)

    # Plot input-output relationship if requested
    if plot:
        plt.scatter(task.x_expanded[1000:2000], sampled_data[0, 1000:2000] * 1e3)
        plt.ylabel('Power (mW)')
        plt.xlabel('Input (a.u.)')
        plt.grid()
        plt.show()

    # Create delay taps
    taping_range = np.arange(-taps*masked_samples +1, 1, 1)

    # Construct feature matrix with temporal taps
    cntr = task.data_starting_point + masked_samples
    for i in range(0, n_symbols):
        for k in range(0, nodes):
            out[i, k * masked_samples * taps:(k + 1) * masked_samples * taps] = \
                X[k, cntr + taping_range]
            cntr += masked_samples

    return out


def feedforward_equalizer(x, task, memory_capacity_calculations=0):
    """
    Train and evaluate ridge regression model.
    
    Args:
        x: Feature matrix
        task: SeriesGenerator object
        memory_capacity_calculations: Flag for memory capacity mode
        
    Returns:
        Either NMSE scores or predictions based on mode
    """
    # Split data into training and testing
    x_train = x[0:task.n_training, :]
    y_train = task.y[task.n_ignore:task.n_ignore + task.n_training]

    """ Perform the Equalization """
    def normalized_mean_square_error(yp, y):
        """Calculate normalized mean squared error."""
        c_sum = 0
        var_y = np.std(y) ** 2
        Nd = len(y)
        for i in range(0, len(y)):
            c_sum += (yp[i] - y[i]) ** 2

        return c_sum / (Nd * var_y)

    # Train ridge regression model
    ridge = Ridge(alpha=1e-4)
    ridge.fit(x_train, y_train)
    y_train_predict = ridge.predict(x_train)

    # Calculate training NMSE
    nmse_train = normalized_mean_square_error(y_train_predict, y_train)
    
    # Evaluate on test set
    x_test = x[task.n_training:task.n_training + task.n_testing, :]
    y_test = task.y[task.n_ignore + task.n_training: task.n_ignore + task.n_training + task.n_testing]
    y_test_predict = ridge.predict(x_test)
    nmse_test = normalized_mean_square_error(y_test_predict, y_test)

    if memory_capacity_calculations:
        return y_test, y_test_predict  # For memory capacity calculation
    else:
        return nmse_train, nmse_test  # For normal operation


def memory_capacity(x, task, memory_depth=13):
    """
    Calculate memory capacity of the system.
    
    Args:
        x: Feature matrix
        task: SeriesGenerator object
        memory_depth: Maximum delay to evaluate
    """
    memory_capacity_vector = np.zeros(memory_depth)
    y_hold = task.y + 0  # Save original target

    # Evaluate memory for different delays
    for d in range(0, memory_depth):
        task.y = np.roll(y_hold, d)  # Shift target by delay d
        y_desired, y_predict = feedforward_equalizer(x, task, memory_capacity_calculations=1)
        r = np.corrcoef(y_desired, y_predict)  # Calculate correlation
        memory_capacity_vector[d] = r[0, 1]  # Store correlation coefficient

    # Restore original target
    task.y = y_hold + 0

    # Plot memory capacity curve
    plt.plot(memory_capacity_vector)
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Lag')
    plt.grid()
    plt.show()