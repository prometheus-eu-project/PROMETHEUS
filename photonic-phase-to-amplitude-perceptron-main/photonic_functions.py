# Import necessary libraries
# from scipy import signal
# from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
# from digital import pam-mod
from tqdm import tqdm  # For progress bars
# from problem_loader import rz_pulse


def plot_signal_over_transfer_function(inp, tf, simulation_bandwidth):
    """
    Plot the input signal spectrum and transfer function in dB scale.
    
    Args:
        inp: Input signal (time domain)
        tf: Transfer function (frequency domain)
        simulation_bandwidth: Frequency range for plotting
    """
    # Compute input spectrum and convert to dB scale
    inp_spectrum = np.fft.fftshift(np.fft.fft(inp))
    P_spectrum_inp_dB = 10 * np.log10(np.abs(inp_spectrum) ** 2)

    # Convert transfer function to dB scale
    H_dB = 10 * np.log10(np.abs(tf) ** 2)

    # Plot both spectra
    plt.plot(simulation_bandwidth, P_spectrum_inp_dB)
    plt.plot(simulation_bandwidth, H_dB)
    plt.show()


def compute_ring_phase_shift(detuning, radius):
    """
    Compute phase shift in a ring resonator due to detuning.
    
    Args:
        detuning: Frequency detuning from resonance
        radius: Radius of the ring resonator
        
    Returns:
        Phase shift in radians
    """
    n = 4.18  # Refractive index
    c_speed = 299792458  # Speed of light
    L = 2 * np.pi * radius  # Ring circumference
    T = L / (c_speed / n)  # Round-trip time

    return 2 * np.pi * detuning * T


def run_the_laser(wavelength, n_samples, dt, bias_current, phase_noise=True, Pwr_dBm=1):
    """
    Simulate laser dynamics using rate equations.
    
    Args:
        wavelength: Laser wavelength in meters
        n_samples: Number of time samples
        dt: Time step
        bias_current: Laser bias current
        phase_noise: Whether to include phase noise
        Pwr_dBm: Output power in dBm
        
    Returns:
        Complex laser field
    """
    # Laser parameters
    g1 = 1.2e4  # Differential gain coefficient
    s = 5e-7  # Gain saturation coefficient
    tph = 2e-12  # Photon lifetime
    tn = 2e-9  # Carrier lifetime
    N0 = 1.5e8  # Transparency carrier density
    aj = 2.5  # Linewidth enhancement factor
    bb = 200  # Spontaneous emission factor
    Ib = 17e-3  # Reference current
    Po = 5e-3  # Reference power
    q = 1.6e-19  # Electron charge

    # Physical constants
    h = 6.63e-34  # Planck's constant
    f0 = 3e8 / wavelength  # Optical frequency
    nsp = 1.5  # Spontaneous emission factor
    group_velocity = 3e8 / 4  # Group velocity in waveguide
    mirror_losses = 20e2  # Mirror losses
    cavity_length = 300e-6  # Laser cavity length
    round_trip_time = 2 * cavity_length / group_velocity  # Cavity round-trip time
    output_factor = h * f0 * group_velocity * mirror_losses / 2  # Output coupling factor
    angular_frequency = 2 * np.pi * f0  # Angular frequency

    threshold_current = (q / tn) * (N0 + 1 / (g1 * tph))  # Threshold current

    # Initialize arrays
    carriers = np.zeros(n_samples)  # Carrier density
    internal_field = np.zeros(n_samples, dtype='csingle')  # Complex field

    # Main simulation loop
    iterator = tqdm(range(n_samples - 1))
    for i in iterator:
        # Calculate gain and phase change
        GG = g1 * (carriers[i] - N0) / (1 + s * np.abs(internal_field[i]) ** 2)
        ae = 0.5 * (1 + aj * 1j) * (GG - 1 / tph)  # Complex rate
        be = np.sqrt(4 * carriers[i] * bb / dt) * (np.random.rand() + 1j * np.random.rand())  # Noise term
        
        # Carrier rate equation terms
        an = -1 / tn - g1 * np.abs(internal_field[i]) ** 2 / (1 + s * np.abs(internal_field[i]) ** 2)
        bn = bias_current / q + g1 * N0 * np.abs(internal_field[i]) ** 2 / (1 + s * np.abs(internal_field[i]) ** 2)

        # Update field and carriers
        internal_field[i + 1] = internal_field[i] * np.exp(ae * dt) + (be / ae) * (np.exp(ae * dt) - 1)
        carriers[i + 1] = carriers[i] * np.exp(an * dt) + (bn / an) * (np.exp(an * dt) - 1)

    # Remove phase noise if requested
    if phase_noise == False:
        internal_field *= np.exp(-1j * np.angle(internal_field))

    # Normalize power
    Pwr = 1e-3 * 10 ** (Pwr_dBm / 10)  # Convert dBm to Watts
    internal_field *= np.sqrt(Pwr) / np.sqrt(np.mean(np.abs(internal_field) ** 2))

    return internal_field


def span_single_polar(field, n_space_samples, g, dz, Linear):
    """
    Propagate field through a single polarization fiber span.
    
    Args:
        field: Input field
        n_space_samples: Number of spatial steps
        g: Nonlinear coefficient
        dz: Step size
        Linear: Linear propagation operator
        
    Returns:
        Propagated field
    """
    for k in range(0, n_space_samples):
        # Nonlinear step
        intensity = np.abs(field) ** 2
        field = field * np.exp(1j * g * intensity * dz)
        
        # Linear step in frequency domain
        field_spectrum = np.fft.fft(field)
        field_spectrum = np.fft.fftshift(field_spectrum)
        field_spectrum = Linear * field_spectrum
        field_spectrum = np.fft.ifftshift(field_spectrum)
        field = np.fft.ifft(field_spectrum)

    return field


def transmission(field, fiber_length, dz, simulation_bandwidth, central_wavelength):
    """
    Simulate optical fiber transmission with amplification.
    
    Args:
        field: Input optical field
        fiber_length: Total fiber length
        dz: Simulation step size
        simulation_bandwidth: Frequency array
        central_wavelength: Signal wavelength
        
    Returns:
        Field after transmission
    """
    # Physical constants
    h = 6.626070040e-34  # Planck's constant
    nsp = 1.5  # Spontaneous emission factor

    n_temporal_samples = len(field)
    n_space_samples = int(fiber_length / dz)  # Number of spatial steps

    df = simulation_bandwidth[1] - simulation_bandwidth[0]  # Frequency resolution
    dt = 1 / (df * n_temporal_samples)  # Time resolution

    # Spectral properties
    central_frequency = 3e8 / central_wavelength
    w = 2 * np.pi * (simulation_bandwidth - central_frequency)  # Angular frequency detuning

    # Fiber parameters
    D = 4e-6  # Dispersion parameter
    b2 = -(central_wavelength ** 2) * D / (2 * np.pi * 3e8)  # Second order dispersion
    b3 = 0  # Third order dispersion
    g = 1.3e-3  # Nonlinear parameter
    fiber_losses = 0.34 / (4.343 * 1000)  # Loss coefficient
    dz = 1e3  # Simulation step size

    # Linear propagation operator
    Linear = np.exp(-fiber_losses * dz / 2.0 + 1j * ((w ** 2) * b2 * dz / 2.0 + (w ** 3) * b3 * dz / 6.0))

    # Propagate through fiber
    field = span_single_polar(field, n_space_samples, g, dz, Linear)

    """ Coupling Losses """
    field *= np.sqrt(0.4)  # 40% coupling efficiency

    "Amplify the Signal at the output of the fiber"
    gain_amplifier = 10  # Amplifier gain
    field *= np.sqrt(gain_amplifier)  # Apply gain

    # Add amplifier noise
    noise_amplifier_spectral_density = h * central_wavelength * nsp * (gain_amplifier - 1) / dt
    x1, x2 = np.random.randn(len(field)), np.random.randn(len(field))
    field += np.sqrt(noise_amplifier_spectral_density / 2.0) * (x1 + 1j * x2)

    return field


def butterworth_filter(x, order, sampling_frequency, filter_bandwidth):
    """
    Apply Butterworth filter in frequency domain.
    
    Args:
        x: Input signal
        order: Filter order
        sampling_frequency: Sampling rate
        filter_bandwidth: Filter bandwidth
        
    Returns:
        Filtered signal
    """
    nyquist_freq = sampling_frequency / 2
    f = np.linspace(-nyquist_freq, nyquist_freq, len(x))
    H = 1 / (1 + 1j * (f / filter_bandwidth) ** order)  # Butterworth transfer function

    # Filter in frequency domain
    x_spectrum = np.fft.fftshift(np.fft.fft(x))
    out_spectrum = H * x_spectrum
    out = np.fft.ifft(np.fft.ifftshift(out_spectrum))

    return np.real(out)


class Beamsplitter:
    """Class representing a beam splitter optical component."""

    def __init__(self, coupling) -> None:
        """
        Initialize beamsplitter with given coupling coefficient.
        
        Args:
            coupling: Power coupling coefficient (0-1)
        """
        self.transfer_function = None
        self.coupling_coeff = np.sqrt(coupling)  # Field coupling coefficient

    def compute_transfer_matrix(self):
        """Compute the transfer matrix of the beamsplitter."""
        cross = np.sqrt(1 - self.coupling_coeff ** 2)  # Cross-coupling coefficient

        # Beamsplitter transfer matrix
        T = np.array([[self.coupling_coeff, -1j * cross], [-1j * cross, self.coupling_coeff]])

        self.transfer_function = T

    def propagate_field(self, field):
        """
        Propagate field through beamsplitter.
        
        Args:
            field: Input field (single port)
            
        Returns:
            Output fields from both ports
        """
        in_port = np.array([field, np.zeros(len(field), dtype='csingle')])  # Second port empty
        output = np.matmul(self.transfer_function, in_port)  # Apply transfer matrix

        return output


class Photodiode:
    """Class representing a photodetector with noise sources."""

    def __init__(self, B, RL=50, RS=1, I_dark=1e-9, temperature=300) -> None:
        """
        Initialize photodiode with given parameters.
        
        Args:
            B: Bandwidth
            RL: Load resistance
            RS: Responsivity
            I_dark: Dark current
            temperature: Operating temperature
        """
        self.B = B  # Bandwidth
        self.RL = RL  # Load resistance
        self.RS = RS  # Responsivity
        self.I_dark = I_dark  # Dark current
        self.q = 1.6e-19  # Electron charge
        self.h = 6.626070040e-34  # Planck's constant
        self.c = 299792458  # Speed of light
        self.boltzmann_constant = 1.380649e-23  # Boltzmann constant
        self.temperature = temperature  # Temperature in Kelvin

    def read(self, field, sample_freq, enable_noise, central_wavelength=1550e-9):
        """
        Detect optical field and convert to electrical signal.
        
        Args:
            field: Input optical field
            sample_freq: Sampling frequency
            enable_noise: Whether to include noise
            central_wavelength: Optical wavelength
            
        Returns:
            Filtered photocurrent
        """
        new_field = field + 0  # Copy input field
        dt = 1 / sample_freq  # Time step

        # Calculate quantum noise power
        nQN = enable_noise * 0.5 * self.h * (self.c / central_wavelength) / dt

        # Add shot noise
        new_field += np.sqrt(nQN / 2) * (np.random.randn(len(field)) + 1j * np.random.randn(len(field)))

        # Convert to photocurrent
        intensity = np.abs(new_field) ** 2
        photocurrent = self.RS * intensity

        # Add thermal noise
        nTH = enable_noise * 4 * self.boltzmann_constant * self.temperature / (self.RL * dt)
        photocurrent += np.sqrt(nTH) * np.random.randn(len(intensity))

        # Calculate SNR (for reference)
        SNR = 10 * np.log10(np.mean(np.abs(photocurrent) ** 2) / nTH)

        # Apply anti-aliasing filter
        filtered_photocurrent = butterworth_filter(photocurrent, 4, sample_freq, self.B)

        return filtered_photocurrent


class MRR:
    """Class representing a microring resonator (MRR)."""

    def __init__(self, radius_vector, coupling_vector, ring_phases=None) -> None:
        """
        Initialize MRR with given parameters.
        
        Args:
            radius_vector: Array of ring radii
            coupling_vector: Array of coupling coefficients
            ring_phases: Optional array of static phase shifts
        """
        if ring_phases is None:
            ring_phases = []

        self.n_stages = len(coupling_vector)  # Number of rings

        # Calculate ring round-trip times
        L_ring = 2 * np.pi * np.array(radius_vector)  # Circumferences
        n = 4.18  # Refractive index
        c = 299792458  # Speed of light

        L_ring = np.append(0.0, L_ring)  # Add zero length for first stage
        self.T_ring = n * L_ring / c  # Round-trip times
        self.L_ring = L_ring  # Store lengths
        self.photon_lifetime = np.zeros(len(self.L_ring))  # Initialize lifetimes

        # Set ring phases (default to zero)
        if len(ring_phases) == 0:
            self.ring_phases = np.zeros(len(self.T_ring))
        else:
            self.ring_phases = np.append(0.0, ring_phases)

        self.coupling_vector = coupling_vector  # Store coupling coefficients

    def compute_transfer_matrix(self, f):
        """
        Compute MRR transfer matrix for given frequency array.
        
        Args:
            f: Frequency array
        """
        alpha_dB_m = 4  # Loss coefficient in dB/m
        n = 4.18  # Refractive index
        c_speed = 299792458  # Speed of light
        gamma = 10 ** (-alpha_dB_m * self.L_ring / 20)  # Loss factor

        """ Create the stage matrices"""
        # Initialize transfer matrix elements
        T11 = np.zeros((self.n_stages, len(f))) + 0*1j
        T12 = np.zeros((self.n_stages, len(f))) + 0*1j
        T21 = np.zeros((self.n_stages, len(f))) + 0*1j
        T22 = np.zeros((self.n_stages, len(f))) + 0*1j

        # Compute the stages of the MRR 
        for stage in range(0, self.n_stages):
            index = self.n_stages -1 - stage  # Work backwards

            # Round-trip phase and loss
            z = np.exp(1j*(2*np.pi*f*self.T_ring[stage] + self.ring_phases[stage]))
            zeta = gamma[stage]/z  # Combined phase and loss

            k = self.coupling_vector[stage]  # Coupling coefficient
            c, s = np.sqrt(1-k), np.sqrt(k)  # Field coupling coefficients
            common_denominator = -1j*s*np.sqrt(zeta)

            # Compute transfer matrix elements for this stage
            T11[index, :] =  1 /common_denominator
            T12[index, :] =  -c /common_denominator
            T21[index, :] = c*zeta /common_denominator
            T22[index, :] = -zeta/common_denominator

        # Calculate the whole matrix by multiplying stage matrices
        h11, h12, h21, h22 = 1 +0*1j, 0 + 0*1j, 0 + 0*1j, 1 + 0*1j  # Initialize as identity

        for stage in range(0, self.n_stages):
            # Matrix multiplication
            h11_new = T11[stage,:]*h11 + T12[stage,:]*h21
            h12_new = T11[stage,:]*h12 + T12[stage,:]*h22
            h21_new = T21[stage,:]*h11 + T22[stage,:]*h21
            h22_new = T21[stage,:]*h12 + T22[stage,:]*h22

            h11, h12, h21, h22 = h11_new, h12_new , h21_new, h22_new

        # Store transfer matrix elements
        self.h11 = h11
        self.h12 = h12
        self.h21 = h21
        self.h22 = h22

    def propagate(self, first_input, second_input=None):
        """
        Propagate fields through MRR.
        
        Args:
            first_input: Input field to first port
            second_input: Optional input to second port
            
        Returns:
            Output fields from through and drop ports
        """
        if second_input is None:
            second_input = []
        if len(second_input) == 0:
            second_input = np.zeros(len(first_input)) + 0 * 1j

        a = self.tf_a
        b = self.tf_b
        c = self.tf_c
        d = self.tf_d

        # Convert to frequency domain
        first_input_spectrum = np.fft.fftshift(np.fft.fft(first_input))
        second_input_spectrum = np.fft.fftshift(np.fft.fft(second_input))

        # Apply transfer functions
        drop_output_spectrum = (first_input_spectrum - b * second_input_spectrum) / a
        through_output_spectrum = c * drop_output_spectrum + d * second_input_spectrum

        # Convert back to time domain
        drop_output = np.fft.ifft(np.fft.ifftshift(drop_output_spectrum))
        through_output = np.fft.ifft(np.fft.ifftshift(through_output_spectrum))

        return through_output, drop_output


class MrrRcNode:
    """Class representing a MRR-based reservoir computing node."""

    def __init__(self, loop_delay, phase, radius_vector, coupling_vector, feedback_strength=1,
                 ring_phases=None, loop_type="T2T") -> None:
        """
        Initialize RC node with given parameters.
        
        Args:
            loop_delay: Feedback loop delay
            phase: Feedback phase shift
            radius_vector: Ring radii
            coupling_vector: Coupling coefficients
            feedback_strength: Feedback strength
            ring_phases: Static ring phase shifts
            loop_type: Feedback configuration type
        """
        if ring_phases is None:
            ring_phases = []
        self.h_rc_drop = None  # Drop port transfer function
        self.h_rc_through = None  # Through port transfer function
        self.loop_delay = loop_delay  # Feedback delay
        self.phase = phase  # Feedback phase
        self.MRR = MRR(radius_vector=radius_vector, coupling_vector=coupling_vector, ring_phases=ring_phases)
        self.feedback_strength = feedback_strength  # Feedback strength
        self.splitter = Beamsplitter(coupling=0.9)  # Internal beamsplitter
        self.loop_type = loop_type  # Feedback configuration type

    def compute_transfer_matrix(self, sample_freq, signal_length, plot=False):
        """
        Compute node transfer functions.
        
        Args:
            sample_freq: Sampling frequency
            signal_length: Signal length
            plot: Whether to plot transfer functions
        """
        nyquist_freq = sample_freq / 2
        f = np.linspace(-nyquist_freq, nyquist_freq, signal_length)

        # Compute MRR transfer functions
        self.MRR.compute_transfer_matrix(f)

        # MRR transfer_function elements
        a = self.MRR.h11
        b = self.MRR.h12
        c = self.MRR.h21
        d = self.MRR.h22

        # Feedback phase delay
        phase_delay = 2 * np.pi * self.loop_delay * f + self.phase
        k = self.feedback_strength
        L = np.sqrt(k)*np.exp(-1j*phase_delay)  # Feedback loop transfer function

        # Calculate transfer functions based on loop type
        if self.loop_type == "T2D":
            denominator = a + b*L*c/(1-d*L)
            nominator = 1
            h_rc_drop = nominator / denominator
            h_rc_through = 0*h_rc_drop

        elif self.loop_type == "T2T":
            nominator = 0.5 * (c / a)
            denominator = 1 + 0.5 * ((c / a) * np.sqrt(k) * np.exp(-1j * phase_delay))
            h_rc_through = nominator / denominator
            h_rc_drop = np.sqrt(2) * h_rc_through / c

        elif self.loop_type == "Fano":
            h_rc_drop = (1 - b * np.sqrt(k) * 1j * np.exp(-1j * phase_delay)) / (np.sqrt(2) * a)
            h_rc_through = (c * h_rc_drop + d * np.sqrt(k) * 1j * np.exp(-1j * phase_delay) / np.sqrt(2))
        else:
            h_rc_drop = []
            h_rc_through = []

        # Plot if requested
        if plot:
            plt.plot(f / 1e12, 10 * np.log10(np.abs(h_rc_drop) ** 2))
            plt.xlabel('Frequency (THz)')
            plt.ylabel('Amplitude (dB)')
            plt.show()

        # Store transfer functions
        self.h_rc_drop = h_rc_drop
        self.h_rc_through = h_rc_through

    def propagate_field(self, field, sample_freq, plot=False):
        """
        Propagate field through the RC node.
        
        Args:
            field: Input field
            sample_freq: Sampling frequency
            plot: Whether to plot spectra
            
        Returns:
            Output fields from drop and through ports
        """
        # Convert to frequency domain
        field_spectrum = np.fft.fftshift(np.fft.fft(field))

        # Apply transfer functions
        out_field_spectrum_drop = field_spectrum * self.h_rc_drop
        out_field_spectrum_through = field_spectrum * self.h_rc_through

        # Convert back to time domain
        out_field_drop = np.fft.ifft(np.fft.ifftshift(out_field_spectrum_drop))
        out_field_through = np.fft.ifft(np.fft.ifftshift(out_field_spectrum_through))

        # Plot spectra if requested
        if plot:
            P_dB = 10 * np.log10(np.abs(field_spectrum + 1e-30) ** 2)
            P_dB -= np.max(P_dB)  # Normalize
            I_transfer = np.abs(self.h_rc_through) ** 2
            I_transfer_drop = np.abs(self.h_rc_drop) ** 2
            nyquist_freq = sample_freq / 2
            f = np.linspace(-nyquist_freq, nyquist_freq, len(field))

            plt.plot(f / 1e12, P_dB, label='signal')
            plt.plot(f / 1e12, 10 * np.log10(I_transfer + 1e-18), label='Through')
            plt.plot(f / 1e12, 10 * np.log10(I_transfer_drop + 1e-18), label='Drop')
            plt.legend(loc="upper left")
            plt.xlabel('Frequency (THz)')
            plt.ylabel('Transmission (dB)')
            plt.show()

        return out_field_drop, out_field_through


class ReservoirLayer:
    """Class representing a reservoir computing layer with multiple nodes."""

    def __init__(self, nodes, t_delay, loop_type, loop_phase, sample_freq, signal_length, bandwidth) -> None:
        """
        Initialize reservoir layer.
        
        Args:
            nodes: Number of nodes
            t_delay: Feedback delay
            loop_type: Feedback configuration type
            loop_phase: Feedback phase
            sample_freq: Sampling frequency
            signal_length: Signal length
            bandwidth: Photodiode bandwidth
        """
        self.parameter_vector = None  # Current parameter vector
        self.reservoir_nodes = None  # List of nodes
        self.n_parameters = None  # Total number of parameters
        self.chart = None  # Parameter structure
        self.nodes = nodes  # Number of nodes
        self.t_delay = t_delay  # Feedback delay
        self.loop_phase = loop_phase  # Feedback phase
        self.loop_type = loop_type  # Feedback configuration
        self.parameters = 0  # Parameter counter
        self.sample_freq = sample_freq  # Sampling frequency
        self.signal_length = signal_length  # Signal length
        self.pd = Photodiode(B=bandwidth)  # Photodetector

    def generate_nodes(self, radius_vectors, coupling_vectors, feedback_strengths, ring_phases):
        """
        Generate reservoir nodes with given parameters.
        
        Args:
            radius_vectors: List of radius arrays for each node
            coupling_vectors: List of coupling arrays for each node
            feedback_strengths: List of feedback strengths
            ring_phases: List of phase arrays
            
        Returns:
            Hyperparameter vector
        """
        reservoir_nodes = []
        chart = []
        hyperparameter_vector = np.array([])

        # Create each node
        for node in range(0, self.nodes):
            reservoir_nodes.append(
                MrrRcNode(self.t_delay, self.loop_phase, radius_vectors[node], coupling_vectors[node],
                          feedback_strengths[node], ring_phases[node], self.loop_type))
            
            # Track parameter counts
            chart.append(len(radius_vectors[node]))
            chart.append(len(coupling_vectors[node]))
            chart.append(1)  # For feedback strength
            chart.append(len(ring_phases[node]))

            # Build parameter vector
            hyperparameter_vector = np.append(hyperparameter_vector, radius_vectors[node])
            hyperparameter_vector = np.append(hyperparameter_vector, coupling_vectors[node])
            hyperparameter_vector = np.append(hyperparameter_vector, feedback_strengths[node])
            hyperparameter_vector = np.append(hyperparameter_vector, ring_phases[node])

        self.chart = np.array(chart)
        self.n_parameters = int(np.sum(chart))
        self.reservoir_nodes = reservoir_nodes

        return hyperparameter_vector

    def update_parameters(self, parameter_vector, plot=False):
        """
        Update node parameters from parameter vector.
        
        Args:
            parameter_vector: New parameters
            plot: Whether to plot transfer functions
        """
        chart_index = 0
        helper = 0

        self.parameter_vector = parameter_vector

        # Distribute parameters to each node
        for node in range(0, self.nodes):
            # Update radii
            self.reservoir_nodes[node].radius_vector = parameter_vector[helper:helper + self.chart[chart_index]]
            helper += self.chart[chart_index]
            chart_index += 1
            
            # Update couplings
            self.reservoir_nodes[node].coupling_vector = parameter_vector[helper:helper + self.chart[chart_index]]
            helper += self.chart[chart_index]
            chart_index += 1
            
            # Update feedback strength
            self.reservoir_nodes[node].feedback_strength = parameter_vector[helper:helper + self.chart[chart_index]]
            helper += self.chart[chart_index]
            chart_index += 1
            
            # Update phases
            self.reservoir_nodes[node].ring_phases = parameter_vector[helper:helper + self.chart[chart_index]]
            helper += self.chart[chart_index]
            chart_index += 1

            # Recompute transfer functions
            self.reservoir_nodes[node].compute_transfer_matrix(sample_freq=self.sample_freq,
                                                               signal_length=self.signal_length,
                                                               plot=plot)

    def propagate(self, input_field, output_port, enable_noise, plot=False):
        """
        Propagate input through reservoir layer.
        
        Args:
            input_field: Input optical field
            output_port: Which port to read ('Drop', 'Through', or 'Differential')
            enable_noise: Whether to include noise
            plot: Whether to plot node responses
            
        Returns:
            Reservoir output matrix (nodes x samples)
        """
        reservoir_output = np.zeros((self.nodes, len(input_field)))

        # Normalize input power
        input_field /= np.sqrt(self.nodes)

        # Process through each node
        for node in range(0, self.nodes):
            # Get outputs from node
            out_field_drop, out_field_through = \
                self.reservoir_nodes[node].propagate_field(field=input_field,
                                                           sample_freq=self.sample_freq,
                                                           plot=plot)

            # Read from specified port
            if output_port == 'Drop':
                detected_power = self.pd.read(field=out_field_drop,
                                              sample_freq=self.sample_freq,
                                              enable_noise=enable_noise)
            elif output_port == 'Through':
                detected_power = self.pd.read(field=out_field_through,
                                              sample_freq=self.sample_freq,
                                              enable_noise=enable_noise)
            elif output_port == 'Differential':
                d_through = self.pd.read(field=out_field_through,
                                         sample_freq=self.sample_freq,
                                         enable_noise=enable_noise)
                d_drop = self.pd.read(field=out_field_drop,
                                      sample_freq=self.sample_freq,
                                      enable_noise=enable_noise)
                detected_power = d_through - d_drop
            else:
                print("Linear output !!!")
                detected_power = self.pd.read(field=input_field, sample_freq=self.sample_freq,
                                              enable_noise=enable_noise)

            # Store node output
            reservoir_output[node, :] = detected_power

        return reservoir_output