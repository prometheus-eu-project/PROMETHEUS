# --- Imports ---
import itertools
import numpy as np
import pickle
import tqdm
import scipy.io as io
from scipy import signal
from scaling_utils import max_min_scaler, standard_scaler
from signal_processing_utils import symbol_error_rate, correlation
from machine_learning_utils import classification
from modulation_utils import QAMModem
from io_utils import read_all_files, read_target_file
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt


# --- Control Parameters for Signal Processing ---
class ControlParameters:
    def __init__(self, 
                 n_filters, 
                 taps=31, 
                 n_ignore=1000, 
                 averaging=5, 
                 sample_freq_upsample=160e9, 
                 sample_freq_signal=32e9,
                 sample_freq_oscilloscope=80e9, 
                 sample_freq_awg=64e9, 
                 low_pass_filter_bandwidth=18e9):
        """
        Class to group and manage configuration parameters for signal processing and filtering.
        """
        self.n_filters = n_filters
        self.taps = taps
        self.n_ignore = n_ignore
        self.averaging = averaging
        self.sample_freq_upsample = sample_freq_upsample
        self.sample_freq_signal = sample_freq_signal
        self.sample_freq_oscilloscope = sample_freq_oscilloscope
        self.sample_freq_awg = sample_freq_awg
        self.low_pass_filter_bandwidth = low_pass_filter_bandwidth


# --- Data Processor Setup Function ---
def prepare_data_processor(config):
    """
    Initializes and returns a configured SignalDataProcessor object for given config.
    """
    control_parameters = ControlParameters(
        n_filters=config['n_filters'], 
        taps=config['taps'], 
        averaging=config["averaging"]
    )

    file_names, phases = read_all_files(directory=config['main_directory'])
    data_inp_awg = read_target_file(directory=config['test_vector_file'])
    modulation = QAMModem(M=config['Modulation'])

    data_processor = SignalDataProcessor(
        config['main_directory'], 
        file_names, 
        modulation, 
        control_parameters
    )

    # Prepare input and reference data for correlation
    data_processor.preprocess_input_data(data_inp_awg)
    data_processor.load_or_compute_correlation_data(apply_filter=True, load_existing_data=True)
    data_processor.phases = phases

    return data_processor


# --- Main Class for Processing Photonic Signals ---
class SignalDataProcessor:
    def __init__(self, data_directory, file_names, modulation_scheme, config_params):
        """
        Initializes internal parameters and filtering configurations.
        """
        self.config_params = config_params
        self.num_samples_to_ignore = self.config_params.n_ignore

        # Design Butterworth filter for low-pass filtering
        low_pass_ratio = self.config_params.low_pass_filter_bandwidth / (self.config_params.sample_freq_oscilloscope / 2)
        b, a = signal.butter(4, low_pass_ratio, btype='low')
        self.butter_filter_params = [b, a]

        self.data_directory = data_directory
        self.file_names = file_names
        self.modulation_scheme = modulation_scheme
        self.correlation_data_paths = []
        self.phases = []

    def resample_signal(self, signal_data, original_sample_freq):
        """
        Upsample a given signal to match the internal processing rate.
        """
        new_length = int(len(signal_data) * self.config_params.sample_freq_upsample / original_sample_freq)
        return signal.resample(signal_data, new_length)

    def preprocess_input_data(self, awg_input_data):
        """
        Align and resample AWG input signal. Apply standard scaling and ignore edge samples.
        """
        sampling_ratio = self.config_params.sample_freq_awg / self.config_params.sample_freq_signal
        processed_input_data = awg_input_data[0::int(sampling_ratio)]
        upsampled_input_data = self.resample_signal(awg_input_data, self.config_params.sample_freq_awg)
        scaled_target_data = standard_scaler(
            processed_input_data[self.num_samples_to_ignore:len(processed_input_data) - self.num_samples_to_ignore]
        )

        self.num_input_samples = len(processed_input_data)
        self.correlation_reference_signal = upsampled_input_data
        self.target_data = scaled_target_data

    def apply_tapping(self, input_data):
        """
        Apply time-domain tapping (sliding window) across signal data to extract temporal features.
        """
        taps = self.config_params.taps
        num_samples = self.num_input_samples
        num_samples_to_ignore = self.num_samples_to_ignore
        tapped_data = np.zeros((num_samples, taps))
        tap_range = np.arange(-int(taps / 2), int(taps / 2) + 1)

        for i in range(num_samples_to_ignore, num_samples - num_samples_to_ignore):
            position = i + tap_range.astype(int)
            tapped_data[i, :] = input_data[position]

        return tapped_data[num_samples_to_ignore:num_samples - num_samples_to_ignore]

    def calculate_weights(self, feature_matrix):
        """
        Trains two regression models (real and imaginary) and evaluates error and weights.
        """
        real_target = np.real(self.target_data)
        imag_target = np.imag(self.target_data)

        real_res, real_pred, real_weights, _ = classification(feature_matrix, real_target)
        imag_res, imag_pred, imag_weights, _ = classification(feature_matrix, imag_target)

        complex_res = real_res + 1j * imag_res
        complex_pred = real_pred + 1j * imag_pred

        mse = np.mean(np.abs(complex_pred - complex_res) ** 2)
        demod_res = self.modulation_scheme.demodulate(complex_res)
        demod_pred = self.modulation_scheme.demodulate(complex_pred)
        ser = symbol_error_rate(demod_res, demod_pred)

        return mse, ser, [real_weights, imag_weights]

    def load_or_compute_correlation_data(self, apply_filter=True, load_existing_data=False, num_files_to_keep=20):
        """
        Compute correlation between each measurement and reference input, or load from disk.
        """
        correlation_data_all_files = []

        if not load_existing_data:
            reference = self.correlation_reference_signal
            for i in tqdm.tqdm(range(len(self.file_names)), desc="Computing correlation files:"):
                corr_per_file = []

                # For each attenuation level
                for k in range(6):
                    file_path = f"{self.data_directory}/{self.file_names[i]}/Measurement_seed341_att{k+1}.mat"
                    data = np.squeeze(io.loadmat(file_path)['Y_PD'])

                    if apply_filter:
                        b, a = self.butter_filter_params
                        data = signal.lfilter(b, a, data)

                    data_resampled = self.resample_signal(data, self.config_params.sample_freq_oscilloscope)
                    corr_result = correlation(data_resampled, reference, self.config_params.averaging)

                    corr_per_file = corr_result if not corr_per_file else np.vstack((corr_per_file, corr_result))

                path = f"D:/ROSS NN data/20240508/correlation_folder/correlation_data{i}.pickle"
                with open(path, 'wb') as f:
                    pickle.dump(corr_per_file, f)

                correlation_data_all_files.append(corr_per_file)
        else:
            # Load existing correlation data paths
            for i in range(len(self.file_names)):
                path = f"D:/ROSS NN data/20240508/correlation_folder/correlation_data{i}.pickle"
                correlation_data_all_files.append(path)

        self.correlation_data_paths = correlation_data_all_files

    def generate_feature_matrices_per_file(self):
        """
        Construct tapped features from the correlation output for each filter trace.
        """
        matrices = []

        for path in self.correlation_data_paths:
            with open(path, 'rb') as file:
                correlation_data = pickle.load(file)

            averaged = np.mean(correlation_data[:self.config_params.averaging], axis=0)
            resample_ratio = self.config_params.sample_freq_upsample / self.config_params.sample_freq_signal
            resampled = averaged[0::int(resample_ratio)]
            tapped = self.apply_tapping(resampled)
            matrices.append(tapped)

        return matrices

    def generate_noisy_combinations(self, selected_combination, num_repetitions=8, num_combinations_to_keep=2000):
        """
        Generate feature sets with noisy averaging and return trained weights and SERs.
        """
        all_noise_combinations = np.array(list(itertools.combinations(range(num_repetitions), self.config_params.averaging)))

        if len(all_noise_combinations) > num_combinations_to_keep:
            all_noise_combinations = all_noise_combinations[np.random.permutation(len(all_noise_combinations))[:num_combinations_to_keep]]

        selected_traces = [pickle.load(open(self.correlation_data_paths[i], 'rb')) for i in selected_combination.astype(int)]

        sers, weights = [], []

        for noise_set in tqdm.tqdm(all_noise_combinations, desc="Computing noisy weight matrices:"):
            feature_matrices = []

            for trace in selected_traces:
                averaged = np.sum([trace[k] for k in noise_set], axis=0) / len(noise_set)
                resample_ratio = self.config_params.sample_freq_upsample / self.config_params.sample_freq_signal
                resampled = averaged[0::int(resample_ratio)]
                tapped = self.apply_tapping(resampled)
                feature_matrices.append(tapped)

            combined_features = np.hstack(feature_matrices)
            mse, ser, feature_weights = self.calculate_weights(combined_features)
            sers.append(ser)
            weights.append(np.hstack(feature_weights))

        return sers, weights

    def evaluate_combinations(self, feature_matrices, combinations):
        """
        Evaluate SER and weights for each PUF filter combination using extracted features.
        """
        sers = []
        weights = np.zeros((len(combinations), self.config_params.taps * self.config_params.n_filters * 2 + 2))

        for i in tqdm.tqdm(range(len(combinations)), desc="Evaluating combinations"):
            selected = [feature_matrices[j] for j in combinations[i]]
            combined = np.hstack(selected)
            mse, ser, w = self.calculate_weights(combined)

            sers.append(ser)
            weights[i, :] = np.hstack(w)

        return np.array(sers), np.array(weights)
