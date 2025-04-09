import numpy as np
from io_utils import read_all_files, save_data, read_previous_data
from combinatorics_utils import *
from distributions_utils import best_fit_distribution, gmm_pdf_cdf, fit_gmm
from scipy.optimize import fsolve
import tqdm
import matplotlib.pyplot as plt
import concurrent.futures

def apply_function_to_list(input_list, func):
    """
    Applies a given function to each element of the input list.

    Parameters:
        input_list (list): List of items to process.
        func (function): Function to apply to each element of the input list.

    Returns:
        list: List of processed elements.
    """
    return [func(item) for item in input_list]

def generate_inter_weights(data_handler, CONFIG):
    """
    Generate inter-PUF weights and SER values for a collection of PUF combinations.

    Parameters:
        data_handler: Object that handles data preprocessing and evaluation.
        CONFIG (dict): Configuration dictionary with experiment parameters.

    Returns:
        Tuple (inter_ser, inter_weights, inter_PUFs)
    """
    # Load all measurement files and their associated phase info
    all_files, phases = read_all_files(directory=CONFIG['main_directory'])

    # Generate all filter combinations of the desired size
    combinations = generate_all_combinations(all_files, CONFIG['n_filters'])

    # Select a subset of combinations for inter-PUF evaluation
    if len(combinations) > CONFIG['n_inter_PUFs']:
        inter_PUFs = reduce_combinations_to_limit(combinations, limit=CONFIG['n_inter_PUFs'])
    else:
        inter_PUFs = combinations

    # Get signal features for each file
    training_data = data_handler.generate_feature_matrices_per_file()

    # Evaluate SER and extract weights for each combination
    inter_ser, inter_weights = data_handler.evaluate_combinations(training_data, inter_PUFs)

    # Save results to disk
    file_prefix = 'Data_Files/'
    inter_read_file = file_prefix + '%d_filters_avg_%d_taps_%d_results.mat' % (
        CONFIG['n_filters'], CONFIG["averaging"], CONFIG['taps']
    )
    save_data(inter_read_file, inter_PUFs, inter_ser, inter_weights)

    return inter_ser, inter_weights, inter_PUFs

def read_inter_weights(CONFIG):
    """
    Load previously saved inter-PUF results.

    Returns:
        Tuple (inter_ser, inter_weights, inter_PUFs)
    """
    inter_read_file = 'Data_Files/%d_filters_avg_%d_taps_%d_results.mat' % (
        CONFIG['n_filters'], CONFIG["averaging"], CONFIG['taps']
    )
    inter_PUFs, inter_ser, inter_weights = read_previous_data(inter_read_file)
    return inter_ser, inter_weights, inter_PUFs

def process_puf(i, PUF, config, data_handler):
    """
    Helper function for multiprocessing to generate intra-PUF samples.

    Returns:
        Tuple (PUF index, ser, weights)
    """
    ser, weights = data_handler.generate_noisy_combinations(
        PUF, 60, config['n_intra_responses_per_PUF']
    )
    return i, ser, weights

def generate_intra_weights(intra_PUFs, data_handler, CONFIG, save=False):
    """
    Generate noisy repeated measurements for intra-PUF analysis.

    Parameters:
        intra_PUFs: List of selected PUF combinations.
        data_handler: Handles signal preprocessing and feature extraction.
        CONFIG: Configuration dictionary.
        save (bool): If True, save results to disk.

    Returns:
        Tuple (intra_ser, intra_weights, intra_PUFs)
    """
    intra_weights = []
    intra_ser = []

    for i, PUF in enumerate(intra_PUFs):
        ser, weights = data_handler.generate_noisy_combinations(PUF, 60, CONFIG['n_intra_responses_per_PUF'])
        intra_weights.append(weights)
        intra_ser.append(ser)

    file_prefix = 'Data_Files/n_filters_%d_n_taps_%d_averaging_%d_logSER_%d' % (
        CONFIG['n_filters'], CONFIG['taps'], CONFIG['averaging'],
        np.log10(CONFIG['ser_thresholds'][0])
    )
    intra_read_file = file_prefix + 'noise_data.mat'

    if save:
        save_data(intra_read_file, intra_PUFs, intra_ser, intra_weights)

    return intra_ser, intra_weights, intra_PUFs

def read_intra_weights(CONFIG):
    """
    Load previously saved intra-PUF results.

    Returns:
        Tuple (intra_ser, intra_weights, intra_PUFs)
    """
    file_prefix = 'Data_Files/n_filters_%d_n_taps_%d_averaging_%d_logSER_%d' % (
        CONFIG['n_filters'], CONFIG['taps'], CONFIG['averaging'],
        np.log10(CONFIG['ser_thresholds'][0])
    )
    intra_read_file = file_prefix + 'noise_data.mat'
    intra_PUFs, intra_ser, intra_weights = read_previous_data(intra_read_file)
    
    return intra_ser, intra_weights, intra_PUFs

def bootstrapping(inter_weights, inter_ser, inter_PUFs, phases, CONFIG, binarizer=None):
    """
    Perform bootstrapped sampling to generate inter-PUF distance distribution.

    Returns:
        Fitted GMM for inter-PUF distances, and raw distance samples.
    """
    n_objects = CONFIG['bootstrapping_n_objects']
    n_samples = CONFIG['bootstrapping_n_samples']
    phase_bound = CONFIG['phase_bound']
    ser_threshold = CONFIG['ser_thresholds']

    # Instantiate the distance computation object
    statistics = DistanceMetrics(CONFIG['distance'], binarizer, CONFIG['num_bits'])

    all_distances = []
    iterator = tqdm.tqdm(range(n_objects))

    for _ in iterator:
        # Sample inter-PUF distances that meet phase/ser criteria
        inter_distances = statistics.compute_filtered_statistics(
            inter_weights, inter_ser, phases, inter_PUFs,
            ser_threshold, phase_bound, max_distances=n_samples
        )
        all_distances.extend(inter_distances)

    inter_distribution = fit_gmm(np.array(all_distances), n_components=5)
    print('I checked %d objects and found %d valid samples.\n' % (n_objects, len(inter_distances)))

    return inter_distribution, inter_distances

def rates_difference(thresholds, intra_distribution, inter_distribution):
    """
    Compute the absolute difference between FAR and FRR at a given threshold.

    Used in root finding for EER.
    """
    _, intra_cdf = gmm_pdf_cdf(thresholds, intra_distribution)
    _, inter_cdf = gmm_pdf_cdf(thresholds, inter_distribution)

    false_acceptance_rate = 1 - intra_cdf
    false_rejection_rate = inter_cdf

    return np.abs(false_acceptance_rate - false_rejection_rate)

def compute_equal_error_rate(intra_distribution, inter_distribution, search_limit):
    """
    Compute Equal Error Rate (EER) by finding the point where
    false acceptance and rejection rates intersect.

    Parameters:
        intra_distribution: GMM for intra-PUF distances
        inter_distribution: GMM for inter-PUF distances
        search_limit (float): Max threshold to consider

    Returns:
        log10(EER)
    """
    # Initial guess range for threshold values
    t0 = np.linspace(0, search_limit, 5)
    roots = []

    # Use root finding to locate points where FAR ≈ FRR
    for guess in t0:
        root = fsolve(lambda x: rates_difference(x, intra_distribution, inter_distribution), x0=guess)[0]
        if 0 < root < 10:
            roots.append(root)

    roots = np.array(roots)
    evaluations = rates_difference(roots, intra_distribution, inter_distribution)
    loc = np.argmin(np.abs(evaluations))
    root = roots[loc]

    _, intra_cdf = gmm_pdf_cdf(root, intra_distribution)
    _, inter_cdf = gmm_pdf_cdf(root, inter_distribution)

    # Compute actual FAR and FRR at optimal threshold
    false_acceptance_rate = 1 - intra_cdf
    false_rejection_rate = inter_cdf
    equal_error_rate = np.max([false_acceptance_rate, false_rejection_rate])

    log_equal_error_rate = np.log10(equal_error_rate + 1e-35)
    
    return log_equal_error_rate
