# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt
from physical_unclonable_function_utils import *
from binarization_utils import *
from combinatorics_utils import *
from distributions_utils import *
from data_processing_utils import *
import time
import seaborn as sns

# --- Configuration Parameters ---
CONFIG = {
    "main_directory": 'D:/ROSS NN data/20240508/25_km',
    "test_vector_file": 'D:/ROSS NN data/20240508/TestVector_seed341.mat',
    "ser_thresholds": [1e-3],                        # Max acceptable Symbol Error Rate
    "phase_bound": [0.6, 0.8],                       # Phase filtering boundaries
    "mc_iterations": 100,                            # (unused here) Number of Monte Carlo iterations
    "num_bins": 30,                                  # Number of bins in histogram
    "n_filters": 3,                                  # Number of filters per PUF
    "taps": 55,                                      # Number of FIR filter taps
    "distance": 'hamming',                           # Distance metric: 'hamming' or 'euclidean'
    "Modulation": 4,                                 # Modulation order (e.g., QPSK)
    'num_bits': 3,                                   # Number of bits used in quantization
    "averaging": 5,                                  # Averaging factor over measurements
    "n_inter_PUFs": int(10e3),                       # Number of inter-PUF combinations to generate
    "n_intra_responses_per_PUF": int(60),            # Number of intra-PUF noise repetitions
    "n_intra_PUFs": 20,                              # Number of intra-PUFs to simulate
    "random_projection_dimension": 5000,             # Projection size before binarization
    "bootstrapping_n_objects": 1000,                 # Bootstrapping samples for inter-distance
    "bootstrapping_n_samples": 500,
    "user_defined_ser": 5e-4,
    "number_of_weights": 100                         # Placeholder, overwritten later
}

# --- Main execution ---
if __name__ == '__main__':

    # Configuration sweep over number of bits and averaging
    step = 0.2
    n_bits = np.arange(3, 6)       # Varying number of bits for quantization
    n_avg = np.arange(5, 6)        # Only one value tested here: 5

    dim1, dim2 = len(n_bits), len(n_avg)
    store_log_eer = np.zeros((dim1, dim2))  # Store Equal Error Rates

    for i_bits in range(dim1):
        for i_avg in range(dim2):

            # Update configuration for current sweep
            CONFIG["averaging"] = n_avg[i_avg]
            CONFIG["num_bits"] = n_bits[i_bits]

            # Initialize distance metric and data processor
            distance_metrics = DistanceMetrics(metric=CONFIG['distance'], num_bits=CONFIG['num_bits'])
            data_processor = prepare_data_processor(CONFIG)

            # Load precomputed inter-PUF weights and SER
            inter_ser, inter_weights, inter_PUFs = read_inter_weights(CONFIG)

            # Select PUFs with acceptable SER
            locs = np.where(inter_ser < CONFIG["ser_thresholds"])[0]
            user_locs = locs + 0
            intra_locs = np.random.permutation(len(user_locs))[:CONFIG["n_intra_PUFs"]]
            intra_pufs = inter_PUFs[user_locs[intra_locs]]

            # Load precomputed intra-PUF weights
            intra_ser, intra_weights, intra_PUFs = read_intra_weights(CONFIG)

            # Update number of weights based on inter_PUF shape
            _, CONFIG['number_of_weights'] = np.shape(inter_weights)

            # Instantiate random projection and binarization processor
            random_binarization_processor = RandomProjectionProcessor(CONFIG, output_bits=256)

            # --- Compute intra distances ---
            if CONFIG["distance"] == "hamming":
                binary_intra_weights = apply_function_to_list(intra_weights, random_binarization_processor.process)
                intra_hamming = apply_function_to_list(binary_intra_weights, distance_metrics.compute_pairwise_distances)
                intra_distances = np.array(intra_hamming).reshape(-1)
                intra_distribution = fit_gmm(intra_distances, n_components=2)

                # Convert inter weights too
                binary_inter_weights = random_binarization_processor.process(inter_weights)

            else:  # Euclidean
                binary_intra_weights = intra_weights
                intra_euclidean = apply_function_to_list(binary_intra_weights, distance_metrics.compute_pairwise_distances)
                intra_distances = np.array(intra_euclidean).reshape(-1)
                intra_distribution = fit_gmm(intra_distances, n_components=2)
                binary_inter_weights = inter_weights + 0

            # --- Compute inter distances via bootstrapping ---
            inter_distribution, inter_distances = bootstrapping(
                binary_inter_weights, inter_ser,
                inter_PUFs, data_processor.phases,
                CONFIG, random_binarization_processor
            )

            # --- Plot Distributions ---
            plt.figure(figsize=(10, 8))
            sns.histplot(intra_distances, bins=30, kde=False, color='blue', label='Intra-Hamming', stat='density', alpha=0.6)
            sns.histplot(inter_distances, bins=30, kde=False, color='orange', label='Inter-Hamming', stat='density', alpha=0.6)

            # Overlay PDFs from GMM
            x_vals = np.linspace(0, 0.7, 1000)
            intra_pdf, _ = gmm_pdf_cdf(x_vals, intra_distribution)
            inter_pdf, _ = gmm_pdf_cdf(x_vals, inter_distribution)
            plt.plot(x_vals, intra_pdf, color='red', label='PDF Intra', linewidth=4)
            plt.plot(x_vals, inter_pdf, color='green', label='PDF Inter', linewidth=4)

            # Final plot formatting
            plt.xlabel("Normalized Hamming Distance", fontsize=24)
            plt.ylabel("Normalized Counts", fontsize=24)
            plt.title("Hamming Distance Distributions", fontsize=24)
            plt.legend(fontsize=24)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.grid(True)
            plt.savefig("figure.svg", bbox_inches='tight')
            plt.show()

            # --- Compute Equal Error Rate ---
            log_eer = compute_equal_error_rate(intra_distribution, inter_distribution, search_limit=np.max(inter_distances))
            print("For %d bits and %.2f phase threshold the log_EER is %.4f\n" %
                  (n_bits[i_bits], CONFIG["phase_bound"][0], log_eer))
            
            store_log_eer[i_bits, i_avg] = log_eer

    # --- Save Results ---
    mdic = {
        'store_log_eer': store_log_eer,
        'n_bits': n_bits,
        'averaging': n_avg,
        'CONFIG': CONFIG
    }
    io.savemat('results_n_bits_n_avg.mat', mdic)