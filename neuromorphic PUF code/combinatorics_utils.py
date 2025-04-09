import itertools
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import scipy.io as io

# --- Combination Utilities ---

def generate_all_combinations(files, num_filters):
    """
    Generate all possible index combinations of file entries, choosing 'num_filters' at a time.

    Parameters:
        files (list): List of file references.
        num_filters (int): Number of filters to include in each combination.

    Returns:
        ndarray: 2D array where each row is a unique combination of file indices.
    """
    num_files = len(files)
    all_combinations = list(itertools.combinations(range(num_files), num_filters))
    return np.array(all_combinations)


def is_combination_match(target_combination, combination_list):
    """
    Check if the given target combination includes any set in the reference list.

    Parameters:
        target_combination (tuple): A specific combination to evaluate.
        combination_list (list): List of reference combinations.

    Returns:
        bool: True if any combination in the list is a subset of the target.
    """
    for comparison_combination in combination_list:
        if set(comparison_combination) <= set(target_combination):
            return True
    return False


def filter_combinations_based_on_previous_results(files, num_filters, results_folder, ser_threshold):
    """
    Load prior SER results and filter combinations below a SER threshold.

    Parameters:
        files (list): List of file names.
        num_filters (int): Number of filters used per combination.
        results_folder (str): Path to .mat file with previous SER results.
        ser_threshold (float): SER threshold for selecting valid combinations.

    Returns:
        ndarray: Valid combinations that met the SER performance criteria.
    """
    # Load previous results from .mat file
    data = io.loadmat(results_folder)
    ser_values = data['ser'][0]
    previous_combinations = data['combs']

    # Find combinations that pass the SER criterion
    valid_combination_indices = np.where(ser_values < ser_threshold)[0]
    valid_combinations = previous_combinations[valid_combination_indices, :]

    all_combinations = generate_all_combinations(files, num_filters)
    filtered_combinations = []

    iterator = tqdm.tqdm(range(len(all_combinations)), desc="Filtering valid combinations")
    for i in iterator:
        combination = all_combinations[i, :]
        if is_combination_match(combination, valid_combinations):
            filtered_combinations.append(combination)

    print(f"Kept {len(filtered_combinations)} out of {len(all_combinations)} combinations")
    return np.array(filtered_combinations)


def reduce_combinations_to_limit(combinations, limit, seed=None):
    """
    Randomly select a limited number of combinations from a larger pool.

    Parameters:
        combinations (ndarray): Full list of combinations.
        limit (int): Maximum number of combinations to return.
        seed (int): Optional seed for reproducibility.

    Returns:
        ndarray: Random subset of combinations up to the specified limit.
    """
    if seed is not None:
        np.random.seed(seed)

    if 0 < limit < len(combinations):
        selected_indices = np.random.permutation(len(combinations))[:limit]
        return combinations[selected_indices, :]

    return combinations

# --- Distance Metrics Class ---

class DistanceMetrics():
    """
    Utility class for computing distances (Euclidean or Hamming) between data vectors.
    """

    def __init__(self, metric='euclidean', binarizer=None, num_bits=8) -> None:
        """
        Initializes the distance calculator with desired metric and settings.

        Parameters:
            metric (str): 'euclidean' or 'hamming'.
            binarizer (optional): A function to binarize data if needed.
            num_bits (int): Used for bit-level comparison.
        """
        self.num_bits = num_bits
        self.binarizer = binarizer

        # Select internal method based on metric
        if metric == 'euclidean':
            self.distance_method = self._euclidean_distance
        elif metric == 'hamming':
            self.distance_method = self._hamming_distance
        else:
            raise ValueError("Invalid metric. Choose either 'euclidean' or 'hamming'.")

    @staticmethod
    def _euclidean_distance(x, y):
        """
        Calculate Euclidean (L2) distance between vectors.

        Parameters:
            x, y (ndarray): Input vectors.

        Returns:
            float or ndarray: Euclidean distance(s).
        """
        if len(np.shape(x)) == 1:
            return np.sqrt(np.sum((x - y) ** 2))
        else:
            return np.sqrt(np.sum((x - y) ** 2, axis=1))

    @staticmethod
    def _hamming_distance(x, y):
        """
        Calculate normalized Hamming distance between binary-like vectors.

        Parameters:
            x, y (ndarray): Input vectors.

        Returns:
            float or ndarray: Hamming distance(s).
        """
        x = 1.0 * x
        y = 1.0 * y
        if len(np.shape(x)) == 1:
            return np.mean(np.abs(x - y))
        else:
            return np.mean(np.abs(x - y), axis=1)

    def compute_pairwise_distances(self, data_list, max_pairs=5000):
        """
        Compute distances for all unique pairs in a dataset, up to a maximum count.

        Parameters:
            data_list (list): List of vectors to compare.
            max_pairs (int): Limit for total number of comparisons.

        Returns:
            ndarray: Distance values for selected pairs.
        """
        n_data = len(data_list)
        pairs = np.array(list(itertools.combinations(range(n_data), 2)))

        # Limit number of computed pairs
        if len(pairs) >= max_pairs:
            pairs = pairs[np.random.permutation(len(pairs))[:max_pairs]]

        distances = []
        for i_idx, j_idx in pairs:
            distances.append(self.distance_method(data_list[i_idx], data_list[j_idx]))

        return np.array(distances)

    def compute_filtered_statistics(
        self, data_list, ser, phases, combinations,
        ser_threshold=3e-3, phase_bounds=[0, np.pi], max_distances=2000):
        """
        Compute distances only between PUFs that meet certain SER and phase diversity criteria.

        Parameters:
            data_list (list): Vector representations (e.g., weights or hashes).
            ser (array): SER values associated with each PUF.
            phases (array): Phase values for each file.
            combinations (array): File indices used in each PUF.
            ser_threshold (float): Maximum acceptable SER.
            phase_bounds (list): Min/max phase difference for valid comparison.
            max_distances (int): Max number of distances to return.

        Returns:
            ndarray: Filtered pairwise distances.
        """
        combinations = combinations.astype(int)
        ser = np.array(ser[0])
        data_list = np.array(data_list)

        # Select PUFs with acceptable error rate
        valid_indices = np.where(ser < ser_threshold)[0]
        filtered_data = data_list[valid_indices]
        filtered_combinations = combinations[valid_indices]

        n_data = len(filtered_data)
        n_iterations = 10000
        distances = []
        attempts = 0
        collected = 0

        while collected < max_distances and attempts < n_iterations:
            sample_size = 2 * max_distances
            r1 = np.random.randint(0, n_data, sample_size)
            r2 = np.random.randint(0, n_data, sample_size)

            # Select pairs and their corresponding phase info
            data1, data2 = filtered_data[r1], filtered_data[r2]
            comb1, comb2 = filtered_combinations[r1], filtered_combinations[r2]

            phase1 = np.arctan(np.sin(phases[comb1]) / np.cos(phases[comb1]))
            phase2 = np.arctan(np.sin(phases[comb2]) / np.cos(phases[comb2]))
            phase_diff = np.abs(phase1 - phase2)

            # Identify valid pairs based on phase difference
            valid_mask = (phase_diff >= phase_bounds[0])
            valid_indices = np.where(np.prod(valid_mask, axis=1) != 0)[0]

            if len(valid_indices) > 0:
                valid_data1 = data1[valid_indices]
                valid_data2 = data2[valid_indices]
                distances.extend(self.distance_method(valid_data1, valid_data2))
                collected = len(distances)

            attempts += 1

        if attempts == n_iterations:
            print("Could not find enough valid solutions.")

        return np.array(distances[:max_distances])
