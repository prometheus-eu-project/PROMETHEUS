import numpy as np
from scipy import stats
from scipy.fftpack import fft
import scipy.io as io
import hashlib
from datasketch import MinHash
import matplotlib.pyplot as plt
from scaling_utils import max_min_scaler

# ------------------------------
# Distribution and Quantization Utilities
# ------------------------------

def transform_to_uniform(data):
    """
    Convert data to a uniform distribution [0, 1] using rank-based normalization.

    Parameters:
        data (np.ndarray): Input 2D array.

    Returns:
        np.ndarray: Uniformly distributed array of same shape.
    """
    ranks = stats.rankdata(data, axis=-1)
    _, n = np.shape(data)
    return (ranks - 1) / (n - 1)


def gray_code_lookup(n_bits):
    """
    Generate a lookup table to convert integers to Gray code.

    Parameters:
        n_bits (int): Number of bits for the Gray code.

    Returns:
        np.ndarray: Array of shape (2^n_bits, n_bits) representing Gray codes.
    """
    num_levels = 2 ** n_bits
    levels = np.arange(num_levels, dtype=np.uint8)
    gray_levels = levels ^ (levels >> 1)  # Convert to Gray code
    return np.unpackbits(gray_levels[:, None], axis=-1, bitorder='big')[:, -n_bits:]


# ------------------------------
# MinHash and Cryptographic Hashing
# ------------------------------

def minhash_rowwise(matrix, num_hashes=256):
    """
    Apply MinHash to each row of a binary matrix to generate binary signatures.

    Parameters:
        matrix (np.ndarray): Binary matrix (0/1).
        num_hashes (int): Number of hash functions.

    Returns:
        np.ndarray: Matrix of MinHash signatures.
    """
    n_rows, _ = matrix.shape
    minhash_signatures = []

    for row in matrix:
        m = MinHash()
        for idx, value in enumerate(row):
            if value == 1:
                m.update(str(idx).encode('utf8'))

        # Convert digest to binary
        signature = np.array([1 if m.digest()[i] & (1 << (7 - j)) else 0
                              for i in range(len(m.digest()))
                              for j in range(8)])
        minhash_signatures.append(signature)

    return np.array(minhash_signatures)


def cryptographic_hash_rowwise_binary(matrix, hash_dim=256):
    """
    Apply SHA-256 hash function row-wise to a binary matrix.

    Parameters:
        matrix (np.ndarray): Input binary matrix.
        hash_dim (int): Must be 256 (for SHA-256).

    Returns:
        np.ndarray: Matrix of binary hashes.
    """
    if hash_dim != 256:
        raise ValueError("Only SHA-256 (256-bit hash) is supported.")
    
    binary_hashes = []

    for row in matrix:
        row_str = ''.join(map(str, row))
        hash_digest = hashlib.sha256(row_str.encode('utf-8')).digest()
        binary_bits = np.unpackbits(np.frombuffer(hash_digest, dtype=np.uint8))
        binary_hashes.append(binary_bits)

    return np.array(binary_hashes)


# ------------------------------
# XOR-based Hashing
# ------------------------------

def xor_successive(matrix):
    """
    Apply XOR operation between successive rows.

    Parameters:
        matrix (np.ndarray): Input binary matrix.

    Returns:
        np.ndarray: XOR-ed matrix.
    """
    n_rows, n_bits = np.shape(matrix)
    new_matrix = 0 * matrix
    new_matrix[0:-2, :] = matrix[0:-2, :] ^ matrix[1:-1, :]
    new_matrix[-1, :] = matrix[-1, :] ^ matrix[0, :]
    return new_matrix


def xor_hash(matrix, hash_dim, random_seed=None):
    """
    Apply XOR-based hash to each row by randomly combining bits.

    Parameters:
        matrix (np.ndarray): Input binary matrix.
        hash_dim (int): Number of hash bits.
        random_seed (int): Optional seed for reproducibility.

    Returns:
        np.ndarray: Hashed matrix.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_rows, n_features = matrix.shape
    selected_indices = np.random.randint(0, n_features, size=(hash_dim, 2))
    binary_hashes = np.zeros((n_rows, hash_dim), dtype=int)

    for i in range(hash_dim):
        idx1, idx2 = selected_indices[i]
        binary_hashes[:, i] = matrix[:, idx1] ^ matrix[:, idx2]

    return binary_hashes


def quantize_float_to_bits(f, num_bits):
    """
    Convert float matrix to quantized binary using Gray code.

    Parameters:
        f (np.ndarray): Input float array.
        num_bits (int): Number of bits for quantization.

    Returns:
        np.ndarray: Binary representation using Gray code.
    """
    uniform_data = transform_to_uniform(f)
    quantized = np.round((2 ** num_bits - 1) * uniform_data).astype(int)
    gray_table = gray_code_lookup(num_bits)
    bit_series = gray_table[quantized]

    rows, n_values = quantized.shape
    return np.reshape(bit_series, (rows, num_bits * n_values))


# ------------------------------
# Bit-Level Compression
# ------------------------------

def vectorized_xor(matrix, group_size):
    """
    XOR all chunks of each row to compress into fixed-length binary vector.

    Parameters:
        matrix (np.ndarray): Input binary matrix (M, N).
        group_size (int): Size of each chunk to XOR (e.g., 256).

    Returns:
        np.ndarray: Compressed XOR result per row (M, group_size).
    """
    M, N = matrix.shape
    pad_length = (group_size - (N % group_size)) % group_size
    padded_matrix = np.pad(matrix, ((0, 0), (0, pad_length)), mode='constant', constant_values=0)

    num_chunks = padded_matrix.shape[1] // group_size
    reshaped = padded_matrix.reshape(M, num_chunks, group_size)

    return np.bitwise_xor.reduce(reshaped, axis=1)


def hash_with_tolerance(num, epsilon=0.1):
    """
    A hash function that groups similar floats together into the same bin.

    Parameters:
        num (float): Input float value.
        epsilon (float): Bin size.

    Returns:
        int: Bin ID (hash result).
    """
    return int(num / epsilon)


# ------------------------------
# Random Projection Processor
# ------------------------------

class RandomProjectionProcessor:
    """
    Class to apply FFT-based random projection and quantization to generate binary fingerprints.
    """

    def __init__(self, CONFIG, num_bits=8, output_bits=256, random_seed_for_matrices=888):
        """
        Initialize projection matrix and masks.

        Parameters:
            CONFIG (dict): Contains input_dim and projection_dim values.
            num_bits (int): Number of bits per quantized value.
            output_bits (int): Output dimensionality.
            random_seed_for_matrices (int): Seed for reproducibility.
        """
        self.input_dim = CONFIG['number_of_weights']
        self.random_projection_dim = CONFIG['random_projection_dimension']
        self.num_bits = CONFIG['num_bits']
        self.output_bits = output_bits
        self.random_seed_for_matrices = random_seed_for_matrices

        if random_seed_for_matrices is not None:
            np.random.seed(random_seed_for_matrices)

        # Gaussian random projection matrix
        self.random_matrix = np.random.normal(0, 1, (self.input_dim, self.random_projection_dim)) / np.sqrt(self.random_projection_dim)

        # Diagonal matrix for sign flipping (Fastfood transform)
        self.U = np.diag(np.random.choice([1, -1], size=self.input_dim, p=[0.5, 0.5]))

        # Sampling matrix S
        indices = np.random.choice(self.input_dim, size=self.random_projection_dim, replace=True)
        self.S = np.zeros((self.random_projection_dim, self.input_dim))
        for i, idx in enumerate(indices):
            self.S[i, idx] = 1

        self.output_mask = np.random.permutation(self.random_projection_dim)[:output_bits]
        np.random.seed()

    def process(self, data, data_file_ldir=None, save=False):
        """
        Apply FFT + projection + quantization pipeline to generate binary signature.

        Parameters:
            data (np.ndarray): Float matrix input (weights).
            data_file_ldir (str): Optional path to load precomputed binary output.
            save (bool): If True, save binary output to .mat file.

        Returns:
            np.ndarray: Binary signature.
        """
        if data_file_ldir is None:
            if data.ndim == 1:
                data = data.reshape(1, -1)

            FFT = fft(data, axis=-1)
            modulated = FFT @ self.U
            projected = modulated.real @ self.S.T
            binary_data = quantize_float_to_bits(projected, self.num_bits)

            if save:
                io.savemat(f'Data_Files/binary_data_{self.random_seed_for_matrices}.mat', {'binary_data': binary_data})
        else:
            binary_data = io.loadmat(data_file_ldir)['binary_data']

        # Apply bitmask to reduce dimensionality
        return binary_data[:, self.output_mask]
