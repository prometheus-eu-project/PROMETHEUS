import os
import re
import numpy as np
from scipy import io  # For loading/saving MATLAB .mat files

def read_all_files(directory):
    """
    Read all filenames from a directory and extract phase values encoded in filenames.

    Expected filename format: something like "phase0.45_...". The function extracts
    the numeric value after "phase" using a regex.

    Parameters:
        directory (str): Path to the directory containing measurement files.

    Returns:
        tuple:
            - all_files (list of str): List of all filenames in the directory.
            - phases (ndarray of float): Array of extracted phase values.
    """
    all_files = os.listdir(directory)  # List all files in directory
    phases = []

    for file_name in all_files:
        # Extract a float from filenames containing "phaseX.XX"
        match = re.search(r'phase(\d+\.\d+)', file_name)
        if match:
            phases.append(float(match.group(1)))  # Append extracted phase value
        else:
            print(f"No match found in filename: {file_name}")  # Warn if no phase info found

    return all_files, np.array(phases)


def read_target_file(directory):
    """
    Read a specific MATLAB .mat file that contains a target signal matrix.

    The expected structure is:
        data['tv']['X_pulseShapedNew'][0][0]

    Parameters:
        directory (str): Full path to the .mat file.

    Returns:
        ndarray: 1D or 2D NumPy array with pulse-shaped target data.
    """
    data = io.loadmat(directory)  # Load MATLAB file
    return np.squeeze(data['tv']['X_pulseShapedNew'][0][0])  # Extract target waveform


def read_previous_data(directory):
    """
    Load previously saved PUF experiment data: combinations, SER, and weights.

    Parameters:
        directory (str): Path to the .mat file.

    Returns:
        tuple:
            - combinations (ndarray or None): Array of combination indices (if saved).
            - ser (ndarray): Symbol Error Rates.
            - weights (ndarray): Feature weights from experiment.
    """
    data = io.loadmat(directory)

    # Handle optional presence of 'combs' key
    combinations = data['combs'] if 'combs' in data else None
    ser = data['ser']
    weights = data['weights']

    return combinations, ser, weights


def save_data(directory, combinations, ser, weights):
    """
    Save PUF data to a .mat file, including:
    - Filter combinations (optional)
    - SER values
    - Extracted weight vectors

    Parameters:
        directory (str): Full path to save .mat file.
        combinations (ndarray): PUF filter combinations used.
        ser (ndarray): Symbol Error Rate values.
        weights (ndarray): Extracted or trained feature weights.

    Returns:
        None
    """
    data_to_save = {
        'combs': combinations,
        'weights': weights,
        'ser': ser
    }

    io.savemat(directory, data_to_save)  # Save to MATLAB .mat format
