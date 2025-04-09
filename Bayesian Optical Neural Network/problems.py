from itertools import permutations
import numpy as np
from scipy.sparse.construct import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 


def normalize_inputs(inputs, feature_axis=1):
    """
    Normalize inputs.
    :param inputs: The input features.
    :param feature axis: Define the axis (default = 1).
    :return output: The normalized inputs.
    
    """
    if feature_axis == 1:
        n_features, n_examples = inputs.shape
    elif feature_axis == 0:
        n_examples, n_features = inputs.shape
    for i in range(n_features):
        l1_norm = np.mean(np.abs(inputs[i, :]))
        inputs[i, :] /= l1_norm
    return inputs

def prepare_iris(bias=False):

    """
    Prepare the Iris dataset.
    :param bias: If True it inserts one additional input for bias. Default = False.
    :return x_train: The training features in complex field representation.
    :return t_train: The target training vector in one-hot encoding.
    :return x_test: The testing features in complex field representation. 
    :return t_test: The target testing vector in one-hot encoding.
    """

    iris = load_iris()
    x = iris['data']
    y = iris['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

    n_training_examples, n_features = np.shape(x_train)
    n_testing_examples, _ = np.shape(x_test)
    n_classes = np.max(y) + 1

    t_train = np.zeros((n_training_examples, n_classes))
    t_test = np.zeros((n_testing_examples, n_classes))
    for el, y_el in enumerate(y_train):
        t_train[el, y_el] = 1.
    for el, y_el in enumerate(y_test):
        t_test[el, y_el] = 1.
    
    x_train = np.transpose(x_train) +0*1j
    x_test = np.transpose(x_test) + 0*1j
    t_train = np.transpose(t_train)
    t_test = np.transpose(t_test)

    return x_train, t_train, x_test, t_test

def prepare_mnist(freq_radius, bias=False):
    
    """
    Prepare the MNIST dataset.
    :param freq_radius: Set the radius for acquisition of samples from the 2D Fourier transform.
    :param bias: If True it inserts one additional input for bias. Default = False.
    :return x_train: The training features in complex field representation.
    :return t_train: The target training vector in one-hot encoding.
    :return x_test: The testing features in complex field representation. 
    :return t_test: The target testing vector in one-hot encoding.
    """
    train_set = datasets.MNIST(root="/mnist/",
                            train=True,
                            transform=transforms.ToTensor,
                            download=True)

    test_set = datasets.MNIST(root="/mnist/",
                            train=False,
                            transform=transforms.ToTensor,
                            download=True)

    x_train_raw = train_set.data.numpy()
    y_train = train_set.targets.numpy()
    num_train = len(y_train)
    n_classes = np.max(y_train) + 1

    x_test_raw = test_set.data.numpy()
    y_test = test_set.targets.numpy()
    num_test = len(y_test)

    x_train_ft = np.fft.fftshift(np.fft.fft2(x_train_raw), axes=(1, 2))
    x_test_ft = np.fft.fftshift(np.fft.fft2(x_test_raw), axes=(1, 2))

    def fourier(x_train_ft, x_test_ft, freq_radius):
        min_r, max_r = 14 - freq_radius, 14 + freq_radius
        x_train = x_train_ft[:, min_r:max_r, min_r:max_r]
        x_test = x_test_ft[:, min_r:max_r, min_r:max_r]
        
        units = int((2*freq_radius)**2)
        return x_train, x_test, units
    
    x_train, x_test, units = fourier(x_train_ft, x_test_ft, freq_radius)

    x_train=normalize_inputs(x_train.reshape((num_train, units)).astype(np.complex64))
    x_test= normalize_inputs(x_test.reshape((num_test, units)).astype(np.complex64))

    t_train = np.zeros((num_train, n_classes))
    t_test = np.zeros((num_test, n_classes))
    for el, y_el in enumerate(y_train):
        t_train[el, y_el] = 1.
    for el, y_el in enumerate(y_test):
        t_test[el, y_el] = 1.
    
    x_train = np.transpose(x_train)
    x_test = np.transpose(x_test)
    t_train = np.transpose(t_train)
    t_test = np.transpose(t_test)

    if bias:
        bias_value =  1+0*1j 
        x_train = np.vstack((x_train, bias_value*np.ones(num_train)))
        x_test = np.vstack((x_test, bias_value*np.ones(num_test)))

    return x_train, t_train, x_test, t_test

