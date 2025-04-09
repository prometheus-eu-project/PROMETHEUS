import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.shape_base import tile
from photonics import*
from bayesian import*
from analysis import*
from problems import prepare_iris, prepare_mnist
from scipy.signal import savgol_filter
from tqdm import std, tqdm

dimensions = [4]
layers = len(dimensions)

x_train, t_train, x_test, t_test = prepare_iris()

network = Network(
                [ClementsLayer(dimensions[0]),
                 ActivationFunction(dimensions[0], type='AbsSquare')],
                 loss=CategoricalCrossEntropy
                )

problem_set = (x_train, t_train, x_test, t_test)
method = 'bits-back'

x_train, t_train, x_test, t_test = problem_set[0], problem_set[1], problem_set[2], problem_set[3]

epochs = 1000
batch_percent = 10/120
eta = 5e-3


theta = initialize_theta(dimensions)
phi_init = initialize_on_chip_meshes(theta, dimensions, type='zeros')

phi_measured = phi_init + np.zeros(len(phi_init))
theta, losses, accuracies = neural_net_training(epochs, batch_percent, eta, x_train, t_train, theta, dimensions, network, phi_char=phi_measured, sigma_prior=1, method = method) 

plt.plot(losses)
plt.title("Loss function")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()

plt.plot(accuracies)
plt.title("Accuracies")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

ph, dph_norm, _ = load_on_chip(phi_init, phi_measured, theta, dimensions, method)
_, accuracy = evaluate_testing(x_test, t_test, ph, network)
            
print("The testing accuracy is %.4f" %accuracy)