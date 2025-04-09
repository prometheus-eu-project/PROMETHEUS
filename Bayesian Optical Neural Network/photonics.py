from re import S
from numpy.lib.function_base import gradient
from numpy.testing._private.utils import requires_memory
import torch
from torch.autograd import Variable
import numpy as np
from bayesian import*
from tqdm import tqdm
import matplotlib.pyplot as plt
import time 

class Layer():
    """
    Represents a single layer of the Clements Mesh.   
    """
    def __init__(self,n ,ports) -> None:
        """
        Initialize the Layer
        :param n: number of input ports
        :param n: indexes of the ports that are associated with this layer
        """
        self.phi_ext = np.zeros(int(0.5*len(ports)))
        self.phi_mzi = np.zeros(int(0.5*len(ports)))
        self.n = n
        self.phi_ext_power_f = np.zeros(int(0.5*len(ports)))
        self.phi_mzi_power_f = np.zeros(int(0.5*len(ports)))
        self.phi_ext_power_b = np.zeros(int(0.5*len(ports)))
        self.phi_mzi_power_b = np.zeros(int(0.5*len(ports)))

        self.first_ports = np.arange(ports[0], ports[len(ports)-1], 2)

    def update(self, phi_ext, phi_mzi):
        """
        Update the phase shifter values on the layer
        :param phi_ext: New values for the external phase shifters
        :param phi_mzi: New values for the MZIs
        """
        self.phi_ext = phi_ext
        self.phi_mzi = phi_mzi
    
    def get_partial_transfer_matrices(self):
        """
        Return the transfer matrices of the beamsplitters, external phase shifters and MZIs.
        :return t_ext: The transfer matrix associated with the external phase shifters 
        :return t_bs1: The transfer matrix associated with the first beamsplitter
        :return t_theta: The transfer matrix associated with the MZI
        :return t_bs2: The transfer matrix associated with the second beamsplitter
        """
        
        t_ext = np.eye(self.n, dtype='csingle')
        t_bs1 = np.eye(self.n, dtype='csingle')
        t_theta = np.eye(self.n, dtype='csingle')
        t_bs2 = np.eye(self.n, dtype='csingle')

         # Beamsplitters

        t_bs1[self.first_ports, self.first_ports] = 1/np.sqrt(2)
        t_bs1[self.first_ports+1, self.first_ports] = 1/np.sqrt(2)*1j
        t_bs1[self.first_ports, self.first_ports + 1] = 1/np.sqrt(2)*1j
        t_bs1[self.first_ports + 1, self.first_ports + 1] = 1/np.sqrt(2)

        t_bs2 = t_bs1

        # External phase shifter

        t_ext[self.first_ports, self.first_ports] = np.exp(1j*self.phi_ext)

        # MZI

        t_theta[self.first_ports, self.first_ports] = np.exp(1j*self.phi_mzi)

        return t_ext, t_bs1, t_theta, t_bs2
    
    def forward_propagate(self, inp):
        """
        Forward propagate information through the layer
        :param inp: The complex field that is inserted in the layer
        :return out: The complex field at the output of the layer 
        """
        
        t_ext, t_bs1, t_theta, t_bs2 = self.get_partial_transfer_matrices()

        # Forward Propagate

        out_phi = np.dot(t_ext, inp)
        out_bs1 = np.dot(t_bs1, out_phi)
        out_theta = np.dot(t_theta,out_bs1)
        out = np.dot(t_bs2,out_theta)
        
        # Store Field for the Adjoint Method

        self.phi_ext_power_f = np.sum(np.abs(out_phi[self.first_ports, :]) ** 2, axis=1)
        self.phi_mzi_power_f = np.sum(np.abs(out_theta[self.first_ports, :])**2, axis=1)

        return out
    
    def backward_propagate(self, delta):
        """
        Backpropagate the field through the layer
        :param delta: The complex field that is inserted at the output of the layer
        :return inp: The complex field at the input of the layer
        """
        t_ext, t_bs1, t_theta, t_bs2 = self.get_partial_transfer_matrices()

        # Back-propagate

        out = delta

        out_theta = np.dot(np.transpose(t_bs2), out)
        out_bs1 = np.dot(np.transpose(t_theta), out_theta)
        out_phi = np.dot(np.transpose(t_bs1), out_bs1)
        inp = np.dot(np.transpose(t_ext), out_phi)
        
        
        # Store Field for the Adjoint Method

        self.phi_ext_power_b = np.sum(np.abs(out_phi[self.first_ports, :]) ** 2, axis=1)
        self.phi_mzi_power_b = np.sum(np.abs(out_theta[self.first_ports, :]) ** 2, axis=1)

        return inp

class DelayLayer():
    """
    The additional external phase shifters needed to acquire a unitary matrix
    """

    def __init__(self, n, ports) -> None:
        """
        Initialize the last layer of external phase shifters
        :param n: The number of input ports.
        :param ports: The ports associated with the external phase shifters at the last layer.
        """

        self.phi_ext = np.zeros(int(0.5*len(ports)))
        self.n = n
        self.phi_ext_power_f = np.zeros(int(0.5*len(ports)))
        self.phi_ext_power_b = np.zeros(int(0.5*len(ports)))
    
        self.first_ports = np.arange(0, self.n)

    def update(self, phi_ext):
        """
        Update the phase shifter values on the layer
        :param phi_ext: New values for the external phase shifters
        """
        self.phi_ext = phi_ext
    
    def transfer_matrix(self):
        """
        Return the transfer matrices of the external phase shifters.
        :return t_ext: The transfer matrix associated with the external phase shifters 
        """
        
        t_ext = np.eye(self.n, dtype='csingle')
        t_ext[self.first_ports, self.first_ports] = np.exp(1j*self.phi_ext)

        return t_ext
    
    def forward_propagate(self, inp):
        """
        Forward propagate information through the layer
        :param inp: The complex field that is inserted in the layer
        :return out: The complex field at the output of the layer 
        """

        t_ext  = self.transfer_matrix()
        
        out = inp
        out = np.dot(t_ext, out)
        self.phi_ext_power_f = np.sum(np.abs(out[self.first_ports, :]) ** 2, axis=1)  # Store Fields for In Situ Backpropagatiom

        return out
    
    def backward_propagate(self, delta):
        """
        Backpropagate the field through the layer
        :param delta: The complex field that is inserted at the output of the layer
        :return inp: The complex field at the input of the layer
        """

        
        t_ext = self.transfer_matrix()
        
        self.phi_ext_power_b = np.sum(np.abs(delta[self.first_ports, :])**2, axis=1)
        inp = np.dot(np.transpose(t_ext), delta)

        return inp
    
class ClementsLayer():
    "A single Clements Mesh structure implementing a unitary matrix."
    def __init__(self, n) -> None:
        """
        Initialize the Clements layer
        :param n: number of input ports
        :property topology: A list containing all the layer of the Clements Mesh.
        :property identity: A string defining the object as layer.
        :property x: The previous complex field input. This is needed for the back-propagation.
        """
        self.n = n
        self.topology = []
        self.identity = 'layer'
        self.x = []

        ## Create structure
        
        if np.mod(self.n, 2) == 1:

            for layer_index in range(0, self.n):

                if np.mod(layer_index, 2) == 0:

                    ports = np.arange(0, self.n-1)
                    self.topology.append(Layer(self.n, ports))

                else:

                    ports = np.arange(1, self.n)
                    self.topology.append(Layer(self.n, ports))
        else:

            for layer_index in range(0, self.n):

                if np.mod(layer_index, 2) == 0:
                    
                    ports = np.arange(0, self.n)
                    self.topology.append(Layer(self.n, ports))

                else:
                    ports = np.arange(1, self.n-1)
                    self.topology.append(Layer(self.n, ports))
        
        ports = np.zeros(2*self.n, dtype=int)
        ports[0:2*self.n-1:2] = np.arange(0, self.n)
        ports[1:2*self.n:2] = np.arange(0, self.n)
        
        self.topology.append(DelayLayer(self.n, ports))

    def update(self, phi):

        """
        Update all the phase shifters on the Clements mesh.
        :param phi: The new phase shifter vector.
        """

        helper = 0

        for layer_index in range(0, self.n):
            phi_ext_range = np.arange(helper, helper + 2*len(self.topology[layer_index].phi_ext) - 1, 2) 
            phi_mzi_range = np.arange(helper + 1, helper + 2*len(self.topology[layer_index].phi_ext), 2)
            self.topology[layer_index].update(phi[phi_ext_range], phi[phi_mzi_range])
            helper += 2*len(self.topology[layer_index].phi_ext)
        
        phi_delay_range =  np.arange(helper, helper + len(self.topology[self.n].phi_ext))
        self.topology[self.n].update(phi[phi_delay_range]) 
    
    def forward_propagate(self, inp):
        """
        Forward propagate a complex field through the  Clements mesh.
        :param inp: The complex field input.
        :return out: The complex field at the output ports of the Clements Mesh.
        """
        
        self.x = inp
        out = inp

        for layer in self.topology:
            out = layer.forward_propagate(out)
        
        return out

    def backward_propagate(self, delta):
        """
        Back-propagate a complex field through the Clements mesh.
        :param delta: The complex field inserted at the output of the Clements mesh.
        :return out: The complex field at the input of the Clements mesh.
        """

        out = delta

        for layer in reversed(self.topology):
            out = layer.backward_propagate(out)
            
        return out

    def transfer_function(self):
        """
        Returns the unitary matrix that is implemented by the Clements mesh.
        :return t_mat: The unitary matrix.
        """

        inp = np.eye(self.n) + 0*1j

        t_mat = self.forward_propagate(inp)

        return t_mat

    def phase_shifter_power_measurements(self, direction):
        """
        Returns the power measurements at each phase shifter for the adjoint method.
        :param direction: String with two values "forward", "backward", that label the direction of propagation.
        :return phase_power: A vector containing all the power measurements.
        """

        phase_power = []


        if direction == 'forward':

            for layer_index in range(0, self.n):

                layer_phases = np.zeros(2*len(self.topology[layer_index].phi_ext))
                layer_phases[0:2*len(self.topology[layer_index].phi_ext_power_f)-1:2] =  self.topology[layer_index].phi_ext_power_f
                layer_phases[1:2*len(self.topology[layer_index].phi_mzi_power_f):2] =  self.topology[layer_index].phi_mzi_power_f
                phase_power = np.concatenate((phase_power, layer_phases))
                
            phase_power = np.concatenate((phase_power, self.topology[self.n].phi_ext_power_f))

        else:

            for layer_index in range(0, self.n):

                layer_phases = np.zeros(2*len(self.topology[layer_index].phi_ext))
                layer_phases[0:2*len(self.topology[layer_index].phi_ext_power_b)-1:2] =  self.topology[layer_index].phi_ext_power_b
                layer_phases[1:2*len(self.topology[layer_index].phi_ext_power_b):2] =  self.topology[layer_index].phi_mzi_power_b
                phase_power = np.concatenate((phase_power, layer_phases))
                
            phase_power = np.concatenate((phase_power, self.topology[self.n].phi_ext_power_b))

        return phase_power
    

class ActivationFunction():

    """
    The non-linear activation functions.
    """

    def __init__(self, n, type) -> None:
        """
        Initialize an activation function object.
        :param n: The number of input ports.
        :param type: Choose the type of the activation function. Supports "Abs" for |z|, "AbsSquare" for |z|^2, "EO" for the MZI based electro-optic function.
        :property dfdRe: The gradient of the activation function with respect to the real part.
        :property dfdIm : The gradient of the activation function with respect to the imaginary part.
        :property identity: A string stating that this object is an "activation" function.
        """
        self.n = n
        self.type = type
        self.dfdRe = []
        self.dfdIm = []
        self.identity = 'activation'

    def forward_propagate(self, inp):
        """
        Forward propagate a complex field through the activation function.
        :param inp: The complex field at the input of the activation functions.
        :property gradient: The gradients with respect to the complex field.
        :return out: The output of the activation functions.
        """

        a, b = torch.tensor(np.real(inp)), torch.tensor(np.imag(inp))

        z = Variable(torch.complex(a, b), requires_grad=True)

        if self.type == 'Abs':
            out = torch.complex(torch.abs(z), torch.zeros_like(a))
        elif self.type == 'AbsSquare':
            out = torch.complex(torch.pow(torch.abs(z),2), torch.zeros_like(a))
        elif self.type == 'EO':

            alpha = torch.tensor(0.1)
            phi_b = torch.tensor(1*np.pi)
            g = torch.tensor(0.05*np.pi)

            dph = phi_b + g*z*torch.conj(z)

            out = 1j * torch.sqrt(1 - alpha) * 1j * torch.exp(-1j*dph/2.0)*torch.cos(dph/2.0)*z

        else:
            out = z
        
        form = torch.complex(torch.ones_like(a), torch.zeros_like(b))
        out.backward(form)
       
        self.gradient =  np.conj(z.grad.data.numpy())

        return out.detach().numpy()
    
    def backward_propagate(self, gamma):

        """
        Back-propagate the field through the activation functions.
        :param gamma: The field at the output ports of the activation functions.
        :return delta: The field at the input ports of the activation functions.
        
        """
        
        delta = gamma*self.gradient

        return delta
    
    def plot_activation(self, z_range):

        """
        Plot the activation function.
        :param z_range: The range of the field amplitude for the plot diagram.
        """

        f = self.forward_propagate(z_range + 0*1j)

        plt.plot(z_range, np.abs(f))
        plt.title('Activation Function :' + self.type)
        plt.xlabel('Input Amplitude')
        plt.ylabel('Output Amplitude')
        plt.show()

class Network():

    """
    An object for the whole optical neural network.
    """

    def __init__(self, model, loss) -> None:
        """
        Initialie the object Network.
        :param model: A list with all the components(linear, non-linear layers) of the neural network.
        :param loss: An object defining the loss function to be used for evaluations. Either MeanSquaredError or CategoricalCrossEntropy.
        """
        self.model = model
        self.loss = loss

    def compute_limits(self):

        limits = []

        for component in self.model:
            if component.identity == 'layer':
                
                 n = component.n
                 
                 for layer_index in range(0, n):

                    layer_limits = np.zeros(2*len(component.topology[layer_index].phi_ext))
                    layer_limits[0:2*len(component.topology[layer_index].phi_ext)-1:2] =  2*np.pi
                    layer_limits[1:2*len(component.topology[layer_index].phi_ext):2] =  np.pi
                    limits = np.concatenate((limits, layer_limits))

                 limits = np.concatenate((limits, 2*np.pi*np.ones(n)))
        
        return limits


    def forward_propagate(self, inp):
        """
        Forward propagate a complex field through the optical neural network.
        :param inp: The input complex field.
        :return output: The output power.
        """

        out = inp

        for component in self.model:
            out = component.forward_propagate(out)

        return out

    def backward_propagate(self, delta):
        """
        Perform back-propagation with the adjoint method through the optical neural network.
        :param delta: The complex field applied at the output of the neural network.
        :return inp: The complex field that results at the input of the neural network.
        """

        inp = delta

        for i in range(len(self.model)-1, -1, -1):
            component = self.model[i]
            inp = component.backward_propagate(inp)
        
        return inp

    def monitor_phase_shifter_power(self, direction)->np.ndarray:
        """
        Monitor the power at the output of all phase shifters of the optical neural network.
        :param direction: Choose the direction of the propagation. Either "forward" or "backward".
        :return phase_power: A vector containing all the power measurements for all the phase shifters.
        """
        
        phase_power = []

        for component in self.model:
            if component.identity == 'layer':

                phase_power = np.concatenate((phase_power, component.phase_shifter_power_measurements(direction)))
        
        return phase_power
    
    def upload_phase_shifter_vector(self, phi):
        """
        Upload a new phase shifter vector at the Clements layers of the optical neural newtork.
        :param phi: A vector containing all the phase shifter values.
        """

        cntr = 0

        for component in self.model:
            if component.identity == 'layer':
                n = component.n
                phi_component = phi[cntr:cntr+ int(n*(n-1)+ n)]
                component.update(phi_component)
                cntr += int(n*(n-1) + n)


    def adjoint_method_gradients(self, x_train, t_train):
        """
        Perform the adjoint method to determine the gradients of the phase shifters.
        :param inp: The inputs of the training set in a complex field representation.
        :param t_train: The targets of the training set in an one-hot vector representation.
        :return loss: The loss of the current evaluation.
        :return gradients: The gradients of the loss function with respect to all the phase shifters.

        """

        # Forward_propagate
        loss_fun = self.loss()

        out = self.forward_propagate(x_train)
        loss = loss_fun.loss(out, t_train)

        power_forward = self.monitor_phase_shifter_power('forward')

        # Backward_propagate
        n_classes, _ = np.shape(t_train)
        dloss = np.zeros(np.shape(out)) + 0*1j
        dloss[0:n_classes, :] = loss_fun.gradient + 0*1j
       
        delta = dloss
        for component in reversed(self.model):
            delta = component.backward_propagate(delta)
            if component.identity == 'layer':
                inp = np.conj(1j*delta) + component.x
                _ = component.forward_propagate(inp)

        
        power_backward = self.monitor_phase_shifter_power('backward')
        power_total = self.monitor_phase_shifter_power('forward')

        gradients = 0.5*(power_total - power_forward - power_backward)
        

        return loss, gradients


def number_of_phase_shifters_for_unitary_matrix(n):
    """
    A short function that determines the number of phase shifters needed for an n x n unitary matrix.
    :param n: The number of input ports.
    :return int(total): The number of all the phases needed to tune the unitary matrix.
    """
    return int(2*n*(n-1)/2 + n)

def total_number_of_phase_shifters(dimensions):
    """
    A short function that determines the total number of phase shifters in an optical neural network.
    :param dimensions: A list containing the number of inputs for each Clements mesh of the optical neural network.
    :return int(total): The number of all phase shifters in the optical neural network.
    """
    total = 0
    for layer in range(0, len(dimensions)):
        total += number_of_phase_shifters_for_unitary_matrix(dimensions[layer])
    
    return int(total)

def number_of_random_variables(n):
    
    """
    A short function that determines the total number of random variables needed to define the posterior of a single Clements Mesh.
    :param n: The number of input ports.
    :return : The number of random variables.
    """
    return 2*number_of_phase_shifters_for_unitary_matrix(n)

def initialize_theta(dimensions):

    """
    Initialize the theta-vector containing all the random variables to be trained by the Bayesian algorithm.
    :param dimensions: A list containing the number of inputs for each Clements mesh of the optical neural network.
    :return theta: The initialized theta-vector.
    """

    theta = []
    rho_mean = []
    rho_phi = []

    for i in range(0, len(dimensions)):
        n_phase_shifters = number_of_phase_shifters_for_unitary_matrix(dimensions[i])
        rho_mean_phi_layer = -1 + 2*np.random.rand(n_phase_shifters)
        rho_phi_layer = np.log(np.exp(0.2*np.ones(n_phase_shifters)) - 1)

        rho_mean = np.concatenate((rho_mean, rho_mean_phi_layer))
        rho_phi = np.concatenate((rho_phi, rho_phi_layer))

    theta = np.concatenate((theta, rho_mean, rho_phi))
    
    return theta


def unpack_theta(theta, dimensions):
    """
    Unpack the theta-vector containing all the variables to be trained by the Bayesian algorithm.
    :param theta: The theta-vector.
    :param dimensions: A list containing the number of inputs for each Clements mesh of the optical neural network.
    :return rho_mean: The free variable vector that through the softplus function defines the mean values of the phase shifters.
    :return rho_mean: The free variable vector that through the softplus function defines the stabdard deviation values of the phase shifters. 
    """

    rho_mean= []
    rho_std = []

    cntr = 0
    for i in range(0, len(dimensions)):
        n_rv = number_of_random_variables(dimensions[i])
        cntr += n_rv

    rho_mean = theta[0:int(cntr/2)]
    rho_std = theta[int(cntr/2):cntr]

    return rho_mean, rho_std


def neural_net_training(epochs, batch_ratio, eta, x_train, t_train, theta, dimensions, network, method, sigma_prior=1, phi_char = None, suppress=False):

    """
    A function tha trains the whole optical neural network.
    :param epochs: The numbre of epochs.
    :param batch_ratio: The portion of the training set that will be used as a mini batch.
    :param eta: The training step size.
    :param x_train: The input training features in a complex field representation.
    :param t_train: The targets of the training set in a one-hot vector representation.
    :param theta: The theta-vector containing all the trainable variables of the posterior. 
    :param dimensions: A list containing the number of inputs for each Clements mesh of the optical neural network.
    :param network: The object that represents the whole optical neural network.
    :param method: Three methods are supported. Choose "bits-back" for the full Bayesian training, "regularization" for training with L2 regularization, "classical" for conventional training.
    :param sigma_prior: Choose the standard deviation value for the prior.
    :param phi_char: Insert the passive offsets determined by a pre-characterization routine. If None (the default), it is assumed that the passive offsets are [0,0, ... 0]. 
    :param suppress: Choose whether to show or hide a bar showing the training progress. If False (the default), it is suppressed, whereas for True it is presented.
    :return theta: The trained theta-vector containing all the variables for the approximate posterior.
    :return loss: Return a vector containing the evaluations of the loss function for each epoch.
    :return accuracy: Return a vector containing the evaluations of the accuracy for each epoch. 
    """

    limits = network.compute_limits()

    # Initialize random variable storage
    n_data, n_train = np.shape(x_train)
    n = network.model[0].n

    diff = n - n_data
    if diff != 0:
        add = np.zeros((diff, n_train))
        x_train = np.row_stack((x_train, add))


    gradient_theta = np.zeros(len(theta))
    m_theta = np.zeros(len(theta))
    v_theta = np.zeros(len(theta))

    losses = np.zeros(epochs)
    accuracies = np.zeros(epochs)

    _, n_examples = np.shape(t_train)
    batch_length = int(batch_ratio*n_examples)

    n_batches = int(n_examples/batch_length)

    if suppress == False:
        iterator = tqdm(range(epochs))
    else:
        iterator = range(epochs)

    time_counter = 0

    def sigmoid(x, par):
        return par*(1/(1+np.exp(-x)))
    
    def dsigmoid(x, par):
        return par*np.exp(-x)/(1+np.exp(-x))**2
    
    def softplus(x):
        return np.log(1 + np.exp(x))
    
    def dsoftplus(x):
        return (1/(1+np.exp(-x)))

    for epoch in iterator:

        batch_losses = []
        batch_accuracy = []

        for batch in range(0, n_batches):
            
            from_batch = int(batch*batch_length)
            to_batch = int((batch+1)*batch_length)
            x_mini_batch = x_train[:, from_batch:to_batch]
            t_mini_batch = t_train[:, from_batch:to_batch]

            pi = 2**(n_batches - batch)/(2**(n_batches) - 1)

            rho_mean, rho_phi = unpack_theta(theta, dimensions)
            if method == 'bits-back':
                eps = np.random.randn(len(rho_mean))
            else:
                eps = 0

            # The classical method
            mean_phi = phi_char + sigmoid(rho_mean, limits)
            std_phi =  softplus(rho_phi)

            phi = mean_phi + eps*std_phi

            network.upload_phase_shifter_vector(phi)
            cost, adjoint_gradients = network.adjoint_method_gradients(x_mini_batch, t_mini_batch)

            # Additional steps
            if method == "bits-back": 
                
                mean_phi_tensor = Variable(torch.tensor(mean_phi), requires_grad=True)
                std_phi_tensor = Variable(torch.tensor(std_phi), requires_grad=True)
                phi_tensor = Variable(torch.tensor(phi), requires_grad=True)
                if phi_char is None:
                    mean_prior_tensor = torch.zeros_like(mean_phi_tensor)
                else:
                    mean_prior_tensor = torch.tensor(phi_char)
                
                sigma_prior_tensor = sigma_prior*torch.ones_like(std_phi_tensor)
                kl = normal_kl_divergence(mean_phi_tensor, std_phi_tensor, phi_tensor, mean_prior_tensor, sigma_prior_tensor)*torch.tensor(pi)
                kl.backward()

                partial_phi_grad =phi_tensor.grad.data.numpy()
                partial_mean_grad = mean_phi_tensor.grad.data.numpy()
                partial_std_grad = std_phi_tensor.grad.data.numpy()

                mean_grad = partial_mean_grad + partial_phi_grad + adjoint_gradients
                std_grad = partial_std_grad + eps*(partial_phi_grad + adjoint_gradients) 
                rho_grad = dsoftplus(rho_phi)*std_grad
                rho_mean_grad = dsigmoid(rho_mean, limits)*mean_grad


                gradient_theta = np.concatenate((rho_mean_grad, rho_grad))

                # Update theta

                adam_gradients, m_theta, v_theta = adam_optimizer(gradient_theta, m_theta, v_theta, time_counter)
                theta -= eta*adam_gradients

                cost += kl.detach().item()

            elif method=='regularization':
                
                if phi_char is None:
                    z = phi/sigma_prior
                else:
                    z = (phi - phi_char)/sigma_prior
                cost_phi = pi*(len(phi)*np.log(np.sqrt(2*np.pi)*sigma_prior) + 0.5*np.sum(z**2))

                grad_cost_phi = pi*z/sigma_prior

                rho_mean_grad = dsigmoid(rho_mean, limits)*(grad_cost_phi + adjoint_gradients)

                gradient_theta[0: int(len(theta)/2)] = rho_mean_grad

                adam_gradients, m_theta, v_theta = adam_optimizer(gradient_theta, m_theta, v_theta, time_counter)
                theta -= eta*adam_gradients

                cost += cost_phi
            
            else:

                rho_mean_grad = dsigmoid(rho_mean, limits)*(adjoint_gradients)
                gradient_theta[0: int(len(theta)/2)] = rho_mean_grad
            
            
            adam_gradients, m_theta, v_theta = adam_optimizer(gradient_theta, m_theta, v_theta, time_counter)
            theta -= eta*adam_gradients
            batch_losses.append(cost)

            batch_accuracy.append(calculate_accuracy(network, x_mini_batch, t_mini_batch))
            time_counter += 1

        losses[epoch] = np.mean(np.array(batch_losses))

        accuracies[epoch] = np.mean(np.array(batch_accuracy))
        
        if suppress==False:
            iterator.set_description("ℒ = %.4f, A = %.4f"%(losses[epoch], accuracies[epoch]), refresh=False)

    return theta, losses, accuracies


def evaluate_testing(x_test, t_test, phi, network):
    """
    Evaluate the testing set.
    :param x_train: The input training features in a complex field representation.
    :param t_train: The targets of the training set in a one-hot vector representation.
    :param phi: The phase shifter values that are uploaded on the network.
    :param network: The object that represents the whole optical neural network.
    """
    
    loss_fun = network.loss()

    network.upload_phase_shifter_vector(phi)

    out = network.forward_propagate(x_test)
    loss = loss_fun.loss(out, t_test)

    accuracy = calculate_accuracy(network, x_test, t_test)

    return loss, accuracy
    
def scale_phases(dphi_unscaled):
    """
    Scale the phases so as to be in the region [-π, π].
    :param dphi_unscaled: The unscaled phases.
    :param dphi_scaled: The scaled phases.
    """

    dphi_scaled = dphi_unscaled + np.zeros(len(dphi_unscaled))

    wh_up = np.where(dphi_scaled>2*np.pi)[0]

    if len(wh_up) != 0:
        m_up = np.ceil((dphi_scaled[wh_up] - 2*np.pi)/(2*np.pi))
        dphi_scaled[wh_up] -= m_up*2*np.pi

    return dphi_scaled
 

def initialize_on_chip_meshes(theta, dimensions, type='uniform'):
    """
    Determine the passive phases due to fabrication errors.
    :param theta: The trained theta-vector containing all the variables for the approximate posterior.
    :param dimensions:  A list containing the number of inputs for each Clements mesh of the optical neural network.
    :param type: Set "uniform" (default) for uniform values between (-π, π), set "gaussian" for Gaussian fabrication error around zero mean, set "zeros" for clear zeros.
    :return phi: Return a vector with the passive offsets.
    """

    mean_phi, _ = unpack_theta(theta, dimensions)
    
    if type == 'zeros':
        phi = np.zeros(len(mean_phi))
    elif type == 'gaussian':
        phi = 0.01*np.random.randn(len(mean_phi))
    else:
        phi = 2*np.pi*np.random.rand(len(mean_phi))

    return phi

def load_on_chip(phi_init, phi_measured, theta, dimensions, method, thermal_crosstalk=0, quantization=None, limits=None):
    """
    Load the phase vector values on the physical chip.
    :param phi_init: The passive offset vector of the chip.
    :param phi_measured: The measured passive offsets determined by the precharacterization procedure. Set a zero vector if no precharacterization procedure took place.
    :param theta: The trained theta-vector containing all the variables for the approximate posterior.
    :param dimensions:  A list containing the number of inputs for each Clements mesh of the optical neural network.
    :param method: Three methods are supported. Choose "bits-back" for the full Bayesian training, "regularization" for training with L2 regularization, "classical" for conventional training.
    :param thermal_crosstalk: Define the thermal crosstalk coefficient. (Default = 0).
    :param quantization: Define the bit-precision of the actuators in bits. If None (default), infinite precision is assumed.
    :return phi_chip: The phase vector as it truly is on-chip. 
    :return dphi_norm : The L1 norm determining the power consumption.
    :return n_prunned: In case of the Bayesian procedure, it returns the number of prunned actuators.
    """

    def sigmoid(x, par):
        return par*(1/(1+np.exp(-x)))
    
    def softplus(x):
        return np.log(1 + np.exp(x))

    rho_mean, rho_phi = unpack_theta(theta, dimensions)

    std_phi = softplus(rho_phi)

    if limits is None:
        mean_phi = phi_measured +  softplus(rho_mean)
    else:
        mean_phi = phi_measured + sigmoid(rho_mean, limits)

    dph = np.zeros(len(phi_init))

    z = (phi_measured - mean_phi)/std_phi
    u =  0.9  + 0*np.random.rand(len(z))

    dph = mean_phi - phi_measured

    if method == 'bits-back':

        for phase in range(0, len(phi_init)):
            if np.abs(z[phase])< 1:
                dph[phase] = 0
            else:
                dph[phase] = mean_phi[phase] - u[phase]*std_phi[phase] - phi_measured[phase]

    # Quantization Error
    if quantization is not None:
        levels = 2*np.pi/(2**quantization)

        quantization_integer = np.floor(dph/levels)
                
        dph_quantized = np.multiply(quantization_integer, levels)
    else:
        dph_quantized = dph + np.zeros(len(dph))
    

    TC = np.zeros((len(dph), len(dph)))
    cntr = 0
    for d in dimensions:
        shape = np.shape(TC[cntr: int(d**2) + cntr, cntr: int(d**2) + cntr ])
        TC[cntr: int(d**2) + cntr, cntr: int(d**2) + cntr ] = thermal_crosstalk*np.random.rand(shape[0], shape[1])
        cntr += int(d**2)
    for i in range(0, len(dph)):
        TC[i, i] = 1

    dph_thermal = np.dot(TC, dph_quantized)
    dph_norm = np.sum(np.abs(dph_quantized))
    
    ph_thermal = phi_init + dph_thermal
    
    n_prunned = len(np.where(dph_quantized == 0)[0])

     
    return ph_thermal, dph_norm, n_prunned
        

    


        








