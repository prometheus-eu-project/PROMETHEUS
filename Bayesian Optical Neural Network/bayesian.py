from matplotlib.pyplot import axis
import numpy as np
import torch
from torch.autograd import Variable, grad

def calculate_accuracy(neural_net, x, t_true):
    """
    Determine the accuracy of the testing set. 
    :param neural net: The object that defines the neural network.
    :param x: The inputs in complex field representation.
    :param t_true: The target vectors in one-hot vector representation.
    :return accuracy: The evaluated accuracy.
    """
    out = neural_net.forward_propagate(x)

    dim, _ = np.shape(t_true)
    out_cropped = out[0:dim, :]

    t_pred = np.exp(out_cropped)/np.sum(np.exp(out_cropped), axis=0)

    accuracy = np.mean( np.argmax(t_pred, axis=0) == np.argmax(t_true, axis=0))

    return accuracy

class CategoricalCrossEntropy():
    """
    An object that implements the Categorical Cross Entropy function.
    """

    def __init__(self) -> None:
      """
      Initialize the Categorical Cross Entropy object.
      :property gradient: The gradient of the loss function with respect to the outputs.
      """
      self.gradient = []
    
    def loss(self, output, t_true):
        """
        Compute the loss.
        :param output: The outputs of the neural network. 
        :param t_true: The target vectors in one-hot vector representation.
        :return loss: Return the computed loss.
        """

        dim, _ = np.shape(t_true)
        out_cropped = np.real(output[0:dim, :])

        z = Variable(torch.tensor(out_cropped), requires_grad=True)
        
        t_pred = torch.softmax(z, dim=0)

        loss = - torch.sum(torch.tensor(t_true)*torch.log(t_pred))
        loss.backward()

        self.gradient = z.grad.data.numpy()

        return loss.detach().item()

class MeanSquareError():
    def __init__(self) -> None:
        self.gradient = []
    
    def loss(self, output, t_true):
        """
        Compute the loss.
        :param output: The outputs of the neural network. 
        :param t_true: The target vectors in one-hot vector representation.
        :return loss: Return the computed loss.
        """
        
        dim, _ = np.shape(t_true)
        out_cropped = np.real(output[0:dim, :])

        z = Variable(torch.tensor(out_cropped), requires_grad=True)
        
        t_pred = z/torch.norm(z, p=1, dim=0)

        loss = 0.5*torch.sum((t_pred - torch.tensor(t_true))**2)

        loss.backward()

        self.gradient = z.grad.data.numpy()

        return loss.detach().item()

def normal_kl_divergence(mean_post, std_post, sample, mean_prior, sigma_prior):
    """
    Computes the kullbag-leiberg divergence between the Gaussian prior and the Gaussian posterior.
    :param mean_post: The mean vector of the Gaussian posterior.
    :param std_post: The standard deviation vector of the Gaussian posterior
    :param mean_prior: The mean vector of the Gaussian prior.
    :param sigma_prior: The standard deviation value of the prior (a single value not tensor).
    :return KL: The Kullbag-Leiberg divergence.
    """
    
    q_log = torch.distributions.Normal(mean_post, std_post).log_prob(sample)
    std_prior = sigma_prior*torch.ones_like(std_post)
    
    p_log = torch.distributions.Normal(mean_prior, std_prior).log_prob(sample)
    
    return torch.sum(q_log- p_log)

def adam_optimizer(gradients, m, v, t, b1=0.9, b2=0.999, e=1e-8):
    """
    The Adam optimizer.
    :param gradients: The gradients as they are calculated by the back-propagation algorithm.
    :param m: The first momentum vector.
    :param v: The second momentum vector.
    :param t: The running epoch.
    :return adam_gradients: The gradients as they are determined by the Adam optimizer.
    :return m: The new first momentum vector.
    :return v: The new second momentum vector.
    """

    m = b1*m + (1-b1)*gradients
    v = b2*v + (1-b2)*(gradients**2)
    mhat = m/(1-b1**(t+1))
    vhat = v/(1-b2**(t+1))

    adam_gradients = mhat/(np.sqrt(vhat) + e)

    return adam_gradients, m, v