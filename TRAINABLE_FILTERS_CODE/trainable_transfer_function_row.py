import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy as sp
import snntorch.surrogate as sr

from torchaudio import transforms
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

from abc import ABC
from FilterClasses.utilities import *

from FilterClasses.TrainableTransferFunction import TrainableTransferFunctionReal, TrainableTransferFunctionComplex





from torchvision import transforms as tvsf
from torchvision import datasets

transform = tvsf.transforms.Compose([
tvsf.transforms.Resize((28, 28)),
tvsf.transforms.Grayscale(),
tvsf.transforms.ToTensor(),
tvsf.transforms.Lambda(lambda x: x * np.pi)])



batch_size = 128
data_path='./data/mnist'
dtype = torch.float

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

   
def uprade_progress_accuracy(progress, current_epoch, total_epochs, accuracy):
    '''
    @brief Function used to update the training progress

    Parameters
    ----------

    - progress : Batch Index divided by the total number of batches
    - current_epoch : Current training epoch
    - total_epochs : Total number of epochs
    - accuracy : Measured accuracy
    '''
    bar_length = 50
    block = int(round(bar_length * progress)) 
    text = "\r Epoch [{}/{}] Progress: [{}] {:.2f}% | Accuracy {:.2f} ".format(current_epoch,total_epochs, "=" * block + "." * (bar_length - block), progress * 100, accuracy)
    sys.stdout.write(text)
    sys.stdout.flush()
import torch.nn.functional as F
if __name__ == "__main__":
    
    
    
    from torchvision import transforms as tvsf
    from torchvision import datasets

    transform = tvsf.transforms.Compose([
    tvsf.transforms.Resize((28, 28)),
    tvsf.transforms.Grayscale(),
    tvsf.transforms.ToTensor(),
    tvsf.transforms.Lambda(lambda x: x * np.pi)])



    batch_size = 128
    data_path='./data/mnist'
    dtype = torch.float

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    
    accuracy_sum = 0
    test_accuracies = []
    
    for trial in range(10):
        # Set a different seed for shuffling
        torch.manual_seed(trial)


        train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

        loss_fn = nn.CrossEntropyLoss()
        
        # NORMAL in case of simple FC layer
        network_type = "FILTER"

        net = TrainableTransferFunctionComplex(2*784,6,10,10,True).to(device)

  
        optimizer = torch.optim.Adam(
            net.parameters(), lr=1e-3)
        epochs = 50
        caches = []

        for epoch in range(epochs):
            total = 0
            acc = 0
            running_loss = 0.
            net.train()
            
            for i, data in enumerate(train_loader):
                inputs_before, labels = data
                batch_loss = 0

                # Flatten the image using both orientations, stacked one after the other
                inputs_row_based = kernel_based_split_vectorized3d_row_based_arhs(inputs_before[:,0],4,4)
                inputs_row_based = inputs_row_based.permute(1,0,2).flatten(start_dim=1)
                
               
                if network_type != "NORMAL":
                    # Upsample the data at 50 GBaud pixel rate
                    phi_row, _ , _ ,_,_,_ = prepare_image(inputs_row_based, 50)

                    # Perform amplitude modulation
                    amplitude_modulated_row = torch.tensor(np.sqrt(1+0.5*phi_row)*np.exp(1j*0)).to(device)
                    
                    # Perform the forward pass
                    output = net(amplitude_modulated_row.real).to(device)
                else:
                    merged_tensor = torch.stack((inputs, inputs_row_based), dim=2)
                    merged = merged_tensor.view(batch_size,-1)
                    output = net(merged.to(device))
                labels = labels.to(device)

                #Calculate the loss
                loss = loss_fn(output, labels)
                
                # Zero the gradients (SGD)
                optimizer.zero_grad()
                
                # Backpropagation
                loss.backward()

                # Update the weights
                optimizer.step()
                
                # Calculate the loss and accuracy
                running_loss += loss.item()
                batch_loss += loss.item()
                # Store loss history for future plotting
                predicted = output.cpu()
                labels_cpu = labels.cpu()
                predicted = output.argmax(dim=1)  # Get index of max log-probability
                acc += (predicted == labels).sum().item()
                total += predicted.size(0)
                uprade_progress_accuracy((i+1) / len(train_loader),epoch,epochs, 100*acc/total)
            if epoch == 0:
                caches = []
                import copy
                for idx in range(0,net.num_filters):
                    caches.append(copy.deepcopy(getattr(net,f"cache_{idx}")))
                
        test_acc = 0
        total_test = 0
        with torch.no_grad():
            net.eval()
            for inputs, labels in test_loader:

                inputs_before = inputs
                batch_loss = 0
                

                inputs_row_based = kernel_based_split_vectorized3d_row_based_arhs(inputs_before[:,0],4,4)
                inputs_row_based = inputs_row_based.permute(1,0,2).flatten(start_dim=1)
              
                if network_type != "NORMAL":

                    

                    phi_row, _ , _ ,_,_,_ = prepare_image(inputs_row_based, 50)
                    amplitude_modulated_row = torch.tensor(np.sqrt(1+0.5*phi_row)*np.exp(1j*0)).to(device)
                    
                    output = net(amplitude_modulated_row.real).to(device)
                
                
                else:
                    merged_tensor = torch.stack((inputs, inputs_row_based), dim=2)
                    merged = merged_tensor.view(batch_size,-1)
                    output = net(merged.to(device))
                    
                labels = labels.to(device)
                loss = loss_fn(output, labels)
                
                running_loss += loss.item()
                batch_loss += loss.item()
                # Store loss history for future plotting
                predicted = output.cpu()
                labels_cpu = labels.cpu()
                predicted = output.argmax(dim=1)  # Get index of max log-probability
                test_acc += (predicted == labels).sum().item()
                total_test += predicted.size(0)        
        print(f"\t Trial {trial} : Test Accuracy: {100 * test_acc / total_test:.2f}%")
        accuracy_sum += 100 * test_acc / total_test
        test_accuracies.append(100*test_acc/total_test)
    print(f"Average accuracy {np.average(test_accuracies)} {accuracy_sum / 10}")
    print(f"std {np.std(test_accuracies)}")
        