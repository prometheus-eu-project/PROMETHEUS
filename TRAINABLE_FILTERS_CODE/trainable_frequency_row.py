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

from FilterClasses.TrainableFrequency import TrainableFrequency

# Filters configuration, it must be a dictionary that contains
# The central frequency and bw in **GHz** along with a "trainable" key that denotes if its trainable or not
config = {
    "filter_0":{
        "f":{"value":3.0, "trainable":False},
        "bw":{"value":7.0, "trainable":False}
    },
    
    "filter_1":{
        "f":{"value":10.0, "trainable":False},
        "bw":{"value":7.0, "trainable":False}
    },
    
    "filter_2":{
        "f":{"value":17.0, "trainable":False},
        "bw":{"value":7.0, "trainable":False}
    },
    "filter_3":{
        "f":{"value":24.0, "trainable":False},
        "bw":{"value":7.0, "trainable":False}
    },
    "filter_4":{
        "f":{"value":32.0, "trainable":False},
        "bw":{"value":7.0, "trainable":False}
    },
    "filter_5":{
        "f":{"value":39.0, "trainable":False},
        "bw":{"value":7.0, "trainable":False}
    },
    "filter_6":{
        "f":{"value":46.0, "trainable":False},
        "bw":{"value":7.0, "trainable":False}
    },
    
    
}




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


        network_type = "FILTER"


        net = TrainableFrequency(2*784,10,config,True).to(device)

        prev_params = {name: param.clone().detach() for name, param in net.frequencies.named_parameters()}
        prev_params_bw = {name: param.clone().detach() for name, param in net.bws.named_parameters()}

        optimizer = torch.optim.Adam(
            net.parameters(), lr=1e-3)
        
        epochs = 800
        caches = []

        for epoch in range(epochs):
            total = 0
            acc = 0
            running_loss = 0.
            net.train()
            
            for i, data in enumerate(train_loader):
                inputs_before, labels = data
                batch_loss = 0
                
                inputs_row_based = kernel_based_split_vectorized3d_row_based_arhs(inputs_before[:,0],4,4)

                
               
                if network_type != "NORMAL":
                    phi_row, _ , _ ,_,_,f = prepare_image(inputs_row_based, 100)

                    amplitude_modulated_row = torch.tensor(np.sqrt(1+0.5*phi_row)*np.exp(1j*0)).to(device)
                    
                    output = net(amplitude_modulated_row.real).to(device)
                else:
                    merged_tensor = torch.stack((inputs, inputs_row_based), dim=2)
                    merged = merged_tensor.view(batch_size,-1)
                    output = net(merged.to(device))
                labels = labels.to(device)
                loss = loss_fn(output, labels)
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()
                
                running_loss += loss.item()
                batch_loss += loss.item()
                # Store loss history for future plotting
                predicted = output.cpu()
                labels_cpu = labels.cpu()
                predicted = output.argmax(dim=1)  # Get index of max log-probability
                acc += (predicted == labels).sum().item()
                total += predicted.size(0)
                uprade_progress_accuracy((i+1) / len(train_loader),epoch,epochs, 100*acc/total)

            # Code used to show any difference between the epochs on central frequencies and BWs, can be commented out as it was used mainly for debugging issues
            print("\n")
            for (name_f, param_f), (name_bw, param_bw) in zip(net.frequencies.named_parameters(), net.bws.named_parameters()):
                prev_f, new_f, diff_f = prev_params[name_f].item(), param_f.item(), torch.norm(param_f - prev_params[name_f]).item()
                prev_bw, new_bw, diff_bw = prev_params_bw[name_bw].item(), param_bw.item(), torch.norm(param_bw - prev_params_bw[name_bw]).item()

                print(f"{name_f}: Freq → Prev: {prev_f:.6f}, New: {new_f:.6f}, Δ: {diff_f:.6f} | " f"BW → Prev: {prev_bw:.6f}, New: {new_bw:.6f}, Δ: {diff_bw:.6f}")
                    


            prev_params = {name: param.clone().detach() for name, param in net.frequencies.named_parameters()}
            prev_params_bw = {name: param.clone().detach() for name, param in net.bws.named_parameters()}
            
        

                
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

                    

                    phi_row, _ , _ ,_,_,_ = prepare_image(inputs_row_based, 100)
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
        