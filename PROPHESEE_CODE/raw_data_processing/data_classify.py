import h5py
import numpy as np
from torch.utils.data import DataLoader
import spiking_network as sn
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF
import matplotlib.pyplot as plt
import sys
import h5py
import numpy as np
from torch.utils.data import DataLoader
import spiking_network as sn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF
import snntorch.spikeplot as splt
import matplotlib.pyplot as plt


def uprade_progress_accuracy(progress, current_epoch, total_epochs, accuracy):
    bar_length = 50
    block = int(round(bar_length * progress)) 
    text = "\r Epoch [{}/{}] Progress: [{}] {:.2f}% | Accuracy {:.5f} ".format(current_epoch,total_epochs, "=" * block + "." * (bar_length - block), progress * 100, accuracy)
    sys.stdout.write(text)
    sys.stdout.flush()

# First, we load the datasets created by the `data_processing_visualization.py`

with h5py.File('/home/gmoustakas/Documents/prophesee_processing/20.hdf5', 'r') as hdf:
    f20_data = hdf['data'][:]
    f20_labels = hdf['labels'][:]

with h5py.File('/home/gmoustakas/Documents/prophesee_processing/12.hdf5', 'r') as hdf:
    f12_data = hdf['data'][:]
    f12_labels = hdf['labels'][:]

with h5py.File('/home/gmoustakas/Documents/prophesee_processing/16.hdf5', 'r') as hdf:
    f16_data = hdf['data'][:]
    f16_labels = hdf['labels'][:]


# Concataenate the data and labels
unified_data = np.concatenate((f20_data,f16_data,f12_data),axis=0)
unified_labels = np.concatenate((f20_labels,f16_labels, f12_labels), axis=0)


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(unified_data, unified_labels, test_size=0.2,stratify=unified_labels, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.uint16)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.uint16)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 64  # Set your desired batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)



#======================= After loading, create the SNN ===========================#


device = torch.device("cuda")
network = sn.Network().to(device=device)

optimizer = torch.optim.Adam(network.parameters(), lr=0.001, betas=(0.9, 0.999))


print(len(train_loader))

loss_fn = SF.ce_rate_loss()

for epoch in range(50):
    total = 0
    acc = 0
    batch_idx = 0
    for X,Y in train_loader:
        batch_loss = 0
        X = X.to(device)
        Y = Y.to(device)
        
        spikes_out,mem_out = network(X)
        
        loss_val = loss_fn(spikes_out, Y.long())
        
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        batch_loss += loss_val.item()
        
        # Return the predicted labels and the actual labels from the GPU -> CPU
        predicted = spikes_out.cpu()
        labels_cpu = Y.cpu()


        acc += SF.accuracy_rate(predicted, labels_cpu) * predicted.size(1)
        total += predicted.size(1)

        
        uprade_progress_accuracy((batch_idx+1) / len(train_loader),epoch,50,100*acc/total)
        batch_idx +=1
        


#====================== EVALUATE ON TEST SET ======================#
test_acc = 0
total_test = 0
with torch.no_grad():
    network.eval()
    for i, data in enumerate(test_loader):
        batch_loss = 0
        inputs, labels = data

        

        inputs = inputs.to(device)

        # forward pass
        spk_rec, mem_rec = network(inputs)

        predicted = spk_rec.cpu()
        labels_cpu = labels.cpu()
        test_acc += SF.accuracy_rate(predicted, labels_cpu) * predicted.size(1)
        total_test += predicted.size(1)



print(f"Test accuracy: {100*test_acc/total_test}")