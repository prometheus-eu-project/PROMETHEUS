import os
from neuronal_models import *
import gpu_stat
import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader,TensorDataset
from snntorch import functional as SF
from sklearn.model_selection import train_test_split
import utilities as snn_utils
import os
import numpy as np
import snntorch as sn
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from utilities import uprade_progress_accuracy

def rate_code_3d(array : np.ndarray) -> np.ndarray:
    """
    Converts a 3D input array of event timings into a rate-coded 4D array representation.

    The function generates a binary 4D array where each frame corresponds to whether a given event in 
    the input array occurred before or at that frame. This is a common technique in spiking neural networks 
    to represent the timing of events or spikes over a set of frames.

    Args:
        array (np.ndarray): A 3D NumPy array of shape (batch_size, spatial_size, time_steps) representing 
                             event timings or spikes. The values in the array represent the time at which 
                             the event occurred.

    Returns:
        np.ndarray: A 4D NumPy array of shape (batch_size, spatial_size, time_steps, max_frames), where 
                    each entry is a binary value indicating if the event occurred before the frame.
        int: The maximum number of frames (`max_frames`), corresponding to the latest event across all input data.


    """
                    
    max_frames = int(array.max())  # The maximum number of frames needed

    # Pre-allocate the output array with zeros
    out_array = np.zeros((array.shape[0], array.shape[1], array.shape[2], max_frames))

    # Create an index grid
    frames = np.arange(max_frames).reshape(1, 1, 1, -1)  # Shape (1,1,1,max_frames)

   
    out_array = (frames < array[..., None]).astype(np.float32)
    

    return out_array, max_frames

os.environ['KMP_DUPLICATE_LIB_OK']='True'
dtype = torch.float32
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

plt.ioff()
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dtype = torch.float32
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

PATH = "/home/gmoustakas/Documents/prophesee_processing/MENELAOS_NETWORKS/balls/"
x,y = snn_utils.load_dataset(PATH)



#x,numsteps = rate_code_3d(x) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create TensorDataset objects
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

# Create DataLoader objects
batch_size = 64  # Adjust batch size as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)



loss_fn = SF.ce_rate_loss()




network = SLSTMNET("ROWS_COLUMNS").to(device)



optimizer = torch.optim.Adam(network.parameters(), lr=0.001, betas=(0.9, 0.999))
epochs = 50
from torchinfo import summary

power_consumption_mws = []

gpu_stat.initNVML()
for epoch in range(epochs):
    total = 0
    acc = 0
    batch_idx = 0
    batch_consumptions = []
    for X,Y in train_loader:
        X = X.to(device)
        Y = Y.to(device)
        batch_loss = 0
        spk_rec,mem_rec = network(X)
        
        loss_val = loss_fn(spk_rec, Y.long())
        
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        batch_loss += loss_val.item()
        
        # Return the predicted labels and the actual labels from the GPU -> CPU
        predicted = spk_rec.cpu()
        labels_cpu = Y.cpu()


        acc += SF.accuracy_rate(predicted, labels_cpu) * predicted.size(1)
        total += predicted.size(1)
        
        uprade_progress_accuracy((batch_idx+1) / len(train_loader),epoch,epochs,100*acc/total)
        batch_idx +=1
        batch_consumptions.append(gpu_stat.getPowerConsumption() / 1000)


    power_consumption_mws.append(np.mean(batch_consumptions))


gpu_stat.shutdownNVML()

print(f"Consumption in  W {np.mean(power_consumption_mws)}")
test_acc = 0 
total_test = 0
with torch.no_grad():
    network.eval()
    for i, data in enumerate(test_loader):
        batch_loss = 0
        inputs, labels = data
        inputs = inputs.to(device)

        # forward pass
        spk_rec, mem_rec = network( inputs)

        predicted = spk_rec.cpu()
        labels_cpu = labels.cpu()
        test_acc += SF.accuracy_rate(predicted, labels_cpu) * predicted.size(1)
        total_test += predicted.size(1)

print(f"Test accuracy: {100*test_acc/total_test}")
#==== work on confusion matrices ====#

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

xtest = x_test_tensor.to(device)
ytest = y_test_tensor.to(device)

spk_rec, mem_rec = network(xtest)
labels_cpu = ytest.cpu()

_, predicted = spk_rec.cpu().sum(dim=0).max(1)
cm = confusion_matrix(labels_cpu, predicted, normalize="true")
disp = ConfusionMatrixDisplay(cm).plot()
fig, ax = plt.subplots()
disp.plot(cmap='Blues', ax=ax)
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')  # Save with high resolution
plt.close() 

#===== apply UMAP now on predictions to visualize the learnable parameters =====#
import umap
aggregated_features = spk_rec.sum(dim=0)  # Shape: [Batch, Features]
features_numpy = aggregated_features.detach().cpu().numpy()  # Convert to NumPy array

umap_embedding = umap.UMAP(n_neighbors=35, min_dist=0.085, n_components=2).fit_transform(features_numpy)
# Fit and transform the features

class_to_color = {0: 'red', 1: 'green', 2: 'blue'}
idx_to_class = {0:"12um", 1 :"16um", 2:"20um"}
colors = [class_to_color[label] for label in labels_cpu.tolist()]
plt.figure(figsize=(8, 6))
plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=colors)
handles = []
for class_label, color in class_to_color.items():
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'{idx_to_class[class_label]}'))

plt.legend(handles=handles, title="Classes", loc='best')

# Title and plot adjustments
plt.title('UMAP Clustering with Specific Class Colors')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')



plt.savefig('umap.png', dpi=300, bbox_inches='tight')  # Save with high resolution
plt.close() 