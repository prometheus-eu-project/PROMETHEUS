import h5py
import numpy as np
from tkinter import filedialog,simpledialog
import os
import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import re
from datetime import datetime
import copy           
from tkinter import PhotoImage
import glob
from snntorch import functional as SF
import torch
import torch.nn as nn
from typing import List
import snntorch as snn
import torchsummary 
import snntorch.spikegen as sgen
import torch.nn.functional as F
import matplotlib.pyplot as plt
import h5py
import numpy as np
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from snntorch import functional as SF
import matplotlib.pyplot as plt
import sys
import h5py
import sklearn.utils as utils
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader,random_split
from sklearn.model_selection import train_test_split

def concatenate_dicts(
    dict_data: dict[str, List[np.ndarray]],
    dict_labels: dict[str, List[np.ndarray]],
    dict_diameters: dict[str, List[np.ndarray]]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    '''
    Concatenates lists of NumPy arrays stored in dictionaries into single NumPy arrays.

    This function iterates through the provided dictionaries (`dict_data` and `dict_labels`), concatenates the lists of NumPy arrays for each key, and returns new dictionaries with the concatenated arrays. If the list for any key is empty, it returns an empty NumPy array for that key to avoid errors.

    Parameters
    ----------
    dict_data : dict[str, List[np.ndarray]]
        A dictionary where each key maps to a list of NumPy arrays. The arrays will be concatenated along axis 0.

    dict_labels : dict[str, List[np.ndarray]]
        A dictionary where each key maps to a list of label arrays. The arrays will be concatenated along axis 0.

    dict_diameters : dict[str, List[np.ndarray]]
        A dictionary where each key maps to a list of diameter arrays. Currently not used in the function but provided for compatibility.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]
        A tuple containing the concatenated dictionaries:
        - `dict_data`: Concatenated data arrays.
        - `dict_labels`: Concatenated label arrays.
        - `dict_diameters`: The input `dict_diameters`, which is not modified.

    Example
    -------
    dict_data = {"class1": [data_array1, data_array2], "class2": [data_array3]}
    dict_labels = {"class1": [labels_array1, labels_array2], "class2": [labels_array3]}
    dict_diameters = {"class1": [diameter_array1], "class2": [diameter_array2]}
    concatenated_data, concatenated_labels, _ = concatenate_dicts(dict_data, dict_labels, dict_diameters)
    '''
    for key in dict_data:
        dict_data[key] = np.concatenate(dict_data[key], axis=0) if dict_data[key] else np.array([])  # Avoid empty array errors
        dict_labels[key] = np.concatenate(dict_labels[key], axis=0) if dict_labels[key] else np.array([])  # Avoid empty array errors

    return dict_data, dict_labels


def merge_datasets(datasets_path: str, speed : str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    '''
    Merges datasets from multiple HDF5 files

    This function iterates through directories for each class (`"20um"`, `"16um"`, `"12um"`) and merges data, labels, and diameters stored in `.h5` files.
    It then calls `concatenate_dicts` to concatenate lists of arrays into single arrays for each class to perform classification later.

    Parameters
    ----------
    datasets_path : str
        The path to the dataset directory that contains subdirectories for each class.
    
    speed : str
        The particle speed (0.02 or 0.2)

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]
        A tuple containing the dictionaries with concatenated data, labels, and diameters:
        - `dict_data`: Dictionary of concatenated data arrays.
        - `dict_labels`: Dictionary of concatenated label arrays.
        - `dict_diameters`: Dictionary of concatenated diameter arrays (passed but not used in this function).
    '''
    
    
    
    classes_dictionary_data ={"20um":[],
     "16um":[],
     "12um":[]}
    classes_dictionary_labels = {"20um":[],
     "16um":[],
     "12um":[]}
    
    classes_dictionary_diameters = {"20um":[],
     "16um":[],
     "12um":[]}
    
    # for each class...
    for key,value in enumerate(classes_dictionary_data):
        print(f"Merging class {value}")
        directory = f"./{value}/{speed}/"
        print(f"\tentering.. {directory}")
        # find all the .h5 files inside this directory
        files = glob.glob(os.path.join(directory, f"*.h5"))
        for file in files:
            print(f"\t\tFOUND {file}")

            # for each .h5py file, read the data and append it to appropriate list
            with h5py.File(file, 'r') as f:

                diameters = f["diameters"][:]
                data = f['data'][:]  
                labels = f['labels'][:]  
                
                classes_dictionary_data[value].append(data)
                classes_dictionary_labels[value].append(labels)
                classes_dictionary_diameters[value].append(diameters)

        print(f"\texiting... {directory}")

    return concatenate_dicts(classes_dictionary_data,classes_dictionary_labels,None)
    



class LeNet5(nn.Module):
        """
        Typical LeNet V5 Architecture with some adjustments to match the PMMA particle.

        Based on Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," in *Proceedings of the IEEE*, vol. 86, no. 11, pp. 2278-2324, Nov. 1998. DOI: 10.1109/5.726791.

    
        """

        def __init__(self, num_classes):
            super(LeNet5, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm2d(6),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2, stride = 2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2, stride = 2))
            self.fc = nn.Linear(18496, 120)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(120, 84)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(84, num_classes)
            
        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            out = self.relu(out)
            out = self.fc1(out)
            out = self.relu1(out)
            out = self.fc2(out)
            return out
    


class SLSTMNET(nn.Module):

    '''
    Spiking LSTM Network used to perform the classification on the event frames
    It consists of 1 SLSTM layer, 1 Fully Connected layer and 1 LIF layer

    Currently two modes of integration were applied
    1. Rows and Columns : where each row and column are stacked and passed as input yielding 300 features
    2. Rows or Columns : where either rows or columns are passed as input yielding 150 features
    

    '''


    def __init__(self,mode="FULL"):
        super().__init__()
        
        if mode == "ROWS_COLUMNS":
            self.slstm1 = snn.SLSTM(300,50,learn_threshold=True)
        else:
            self.slstm1 = snn.SLSTM(150,50,learn_threshold=True)

        self.fc1 = nn.Linear(50,3)
        self.lif1 = snn.Leaky(beta=0.95,threshold=1.0,learn_beta=True, learn_threshold=True)
        self.timesteps = 200
        self.mode = mode
    def forward(self,x):
        syn, mem1 = self.slstm1.init_slstm()

        mem = self.lif1.init_leaky()
        spk_rec = []
        mem_rec = []

        if self.mode == "ROWS_COLUMNS":
            
            for index in range(x.shape[1]):
                row = x[:,index,:] # fetch row at index
                col = x[:,:,index] # fetch column at index
                
                row_col = torch.concatenate((row,col),dim=1).squeeze(2) # concatenate them
                spk2, syn, mem1= self.slstm1(row_col, syn, mem1) # pass them as input to the SLSTM

                # weight the spikes using a linear layer
                fc2_out = self.fc1(spk2)

                # pass them as inputs to the classifying final layer
                spk,mem = self.lif1(fc2_out)

                spk_rec.append(spk)
                mem_rec.append(mem)
        elif self.mode == "INTEGRATE_ROWS":
            # TO integrate the rows, we iterate over the columns, using each column as a separate timestep
            for col in range(x.shape[1]):
           
                spk2, syn, mem1= self.slstm1(x[:,:,col].squeeze(2), syn, mem1)

                fc2_out = self.fc1(spk2)
                spk,mem = self.lif1(fc2_out)

                spk_rec.append(spk)
                mem_rec.append(mem)
        elif self.mode == "INTEGRATE_COLUMNS":
            # TO integrate the columns, we iterate over the rows, using each row as a separate timestep
            for row in range(x.shape[1]):
           
                spk2, syn, mem1= self.slstm1(x[:,row,:].squeeze(2), syn, mem1)

                fc2_out = self.fc1(spk2)
                spk,mem = self.lif1(fc2_out)

                spk_rec.append(spk)
                mem_rec.append(mem)
                

        return torch.stack(spk_rec), torch.stack(mem_rec)


    

speed = "0.02"

data,labels = merge_datasets("./",speed)


print(data["12um"].shape)

class VideoPlayer:
    '''
    Class used to perform visualization on the dataset prior to classification, it is usesful to make sure that data are not wrong/distorted
    
    It creates a GUI element with two navigation buttons that display
    1. The PMMA Particle event frame
    2. The class of the particular frame

    Note
    ----
    * Since we are passing the stacked frames, the first X belong to class 0, the next X to class 1 and the final X to class 2 ( they are NOT shuffled here, rather, they are shown sequentially)

    '''
    def __init__(self, root, frames,labels, fps=30):
        self.root = root
        self.frames = copy.deepcopy(frames)
        self.labels = copy.deepcopy(labels)
        self.frame_idx = 0
        self.fps = fps
        self.delay = int(1000 / fps)  # Convert FPS to milliseconds


        self.root.geometry("400x400")
        self.root.resizable(False, False)  # Disable resizing                

        # Create a label for displaying the video frames
        self.label = Label(root)
        self.label.pack(expand=True)

        self.index_label = Label(root, text="Frame: 0", font=("Arial", 12), fg="white", bg="black")
        self.index_label.place(x=10, y=10)  # Position at top-left corner
        self.frame_label = Label(root, text="Label: None", font=("Arial", 12), fg="white", bg="black")
        self.frame_label.place(x=10, y=40)  # Position at top-left corner

        self.left_button = tk.Button(root, text="←", font=("Arial", 16), command=self.prev_frame)
        self.left_button.place(x=10, y=350)  # Position at the bottom-left corner

        self.right_button = tk.Button(root, text="→", font=("Arial", 16), command=self.next_frame)
        self.right_button.place(x=300, y=350)  # Position at the bottom-right corner



        self.indices_to_keep = []

        self.root.bind("<Left>", self.prev_frame_key)  # Left arrow key -> Move back
        self.root.bind("<Right>", self.next_frame_key)

        # Start the video loop
        self.update_frame()


    def next_frame(self, event = None):
        """Move to the next frame"""
        if self.frame_idx < len(self.frames) - 1:
            self.frame_idx += 1
        else:
            self.frame_idx = 0  # Loop back to the start
        self.update_frame()

    def prev_frame(self, event=None):
        """Move to the previous frame"""
        if self.frame_idx > 0:
            self.frame_idx -= 1
        else:
            self.frame_idx = len(self.frames) - 1  # Loop to the last frame
        self.update_frame()
    


    def next_frame_key(self, event):
        """Move to the next frame with the right arrow key"""
        self.next_frame()

    def prev_frame_key(self, event):
        """Move to the previous frame with the left arrow key"""
        self.prev_frame()


    def update_frame(self):
        if self.frame_idx < len(self.frames):
            frame = self.frames[self.frame_idx]

            # Convert (150,150,1) → (150,150)
            frame = np.squeeze(frame, axis=-1)
            frame = (frame * 255).astype(np.uint8) 



            # Convert NumPy array to ImageTk format
            image = Image.fromarray(frame).resize((150, 150))  # Resize for better visibility
            image = ImageTk.PhotoImage(image)

            
            # Update the label
            self.label.config(image=image)
            self.label.image = image

            self.index_label.config(text=f"Frame: {self.frame_idx}")
            self.frame_label.config(text=f"Label: {int(self.labels[self.frame_idx])}")


# To create a balanced dataset:
# 1. Find the class with the smallest number of samples.
# 2. Randomly shuffle all classes and keep only the number of samples equal to the smallest class size found in step 1.
# 3. Concatenate the labels and data into a unified dataset.

# Fetch the shapes of each class 
num_data_20 = data["20um"].shape[0]
num_data_16 = data["16um"].shape[0]
num_data_12 = data["12um"].shape[0]

shapes = [num_data_20,num_data_16,num_data_12]

# Step 1. Find the class which has the smallest number of available particles
print(f"Min class is {np.argmin(shapes)} with {shapes[np.argmin(shapes)]} frames {shapes}")
min_class = np.argmin(shapes)
samples = shapes[min_class]



# Step 2. Randomly shuffle all the classes and keep only `samples` from each class
f20_data_sampled = data["20um"][np.random.choice(data["20um"].shape[0], samples, replace=False)]
f20_labels_sampled = labels["20um"][np.random.choice(labels["20um"].shape[0], samples, replace=False)]

f16_data_sampled = data["16um"][np.random.choice(data["16um"].shape[0], samples, replace=False)]
f16_labels_sampled = labels["16um"][np.random.choice(labels["16um"].shape[0], samples, replace=False)]

f12_data_sampled = data["12um"][np.random.choice(data["12um"].shape[0], samples, replace=False)]
f12_labels_sampled = labels["12um"][np.random.choice(labels["12um"].shape[0], samples, replace=False)]

# Combine the sampled data into a single dataset
data_combined = np.concatenate((f20_data_sampled, f16_data_sampled, f12_data_sampled), axis=0)
labels_combined = np.concatenate((f20_labels_sampled, f16_labels_sampled, f12_labels_sampled), axis=0)



# Here, we aim to create overlapping histograms to visualize the class overlap.
# This is done to demonstrate the need for machine learning-based methods,
# as there is significant overlap between all the classes.


total_events_20 = np.sum(data["20um"],axis=(1,2))
total_events_16 = np.sum(data["16um"],axis=(1,2))
total_events_12 = np.sum(data["12um"],axis=(1,2))

bins = range(0, 2500, 100)

# Plot histograms with transparency (alpha)
plt.hist(total_events_20, bins=bins, alpha=0.5, label="20", color="blue", edgecolor="black")
plt.hist(total_events_16, bins=bins, alpha=0.5, label="16", color="red", edgecolor="black")
plt.hist(total_events_12, bins=bins, alpha=0.5, label="12", color="green", edgecolor="black")

# Labels and legend
plt.xlabel("Sum of Pixels")
plt.ylabel("Frequency")
plt.title("Overlapping Histograms")
plt.legend()  # Show labels

plt.show()



root = tk.Tk()
root.title("PMMA Visualizer")

player = VideoPlayer(root, data_combined, labels_combined, fps=60)
root.mainloop()

#=========================== CLASSIFICATION  PART ==============================#

def uprade_progress_accuracy(progress, current_epoch, total_epochs, accuracy):
    bar_length = 50
    block = int(round(bar_length * progress)) 
    text = "\r Epoch [{}/{}] Progress: [{}] {:.2f}% | Accuracy {:.5f} ".format(current_epoch,total_epochs, "=" * block + "." * (bar_length - block), progress * 100, accuracy)
    sys.stdout.write(text)
    sys.stdout.flush()


# Split the combined dataset with a test size of 30%
X_train, X_test, y_train, y_test = train_test_split(data_combined, labels_combined, test_size=0.3)


# Create appropriate tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train,dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test,dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create appropriate DataLoaders
batch_size = 64  # Set your desired batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

device = torch.device("cuda")



network_top = "LENET"

if network_top == "SPIKING":
    net = SLSTMNET("ROWS_COLUMNS").to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))


    loss_fn = SF.ce_rate_loss()
    epochs = 50
    for epoch in range(epochs):
        total = 0
        acc = 0
        batch_idx = 0
        for X,Y in train_loader:
            batch_loss = 0
            Y = torch.reshape(Y, (Y.shape[0],))
            X = X.to(device)
            Y = Y.to(device)
         
            spk_rec,mem_rec = net(X)

            loss_val = loss_fn(spk_rec, Y.long())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            batch_loss += loss_val.item()
            

            
            predicted = spk_rec.cpu()
            labels_cpu = Y.cpu()


            acc += SF.accuracy_rate(predicted, labels_cpu) * predicted.size(1)
            total += predicted.size(1)
            
            uprade_progress_accuracy((batch_idx+1) / len(train_loader),epoch,epochs,100*acc/total)
            
            batch_idx +=1
            
    #====================== EVALUATE ON TEST SET ======================#
    test_acc = 0
    total_test = 0
    with torch.no_grad():
        net.eval()
        for i, data in enumerate(test_loader):
            batch_loss = 0
            inputs, labels = data
            labels = torch.reshape(labels, (labels.shape[0],))
        

            inputs = inputs.to(device)
            spk_rec, mem_rec = net( inputs)

            predicted = spk_rec.cpu()
            labels_cpu = labels.cpu()
            test_acc += SF.accuracy_rate(predicted, labels_cpu) * predicted.size(1)
            total_test += predicted.size(1)
            
else:
    net = LeNet5(3).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))



    loss_fn = nn.CrossEntropyLoss()

    epochs = 50
    for epoch in range(epochs):
        total = 0
        acc = 0
        batch_idx = 0
        for X,Y in train_loader:
            batch_loss = 0
            Y = torch.reshape(Y, (Y.shape[0],))
            X = torch.reshape(X,(X.shape[0],1,X.shape[1],X.shape[2]))
            X = X.to(device)
            Y = Y.to(device)
            
            out = net(X)
            loss_val = loss_fn(out, Y.long())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            batch_loss += loss_val.item()
            
            predicted = out.cpu()
            labels_cpu = Y.cpu()
            predicted = predicted.argmax(dim=1)  # Get index of max log-probability
            acc += (predicted == labels_cpu).sum().item()
            total += predicted.size(0)
          
            
            uprade_progress_accuracy((batch_idx+1) / len(train_loader),epoch,50,100*acc/total)
            batch_idx +=1
            


    #====================== EVALUATE ON TEST SET ======================#
    test_acc = 0
    total_test = 0
    with torch.no_grad():
        net.eval()
        for i, data in enumerate(test_loader):
            batch_loss = 0
            inputs, labels = data
            labels = torch.reshape(labels, (labels.shape[0],))
            inputs = torch.reshape(inputs,(inputs.shape[0],1,inputs.shape[1],inputs.shape[2]))
        

            inputs = inputs.to(device)
           
            out = net(inputs)
            
            predicted = out.cpu()

            predicted = predicted.argmax(dim=1)  # Get index of max log-probability
            test_acc += (predicted == labels).sum().item()
            total_test += predicted.size(0)

            


print(f"Test accuracy: {100*test_acc/total_test}")