import numpy as np
import snntorch as sn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from snntorch import spikegen
import torch.nn.functional as F


import sys



def uprade_progress_accuracy(progress, current_epoch, total_epochs, accuracy):
    bar_length = 50
    block = int(round(bar_length * progress)) 
    text = "\r Epoch [{}/{}] Progress: [{}] {:.2f}% | Accuracy {:.2f} ".format(current_epoch,total_epochs, "=" * block + "." * (bar_length - block), progress * 100, accuracy)
    sys.stdout.write(text)
    sys.stdout.flush()




def category_count(one_hot_vectors):
  
  # Create a NumPy array to store the number of occurrences of each category.
  category_counts = np.zeros(shape=(3,), dtype=np.int32)

  # Iterate over the one-hot vectors and count the number of occurrences of each category.
  for one_hot_vector in one_hot_vectors:
    category = np.argmax(one_hot_vector)
    category_counts[category] += 1

  # Return the NumPy array containing the number of occurrences of each category.
  return category_counts

def from_one_hot_vector_to_categories(one_hot_vector):
  """Converts a one-hot vector to a category."""

  # Get the index of the non-zero element in the one-hot vector.
  index = np.argmax(one_hot_vector)

  # The index of the non-zero element is the category of the one-hot vector.
  return index

def from_one_hot_vectors_to_categories_to_multiple_rows(one_hot_vectors):
  #Converts one-hot vectors to categories to multiple rows.

  # Create a NumPy array to store the categories.
  categories = np.zeros(shape=(one_hot_vectors.shape[0],), dtype=np.int32)

  # Iterate over the one-hot vectors and convert each one to a category.
  for i in range(one_hot_vectors.shape[0]):
    categories[i] = from_one_hot_vector_to_categories(one_hot_vectors[i])

  # Return the NumPy array containing the categories.
  return categories



def load_dataset(path):
    """
    Loads the dataset from the specified path, processes it into a suitable format, and returns the image data (X) 
    and corresponding categories (Y) for further use in training or testing.

    The function reads image data and one-hot encoded category labels from files, processes them, 
    and ensures a balanced dataset by selecting an equal number of samples from each category (12um, 16um, 20um).

    Args:
        path (str): The directory path where the dataset files are located. The dataset should contain:
            - `DataY_HotVector.txt`: One-hot encoded category labels for the samples.
            - `DataX_IMAGE.txt`: Image data corresponding to the samples.

    Returns:
        tuple: A tuple containing:
            - `X_IMAGEconc (np.ndarray)`: A 3D NumPy array of shape `(num_samples, 100, 100)` containing the processed image data.
            - `Yconc (np.ndarray)`: A 2D NumPy array of shape `(num_samples, 3)` containing the category labels for the samples.

    

    """
    DataY_HotVector = np.loadtxt(f"{path}/DataY_HotVector.txt")




    # One hot vector
    Y_HV=np.reshape(DataY_HotVector.astype(np.int32), (1908, 3))
    Y_cat = from_one_hot_vectors_to_categories_to_multiple_rows(Y_HV)

    category_counts = category_count(Y_HV)
    #print('We have: \n',category_counts[0], ' of 12um\n',category_counts[1], ' of 16um\n', category_counts[2], ' of 20um\n')

    DataX_IMAGE = np.loadtxt(f"{path}/DataX_IMAGE.txt")
    X_IMAGE=np.reshape(DataX_IMAGE, (1908, 100, 100))

    Y20= Y_cat[0:420]
    Y16= Y_cat[421:1376]
    Y12= Y_cat[1377:1910]

    #print("Y20", Y20.shape,'\n Y16 ', Y16.shape,'\n Y12', Y12.shape)

    minim=min(len(Y12),len(Y16),len(Y20))
    Y12=Y12[:minim]
    Y16=Y16[:minim]
    Y20=Y20[:minim]

    Yconc=np.concatenate((Y20,Y16,Y12),axis=0)
    #print(f"yconc shape {Yconc.shape}")


    X_IMAGE20= X_IMAGE[0:420]
    X_IMAGE16= X_IMAGE[421:1376]
    X_IMAGE12= X_IMAGE[1377:1910]


    minim=min(len(X_IMAGE12),len(X_IMAGE16),len(X_IMAGE20))
    X_IMAGE12=X_IMAGE12[:minim]
    X_IMAGE16=X_IMAGE16[:minim]
    X_IMAGE20=X_IMAGE20[:minim]

    #### X concatenate
    X_IMAGEconc=np.concatenate((X_IMAGE20,X_IMAGE16,X_IMAGE12),axis=0)

    return X_IMAGEconc, Yconc
    

