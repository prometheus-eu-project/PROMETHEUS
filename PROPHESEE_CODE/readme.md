<h1 style="text-align: center;">Prophesee dataset & classification on PMMA particles</h1>

---

## Project Structure

│── `integrated_data_processing/`
│   │── `adamem.py`
│   │── `adasyn.py`
│   │── `classify.py`
│   │── `gpu_stat.so`
│   │── `neuronal_models.py`
│   │── `sgru.py`
│   │── `syn.py`
│   │── `utilities.py`
│── `raw_data_processing/`
│   │── `data_classify.py`
│   │── `data_processing_visualization.py`
│   │── `Input_Dialog.py`
│   │── `spiking_network.py`
### Breakdown of each file

* `integrated_data_processing/` : Folder that contains code for classifying PMMA particles using spiking neural networks from a rate-coded dataset gathered from a Prophesee event-based sensor.
    * `adamem.py` : Adaptive First Order Leaky Integrate and Fire (LIF) neuron implementation.
    * `adasyn.py` : Adaptive Second Order Leaky Integrate and Fire (LIF) neuron implementation.
    * `classify.py` : Classifies PMMA particles:
        1. It loads the pre-processed dataset, which contains event frames where each pixel represents the spike count, i.e., for each X,Y coordinate (pixel) it counts the spikes/events (either positive or negative) that occured in that integration window.
        2. It clips the data, creating a binary representation.
        3. Performs classification using various SNN models defined in `neuronal_models.py`.
    * `gpu_stat.so` : Library that exposes an API call to NVIDIA GPU Driver to measure the power consumption.
    * `neuronal_models.py` : Contains code for various SNN Models used to perform classification vayring from SMLP to Recurrent architectures containing (SLSTM and SGRU)
    * `sgru.py` : Spiking Gated Recurrent Unit implementation.
    * `syn.py` : Second ORder Leaky Integrate and Fire (LIF) neuron implementation.
    * `utilities.py` : Contains utility code used to load the dataset and output accuracy.
    
* `raw_data_processing/` : Folder that contains code for creating an accumulated frame-based dataset from Prophesee event-based sensor raw data. It also contains code for performing classification using spiking neural networks.
    * `data_classify.py` : Loads the extracted dataset from `data_processing_visualization.py` and performs classification using Spiking Neural Network defined in `spiking_network.py`.
    * `data_processing_visualization.py` : Prompts the user for configuration (class, low/high threshold, integration window) using `Input_Dialog.py` along with raw data file and creates frame data by accumulating events based on integration window. Finally uses a straightforward center of mass tracking algorithm to track and crop a **100 x 100** bounding box. Outputs the filtered frames in equivalent *.h5py* file.
    * `Input_Dialog.py` : GUI class that is used to prompt the user for the integration window, the class and low/high thresholds.
    * `spiking_network.py` : Defines the Spiking Neural Network used to classify the PMMA particles. It utilizes two Leaky Integrate-and-Fire (LIF) neurons and a fully connected layer. The first Leaky neuron processes the input spikes, while the second Leaky neuron processes the output weighted spikes from the first LIF. 
    The network processes inputs through a series of layers, accumulating the spikes and membrane potentials across timesteps and returns the final output spikes and membrane potentials.

---

## Dependencies & Installation

To run this project, ensure you have the following Python libraries installed:

### 📊 Numerical & Data Handling  
- [NumPy](https://numpy.org/) - Numerical computing library  
- [h5py](https://www.h5py.org/) - HDF5 file handling library  
- [scikit-learn](https://scikit-learn.org/stable/) - Machine learning utilities  

### 🤖 Deep Learning & Spiking Neural Networks  
- [PyTorch](https://pytorch.org/) - Deep learning framework  
- [SNNtorch](https://snntorch.readthedocs.io/en/latest/) - Spiking neural network library  
- [TorchSummary](https://github.com/sksq96/pytorch-summary) - Model summary utility  

### 🎨 Visualization & Image Processing  
- [Matplotlib](https://matplotlib.org/) - Plotting and visualization library  
- [OpenCV](https://opencv.org/) - Computer vision and image processing library  


## 📌 Installation  

Run the following command to install all dependencies:  

```bash
pip install numpy opencv-python snntorch torch matplotlib scikit-learn h5py torchsummary
```
---

## 📝 Notes  

1. The code provided assumes that files are stored in a predefined path. This might not work for you, so make sure you use the correct paths before running the code.  
2. Ensure you have Python **3.8+** installed for compatibility.  
3. If you encounter issues, try installing dependencies one by one.  
4. If you want CUDA support please take a look at [ PyTorch's official installation guide](https://pytorch.org/multipy/main/setup.html).