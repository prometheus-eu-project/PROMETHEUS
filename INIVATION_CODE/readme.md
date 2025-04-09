<h1 style="text-align: center;">Inivation dataset & classification on PMMA particles</h1>

---

## Project Structure

│── `classify.py`
│── `create_event_data.py`

### Breakdown of each file

* `classify.py` : Dataset Preview & Classification
    1. Merges the dataset for 12um, 16um and 20um PMMA Particles
    2. Creates a balanced dataset by maintaining an equal number of samples for each class.
    3. Shows a preview of the dataset, to ensure its integrity (if needed)
    4. Develops SNN and LeNet-5 networks to classify PMMA particles.
    
* `create_event_data.py` : Dataset extraction from Inivation raw data event stream
    1. Prompts the user for the configuration (integration window, thresholds, input file).
    2. Receives events in batches every `integration_window` μs.
    3. Filters the incoming events based on low and high threshold.
    4. The filtered events are saved in an array of shape `(X, Y, T, P)`, where:

        - `X, Y` are constrained within a **150 × 150** bounding box window.  
        - `T` represents the timesteps, incremented each time a frame is found between the **low** and **high** thresholds.  
        - `P` denotes the polarity, indicating an increase or decrease in pixel brightness.  

---

## Dependencies & Installation

To run this project, ensure you have the following Python libraries installed:

### 📷 Data Processing
- [dv-processing](https://gitlab.com/inivation/dv/dv-processing) - INIVation generic processing algorithms for event cameras.

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

### 🖥️ Graphical User Interface (GUI)  
- [Pillow](https://pillow.readthedocs.io/en/stable/) - Image handling library for Tkinter  


## 📌 Installation  

Run the following command to install all dependencies:  

```bash
pip install numpy opencv-python pillow snntorch torch matplotlib scikit-learn h5py torchsummary
```

To install inivation dv-processing please consult the documentation [DV-Processing Documentation](https://dv-processing.inivation.com/rel_1_7/installation.html)

---

## 📝 Notes  

1. The code provided assumes that files are stored in a predefined path. This might not work for you, so make sure you use the correct paths before running the code.  
2. Ensure you have Python **3.8+** installed for compatibility.  
3. If you encounter issues, try installing dependencies one by one.  
4. If you want CUDA support please take a look at [ PyTorch's official installation guide](https://pytorch.org/multipy/main/setup.html).