<h1 style="text-align: center;">Trainable Filters Code</h1>

---

## Project Structure

│── `FilterClasses/`
│   │── `BaseFilter.py`
│   │── `TrainableFrequency.py`
│   │── `TrainableTransferFunction.py`
│   │── `utility.py`
│── `trainable_frequency_row.py`
│── `trainable_transfer_function_row.py`

### Breakdown of each file:
* `FilterClasses` : Main directory that contains the code for each filter.
    * `BaseFilter.py` :     Abstract Base Filter class that defines the fundamental properties and computations shared by all filter types.
    * `TrainableFrequency.py` : Trainable Frequency and Bandwidth Butterworth module.
    * `TrainableTransferFunction.py` : Custom Trainable Complex Coefficients module.
    * `utility.py` : Contains utility code for upsampling and flattening the images.

* `trainable_frequency_row.py` : Driver code which solves MNIST using trainable frequency module
* `trainable_transfer_function_row.py` : Driver code which solves MNIST using custom trainable coefficients module.

---

## Dependencies & Installation

To run this project, ensure you have the following Python libraries installed:

- [NumPy](https://numpy.org/) - Numerical computing library  
- [Pandas](https://pandas.pydata.org/) - Data analysis and manipulation tool  
- [Matplotlib](https://matplotlib.org/) - Plotting and visualization library  
- [TorchAudio](https://pytorch.org/audio/) - Audio processing library for PyTorch  
- [TorchVision](https://pytorch.org/vision/stable/index.html) - Image processing library for PyTorch  
- [PyTorch](https://pytorch.org/) - Deep learning framework  
- [SciPy](https://scipy.org/) - Scientific computing library  

### Installation

You can install all dependencies using pip:

```bash
pip install numpy pandas matplotlib scipy torch torchvision torchaudio scipy
```

If you want CUDA support please take a look at [ PyTorch's official installation guide](https://pytorch.org/multipy/main/setup.html).
