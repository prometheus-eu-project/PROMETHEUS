# 🧠 Bayesian Optical Neural Network

This project implements a **Bayesian Optical Neural Network (BONN)** in Python. It simulates photonic neural networks using **Clements mesh interferometers** and trains them using **Bayesian inference**, **regularized learning**, or **classical gradient descent**.

---

## 🚀 Features

- 🌀 Optical neural network using Clements mesh topology  
- 🔁 Support for complex-valued forward/backward propagation  
- 🧮 Custom loss functions (Cross Entropy, MSE)  
- 📈 Training via:
  - Bayesian "Bits-Back" method
  - L2 Regularization
  - Classical Adjoint Optimization
- 📊 Dataset support: **Iris** and **MNIST**
- 🔧 Simulates fabrication errors, quantization, thermal crosstalk

---

## 📦 Installation

```bash
git clone https://github.com/your-username/bayesian-optical-neural-network.git
cd bayesian-optical-neural-network
pip install -r requirements.txt
```

### 🧩 Dependencies

- `numpy`
- `scipy`
- `matplotlib`
- `torch`
- `sklearn`
- `torchvision`
- `tqdm`

---

## 🧪 Usage

The main script is `main.py`. It uses the **Iris dataset** by default:

```bash
python main.py
```

You’ll see training loss and accuracy plots for 1000 epochs. Final test accuracy is printed at the end.

---

## 🧠 Training Methods

You can train using one of three approaches:
- `method='bits-back'` – full Bayesian training using KL divergence
- `method='regularization'` – L2 regularized loss
- `method='classical'` – deterministic training without uncertainty modeling

Set this in `main.py`:
```python
method = 'bits-back'  # or 'regularization' / 'classical'
```

---

## 📚 Datasets

### Iris (default)
- 4 input features
- 3 classes (setosa, versicolor, virginica)

### MNIST (optional)
To use MNIST, replace `prepare_iris()` with `prepare_mnist(freq_radius=5)`.

---

## 📊 Output

- `losses`: Training loss per epoch
- `accuracies`: Training accuracy per epoch
- Final test accuracy is displayed
- Optional: Save and visualize trained phase shifters

---

## 🛠 Project Structure

```
├── main.py             # Entry point
├── bayesian.py         # Loss functions, KL divergence, Adam optimizer
├── photonics.py        # Optical layers, activation functions, training logic
├── problems.py         # Dataset loaders (Iris, MNIST)
└── README.md           # Project documentation
```

---

## 🧠 Background

This implementation simulates optical computing by modeling how photonic phase shifters can implement neural networks. Using the **Adjoint Method**, it performs backpropagation by measuring optical powers — a technique aligned with current photonic hardware research.

---

## 🤝 Contributing

PRs welcome! Please include:
- Clear descriptions of the change
- Tests or reproducible results where possible

---

## 📄 License

MIT License © 2025
