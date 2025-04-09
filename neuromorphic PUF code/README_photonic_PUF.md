# 🧠 Dual-Use Photonic System for Neuromorphic Equalization & PUF-Based Authentication

This repository contains the code supporting our experimental work on the **dual use of a reconfigurable photonic mesh** as both a **neuromorphic equalizer** for optical communication signals and a **photonic physical unclonable function (PUF)** for hardware-based authentication.

We demonstrate that a **spectrum slicing self-coherent transceiver**, based on an **optical microring mesh**, can mitigate dispersion effects in a 32 Gbaud/s QPSK link over 25 km of fiber, while simultaneously producing **unique, unclonable digital fingerprints** from fabrication-induced variability in photonic components.

---

## 📦 Features

- 🔁 **Photonic Mesh Equalizer**: Learns to mitigate dispersion in 32 Gbaud QPSK transmission.
- 🔐 **Physical Unclonable Function (PUF)**: Extracts digital responses from unique photonic weights.
- 🧬 **Dual-Use Photonic Node**: Simultaneous computation and security functionality.
- 📉 **Security Metrics**: Equal Error Rate (EER), Inter/Intra distance, and bit-level stability.
- 📊 **Statistical Modeling**: Gaussian Mixture Model (GMM) fitting for distribution analysis.
- 📈 **Fingerprint Binarization**: Quantization, SHA-256, XOR hashing, Gray codes, MinHash.

---

## 🗂 Code Structure

| File | Description |
|------|-------------|
| `main.py` | Runs full evaluation loop with EER plots |
| `data_processing_utils.py` | Signal pre-processing, correlation, resampling, tapping |
| `physical_unclonable_function_utils.py` | PUF generation, noise analysis, EER computation |
| `binarization_utils.py` | Float-to-binary transforms, quantization, hashing |
| `combinatorics_utils.py` | Combines filter responses, manages valid sets |
| `modulation_utils.py` | QAM/PSK modulation and demodulation |
| `distributions_utils.py` | GMM fitting and KS test distribution comparison |
| `machine_learning_utils.py` | Ridge regression equalizer and decision logic |
| `io_utils.py` | Reads/writes `.mat` files and waveform data |

---

## 🧪 Running the Code

Update paths in `CONFIG` inside `main.py`, then run:

```bash
python main.py
```

Outputs:
- `results_n_bits_n_avg.mat`: EER and SER metrics
- `figure.svg`: Histogram comparing inter-/intra-PUF distances
- Log-EER values printed for each setting

---

## ⚙️ Key Parameters (`CONFIG`)

| Name | Description |
|------|-------------|
| `Modulation` | Modulation order (e.g. 4 for QPSK) |
| `n_filters` | Number of filters per mesh combination |
| `taps` | Time-domain taps for feature vectors |
| `averaging` | Signal averaging depth |
| `num_bits` | Quantization bits for binarization |
| `random_projection_dimension` | Dimension for random feature hashing |
| `n_inter_PUFs` / `n_intra_PUFs` | Number of identities / repetitions for PUF |
| `phase_bound` | Allowed phase range for valid PUF distance samples |

---

## 🛡 Security Analysis

- **EER < 10⁻⁸** with SHA-based binarization
- Strong **intra-class repeatability** and **inter-class separation**
- GMM modeling shows clear statistical distinction between devices

---

## 🔬 Experimental Setup

- 32 Gbaud QPSK signal over 25 km SSMF
- Photonic mesh: reconfigurable ring resonator array
- DSP backend: Ridge regression for equalization + binarization for fingerprinting
- Hardware: Self-coherent transceiver with spectrum slicing

---

## 📄 License

This project is released under the **MIT License**.
