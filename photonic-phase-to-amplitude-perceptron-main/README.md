# 💡 Photonic Reservoir Computing with MRR-Based Delay Loop

This repository implements a **time-delayed photonic reservoir computing (RC)** architecture for time-series prediction tasks such as **NARMA10** and the **Santa Fe laser dataset**. The system leverages a **microring resonator (MRR)** as the nonlinear processing node, integrated within a feedback loop to provide memory.

A key innovation in this work is the **use of phase encoding**, which, when propagated through the MRR, is **nonlinearly transformed into an amplitude-modulated signal** due to the spectral characteristics of the MRR's transfer function. This enables rich feature generation from temporally multiplexed inputs.

---

## ✨ Key Features

- 📡 **Phase Encoding**: Digital inputs are encoded in the phase of the optical carrier.
- 🔁 **Time-Delayed Loop**: A feedback loop introduces temporal recurrence, enabling memory.
- 🌐 **MRR-Based Nonlinearity**: Phase-encoded signals undergo amplitude transformation shaped by the MRR transfer function.
- 🎛️ **Customizable Photonic Reservoir**: Configurable parameters include loop type, detuning, feedback strength, ring radius, and coupling.
- 📊 **Evaluation Tools**: Includes NMSE performance metrics and memory capacity analysis.

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/photonic-reservoir-mrr.git
cd photonic-reservoir-mrr
pip install numpy matplotlib scipy scikit-learn tqdm
```

---

## ⚙️ System Overview

### Architecture
- **Input Processing**: Digital signal is masked and expanded in time via ZOH or RZ encoding.
- **Digital-to-Analog Conversion**: Signal is modulated (AM or PM) and filtered to match photonic hardware specs.
- **Photonic Node**: A microring resonator (MRR) with a feedback loop simulates a nonlinear node with memory.
- **Detection**: A photodiode with configurable noise and bandwidth captures the system’s response.
- **Readout Layer**: Linear regression (ridge) is used to map reservoir states to the target output.

---

## 🚀 Running the Code

### Standard NARMA10 Benchmark

```bash
python main.py
```

### Parameter Sweep on Santa Fe Dataset

```bash
python main_scanner.py
```

---

## 🧠 Parameters of Interest

| Parameter | Description |
|----------|-------------|
| `loop_type` | Feedback configuration (`T2D`, `T2T`, `Fano`) |
| `modulation_type` | `'AM'` or `'PM'` for amplitude or phase modulation |
| `ring_phases` | Static detuning phases per ring |
| `radius_vectors` / `coupling_vectors` | MRR structural parameters |
| `feedback_strengths` | Controls memory depth |
| `loop_phase` | Adds additional phase to feedback path |

---

## 📈 Example Output

```bash
The training NMSE is 3.45E-03
The testing NMSE is 4.21E-03
```

A memory capacity plot is also displayed after training.

---

## 🧪 Tasks Supported

- ✅ **NARMA10**: Synthetic nonlinear benchmark
- ✅ **Santa Fe**: Real-world chaotic laser dataset

---

## 🧬 Core Idea

This architecture exploits **nonlinear transformation of phase to amplitude** by an MRR-based filter. The loop introduces delayed self-interaction (echo state property), and the **transfer function of the MRR filters** imposes rich nonlinear dynamics over time-multiplexed inputs — ideal for RC.

---

## 📚 File Overview

```
├── main.py              # Run a single experiment (e.g., NARMA10)
├── main_scanner.py      # Run parameter sweeps (e.g., detuning vs loop phase)
├── digital.py           # PAM modulation, signal masking, Butterworth filters
├── electronics.py       # Modulation and D/A conversion
├── photonic_functions.py# MRR models, laser dynamics, photodetectors
├── problem_loader.py    # Time-series data generators and post-processing
├── README.md            # This file
```

---

## 📄 License

This project is licensed under the **MIT License**.
