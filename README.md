# Blood-Brain Barrier: Quantized 1.58-bit Mixture of Experts for Blood-Brain Barrier Prediction

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🧠 Overview: Molecular AI for In-Silico Laboratory Screening
This project introduces an innovative **Molecular AI** framework designed to function as a "Digital Twin" of the Blood-Brain Barrier (BBB). Traditionally, determining if a new drug can reach the brain requires expensive, time-consuming wet-lab experiments. 

Our solution provides a high-speed, **In-Silico Laboratory** alternative. By utilizing a specialized **Mixture of Experts (MoE)** architecture, the model simulates the chemical logic of the BBB, allowing researchers to screen thousands of compounds in seconds. 

### 🔬 Innovation: The 1.58-bit Advantage
While most molecular models require high-end supercomputers, this project implements **1.58-bit Ternary Quantization**. This allows complex pharmacological reasoning to be performed with extreme hardware efficiency, effectively bringing "Lab-Grade" intelligence to low-power edge devices.

This research project demonstrates that high-precision pharmacological modeling is possible even under extreme quantization, making it suitable for edge-deployment in diagnostic medical devices.

## 🚀 Performance
* **Accuracy:** 88.03%
* **AUC-ROC:** 0.9309
* **F1-Score (Passed):** 0.91
* **Architecture:** 1.58-bit Quantized Weights + 16-Expert MoE Layer
* **VRAM Savings:** ~95% reduction compared to standard FP32 models.

## 🛠️ Tech Stack
- **Deep Learning:** PyTorch
- **Cheminformatics:** RDKit
- **Visualization:** Matplotlib, Scikit-Learn (t-SNE)
- **Quantization:** BitNet-inspired Ternary Logic

## 📂 Project Structure
- `models/`: MoE layer implementation and ternary quantization logic.
- `utils/`: Featurization (Morgan Fingerprints + Physicochemical Descriptors), Data Loaders, and Evaluation scripts.
- `data/`: B3DB classification dataset.
- `test_brain.py`: Inference script for real-world drug testing (e.g., Caffeine vs. Dopamine).

## 📊 Visualizing the Latent Space
The model effectively clusters molecules based on their "Chemical Logic." Below is a t-SNE projection of the 2055-dimensional feature space as interpreted by the 16 Experts:

![Performance Dashboard](visuals/performance_dashboard.png)

## 🧪 Quick Start
1. Install dependencies: `pip install torch rdkit pandas matplotlib scikit-learn`
2. Run inference: `python test_brain.py`
3. Generate metrics: `python -m utils.evaluate`
