# QuantumNet: Diabetic Retinopathy Detection & Quantum Scoring

A hybrid classical-quantum deep learning web application for **Diabetic Retinopathy (DR)** detection and scoring using **ResNet-50**, **Principal Component Analysis (PCA)**, and a **4-Qubit Variational Quantum Circuit (VQC)** powered by **Google Cirq**.

---

## 🌟 Key Features
- **Binary Retinopathy Detection:** Accurately classifies retinal fundus images into **Diabetic Retinopathy (DR)** or **No Diabetic Retinopathy (Non-DR)**.
- **Continuous Quantum Scoring:** Produces a calibrated DR probability score ($0.0\%$ to $100.0\%$), Non-DR confidence score, and clinical risk tiering (Normal, Borderline, Moderate, Severe).
- **Quantum Circuit Simulation:** Inspect the 4-qubit Cirq circuit, rotation angles $\theta$, data encoding $R_y(x_k \cdot \pi/2)$, and Pauli-Z expectation values $\langle Z_q \rangle$.
- **Instant Demo Mode:** Includes realistic synthetic fundus samples (Normal and DR) for zero-setup 1-click testing.
- **Custom Model Checkpoints:** Drag-and-drop or copy your Google Colab trained checkpoints (`best.pkl`, `pca_2048_to_4.joblib`, `best_finetuned_resnet.weights.h5`).
- **Research Benchmark Display:** Compares against the research paper baseline (**96.00%** vs 94.11% accuracy, confusion matrix with TN=262, FP=9, FN=13, TP=266).

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Sample Retinal Images (if not already created)
```bash
python sample_generator.py
```

### 3. Launch the Streamlit Interface
```bash
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

---

## 📁 Project Structure
```text
quantumnet_retinopathy_app/
├── app.py                   # Streamlit web application
├── pipeline.py              # End-to-end inference coordinator
├── quantum_model.py         # 4-Qubit Cirq VQC architecture & Pauli-Z simulation
├── sample_generator.py      # Retinal image generator for immediate demo
├── requirements.txt         # Dependencies
├── README.md                # Documentation & instructions
├── models/                  # Store your trained Colab checkpoints here
│   ├── best.pkl             # Trained VQC weights (theta, readout_w, readout_b)
│   ├── pca_2048_to_4.joblib # Trained PCA model
│   ├── pca_min.npy          # Min values for [-1, 1] normalization
│   ├── pca_max.npy          # Max values for [-1, 1] normalization
│   └── best_finetuned_resnet.weights.h5 (or resnet50_best.keras)
└── samples/
    ├── sample_normal_retina.png
    └── sample_dr_retina.png
```

---

## 🔬 Architecture Overview
1. **Input:** Retinal Fundus image resized to $224 \times 224 \times 3$.
2. **Backbone:** ResNet-50 feature extractor up to `global_average_pooling` $\rightarrow$ 2048-dimensional representation.
3. **Dimensionality Reduction:** PCA maps 2048 features $\rightarrow$ 4 principal components.
4. **Quantum Normalization:** Min-Max scaling to $[-1, 1]$ mapped to rotation angles $x_k \cdot \pi/2$.
5. **Cirq 4-Qubit VQC:**
   - Hadamard gates on all 4 qubits.
   - $R_y(x_k \cdot \pi/2)$ state encoding.
   - 6 Variational Layers with trainable $R_y(\theta)$ and CNOT entanglement chain (0-1, 1-2, 2-3).
   - Pauli-Z measurements $\langle Z_0 \rangle, \langle Z_1 \rangle, \langle Z_2 \rangle, \langle Z_3 \rangle$.
6. **Classical Readout:** $\text{logit} = \sum w_i \langle Z_i \rangle + b \longrightarrow P(\text{DR}) = \sigma(\text{logit})$.
