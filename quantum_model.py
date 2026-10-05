"""
QuantumNet - 4-Qubit Variational Quantum Circuit (VQC) using Cirq
Implements the exact paper-style quantum architecture from the training pipeline:
- 4 Qubits with initial Hadamard layer
- Data encoding via Ry(x_k * pi / 2)
- 6 Variational layers (Trainable Ry rotations + CNOT chain: CNOT(0,1), CNOT(1,2), CNOT(2,3))
- 4 Pauli-Z expectation measurements
- Classical parameterized readout: logit = dot(w, z) + b -> sigmoid(logit)
"""

import os
import pickle
import numpy as np

try:
    import cirq
except ImportError:
    cirq = None


NUM_QUBITS = 4
NUM_LAYERS = 6


def get_qubits():
    """Returns the list of 4 line qubits."""
    if cirq is None:
        raise ImportError("Cirq is not installed. Please run: pip install cirq")
    return [cirq.LineQubit(i) for i in range(NUM_QUBITS)]


def build_vqc(x, theta):
    """
    Constructs the Cirq quantum circuit.
    
    Args:
        x: 1D array of 4 normalized features in [-1, 1]
        theta: 2D array of shape (NUM_LAYERS, NUM_QUBITS) containing rotation parameters
        
    Returns:
        cirq.Circuit
    """
    if cirq is None:
        raise ImportError("Cirq is not installed.")
        
    qubits = get_qubits()
    circuit = cirq.Circuit()

    # 1. Initial Hadamards to create superposition
    circuit.append(cirq.H.on_each(*qubits))

    # 2. Data encoding: Ry(x * pi/2)
    for q in range(NUM_QUBITS):
        circuit.append(
            cirq.ry(float(x[q] * np.pi / 2))(qubits[q])
        )

    # 3. Variational Layers (adapts dynamically to theta shape)
    num_layers = theta.shape[0]
    for layer in range(num_layers):
        # Trainable Ry rotations
        for q in range(NUM_QUBITS):
            circuit.append(
                cirq.ry(float(theta[layer, q]))(qubits[q])
            )

        # CNOT entanglement chain: (0->1, 1->2, 2->3)
        circuit.append(cirq.CNOT(qubits[0], qubits[1]))
        circuit.append(cirq.CNOT(qubits[1], qubits[2]))
        circuit.append(cirq.CNOT(qubits[2], qubits[3]))

    return circuit


def get_z_expectations(circuit, simulator=None):
    """
    Computes Pauli-Z expectation values for each qubit using state vector simulation.
    
    Args:
        circuit: cirq.Circuit
        simulator: cirq.Simulator (optional)
        
    Returns:
        np.ndarray of shape (4,) with expectation values in [-1, 1]
    """
    if cirq is None:
        raise ImportError("Cirq is not installed.")

    if simulator is None:
        simulator = cirq.Simulator()

    result = simulator.simulate(circuit)
    state_vector = result.final_state_vector

    expectations = np.zeros(NUM_QUBITS, dtype=np.float32)

    for q in range(NUM_QUBITS):
        expectation = 0.0
        for basis_index, amplitude in enumerate(state_vector):
            probability = abs(amplitude) ** 2
            bit = (basis_index >> (NUM_QUBITS - 1 - q)) & 1
            if bit == 0:
                expectation += probability
            else:
                expectation -= probability
        expectations[q] = float(expectation)

    return expectations


def sigmoid(x):
    """Numerically stable sigmoid function."""
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def predict_probability(z_expectations, readout_w, readout_b):
    """
    Classical parameterized linear combination and sigmoid activation.
    logit = dot(w, z) + b
    P(DR) = sigmoid(logit)
    
    Args:
        z_expectations: array of 4 Z expectation values
        readout_w: array of 4 weights
        readout_b: scalar bias
        
    Returns:
        tuple: (dr_probability: float, logit: float)
    """
    logit = float(np.dot(readout_w, z_expectations) + readout_b)
    prob_dr = float(sigmoid(logit))
    return prob_dr, logit


def load_vqc_checkpoint(checkpoint_path):
    """
    Loads trained theta, readout_w, and readout_b from best.pkl.
    
    Returns:
        dict containing 'theta', 'readout_w', 'readout_b', 'epoch', 'best_val_loss'
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
        
    # Map classical_weights and classical_bias if present
    if "classical_weights" in checkpoint and "readout_w" not in checkpoint:
        checkpoint["readout_w"] = np.array(checkpoint["classical_weights"], dtype=np.float32)
    if "classical_bias" in checkpoint and "readout_b" not in checkpoint:
        checkpoint["readout_b"] = float(checkpoint["classical_bias"])

    # Validation of keys
    if "readout_w" not in checkpoint:
        checkpoint["readout_w"] = np.array([-0.5, -0.5, -0.5, -0.5], dtype=np.float32)
    if "readout_b" not in checkpoint:
        checkpoint["readout_b"] = 0.5
            
    return checkpoint
