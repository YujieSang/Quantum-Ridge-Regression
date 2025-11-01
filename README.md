# Adaptive HHL (QPE-based) Linear Regression — Simulator Version

This project implements a **quantum linear regression solver** using the **Harrow–Hassidim–Lloyd (HHL)** algorithm with **Quantum Phase Estimation (QPE)**, running fully on a simulator (Qiskit Aer). It reproduces the behavior of ridge regression by solving the normal equations ( (X^T X + \lambda I) \beta = X^T y ) quantum-mechanically.

---

## 🧮 Core Idea

The HHL algorithm uses a quantum circuit to prepare a state proportional to the solution ( |x\rangle ) of a linear system ( A|x\rangle = |b\rangle ). Here:

* ( A = X^T X + \lambda I )
* ( b = X^T y )

It then infers regression coefficients from the post-selected ancilla state representing ( A^{-1}b ).

---

## ⚙️ Main Components

### 1. **Feature Standardization**

Before forming ( A, b ), all features are standardized to zero mean and unit variance. The intercept term is always included, and the coefficients are later unstandardized to original data units.

### 2. **Adaptive Quantum Parameters**

The algorithm adaptively chooses:

* **t** (evolution time) ≈ ( 2\pi/\lambda_{max} )
* **m_bits** (number of phase estimation qubits)
* **C** (reciprocal scaling constant)

This ensures stability and accurate eigenvalue resolution for a given dataset.

### 3. **Quantum Phase Estimation (QPE)**

Constructs a controlled ( e^{iAt} ) evolution operator, encoding eigenvalues as phases on the QPE register.

### 4. **Reciprocal Lookup Rotation**

Applies controlled-( R_y(\theta) ) rotations with ( \theta = 2\sin^{-1}(C/\lambda) ) to encode ( 1/\lambda ) factors. Phase bins are centered for better numerical stability.

### 5. **Postselection and Extraction**

After uncomputing the QPE register, the ancilla qubit is postselected on (|1\rangle). The system register encodes (|x\rangle), which is extracted from the simulated statevector.

### 6. **Visualization**

Plots both classical and quantum regression results:

* **1D:** Overlayed line fits.
* **2D:** 3D planes representing classical and quantum fits.

---

## 📈 Workflow Overview

1. **Load CSV Data** — specify feature and target columns.
2. **Standardize Data** — compute ( X, y ) in normalized form.
3. **Form Normal Equations** — compute ( A, b ).
4. **Construct HHL Circuit** — build full QPE + reciprocal + inverse-QPE circuit.
5. **Simulate Statevector** — use Qiskit Aer to obtain (|\psi_{final}\rangle).
6. **Postselect Ancilla = 1** — extract and normalize the system state.
7. **Compare with Classical Regression** — compute fidelity and fit quality.

---

## 🔍 Key Outputs

* **`x_classical`** — classical regression coefficients (unstandardized).
* **`x_quantum`** — quantum coefficients inferred from HHL simulation.
* **`fidelity`** — overlap between quantum and classical solution states.
* **`prob_post`** — postselection success probability.

---

## 🧠 Diagnostics & Adaptivity

* Automatically scales ridge regularization (( \lambda_{auto} \approx 10^{-3}\bar{A}_{ii} )).
* Prints eigenvalues, condition number, QPE bin width, and postselection probability.
* Warns when ancilla success probability is low (< 0.05).

---

## 🧰 Example Usage

```bash
python adaptive_hhl_regression.py
```

Generates synthetic 2D data and fits both classical and quantum regressions, displaying fidelity and parameter alignment.

---

## 🧩 Requirements

* Python ≥ 3.10
* Qiskit ≥ 1.0
* Matplotlib (for visualization)
* NumPy / Pandas

Install via:

```bash
pip install qiskit qiskit-aer numpy pandas matplotlib
```

---

## 🧾 Notes

* The simulator version uses **`Statevector.from_instruction`** — exact simulation of the HHL state.
* Error resilience and measurement noise are **not** modeled here (unlike the Runtime version).
* Works with both 1D and 2D regression datasets.

---

## 📚 References

* Harrow, A. W., Hassidim, A., & Lloyd, S. (2009). *Quantum algorithm for linear systems of equations*. Nature, 460(7252), 56–60.
* Schuld, M., Sinayskiy, I., & Petruccione, F. (2016). *Prediction by linear regression on a quantum computer*. Phys. Rev. A 94, 022342.

---

## 🧭 Future Extensions

* Replace Aer simulator with **Qiskit IBM Runtime EstimatorV2** for execution on real hardware.
* Add **Zero-Noise Extrapolation (ZNE)** and **M3 mitigation** for robustness.
* Implement multi-qubit tomography for full postselection validation.

---
