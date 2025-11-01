# Quantum Ridge Regression (via HHL and QPE)

This project implements a **quantum ridge regression solver** using the **Harrow–Hassidim–Lloyd (HHL)** algorithm with **Quantum Phase Estimation (QPE)**, running fully on a simulator (Qiskit Aer). It reproduces the behavior of ridge regression by solving the normal equations ( (X^T X + \lambda I) \beta = X^T y ) quantum-mechanically.

---

## How to run

You will need following packages for python.
* Qiskit 
* Matplotlib
* NumPy
* Pandas
* qiskit_aer

Change feature_cols=['std_ion','nvalence_avg'], y_col='formation_energy' to match your own csv file, or to feature_cols=['a','b'], y_col='y' to use generated dataset. If you are using your own csv file, put it under same folder and open the folder, if not the program will generate a sample dataset and you can adjust parameter of the generation process near the end of the code.  

Since this runs on simulator, we read the statevector directly. If you want to run on real quantum computer, you will need to implement a full tomography at the end to estimate the statevector postselection.

## Core Idea

The HHL algorithm uses a quantum circuit to prepare a state proportional to the solution ( |x\rangle ) of a linear system ( A|x\rangle = |b\rangle ). Here:

* ( A = X^T X + \lambda I )
* ( b = X^T y )

It then infers regression coefficients from the post-selected ancilla state representing ( A^{-1}b ).

---

## Main Components

### 1. **Feature Standardization**

Before forming ( A, b ), all features are standardized to zero mean and unit variance. The intercept term is included, and the coefficients are later unstandardized to original data units.

### 2. **Adaptive Quantum Parameters**

The algorithm adaptively chooses:

* **t** (evolution time) ≈ ( 2\pi/\lambda_{max} )
* **m_bits** (number of digit for binary fraction for phase estimation qubits)
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




## References

* Harrow, A. W., Hassidim, A., & Lloyd, S. (2009). *Quantum algorithm for linear systems of equations*. Nature, 460(7252), 56–60.
* Schuld, M., Sinayskiy, I., & Petruccione, F. (2016). *Prediction by linear regression on a quantum computer*. Phys. Rev. A 94, 022342.


