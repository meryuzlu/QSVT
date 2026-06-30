# Manual Quantum Singular Value Transformation (QSVT)

Quantum Singular Value Transformation (QSVT) is one of the most powerful algorithmic frameworks in quantum computing, providing a unified approach to polynomial transformations of matrices and Hamiltonians. Rather than relying on built-in QSVT routines, this project develops the complete QSVT pipeline from first principles.

Starting from a simple Hermitian Hamiltonian, the project constructs a manual block encoding, builds the QSVT operator using projector-controlled phase operators, verifies polynomial spectral transformations against exact classical computations, and concludes with Quantum Phase Estimation (QPE) spectroscopy. Every stage of the implementation is validated numerically to demonstrate the correctness of the construction.

---

# What is Implemented Manually?

With the exception of `qml.poly_to_angles`, which computes the QSP phase angles from the target polynomial, the QSVT pipeline is constructed manually.

In particular, the following components are implemented explicitly:

- Construction of a Hermitian Hamiltonian
- Spectral functional calculus
- Manual block encoding
- Projector-controlled phase operators
- Manual QSVT construction
- Polynomial spectral transformations
- Quantum Phase Estimation (QPE)
- Verification procedures

PennyLane's built-in `qml.QSVT` routine is used solely as an independent verification of the manually constructed implementation.

---

# Project Workflow

The notebook is organized into seven parts.

## Part I — Constructing a Hermitian Hamiltonian

- Construct a simple Hermitian Hamiltonian.
- Compute its eigenvalues and eigenvectors.
- Verify the spectral decomposition.

---

## Part II — Spectral Functional Calculus

- Apply polynomial functions to the Hamiltonian.
- Verify the Spectral Theorem numerically.
- Compare direct matrix evaluation with the spectral decomposition.

---

## Part III — Manual Block Encoding

- Construct a block encoding without using built-in block-encoding routines.
- Verify that the encoded block reproduces the original Hamiltonian.

---

## Part IV — Manual Quantum Singular Value Transformation

- Compute the QSP phase angles.
- Assemble the QSVT operator manually from projector-controlled phase operators.
- Compare the manual implementation with exact classical polynomial transformations.
- Verify the implementation against PennyLane's built-in `qml.QSVT`.

---

## Part V — Additional Polynomial Transformations

Apply several different polynomial transformations to demonstrate that the manual QSVT construction correctly implements a variety of spectral transformations.

---

## Part VI — Quantum Phase Estimation Spectroscopy

- Perform Quantum Phase Estimation.
- Recover the Hamiltonian spectrum.
- Compare the estimated energies with the exact eigenvalues.
- Explain the small estimation errors resulting from the finite precision of the counting register.

---

## Part VII — Quantum Phase Estimation from a Non-Eigenstate

Apply Quantum Phase Estimation to a superposition state and observe how the resulting probability distribution reflects the spectral decomposition of the input state.

---

# Numerical Results

The implementation is validated throughout the notebook using independent numerical tests.

The manual QSVT operator reproduces the exact polynomial transformation of the Hamiltonian up to floating-point precision. Independent verification using PennyLane's built-in `qml.QSVT` routine produces agreement at machine precision, confirming the correctness of the manual implementation.

Quantum Phase Estimation successfully reconstructs the Hamiltonian spectrum. The remaining energy estimation errors are consistent with the finite precision of the counting register and are expected from the theoretical behavior of QPE.

---

# Repository Contents

| File | Description |
|------|-------------|
| `Manual_QSVT.ipynb` | Fully documented notebook containing explanations, derivations, numerical experiments, and visualizations. |
| `qsvt_final.py` | Python implementation of the complete project. |
| `README.md` | Project documentation. |

---

# Example Usage

Clone the repository

```bash
git clone https://github.com/meryuzlu/QSVT.git
```

Open `Manual_QSVT.ipynb` in Jupyter Notebook or Google Colab and execute the cells sequentially.

Alternatively, execute

```bash
python qsvt_final.py
```

to run the complete implementation as a Python script.

Running the project will

- construct a Hermitian Hamiltonian,
- verify the Spectral Theorem numerically,
- build a manual block encoding,
- construct the QSVT operator manually,
- verify the implementation against exact classical computations,
- compare the results with PennyLane's built-in `qml.QSVT`,
- perform Quantum Phase Estimation on eigenstates,
- perform Quantum Phase Estimation on a non-eigenstate, and
- display the numerical results and error analysis.

---

# References

1. Gilyén, A., Su, Y., Low, G. H., & Wiebe, N. *Quantum Singular Value Transformation and Beyond: Exponential Improvements for Quantum Matrix Arithmetics.*

2. PennyLane Documentation  
   https://docs.pennylane.ai/

3. PennyLane QSVT Documentation  
   https://docs.pennylane.ai/en/stable/code/api/pennylane.QSVT.html

4. IBM Quantum Learning  
   https://quantum.cloud.ibm.com/learning
