"""
Project: Spectral Transformations via Manual Block Encoding, Manual QSVT,
and QPE Spectroscopy

This script gives a clean implementation of the final project.

Part I     Construct a simple 2-local Hamiltonian H.
Part II    Verify polynomial functional calculus using the spectral theorem.
Part III   Manually build a block encoding of H.
Part IV    Manually assemble QSVT from the block encoding and phase projectors.
Part V     Run additional polynomial transformations through the same QSVT pipeline.
Part VI    Perform manual QPE spectroscopy on the same Hamiltonian.
Part VII   Run QPE from a non-eigenstate.


The only major PennyLane routine used in the actual QSVT implementation is

    qml.poly_to_angles(...)

which performs the nontrivial classical preprocessing step of computing QSP/QSVT
phase angles from a target polynomial.

The block encoding, projector-controlled phase matrices, QSVT matrix product,
spectral verification, and QPE simulation are implemented manually using NumPy.

PennyLane's qml.QSVT is used only as an optional sanity check to verify that the
manual QSVT matrix assembly agrees with PennyLane's template.
"""

import numpy as np
import pennylane as qml
from scipy.linalg import expm

np.set_printoptions(precision=6, suppress=True)


# =============================================================================
# Printing helpers
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def subsection(title: str) -> None:
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


# =============================================================================
# Basic linear algebra helpers
# =============================================================================

def tensor(*matrices: np.ndarray) -> np.ndarray:
    """Tensor product of several matrices."""
    result = np.array([[1]], dtype=complex)
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


def hermitian_psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    """
    Positive square root of a Hermitian positive semidefinite matrix.

    Small negative eigenvalues caused by floating point roundoff are clipped
    to zero.
    """
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T


def polynomial_on_scalar(x: complex, coeffs: np.ndarray) -> complex:
    """
    Evaluate p(x) = c_0 + c_1 x + ... + c_d x^d.
    Coefficients are listed in increasing order.
    """
    return sum(coeff * x**power for power, coeff in enumerate(coeffs))


def polynomial_on_matrix(matrix: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    Evaluate p(A) = c_0 I + c_1 A + ... + c_d A^d.
    Coefficients are listed in increasing order.
    """
    result = np.zeros_like(matrix, dtype=complex)
    identity = np.eye(matrix.shape[0], dtype=complex)

    for power, coeff in enumerate(coeffs):
        if power == 0:
            result += coeff * identity
        else:
            result += coeff * np.linalg.matrix_power(matrix, power)

    return result


def spectral_polynomial(matrix: np.ndarray, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute p(A) using the spectral theorem.

    If A = V Lambda V*, then p(A) = V p(Lambda) V*.
    """
    eigvals, eigvecs = np.linalg.eigh(matrix)
    transformed_eigvals = np.array(
        [polynomial_on_scalar(lam, coeffs) for lam in eigvals],
        dtype=complex,
    )
    transformed_matrix = eigvecs @ np.diag(transformed_eigvals) @ eigvecs.conj().T
    return transformed_matrix, eigvals, transformed_eigvals


# =============================================================================
# QSVT helpers
# =============================================================================

def manual_block_encoding_for_hermitian_contraction(A: np.ndarray) -> np.ndarray:
    r"""
    Manually construct a block encoding of a Hermitian contraction A.

    If A = A* and ||A|| <= 1, define

        D = sqrt(I - A^2)

    and

        U_A = [[ A,  D],
               [ D, -A]].

    Then U_A is unitary and the top-left block is A.
    """
    if not np.allclose(A, A.conj().T):
        raise ValueError("This block encoding helper assumes A is Hermitian.")

    norm_A = np.linalg.norm(A, 2)
    if norm_A > 1 + 1e-12:
        raise ValueError(f"A must be a contraction. Found ||A|| = {norm_A}.")

    dim = A.shape[0]
    identity = np.eye(dim, dtype=complex)
    D = hermitian_psd_sqrt(identity - A @ A)

    return np.block([[A, D], [D, -A]])


def pcphase_matrix(phi: float, encoded_dim: int, total_dim: int) -> np.ndarray:
    """
    Matrix version of a projector-controlled phase.

    It applies exp(i phi) to the encoded subspace and exp(-i phi) to the
    orthogonal complement.
    """
    diagonal = np.empty(total_dim, dtype=complex)
    diagonal[:encoded_dim] = np.exp(1j * phi)
    diagonal[encoded_dim:] = np.exp(-1j * phi)
    return np.diag(diagonal)


def manual_qsvt_matrix(U_A: np.ndarray, projectors: list[np.ndarray], verbose: bool = False) -> np.ndarray:
    """
    Manually assemble the QSVT matrix by alternating phase projectors with
    U_A and U_A*.

    The operation sequence is

        P_0, U_A, P_1, U_A*, P_2, U_A, P_3, ...

    If operations act on states from left to right in this list, the total
    matrix is accumulated as operation @ total.
    """
    operations: list[tuple[str, np.ndarray]] = []

    for idx, projector in enumerate(projectors[:-1]):
        operations.append((f"projector phase P_{idx}", projector))

        if idx % 2 == 0:
            operations.append(("block encoding U_A", U_A))
        else:
            operations.append(("adjoint block encoding U_A*", U_A.conj().T))

    operations.append((f"projector phase P_{len(projectors) - 1}", projectors[-1]))

    if verbose:
        print("\nManual QSVT operation sequence:")
        for step, (name, _) in enumerate(operations, start=1):
            print(f"  Step {step}: apply {name}")

    total = np.eye(U_A.shape[0], dtype=complex)

    for _, operation in operations:
        total = operation @ total

    return total


def pennylane_qsvt_sanity_check(U_A: np.ndarray, angles: np.ndarray, encoded_dim: int) -> np.ndarray:
    """
    Use PennyLane's qml.QSVT as a sanity check only.

    This is not used as the implementation. The main QSVT implementation in
    this file is manual matrix assembly.
    """
    total_dim = U_A.shape[0]
    num_wires = int(np.log2(total_dim))

    if 2**num_wires != total_dim:
        raise ValueError("The block-encoding dimension must be a power of 2 for this sanity check.")

    wires = list(range(num_wires))
    block_encoding_operator = qml.QubitUnitary(U_A, wires=wires)

    projectors = [
        qml.PCPhase(angle, dim=encoded_dim, wires=wires)
        for angle in angles
    ]

    qsvt_operator = qml.QSVT(block_encoding_operator, projectors)
    return qml.matrix(qsvt_operator, wire_order=wires)


def run_manual_qsvt_experiment(
    A: np.ndarray,
    coeffs: np.ndarray,
    name: str,
    verbose_sequence: bool = False,
    sanity_check: bool = True,
) -> dict:
    """
    Run the full manual QSVT pipeline for one polynomial.

    Returns a dictionary containing the exact polynomial, the QSVT block,
    and error quantities.
    """
    subsection(name)

    encoded_dim = A.shape[0]
    U_A = manual_block_encoding_for_hermitian_contraction(A)
    total_dim = U_A.shape[0]

    exact = polynomial_on_matrix(A, coeffs)
    spectral, eigvals, transformed_eigvals = spectral_polynomial(A, coeffs)

    print("\nEigenvalue transformation:")
    for lam, p_lam in zip(eigvals, transformed_eigvals):
        print(f"  {lam: .6f}  --->  {np.real_if_close(p_lam): .6f}")

    print("\nExact p(A) from matrix powers:")
    print(np.round(exact, 6))

    print("\np(A) from spectral decomposition:")
    print(np.round(spectral, 6))

    spectral_error = np.linalg.norm(exact - spectral)
    print("\nSpectral functional calculus error:")
    print(spectral_error)

    # Remaining major built-in: QSP/QSVT phase-angle synthesis.
    angles = qml.poly_to_angles(coeffs, "QSVT")

    print("\nQSP/QSVT phase angles:")
    print(np.round(angles, 6))

    phase_projectors = [
        pcphase_matrix(angle, encoded_dim=encoded_dim, total_dim=total_dim)
        for angle in angles
    ]

    U_qsvt_manual = manual_qsvt_matrix(U_A, phase_projectors, verbose=verbose_sequence)

    # In this QSVT convention, the target polynomial appears in the real part
    # of the top-left block. The imaginary part is not numerical error.
    qsvt_block = U_qsvt_manual[:encoded_dim, :encoded_dim]
    manual_block = np.real(qsvt_block)
    manual_error = np.linalg.norm(manual_block - exact)

    print("\nReal part of manual QSVT top-left block:")
    print(np.round(manual_block, 6))

    print("\nManual QSVT error:")
    print(manual_error)

    sanity_error = None
    if sanity_check:
        U_qsvt_pl = pennylane_qsvt_sanity_check(U_A, angles, encoded_dim)
        pl_block = U_qsvt_pl[:encoded_dim, :encoded_dim]
        sanity_error = np.linalg.norm(qsvt_block - pl_block)

        print("\nDifference from PennyLane qml.QSVT sanity check:")
        print(sanity_error)

    return {
        "name": name,
        "coeffs": coeffs,
        "exact": exact,
        "spectral": spectral,
        "manual_qsvt_block": manual_block,
        "spectral_error": spectral_error,
        "manual_qsvt_error": manual_error,
        "pennylane_sanity_error": sanity_error,
    }

# =============================================================================
# QPE helpers
# =============================================================================

def inverse_qft_matrix(M: int) -> np.ndarray:
    """Inverse QFT matrix on a counting register of dimension M."""
    y = np.arange(M).reshape((M, 1))
    k = np.arange(M).reshape((1, M))
    return np.exp(-2j * np.pi * y * k / M) / np.sqrt(M)


def qpe_distribution(U: np.ndarray, eigenstate: np.ndarray, num_counting_qubits: int) -> np.ndarray:
    """
    Manual QPE simulation.

    If U|psi> = exp(2πi theta)|psi>, QPE estimates theta.

    The state before inverse QFT is

        (1/sqrt(M)) sum_k |k> U^k |psi>.
    """
    M = 2**num_counting_qubits
    system_dim = U.shape[0]

    state = np.zeros((M, system_dim), dtype=complex)

    current = eigenstate.copy()
    for k in range(M):
        state[k, :] = current / np.sqrt(M)
        current = U @ current

    final_state = inverse_qft_matrix(M) @ state
    return np.sum(np.abs(final_state) ** 2, axis=1)


def theta_to_energy(theta: float, B: float) -> float:
    """
    Convert phase theta back to energy for the scaling

        theta = (E + B)/(2B).

    Hence

        E = 2B theta - B.
    """
    return 2 * B * theta - B


# =============================================================================
# Part I: Construct a 2-local Hamiltonian
# =============================================================================

section("Part I: Construct a simple 2-local Hamiltonian")

I = np.eye(2, dtype=complex)

X = np.array(
    [[0, 1],
     [1, 0]],
    dtype=complex,
)

Y = np.array(
    [[0, -1j],
     [1j, 0]],
    dtype=complex,
)

Z = np.array(
    [[1, 0],
     [0, -1]],
    dtype=complex,
)

# A simple 2-qubit, 2-local Hamiltonian:
#
#     H = 0.3 X⊗X - 0.2 Z⊗I + 0.1 Y⊗Y
#
# Each term acts on at most two qubits.
H = (
    0.3 * tensor(X, X)
    - 0.2 * tensor(Z, I)
    + 0.1 * tensor(Y, Y)
)

print("\nHamiltonian:")
print("  H = 0.3 X⊗X - 0.2 Z⊗I + 0.1 Y⊗Y")

print("\nHamiltonian matrix H:")
print(np.round(H, 6))

print("\nHermitian check ||H - H*||:")
print(np.linalg.norm(H - H.conj().T))

print("\nSpectral norm ||H||:")
print(np.linalg.norm(H, 2))

eigvals_H, eigvecs_H = np.linalg.eigh(H)

print("\nExact eigenvalues of H:")
print(eigvals_H)


# =============================================================================
# Part II: Spectral functional calculus
# =============================================================================

section("Part II: Spectral theorem and polynomial functional calculus")

degree5_poly = np.array([0, 1.0, 0, -0.5, 0, 1 / 3], dtype=float)

H_poly_direct = polynomial_on_matrix(H, degree5_poly)
H_poly_spectral, _, H_transformed_eigvals = spectral_polynomial(H, degree5_poly)

print("\nPolynomial:")
print("  p(x) = x - x^3/2 + x^5/3")

print("\nEigenvalue transformation:")
for lam, p_lam in zip(eigvals_H, H_transformed_eigvals):
    print(f"  {lam: .6f}  --->  {np.real_if_close(p_lam): .6f}")

print("\np(H) from matrix powers:")
print(np.round(H_poly_direct, 6))

print("\np(H) from spectral decomposition:")
print(np.round(H_poly_spectral, 6))

print("\nFunctional calculus error ||p(H)_direct - p(H)_spectral||:")
print(np.linalg.norm(H_poly_direct - H_poly_spectral))


# =============================================================================
# Part III: Manual block encoding of the same Hamiltonian
# =============================================================================

section("Part III: Manual block encoding of H")

U_H = manual_block_encoding_for_hermitian_contraction(H)
dim_H = H.shape[0]
dim_UH = U_H.shape[0]

print("\nBlock encoding formula:")
print("  U_H = [[H, sqrt(I-H^2)], [sqrt(I-H^2), -H]]")

print("\nTop-left block of U_H:")
print(np.round(U_H[:dim_H, :dim_H], 6))

print("\n||U_H* U_H - I||:")
print(np.linalg.norm(U_H.conj().T @ U_H - np.eye(dim_UH)))

print("\n||U_H U_H* - I||:")
print(np.linalg.norm(U_H @ U_H.conj().T - np.eye(dim_UH)))

print("\n||Top-left block - H||:")
print(np.linalg.norm(U_H[:dim_H, :dim_H] - H))


# =============================================================================
# Part IV: Detailed manual QSVT for the degree-5 polynomial
# =============================================================================

section("Part IV: Detailed manual QSVT for p(x)=x-x^3/2+x^5/3")

result_degree5 = run_manual_qsvt_experiment(
    A=H,
    coeffs=degree5_poly,
    name="Detailed walkthrough: degree-5 polynomial transformation of H",
    verbose_sequence=True,
    sanity_check=True,
)


# =============================================================================
# Part V: Additional polynomial transformations
# =============================================================================

section("Part V: Additional polynomial transformations")

print(
    "\nPart IV already gave a detailed walkthrough for the degree-5 polynomial.\n"
    "Here we keep the same Hamiltonian and the same block encoding, but change\n"
    "the target polynomial. This demonstrates the QSVT principle: changing the\n"
    "QSP phase angles changes the spectral transformation."
)

extra_polynomials = {
    "scaled linear p(x)=0.5x": np.array([0, 0.5], dtype=float),
    "safe cubic p(x)=0.5x+0.2x^3": np.array([0, 0.5, 0, 0.2], dtype=float),
}

multi_results = []

for polynomial_name, coeffs in extra_polynomials.items():
    multi_results.append(
        run_manual_qsvt_experiment(
            A=H,
            coeffs=coeffs,
            name=polynomial_name,
            verbose_sequence=False,
            sanity_check=True,
        )
    )

print("\nSummary of additional polynomial transformations:")
print("name                                      | spectral error | manual QSVT error | PL sanity")
print("------------------------------------------|----------------|-------------------|----------")
for result in multi_results:
    print(
        f"{result['name'][:40]:40s} | "
        f"{result['spectral_error']:.3e}   | "
        f"{result['manual_qsvt_error']:.3e}        | "
        f"{result['pennylane_sanity_error']:.3e}"
    )


# =============================================================================
# Part VI: QPE spectroscopy
# =============================================================================

section("Part VI: QPE spectroscopy for the same Hamiltonian")

# QPE estimates phases theta in [0,1). To avoid ambiguity from negative
# energies, shift and rescale H:
#
#     H_scaled = (H + B I)/(2B).
#
# If E is an eigenvalue of H, then theta = (E+B)/(2B).
# QPE estimates theta, and we recover E = 2B theta - B.

B = 1.0
H_scaled = (H + B * np.eye(dim_H)) / (2 * B)
theta_exact_values = (eigvals_H + B) / (2 * B)

U_time = expm(2j * np.pi * H_scaled)

print("\nScaled eigenvalues theta = (E+B)/(2B):")
print(theta_exact_values)

print("\nUnitary check ||U*U - I||:")
print(np.linalg.norm(U_time.conj().T @ U_time - np.eye(dim_H)))

num_counting_qubits = 7
M = 2**num_counting_qubits

print(f"\nNumber of counting qubits: {num_counting_qubits}")
print(f"Phase resolution: 1/{M} = {1/M:.6f}")
print(f"Energy resolution: 2B/{M} = {2 * B / M:.6f}")

qpe_summary = []

for j, E_exact in enumerate(eigvals_H):
    psi = eigvecs_H[:, j]
    probs = qpe_distribution(U_time, psi, num_counting_qubits)

    best_y = int(np.argmax(probs))
    theta_est = best_y / M
    E_est = theta_to_energy(theta_est, B)

    theta_exact = theta_exact_values[j]
    error = abs(E_est - E_exact)

    qpe_summary.append((j, E_exact, theta_exact, best_y, theta_est, E_est, error))

    subsection(f"QPE on eigenstate {j}")
    print(f"Exact energy E       = {E_exact:.6f}")
    print(f"Exact phase theta    = {theta_exact:.6f}")
    print(f"Most likely outcome  = {best_y}")
    print(f"Estimated theta      = {theta_est:.6f}")
    print(f"Estimated energy     = {E_est:.6f}")
    print(f"Energy error         = {error:.6f}")

    print("\nTop QPE outcomes:")
    top_indices = np.argsort(probs)[-5:][::-1]
    for idx in top_indices:
        theta = idx / M
        energy = theta_to_energy(theta, B)
        print(
            f"  y={idx:3d}, theta={theta:.6f}, "
            f"E(theta)={energy:.6f}, probability={probs[idx]:.6f}"
        )

section("QPE spectroscopy summary")

print(
    "state | exact E    | exact theta | y_peak | theta_est | E_est     | error\n"
    "------|------------|-------------|--------|-----------|-----------|----------"
)

for row in qpe_summary:
    j, E_exact, theta_exact, best_y, theta_est, E_est, error = row
    print(
        f"{j:5d} | "
        f"{E_exact:10.6f} | "
        f"{theta_exact:11.6f} | "
        f"{best_y:6d} | "
        f"{theta_est:9.6f} | "
        f"{E_est:9.6f} | "
        f"{error:8.6f}"
    )


# =============================================================================
# Part VII: QPE from a non-eigenstate
# =============================================================================

section("Part VII: QPE from computational basis state |00>")

ket00 = np.array([1, 0, 0, 0], dtype=complex)

overlaps = np.abs(eigvecs_H.conj().T @ ket00) ** 2

print("\nOverlap probabilities of |00> with energy eigenstates:")
for j, prob in enumerate(overlaps):
    print(f"  Eigenstate {j}, E={eigvals_H[j]: .6f}, overlap probability={prob:.6f}")

probs_00 = qpe_distribution(U_time, ket00, num_counting_qubits)

print("\nTop QPE outcomes from |00>:")
top_indices = np.argsort(probs_00)[-8:][::-1]

for idx in top_indices:
    theta = idx / M
    energy = theta_to_energy(theta, B)
    print(
        f"  y={idx:3d}, theta={theta:.6f}, "
        f"E(theta)={energy:.6f}, probability={probs_00[idx]:.6f}"
    )



