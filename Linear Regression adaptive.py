
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Operator, Statevector, DensityMatrix, partial_trace
from qiskit.circuit.library import RYGate
from qiskit_aer import AerSimulator
from mpl_toolkits.mplot3d import Axes3D 



def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    Z = (X - mu) / sigma
    return Z, mu, sigma

def unstandardize_coeffs(beta_std: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """beta_std = [c0, w_std...] with intercept first; return [c, w] in original units."""
    c0 = float(beta_std[0]); w_std = beta_std[1:]
    w = w_std / sigma
    c = c0 - float(np.dot(w_std, mu / sigma))
    return np.concatenate([[c], w])

# Normal equations & padding

def normal_equations(X: np.ndarray, y: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    b = X.T @ y
    return A, b

def next_pow2_ge(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()

def pad_to_pow2(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
    d = A.shape[0]
    P = next_pow2_ge(d)
    if P == d:
        return A, b, d, P
    evals = np.linalg.eigvalsh(A)
    lam_max = float(np.max(evals))
    lam_pad = 1.1 * lam_max
    A_pad = np.zeros((P, P), dtype=float)
    A_pad[:d, :d] = A
    A_pad[d:, d:] = lam_pad * np.eye(P - d)
    b_pad = np.zeros(P, dtype=float)
    b_pad[:d] = b
    return A_pad, b_pad, d, P

# U = exp(i A t)

def unitary_exp_iAt(A: np.ndarray, t: float) -> np.ndarray:
    evals, evecs = np.linalg.eigh(A)
    U = evecs @ np.diag(np.exp(1j * evals * t)) @ evecs.conj().T
    return U

# IQFT + QPE block

def iqft(circ: QuantumCircuit, qreg):
    m = len(qreg)
    for j in range(m // 2):
        circ.swap(qreg[j], qreg[m - 1 - j])
    for j in range(m):
        k = m - 1 - j
        for l in range(k + 1, m):
            circ.cp(-math.pi / (2 ** (l - k)), qreg[l], qreg[k])
        circ.h(qreg[k])

def build_qpe_block(A_pad: np.ndarray, t: float, m_bits: int, eig, sys_regs: List) -> Operator:
    sub = QuantumCircuit(eig, sys_regs, name="QPE(U=e^{iAt})")
    for k in range(m_bits):
        sub.h(eig[k])
    for k in range(m_bits):
        U = unitary_exp_iAt(A_pad, t * (2 ** k))
        CU = Operator(U).to_instruction().control(1)
        sub.append(CU, [eig[k], *sys_regs])
    iqft(sub, eig)
    return sub.to_instruction()

# Reciprocal Ry

def apply_lookup_ry(circ: QuantumCircuit, eig, anc, t: float, m_bits: int, C: float):
    angles = []
    for j in range(2 ** m_bits):
        phi = (j + 0.5) / (2 ** m_bits)  # center-of-bin phase
        lam_est = (2 * math.pi / t) * phi
        theta = 0.0 if lam_est <= 0 else 2.0 * math.asin(min(C / lam_est, 1.0))
        angles.append(theta)
    for j, theta in enumerate(angles):
        if not np.isfinite(theta) or abs(theta) < 1e-12:
            continue
        bits0 = [i for i in range(m_bits) if ((j >> i) & 1) == 0]
        for i in bits0: circ.x(eig[i])
        circ.append(RYGate(theta).control(num_ctrl_qubits=m_bits), [*eig, anc])
        for i in bits0: circ.x(eig[i])

# Adaptive params (t, m_bits, C)

def choose_adaptive_params(A_pad: np.ndarray, safety: float = 0.49,
                           min_bins: int = 4, m_max: int = 10) -> Tuple[float, int, float]:
    evals = np.linalg.eigvalsh(A_pad)
    lam_min = float(np.min(evals))
    lam_max = float(np.max(evals))
    t = safety * (2 * math.pi) / lam_max
    delta_phi = (lam_max - lam_min) * t / (2 * math.pi)
    target = max(delta_phi / float(min_bins), 1e-12)
    m_bits = int(math.ceil(max(1.0, math.log2(1.0 / target))))
    m_bits = min(max(m_bits, 6), m_max)
    C = 0.9 * lam_min
    return t, m_bits, C

# Postselect 

def extract_postselected_system(sv: Statevector, n_sys: int, m_bits: int, d: int) -> Tuple[float, np.ndarray]:
    amps = sv.data.copy()
    anc_pos = n_sys + m_bits
    idx = np.arange(amps.size)
    amps[((idx >> anc_pos) & 1) == 0] = 0.0
    if np.allclose(amps, 0):
        return 0.0, np.full(d, np.nan)
    amps = amps / np.linalg.norm(amps)
    sv_post = Statevector(amps)
    rho = DensityMatrix(sv_post)
    trace_out = list(range(n_sys, n_sys + m_bits)) + [n_sys + m_bits]
    rho_sys = partial_trace(rho, trace_out)
    vals, vecs = np.linalg.eigh(rho_sys.data)
    x_dir = vecs[:, np.argmax(vals)][:d]
    x_dir = x_dir / np.linalg.norm(x_dir)
    p1 = float(sv.probabilities(qargs=[anc_pos])[1])
    return p1, x_dir

# HHL 

def hhl_qpe_full(A: np.ndarray, b: np.ndarray,
                 t: Optional[float] = None, m_bits: Optional[int] = None, C: Optional[float] = None):
    A_pad, b_pad, d, P = pad_to_pow2(A, b)
    n_sys = int(math.log2(P))
    b_state = (b_pad.astype(complex) / np.linalg.norm(b_pad))
    if (t is None) or (m_bits is None) or (C is None):
        t_auto, m_auto, C_auto = choose_adaptive_params(A_pad)
        t = t if t is not None else t_auto
        m_bits = m_bits if m_bits is not None else m_auto
        C = C if C is not None else C_auto
    sys = QuantumRegister(n_sys, 'sys')
    eig = QuantumRegister(m_bits, 'eig')
    anc = QuantumRegister(1, 'anc')
    qc = QuantumCircuit(sys, eig, anc, name='HHL_full')
    qc.initialize(b_state, sys)
    qpe = build_qpe_block(A_pad, t, m_bits, eig, sys)
    qc.append(qpe, [*eig, *sys])
    apply_lookup_ry(qc, eig, anc[0], t, m_bits, C)
    qc.append(qpe.inverse(), [*eig, *sys])
    meta = dict(P=P, d=d, n_sys=n_sys, t=t, m_bits=m_bits, C=C, evals=np.linalg.eigvalsh(A))
    return qc, meta

# Run from CSV 

def run_hhl_from_csv(csv_path: Path, feature_cols: List[str], y_col: str,
                      lam_reg: Optional[object] = 'auto',
                      t: Optional[float] = None, m_bits: Optional[int] = None, C: Optional[float] = None,
                      draw_circuit: bool = True, show_plot: bool = True):
    df = pd.read_csv(csv_path)
    Xraw = df[feature_cols].to_numpy(float)
    y = df[y_col].to_numpy(float)
    Z, muX, sigX = standardize_features(Xraw)
    X = np.column_stack([np.ones(len(Z)), Z])  # intercept first
    names = ['intercept', *feature_cols]

    if isinstance(lam_reg, str) and lam_reg.lower() == 'auto':
        G = X.T @ X
        lam_reg = 1e-3 * (np.trace(G) / G.shape[0])  # ~0.1% of avg diagonal scale

    # Classical in standardized basis
    A, b = normal_equations(X, y, lam=lam_reg)
    x_classical_std = np.linalg.solve(A, b)

    # Quantum in standardized basis
    qc, meta = hhl_qpe_full(A, b, t=t, m_bits=m_bits, C=C)
    if draw_circuit:
        try:
            qc.draw("mpl").savefig("hhl_full_circuit.png", dpi=180, bbox_inches="tight")
            print(f"[saved] circuit → hhl_full_circuit.png")
            plt.close()
        except Exception as e:
            print('Circuit draw skipped:', e)

    sv = Statevector.from_instruction(qc)
    p_post, x_dir_std = extract_postselected_system(sv, meta['n_sys'], meta['m_bits'], meta['d'])
    z = np.real(x_dir_std)
    s = float((z @ x_classical_std) / max(z @ z, 1e-12))
    x_quantum_std = s * z

    # Unstandardize both for reporting/plots
    x_classical = unstandardize_coeffs(x_classical_std, muX, sigX)
    x_quantum   = unstandardize_coeffs(x_quantum_std,   muX, sigX)


    evals = meta['evals']
    lam_min, lam_max = float(np.min(evals)), float(np.max(evals))
    delta_phi = (lam_max - lam_min) * meta['t'] / (2*np.pi)
    bin_width = 1.0 / (2 ** meta['m_bits'])

    print('\n===  HHL Ridge Regression V3 by Yujie Sang===')
    print('features:', names)
    print('λ (ridge) =', lam_reg)
    print('eig(A) =', evals, '| κ(A) =', float(lam_max/lam_min))
    print(f"t = {meta['t']:.6g}, m_bits = {meta['m_bits']}, C = {meta['C']:.6g}")
    print(f"Δφ ≈ {delta_phi:.6f} vs bin = {bin_width:.6f}")
    print('postselect Pr[anc=1] =', p_post)

    print('classical x (orig units) =', x_classical)
    print('quantum   x (orig units) =', x_quantum)

    classical_dir = (x_classical_std / np.linalg.norm(x_classical_std)).astype(complex)
    fidelity = float(np.abs(np.vdot(classical_dir, x_dir_std)) ** 2)
    print('state fidelity in parameter space (standardized) =', fidelity)
    if p_post < 0.05:
        print('[note] Low post-select prob. Consider larger lam_reg or slightly smaller C (≤ λ_min).')

    # Plots
    if show_plot:
        try:
            if len(feature_cols) == 1:
                xvals = Xraw[:, 0]
                yvals = y
                c_c, w_c = float(x_classical[0]), float(x_classical[1])
                c_q, w_q = float(x_quantum[0]),   float(x_quantum[1])
                xs = np.linspace(xvals.min(), xvals.max(), 200)
                plt.scatter(xvals, yvals, s=12, alpha=0.85, label='data')
                plt.plot(xs, c_c + w_c*xs, label='classical fit')
                plt.plot(xs, c_q + w_q*xs, label='quantum fit')
                plt.xlabel(feature_cols[0]); plt.ylabel(y_col)
                plt.title(f'Line fits | Fidelity(std)={fidelity:.4f}  P(anc=1)={p_post:.3f}')
                plt.legend(); plt.show()
            elif len(feature_cols) == 2:
                f0, f1 = Xraw[:,0], Xraw[:,1]
                target = y
                c_c, w0_c, w1_c = x_classical
                c_q, w0_q, w1_q = x_quantum
                fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
                ax.scatter(f0, f1, target, s=12, alpha=0.85, label='data')
                u = np.linspace(f0.min(), f0.max(), 30); v = np.linspace(f1.min(), f1.max(), 30)
                U, V = np.meshgrid(u, v)
                Zc = c_c + w0_c*U + w1_c*V
                Zq = c_q + w0_q*U + w1_q*V
                ax.plot_surface(U, V, Zc, alpha=0.35); ax.plot_surface(U, V, Zq, alpha=0.35)
                ax.set_xlabel(feature_cols[0]); ax.set_ylabel(feature_cols[1]); ax.set_zlabel(y_col)
                ax.set_title(f'Planes | Fidelity(std)={fidelity:.4f}  P(anc=1)={p_post:.3f}')
                plt.show()
        except Exception as e:
            print('Plot skipped:', e)

    # Optional Aer parity check
    try:
        backend = AerSimulator(method='statevector')
        qc_sv = qc.copy(); qc_sv.save_statevector()
        tqc = transpile(qc_sv, backend, optimization_level=0)
        res = backend.run(tqc).result()
        sv2 = res.data(0)['statevector']
        assert np.allclose(sv2, sv.data)
    except Exception as e:
        print('Aer parity check skipped:', e)

    return dict(x_classical=x_classical, x_quantum=x_quantum, fidelity=fidelity,
                prob_post=p_post, names=names, params=meta)

# Configuration for direct run
 
if __name__ == '__main__':
    csv = Path('magneticmoment_Ef_data.csv')
    if not csv.exists():
        rng = np.random.default_rng(7)
        n = 160
        f0 = rng.normal(0, 3.0, n)
        f1 = rng.normal(10, 5.0, n)
        y  = 2.5 + 0.8*f0 - 1.1*f1 + rng.normal(0, 2.0, n)
        pd.DataFrame({'a': f0, 'b': f1, 'y': y}).to_csv(csv, index=False)
        out = run_hhl_from_csv(csv, feature_cols=['a','b'], y_col='y', lam_reg='auto',
                           t=None, m_bits=None, C=None, draw_circuit=False, show_plot=True)
        
    else:
        out = run_hhl_from_csv(csv, feature_cols=['std_ion','nvalence_avg'], y_col='formation_energy', lam_reg='auto',
                           t=None, m_bits=None, C=None, draw_circuit=False, show_plot=True)
    print('\nDone →', out['names'])