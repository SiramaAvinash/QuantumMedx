"""Ideal (statevector) and noisy (Aer + noise model) fidelity-kernel evaluation."""
from __future__ import annotations

import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from .problem import feature_map, kernel_circuit
from .processors import Processor
from .routing import transpile_for
from qiskit.circuit import ParameterVector


def ideal_states(X, n, reps, entanglement):
    p = ParameterVector("x", n)
    qc = feature_map(n, p, reps, entanglement)
    return np.array([Statevector(qc.assign_parameters(dict(zip(p, x)))).data for x in X])


def ideal_kernel(A, B, n, reps=1, entanglement="full") -> np.ndarray:
    SA, SB = ideal_states(A, n, reps, entanglement), ideal_states(B, n, reps, entanglement)
    return np.abs(SA.conj() @ SB.T) ** 2


class NoisyKernel:
    """Transpile the parametrised compute-uncompute circuit ONCE per processor, then bind & run pairs."""

    def __init__(self, proc: Processor, n: int, reps=1, entanglement="full", shots=1024,
                 optimization_level=1, transpile_seed=7, p_2q=None):
        self.proc, self.n, self.shots = proc, n, shots
        self.logical, self.xp, self.yp = kernel_circuit(n, reps, entanglement)
        self.compiled = transpile_for(self.logical, proc, optimization_level, transpile_seed)
        self.sim = AerSimulator(noise_model=proc.noise_model(p_2q=p_2q))
        self.zero = "0" * n
        self.circuits_run = 0
        self.runtime = 0.0

    def _p0(self, pairs, A, B, seed):
        out = np.empty(len(pairs))
        for s in range(0, len(pairs), 400):
            chunk = pairs[s:s + 400]
            circs = [self.compiled.assign_parameters({**dict(zip(self.xp, A[i])), **dict(zip(self.yp, B[j]))})
                     for i, j in chunk]
            res = self.sim.run(circs, shots=self.shots, seed_simulator=seed + s).result()
            for k in range(len(chunk)):
                out[s + k] = res.get_counts(k).get(self.zero, 0) / self.shots
        self.circuits_run += len(pairs)
        return out

    def train_kernel(self, X, seed=0):
        t0 = time.time()
        m = len(X)
        pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
        v = self._p0(pairs, X, X, seed)
        K = np.eye(m)
        for (i, j), val in zip(pairs, v):
            K[i, j] = K[j, i] = val
        self.runtime += time.time() - t0
        return K

    def cross_kernel(self, Xte, Xtr, seed=0):
        t0 = time.time()
        pairs = [(i, j) for i in range(len(Xte)) for j in range(len(Xtr))]
        K = self._p0(pairs, Xte, Xtr, seed + 10_000).reshape(len(Xte), len(Xtr))
        self.runtime += time.time() - t0
        return K

    def random_pairs_p0(self, X, n_pairs=60, seed=0):
        rng = np.random.default_rng(seed)
        idx = [tuple(rng.choice(len(X), 2, replace=False)) for _ in range(n_pairs)]
        return idx, self._p0(idx, X, X, seed + 777)
