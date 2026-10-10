"""Problem Q2: quantum-kernel classifier on a biomedical dataset (Wisconsin breast cancer)."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from . import DATA_DIR


@dataclass
class Split:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    explained_variance: float
    seed: int


def export_raw_data():
    df = load_breast_cancer(as_frame=True).frame
    df.to_csv(DATA_DIR / "raw" / "breast_cancer_wisconsin.csv", index=False)
    return df


def make_split(n_features=5, n_train=40, n_test=30, seed=0, scale=np.pi) -> Split:
    """Stratified subsample -> StandardScaler -> PCA -> MinMax to [0, scale] (all fitted on train only).
    `scale` is the kernel bandwidth: smaller -> smoother, less concentrated kernel."""
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=n_train, test_size=n_test,
                                              stratify=y, random_state=seed)
    sc = StandardScaler().fit(X_tr)
    pca = PCA(n_components=n_features, random_state=seed).fit(sc.transform(X_tr))
    Z_tr, Z_te = pca.transform(sc.transform(X_tr)), pca.transform(sc.transform(X_te))
    mm = MinMaxScaler((0, scale)).fit(Z_tr)
    Z_tr, Z_te = mm.transform(Z_tr), np.clip(mm.transform(Z_te), 0, scale)
    return Split(Z_tr, Z_te, y_tr, y_te, float(pca.explained_variance_ratio_.sum()), seed)


def save_split(split: Split, tag="seed0"):
    cols = [f"pc{i}" for i in range(split.X_train.shape[1])]
    for name, X, y in (("train", split.X_train, split.y_train), ("test", split.X_test, split.y_test)):
        df = pd.DataFrame(X, columns=cols)
        df["label"] = y
        df.to_csv(DATA_DIR / "processed" / f"{tag}_{name}.csv", index=False)


def entangling_pairs(n: int, entanglement: str):
    if entanglement == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if entanglement == "full":
        return list(itertools.combinations(range(n), 2))
    raise ValueError(entanglement)


def feature_map(n: int, params, reps: int = 1, entanglement: str = "full") -> QuantumCircuit:
    """H-P(2x_i) layer + ZZ(2(pi-x_i)(pi-x_j)) interactions, repeated `reps` times."""
    qc = QuantumCircuit(n)
    for _ in range(reps):
        for i in range(n):
            qc.h(i)
            qc.p(2 * params[i], i)
        for i, j in entangling_pairs(n, entanglement):
            qc.cx(i, j)
            qc.p(2 * (np.pi - params[i]) * (np.pi - params[j]), j)
            qc.cx(i, j)
    return qc


def kernel_circuit(n: int, reps: int = 1, entanglement: str = "full"):
    """Compute-uncompute circuit U(y)^dagger U(x)|0>; K(x,y) = P(0...0). Returns (qc, x_params, y_params)."""
    xp, yp = ParameterVector("x", n), ParameterVector("y", n)
    qc = feature_map(n, xp, reps, entanglement).compose(feature_map(n, yp, reps, entanglement).inverse())
    qc.measure_all()
    return qc, xp, yp
