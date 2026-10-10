"""Classification, kernel-quality and statistical helpers."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC


def psd_project(K: np.ndarray):
    """Clip negative eigenvalues (shot/hardware noise can make the Gram matrix indefinite)."""
    w, V = np.linalg.eigh((K + K.T) / 2)
    return (V * np.clip(w, 0, None)) @ V.T, float(w.min())


def fit_eval_precomputed(K_tr, K_te, y_tr, y_te, seed=0, project=True):
    """SVM on a precomputed kernel; C picked by 4-fold CV on the training kernel only."""
    min_eig = float(np.linalg.eigvalsh((K_tr + K_tr.T) / 2).min())
    if project:
        K_tr, _ = psd_project(K_tr)
    cv = StratifiedKFold(4, shuffle=True, random_state=seed)
    gs = GridSearchCV(SVC(kernel="precomputed"), {"C": [0.1, 1, 10, 100]}, cv=cv).fit(K_tr, y_tr)
    pred = gs.predict(K_te)
    score = gs.decision_function(K_te)
    return {
        "accuracy": accuracy_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "auc": roc_auc_score(y_te, score),
        "best_C": gs.best_params_["C"],
        "min_eigenvalue_train": min_eig,
    }


def classical_baseline(X_tr, X_te, y_tr, y_te, seed=0, kernel="rbf"):
    """SVM baseline on the SAME reduced features; inputs re-standardised so the quantum bandwidth scale
    does not handicap the classical model."""
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_tr)
    X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
    cv = StratifiedKFold(4, shuffle=True, random_state=seed)
    grid = {"C": [0.1, 1, 10, 100]}
    if kernel == "rbf":
        grid["gamma"] = ["scale", 0.1, 0.5, 1, 2]
    gs = GridSearchCV(SVC(kernel=kernel), grid, cv=cv).fit(X_tr, y_tr)
    pred = gs.predict(X_te)
    return {"accuracy": accuracy_score(y_te, pred), "f1": f1_score(y_te, pred),
            "auc": roc_auc_score(y_te, gs.decision_function(X_te))}


def kernel_error(K_est, K_ideal, off_diag=False):
    d = np.abs(K_est - K_ideal)
    if off_diag:
        mask = ~np.eye(len(d), dtype=bool)
        d = d[mask]
    return float(d.mean())


def target_alignment(K, y):
    """Centered kernel-target alignment (higher = kernel geometry matches the labels)."""
    yy = np.where(y == 1, 1.0, -1.0)
    Y = np.outer(yy, yy)
    H = np.eye(len(K)) - 1 / len(K)
    Kc = H @ K @ H
    return float((Kc * Y).sum() / (np.linalg.norm(Kc) * np.linalg.norm(Y) + 1e-12))


def offdiag_std(K):
    return float(K[~np.eye(len(K), dtype=bool)].std())


def binomial_se(p, n):
    return float(np.sqrt(max(p * (1 - p), 0) / n))


def mean_sd(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) if len(x) > 1 else 0.0)


def cv_score_precomputed(K_tr, y_tr, seed=0):
    """Training-only 4-fold CV accuracy of an SVM on a precomputed kernel (used for bandwidth selection)."""
    cv = StratifiedKFold(4, shuffle=True, random_state=seed)
    gs = GridSearchCV(SVC(kernel="precomputed"), {"C": [0.1, 1, 10, 100]}, cv=cv).fit(K_tr, y_tr)
    return float(gs.best_score_)
