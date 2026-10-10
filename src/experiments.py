"""Experiment drivers: controlled A/B benchmark, noise sweep, transpiler-seed study."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import TABLE_DIR
from .kernel import NoisyKernel, ideal_kernel
from .metrics import (cv_score_precomputed, binomial_se, classical_baseline, fit_eval_precomputed, kernel_error,
                      offdiag_std, target_alignment)
from .problem import kernel_circuit, make_split
from .routing import circuit_metrics, interaction_distance_stats, transpile_for


def run_benchmark(procs, entanglements=("full", "linear"), seeds=(0, 1, 2), n=5, reps=1,
                  n_train=40, n_test=30, shots=1024, opt_level=1, transpile_seed=7, scale=0.5, verbose=True):
    """Same data, circuits, shots and noise assumptions on every processor (controlled comparison)."""
    rows, kernels = [], {}
    for ent in entanglements:
        for seed in seeds:
            sp = make_split(n, n_train, n_test, seed, scale)
            Ktr_i = ideal_kernel(sp.X_train, sp.X_train, n, reps, ent)
            Kte_i = ideal_kernel(sp.X_test, sp.X_train, n, reps, ent)
            base = dict(seed=seed, entanglement=ent, n_train=n_train, n_test=n_test, shots=shots)
            r = fit_eval_precomputed(Ktr_i, Kte_i, sp.y_train, sp.y_test, seed, project=False)
            rows.append({**base, "backend": "ideal_kernel", **r, "kernel_mae_test": 0.0,
                         "kernel_mae_train_offdiag": 0.0, "alignment_train": target_alignment(Ktr_i, sp.y_train),
                         "offdiag_std": offdiag_std(Ktr_i)})
            kernels[(ent, seed, "ideal_kernel")] = Ktr_i
            if ent == entanglements[0]:  # classical baselines don't depend on the quantum map
                for kind in ("rbf", "linear"):
                    rows.append({**base, "backend": f"classical_{kind}",
                                 **classical_baseline(sp.X_train, sp.X_test, sp.y_train, sp.y_test, seed, kind)})
            for P in procs:
                nk = NoisyKernel(P, n, reps, ent, shots, opt_level, transpile_seed)
                Ktr = nk.train_kernel(sp.X_train, seed)
                Kte = nk.cross_kernel(sp.X_test, sp.X_train, seed)
                r = fit_eval_precomputed(Ktr, Kte, sp.y_train, sp.y_test, seed, project=True)
                cm = circuit_metrics(nk.compiled, nk.logical, P, opt_level, transpile_seed)
                rows.append({**base, "backend": P.name, **r,
                             "kernel_mae_test": kernel_error(Kte, Kte_i),
                             "kernel_mae_train_offdiag": kernel_error(Ktr, Ktr_i, off_diag=True),
                             "alignment_train": target_alignment(Ktr, sp.y_train),
                             "offdiag_std": offdiag_std(Ktr), "runtime_s": nk.runtime,
                             "circuits_run": nk.circuits_run,
                             **{k: cm[k] for k in ("depth", "two_qubit_gates", "swap_count",
                                                   "extra_2q_from_routing", "physical_qubits_used")}})
                kernels[(ent, seed, P.name)] = Ktr
                if verbose:
                    print(f"[{ent:6s} seed {seed}] {P.name:20s} acc={r['accuracy']:.3f} "
                          f"kMAE={rows[-1]['kernel_mae_test']:.4f} t={nk.runtime:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    return df, kernels


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- sample SD over seeds (splits + shot seeds)."""
    metric_cols = ["accuracy", "f1", "auc", "kernel_mae_test", "kernel_mae_train_offdiag",
                   "alignment_train", "offdiag_std", "runtime_s"]
    static_cols = ["depth", "two_qubit_gates", "swap_count", "extra_2q_from_routing", "physical_qubits_used"]
    g = df.groupby(["entanglement", "backend"], dropna=False)
    out = g[metric_cols].agg(["mean", "std"]).round(4)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out = out.join(g[static_cols].first())
    out["n_seeds"] = g["seed"].nunique()
    return out.reset_index()


def noise_sweep(procs, p2_grid=(0.0, 0.001, 0.0025, 0.005, 0.01, 0.02), n=5, reps=1, n_train=24,
                n_test=16, shots=1024, seed=0, entanglement="full", scale=0.5, verbose=True):
    sp = make_split(n, n_train, n_test, seed, scale)
    Ktr_i = ideal_kernel(sp.X_train, sp.X_train, n, reps, entanglement)
    Kte_i = ideal_kernel(sp.X_test, sp.X_train, n, reps, entanglement)
    rows = []
    for P in procs:
        for p2 in p2_grid:
            nk = NoisyKernel(P, n, reps, entanglement, shots, p_2q=p2)
            Ktr, Kte = nk.train_kernel(sp.X_train, seed), nk.cross_kernel(sp.X_test, sp.X_train, seed)
            r = fit_eval_precomputed(Ktr, Kte, sp.y_train, sp.y_test, seed)
            rows.append({"backend": P.name, "p_2q": p2, "accuracy": r["accuracy"], "auc": r["auc"],
                         "kernel_mae_test": kernel_error(Kte, Kte_i), "offdiag_std": offdiag_std(Ktr)})
            if verbose:
                print(f"[sweep] {P.name:20s} p2q={p2:<7} acc={r['accuracy']:.3f}", flush=True)
    return pd.DataFrame(rows)


def transpile_seed_study(procs, seeds=range(10), n=5, reps=1, entanglement="full", n_pairs=60, shots=1024,
                         opt_level=1, scale=0.5):
    """Different layout/routing seeds -> different 2Q counts -> different kernel error (compilation <-> quality)."""
    sp = make_split(n, 40, 30, 0, scale)
    Ki = ideal_kernel(sp.X_train, sp.X_train, n, reps, entanglement)
    logical, _, _ = kernel_circuit(n, reps, entanglement)
    rows = []
    for P in procs:
        for s in seeds:
            nk = NoisyKernel(P, n, reps, entanglement, shots, opt_level, transpile_seed=s)
            idx, p0 = nk.random_pairs_p0(sp.X_train, n_pairs, seed=0)
            ref = np.array([Ki[i, j] for i, j in idx])
            cm = circuit_metrics(nk.compiled, logical, P, opt_level, s)
            rows.append({"backend": P.name, "transpile_seed": s, "kernel_mae": float(np.abs(p0 - ref).mean()),
                         "two_qubit_gates": cm["two_qubit_gates"], "depth": cm["depth"],
                         "swap_count": cm["swap_count"]})
    return pd.DataFrame(rows)


def optimization_level_study(procs, n=5, reps=1, entanglements=("full", "linear"), seed=7):
    rows = []
    for ent in entanglements:
        logical, _, _ = kernel_circuit(n, reps, ent)
        for P in procs:
            for lvl in range(4):
                comp = transpile_for(logical, P, lvl, seed)
                m = circuit_metrics(comp, logical, P, lvl, seed)
                rows.append({"entanglement": ent, "backend": P.name, "opt_level": lvl,
                             **{k: m[k] for k in ("depth", "two_qubit_gates", "swap_count",
                                                  "extra_2q_from_routing")}})
    return pd.DataFrame(rows)


def architecture_table(procs, n=5, reps=1, entanglements=("full", "linear"), opt_level=1, seed=7):
    rows = []
    for P in procs:
        for ent in entanglements:
            logical, _, _ = kernel_circuit(n, reps, ent)
            comp = transpile_for(logical, P, opt_level, seed)
            m = circuit_metrics(comp, logical, P, opt_level, seed)
            rows.append({"entanglement": ent, **P.graph_stats(), **m,
                         **interaction_distance_stats(P, n, ent)})
    return pd.DataFrame(rows)


def select_bandwidth(candidates=(0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 3.14159), n=5, reps=1,
                     entanglements=("full", "linear"), seeds=(0, 1, 2), n_train=40):
    """Pick the feature-scale (kernel bandwidth) by TRAIN-ONLY cross-validation of the ideal kernel."""
    rows = []
    for sc in candidates:
        for ent in entanglements:
            for seed in seeds:
                sp = make_split(n, n_train, 10, seed, sc)
                K = ideal_kernel(sp.X_train, sp.X_train, n, reps, ent)
                rows.append({"scale": sc, "entanglement": ent, "seed": seed,
                             "cv_acc": cv_score_precomputed(K, sp.y_train, seed),
                             "offdiag_mean": float(K[np.triu_indices(len(K), 1)].mean()),
                             "offdiag_std": offdiag_std(K)})
    tab = pd.DataFrame(rows)
    best = tab.groupby("scale").cv_acc.mean().sort_values(ascending=False)
    return tab, float(best.index[0])
