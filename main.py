"""QuantumMedx - end-to-end Phase-1 pipeline (QFF 2026, PS Q2: Quantum Kernel Classifier).

    python main.py            # full benchmark (~10 min on a laptop CPU)
    python main.py --quick    # small smoke run (~1 min)
"""
import argparse
import json
import platform
import time

import numpy as np
import pandas as pd
import qiskit
import qiskit_aer
import sklearn

from src import TABLE_DIR
from src.experiments import (architecture_table, noise_sweep, optimization_level_study, run_benchmark,
                             select_bandwidth, summarize, transpile_seed_study)
from src.problem import export_raw_data, kernel_circuit, make_split, save_split
from src.processors import load_all
from src.routing import transpile_for
from src import visualization as viz
from src.problem import feature_map
from qiskit.circuit import ParameterVector

CONFIG = dict(n_qubits=5, reps=1, n_train=40, n_test=30, shots=1024, opt_level=1, transpile_seed=7,
              seeds=[0, 1, 2], entanglements=["full", "linear"])


def main(quick=False):
    t0 = time.time()
    cfg = dict(CONFIG)
    if quick:
        cfg.update(n_train=16, n_test=10, seeds=[0], shots=512)
    A, B, C = load_all()
    n = cfg["n_qubits"]
    export_raw_data()
    # --- bandwidth selection (train-only CV, ideal kernel; no test data, no noise) ----
    bw_tab, scale = select_bandwidth(n=n, reps=cfg["reps"], entanglements=cfg["entanglements"],
                                     seeds=cfg["seeds"], n_train=cfg["n_train"])
    bw_tab.to_csv(TABLE_DIR / "bandwidth_selection.csv", index=False)
    print(f"selected feature scale (bandwidth) = {scale}", flush=True)
    cfg["scale"] = scale
    save_split(make_split(n, cfg["n_train"], cfg["n_test"], 0, scale), "seed0")

    # --- architecture (no execution) -----------------------------------------
    arch = architecture_table([C, A, B], n, cfg["reps"], cfg["entanglements"], cfg["opt_level"], cfg["transpile_seed"])
    arch.to_csv(TABLE_DIR / "architecture_metrics.csv", index=False)
    print(arch[["processor", "entanglement", "depth", "two_qubit_gates", "swap_count",
                "extra_2q_from_routing"]].to_string(index=False), flush=True)
    optimization_level_study([A, B], n, cfg["reps"]).to_csv(TABLE_DIR / "optimization_level_study.csv", index=False)

    # --- controlled benchmark ------------------------------------------------
    df, kernels = run_benchmark([C, A, B], cfg["entanglements"], cfg["seeds"], n, cfg["reps"], cfg["n_train"],
                                cfg["n_test"], cfg["shots"], cfg["opt_level"], cfg["transpile_seed"], scale)
    df.to_csv(TABLE_DIR / "benchmark_raw.csv", index=False)
    summ = summarize(df)
    summ.to_csv(TABLE_DIR / "final_comparison.csv", index=False)
    for P in (A, B):
        df[df.backend == P.name].to_csv(TABLE_DIR / f"{P.name}_metrics.csv", index=False)
    # tidy A-vs-B table
    rows = []
    for ent in cfg["entanglements"]:
        a = summ[(summ.entanglement == ent) & (summ.backend == "processor_A")].iloc[0]
        b = summ[(summ.entanglement == ent) & (summ.backend == "processor_B")].iloc[0]
        for m in ["accuracy", "auc", "kernel_mae_test", "alignment_train", "offdiag_std", "depth",
                  "two_qubit_gates", "swap_count", "extra_2q_from_routing", "physical_qubits_used", "runtime_s"]:
            va = a.get(f"{m}_mean", a.get(m)); vb = b.get(f"{m}_mean", b.get(m))
            rows.append({"entanglement": ent, "metric": m, "processor_A": va, "processor_B": vb,
                         "difference_B_minus_A": round(float(vb) - float(va), 4)})
    pd.DataFrame(rows).to_csv(TABLE_DIR / "AB_comparison.csv", index=False)

    # --- extra studies -------------------------------------------------------
    sw = noise_sweep([C, A, B], n=n, reps=cfg["reps"], shots=cfg["shots"],
                     n_train=12 if quick else 24, n_test=8 if quick else 16, scale=scale,
                     p2_grid=(0.0, 0.005, 0.02) if quick else (0.0, 0.001, 0.0025, 0.005, 0.01, 0.02))
    sw.to_csv(TABLE_DIR / "noise_sweep.csv", index=False)
    st = transpile_seed_study([A, B], seeds=range(4 if quick else 10), n=n, reps=cfg["reps"],
                              n_pairs=20 if quick else 60, shots=cfg["shots"], scale=scale)
    st.to_csv(TABLE_DIR / "transpile_seed_study.csv", index=False)

    # --- figures -------------------------------------------------------------
    viz.plot_processors([A, B, C])
    viz.plot_graph(A); viz.plot_graph(B)
    logical, _, _ = kernel_circuit(n, cfg["reps"], "full")
    viz.plot_logical_vs_transpiled(logical, {"Processor A": transpile_for(logical, A, 1, 7),
                                              "Processor B": transpile_for(logical, B, 1, 7)})
    xp = ParameterVector("x", n)
    fm = feature_map(n, xp, cfg["reps"], "full").assign_parameters(dict(zip(xp, np.linspace(0.4, 2.8, n))))
    viz.draw_feature_maps(fm, {"A": transpile_for(fm, A, 1, 7), "B": transpile_for(fm, B, 1, 7)})
    viz.plot_ab_performance(df)
    viz.plot_architecture_tradeoffs(arch)
    viz.plot_kernels(kernels); viz.plot_concentration(kernels)
    viz.plot_noise_sweep(sw); viz.plot_gates_vs_error(st)

    env = {"python": platform.python_version(), "qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__,
           "scikit-learn": sklearn.__version__, "numpy": np.__version__, "config": cfg,
           "processors_official": {P.name: P.official_definition for P in (A, B)},
           "runtime_s": round(time.time() - t0, 1)}
    (TABLE_DIR / "run_environment.json").write_text(json.dumps(env, indent=2))
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return df, summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
