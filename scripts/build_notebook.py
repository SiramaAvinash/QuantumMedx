"""Generates main.ipynb (run: python scripts/build_notebook.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: nb.cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# QuantumMedx — Quantum Kernel Classifier for Biomedical Data
**Qiskit Fall Fest 2026 · Phase 1 · Problem Statement Q2 (Quantum Machine Learning — Quantum Kernel Classifier)**

*Choose one PS → build the solution → reproduce on Processor A and B → analyse the architecture effect → reproducible repo.*

| Item | Value |
|---|---|
| PS code / name | **Q2 — Quantum Kernel Classifier** |
| Track | Quantum Machine Learning |
| Input / instance | Wisconsin breast-cancer diagnostic data (569 samples, 30 features), stratified subsample, PCA → 5 features → 5 qubits |
| Core algorithm | Fidelity quantum kernel (ZZ-style feature map, compute–uncompute) + SVM with precomputed kernel |
| Primary metric | Test accuracy (plus ROC-AUC, kernel error vs. ideal, target alignment) |
| Baseline | Ideal (noise-free) quantum kernel and classical RBF / linear SVM on the same 5 features |
| Architecture metrics | depth, two-qubit gates, **measured** SWAPs, extra CX from routing, physical qubits, runtime |

> **Processor definitions.** `processors/processor_A.json` (5-qubit line) and `processor_B.json` (7-qubit heavy-hex-inspired)
> follow the Phase-1 brief. If the official Challenge Kit supplies different files, drop them in `processors/` (same schema) and re-run — nothing else changes.
""")

code("""
import json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
warnings.filterwarnings("ignore")

from src import TABLE_DIR, FIG_DIR
from src.processors import load_all
from src.problem import make_split, kernel_circuit, feature_map, export_raw_data
from src.routing import transpile_for, circuit_metrics, interaction_distance_stats
from src.kernel import ideal_kernel, NoisyKernel
from src.metrics import fit_eval_precomputed
from src.experiments import architecture_table, select_bandwidth
import src.visualization as viz

A, B, C = load_all()
RECOMPUTE = False   # True -> re-run the full benchmark (~7 min) via main.py; False -> load the saved tables
SEED = 0
N_QUBITS = 5
""")

md("## 1. Data\nThe dataset ships with scikit-learn (no download). Preprocessing is fitted on the **training split only**: standardise → PCA(5) → MinMax to `[0, scale]`, where `scale` is the kernel bandwidth.")
code("""
df_raw = export_raw_data()
print(df_raw.shape, "| benign/malignant:", df_raw.target.value_counts().to_dict())
bw_tab, scale = select_bandwidth(n=N_QUBITS)
display(bw_tab.groupby("scale")[["cv_acc","offdiag_mean","offdiag_std"]].mean().round(3))
print("selected bandwidth (train-only CV on ideal kernel):", scale)
split = make_split(N_QUBITS, 40, 30, SEED, scale)
print("PCA explained variance:", round(split.explained_variance, 3))
""")

md("## 2. Processors A and B as graphs")
code("""
viz.plot_processors([A, B, C])
display(Image(str(FIG_DIR/"processors_overview.png")))
display(pd.DataFrame({P.name: P.graph_stats() for P in (A, B, C)}).T)
""")

md("## 3. Logical circuit: feature map and kernel circuit\nThe kernel entry is `K(x,y) = |⟨0|U(y)†U(x)|0⟩|²`, estimated as the probability of measuring all zeros. A *full*-entanglement ZZ map has 10 CX–P–CX blocks, a *linear* one only 4.")
code("""
xp = __import__("qiskit.circuit", fromlist=["ParameterVector"]).ParameterVector("x", N_QUBITS)
fm = feature_map(N_QUBITS, xp, 1, "full").assign_parameters(dict(zip(xp, np.linspace(0.1, 0.9, N_QUBITS))))
fm.draw("mpl", fold=-1, scale=0.6)
""")

md("## 4. Transpile onto Processor A and B and measure the overhead\nSWAPs are **measured** by transpiling again with `swap` allowed in the basis set (same seed); `extra_2q_from_routing` compares against an all-to-all compile with identical settings.")
code("""
arch = architecture_table([C, A, B], N_QUBITS)
cols = ["processor","entanglement","depth","two_qubit_gates","two_qubit_gates_ref","swap_count",
        "extra_2q_from_routing","physical_qubits_used","non_adjacent_pairs","diameter","avg_shortest_path","max_degree"]
display(arch[cols])
""")
code("""
logical, _, _ = kernel_circuit(N_QUBITS, 1, "full")
for P in (A, B):
    comp = transpile_for(logical, P, 1, 7)
    print(P.name, "->", dict(comp.count_ops()), "depth", comp.depth())
display(Image(str(FIG_DIR/"architecture_tradeoffs.png")))
""")

md("## 5. Live mini-benchmark (small, ~10 s): same data, same noise, A vs B\nThe full controlled benchmark below uses 3 seeds, 40 train / 30 test points and 1024 shots per kernel entry.")
code("""
sp = make_split(N_QUBITS, 16, 10, SEED, scale)
Ki_tr = ideal_kernel(sp.X_train, sp.X_train, N_QUBITS); Ki_te = ideal_kernel(sp.X_test, sp.X_train, N_QUBITS)
res = {"ideal": fit_eval_precomputed(Ki_tr, Ki_te, sp.y_train, sp.y_test, SEED, project=False)["accuracy"]}
for P in (A, B):
    nk = NoisyKernel(P, N_QUBITS, 1, "full", 1024)
    Ktr, Kte = nk.train_kernel(sp.X_train, SEED), nk.cross_kernel(sp.X_test, sp.X_train, SEED)
    res[P.name] = fit_eval_precomputed(Ktr, Kte, sp.y_train, sp.y_test, SEED)["accuracy"]
    print(P.name, "mean |dK| =", round(float(np.abs(Kte-Ki_te).mean()),4), "| circuits:", nk.circuits_run)
print(res)
""")

md("## 6. Full controlled A/B benchmark")
code("""
if RECOMPUTE:
    import main; main.main()
summary = pd.read_csv(TABLE_DIR/"final_comparison.csv")
raw = pd.read_csv(TABLE_DIR/"benchmark_raw.csv")
cols = ["entanglement","backend","accuracy_mean","accuracy_std","auc_mean","kernel_mae_test_mean","alignment_train_mean",
        "offdiag_std_mean","depth","two_qubit_gates","swap_count","extra_2q_from_routing","physical_qubits_used","runtime_s_mean"]
display(summary[cols])
""")
code("""
display(Image(str(FIG_DIR/"AB_performance.png")))
display(pd.read_csv(TABLE_DIR/"AB_comparison.csv"))
""")

md("## 7. Why does the kernel degrade? Noise vs. compilation vs. connectivity")
code("""
for f in ["kernel_matrices.png", "kernel_concentration.png", "noise_sweep.png", "gates_vs_kernel_error.png", "logical_vs_transpiled.png"]:
    display(Image(str(FIG_DIR/f)))
display(pd.read_csv(TABLE_DIR/"optimization_level_study.csv"))
""")

md("""
## 8. Interpretation
The written analysis (with the measured numbers, limitations and the connectivity-vs-noise discussion) is in **`report.md`**.
All claims there are backed by the tables in `results/tables/` and the figures in `results/figures/`.

## 9. Reproducibility
`python main.py` regenerates every table and figure (fixed seeds: data split = shot seed = `0,1,2`, transpiler seed `7`).
Library versions used are stored in `results/tables/run_environment.json`.
""")

nbf.write(nb, "main.ipynb")
print("main.ipynb written")
