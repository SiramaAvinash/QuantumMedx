"""All figures for the report (saved to results/figures)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from . import FIG_DIR

COL = {"ideal_kernel": "#2a9d8f", "processor_A": "#e76f51", "processor_B": "#264653",
       "control_all_to_all": "#8d99ae", "classical_rbf": "#f4a261", "classical_linear": "#e9c46a"}
LABEL = {"ideal_kernel": "Ideal kernel", "processor_A": "Processor A", "processor_B": "Processor B",
         "control_all_to_all": "All-to-all control", "classical_rbf": "Classical RBF-SVM",
         "classical_linear": "Classical linear-SVM"}


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_graph(proc, name=None, ax=None):
    g = proc.graph()
    pos = {"processor_A": {i: (i, 0) for i in range(5)},
           "processor_B": {0: (0, 1), 1: (1, 1), 2: (2, 1), 3: (1, 0), 5: (1, -1), 4: (0, -1), 6: (2, -1)}
           }.get(proc.name)
    pos = pos or nx.circular_layout(g)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.2, 3))
    nx.draw(g, pos, ax=ax, with_labels=True, node_size=700, node_color=COL.get(proc.name, "#8d99ae"),
            font_color="white", edge_color="#555", width=2)
    ax.set_title(f"{proc.name}  ({proc.num_qubits} qubits, {g.number_of_edges()} couplers)")
    if own:
        _save(fig, name or f"{proc.name}_graph.png")


def plot_processors(procs):
    fig, axes = plt.subplots(1, len(procs), figsize=(4.4 * len(procs), 3.2))
    for ax, P in zip(np.atleast_1d(axes), procs):
        plot_graph(P, ax=ax)
    _save(fig, "processors_overview.png")


def plot_logical_vs_transpiled(logical, comps: dict):
    """Gate-count composition of the logical vs transpiled kernel circuit."""
    from qiskit import transpile
    fig, ax = plt.subplots(figsize=(8, 4))
    items = {"Logical (generic gates)": transpile(logical, basis_gates=["rz", "sx", "x", "cx"], optimization_level=0)}
    items.update(comps)
    gates = ["cx", "sx", "rz", "x"]
    w = 0.8 / len(items)
    for k, (name, qc) in enumerate(items.items()):
        ops = qc.count_ops()
        ax.bar(np.arange(len(gates)) + k * w, [ops.get(g, 0) for g in gates], w, label=f"{name} (depth {qc.depth()})")
    ax.set_xticks(np.arange(len(gates)) + 0.4 - w / 2)
    ax.set_xticklabels(gates)
    ax.set_ylabel("gate count")
    ax.set_title("Kernel circuit: logical vs transpiled per processor (full entanglement)")
    ax.legend(fontsize=8)
    _save(fig, "logical_vs_transpiled.png")


def draw_feature_maps(logical_fm, comps: dict):
    for name, qc in comps.items():
        qc.draw("mpl", fold=-1, scale=0.5, filename=str(FIG_DIR / f"transpiled_featuremap_{name}.png"))
    logical_fm.draw("mpl", fold=-1, scale=0.6, filename=str(FIG_DIR / "logical_featuremap.png"))
    plt.close("all")


def plot_ab_performance(df):
    ents = list(df.entanglement.unique())
    order = ["classical_linear", "classical_rbf", "ideal_kernel", "control_all_to_all", "processor_A", "processor_B"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    w = 0.8 / len(ents)
    for ax, (metric, title) in zip(axes[:2], [("accuracy", "Test accuracy"), ("auc", "Test ROC-AUC")]):
        for k, ent in enumerate(ents):
            for b in order:
                is_cl = b.startswith("classical")
                if is_cl and k > 0:
                    continue  # classical SVMs do not depend on the quantum feature map -> draw once
                s = df[df.backend == b][metric] if is_cl else df[(df.entanglement == ent) & (df.backend == b)][metric]
                x = order.index(b) + (0.4 - w / 2 if is_cl else k * w)
                ax.bar(x, s.mean(), w * (len(ents) if is_cl else 1), yerr=s.std(ddof=1), color=COL[b],
                       alpha=0.55 if (k == 1 and not is_cl) else 1, capsize=3, edgecolor="k", linewidth=0.5)
        ax.set_xticks(np.arange(len(order)) + 0.4 - w / 2)
        ax.set_xticklabels([LABEL[b].replace(" ", "\n", 1) for b in order], fontsize=7)
        ax.set_ylim(0.4, 1.02)
        ax.set_title(title + " (mean ± SD over seeds)")
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(facecolor="#888", edgecolor="k", label=f"{ents[0]} entanglement (solid)"),
                            Patch(facecolor="#888", alpha=0.55, edgecolor="k", label=f"{ents[1]} entanglement (light)")],
                   fontsize=7, loc="lower right")
    ax = axes[2]
    procs = ["control_all_to_all", "processor_A", "processor_B"]
    for k, ent in enumerate(ents):
        sub = [df[(df.entanglement == ent) & (df.backend == b)].kernel_mae_test for b in procs]
        ax.bar(np.arange(3) + k * w, [s.mean() for s in sub], w, yerr=[s.std(ddof=1) for s in sub],
               color=[COL[b] for b in procs], capsize=3, alpha=1 if k == 0 else 0.55, edgecolor="k", linewidth=0.5)
    ax.set_xticks(np.arange(3) + 0.4 - w / 2)
    ax.set_xticklabels([LABEL[b] for b in procs], fontsize=8)
    ax.set_title("Kernel error vs ideal (mean |ΔK|, test)")
    _save(fig, "AB_performance.png")


def plot_architecture_tradeoffs(arch_df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ents = list(arch_df.entanglement.unique())
    procs = list(arch_df.processor.unique())
    w = 0.8 / len(procs)
    for ax, (col, title) in zip(axes, [("depth", "Circuit depth"), ("two_qubit_gates", "Two-qubit (CX) gates"),
                                       ("swap_count", "Routing SWAPs")]):
        for k, p in enumerate(procs):
            vals = [arch_df[(arch_df.processor == p) & (arch_df.entanglement == e)][col].iloc[0] for e in ents]
            ax.bar(np.arange(len(ents)) + k * w, vals, w, color=COL.get(p), label=LABEL.get(p, p), edgecolor="k", linewidth=.5)
            for x, v in zip(np.arange(len(ents)) + k * w, vals):
                ax.text(x, v, str(v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(np.arange(len(ents)) + 0.4 - w / 2)
        ax.set_xticklabels([f"{e} entanglement" for e in ents])
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    _save(fig, "architecture_tradeoffs.png")


def plot_kernels(kernels, ent="full", seed=0):
    names = ["ideal_kernel", "control_all_to_all", "processor_A", "processor_B"]
    names = [n for n in names if (ent, seed, n) in kernels]
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.4))
    for ax, n in zip(np.atleast_1d(axes), names):
        im = ax.imshow(kernels[(ent, seed, n)], vmin=0, vmax=0.5, cmap="viridis")
        ax.set_title(LABEL[n], fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label="K(x,y) (clipped at 0.5)")
    _save(fig, "kernel_matrices.png")


def plot_concentration(kernels, ent="full", seed=0):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for n in ["ideal_kernel", "processor_A", "processor_B"]:
        K = kernels.get((ent, seed, n))
        if K is None:
            continue
        v = K[np.triu_indices(len(K), 1)]
        ax.hist(v, bins=30, alpha=0.55, color=COL[n], label=f"{LABEL[n]} (σ={v.std():.3f})")
    ax.set_xlabel("off-diagonal kernel value"); ax.set_ylabel("count"); ax.legend(fontsize=8)
    ax.set_title("Kernel concentration: noise shrinks the dynamic range")
    _save(fig, "kernel_concentration.png")


def plot_noise_sweep(sw):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for b, g in sw.groupby("backend"):
        axes[0].plot(g.p_2q * 100, g.accuracy, "o-", color=COL[b], label=LABEL[b])
        axes[1].plot(g.p_2q * 100, g.kernel_mae_test, "o-", color=COL[b], label=LABEL[b])
    axes[0].set_ylabel("test accuracy"); axes[1].set_ylabel("mean |ΔK| vs ideal")
    for ax in axes:
        ax.set_xlabel("two-qubit depolarizing error (%)"); ax.grid(alpha=.3)
    axes[0].legend(fontsize=8); axes[0].set_title("Accuracy vs CX error"); axes[1].set_title("Kernel error vs CX error")
    _save(fig, "noise_sweep.png")


def plot_gates_vs_error(st):
    fig, ax = plt.subplots(figsize=(6, 4))
    for b, g in st.groupby("backend"):
        ax.scatter(g.two_qubit_gates, g.kernel_mae, color=COL[b], label=LABEL[b], s=45, edgecolor="k")
    ax.set_xlabel("two-qubit gates after transpilation (varies with layout/routing seed)")
    ax.set_ylabel("mean |ΔK| vs ideal (60 pairs)")
    ax.set_title("Compilation overhead → kernel error"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    _save(fig, "gates_vs_kernel_error.png")
