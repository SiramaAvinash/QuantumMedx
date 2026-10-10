# QuantumMedx 🧬⚛️
**Quantum kernel classifier for biomedical data — Qiskit Fall Fest 2026, Phase 1 (PS Q2: Quantum Kernel Classifier)**

QuantumMedx builds a fidelity quantum-kernel SVM for breast-cancer diagnosis data, then benchmarks the *same* solution on two processor
architectures (Processor A: 5-qubit line, Processor B: 7-qubit heavy-hex-inspired) under identical noise, measuring how connectivity, routing and
compilation overhead change kernel quality and classification.

> ⚠️ **Before submitting:** `processors/processor_A.json` and `processor_B.json` are built from the Phase-1 brief (5q line / 7q heavy-hex-like, synthetic noise).
> They are marked `"official_definition": false`. If the official Challenge Kit supplies its own files, overwrite them (same schema: `num_qubits, coupling_map, basis_gates, noise{p_1q,p_2q,readout_p01,readout_p10}`)
> and re-run `python main.py`. Also check the Q2 starter kit for the official dataset/metrics — if it differs from the Wisconsin data used here, swap `src/problem.py::make_split`.

## Quick start
```bash
python -m venv .venv && source .venv\Scripts\Activate.ps1      # optional
pip install -r requirements.txt
python main.py            # full benchmark, ~5 min on CPU -> results/tables + results/figures
python main.py --quick    # ~25 s smoke test
python -m pytest tests    # sanity tests
jupyter notebook main.ipynb
```

## What it does
1. **Data** – 569-sample Wisconsin diagnostic set → stratified 40/30 split → StandardScaler → PCA(5) → MinMax to `[0, s]`; bandwidth `s` picked by train-only CV (s = 0.2).
2. **Kernel** – ZZ-style feature map (full or linear entanglement), `K(x,y)=P(0…0)` of `U(y)†U(x)|0⟩`; ideal via statevector, noisy via Aer + depolarising/readout noise.
3. **Classifier** – SVM on precomputed kernel, C by CV; PSD clipping of the noisy Gram matrix.
4. **A/B benchmark** – identical data/shots/noise/seeds on A, B, plus an optional all-to-all *control* to isolate connectivity.
5. **Architecture metrics** – depth, CX count, **measured SWAPs**, extra CX vs all-to-all, physical qubits, runtime, graph statistics.
6. **Studies** – noise sweep, transpiler-seed vs kernel error, optimisation levels, bandwidth selection, classical baselines.

## Key results (3 seeds, full-entanglement map)
| | Depth | CX | SWAPs | Test acc. | Kernel error \|ΔK\| |
|---|---|---|---|---|---|
| All-to-all control | 49 | 38 | 0 | 0.933 ± 0.033 | 0.103 |
| Processor A (line) | 81 | 66 | 12 | 0.933 ± 0.058 | 0.149 |
| Processor B (heavy-hex-like) | 84 | 61 | 9 | 0.933 ± 0.058 | 0.141 |
| Ideal kernel / classical RBF-SVM | – | – | – | 0.956 / 0.933 | 0 / – |

* B needs fewer SWAPs/CX than A for the all-pairs feature map, and its kernel error is lower in proportion (≈ 0.00165 per extra CX on both).
* Accuracy cannot separate A from B with 30 test points (binomial SE ≈ 0.045) — the difference shows up in kernel fidelity and cost.
* With the linear feature map both processors compile to the identical circuit (14 CX, 0 SWAPs): topology only matters when the algorithm's interaction graph does not embed.
* No quantum advantage over classical SVMs is claimed. Full analysis and limitations: **[report.md](report.md)**.

## Repository layout
```
main.py  main.ipynb  report.md  README.md  AI_DISCLOSURE.md  requirements.txt
processors/   processor_A.json  processor_B.json  control_all_to_all.json
src/          problem.py processors.py routing.py kernel.py metrics.py experiments.py visualization.py
data/         raw/ (exported dataset)  processed/ (seed-0 split)
results/      tables/ (CSV/JSON)  figures/ (PNG)
tests/        test_basic.py
scripts/      build_notebook.py
```

## Submission checklist mapping
One PS (Q2) ✔ · core solution ✔ · Processor A & B runs ✔ · same conditions ✔ · A-vs-B table (`results/tables/AB_comparison.csv`, `final_comparison.csv`) ✔ ·
graphs ✔ · README ✔ · `requirements.txt` ✔ · notebook runs from clean env ✔ · AI use disclosed (`AI_DISCLOSURE.md`) ✔ ·
*Remaining for you:* push to a public/shared GitHub repo, use the organizers' repo-name convention, and swap in official processor files if provided.
