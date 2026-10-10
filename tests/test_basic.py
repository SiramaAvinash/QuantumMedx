import numpy as np
from src.kernel import ideal_kernel
from src.problem import kernel_circuit, make_split
from src.processors import load_all
from src.routing import circuit_metrics, transpile_for

A, B, C = load_all()


def test_ideal_kernel_properties():
    sp = make_split(5, 20, 10, 0, scale=0.3)
    K = ideal_kernel(sp.X_train, sp.X_train, 5)
    assert np.allclose(np.diag(K), 1) and np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-9      # fidelity kernel is PSD


def test_routing_overhead_ordering():
    qc, _, _ = kernel_circuit(5, 1, "full")
    m = {P.name: circuit_metrics(transpile_for(qc, P), qc, P) for P in (A, B, C)}
    assert m["control_all_to_all"]["swap_count"] == 0
    assert m["processor_A"]["swap_count"] > 0 and m["processor_B"]["swap_count"] > 0
    assert m["processor_A"]["two_qubit_gates"] > m["control_all_to_all"]["two_qubit_gates"]


def test_linear_map_needs_no_routing_on_line():
    qc, _, _ = kernel_circuit(5, 1, "linear")
    m = circuit_metrics(transpile_for(qc, A), qc, A)
    assert m["swap_count"] == 0
