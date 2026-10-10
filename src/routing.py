"""Transpilation / routing utilities and measured routing overhead."""
from __future__ import annotations

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from .problem import entangling_pairs
from .processors import Processor


def transpile_for(qc: QuantumCircuit, proc: Processor | None, optimization_level: int = 1,
                  seed: int = 7, extra_basis=()) -> QuantumCircuit:
    """Compile onto `proc` (coupling map + basis gates). proc=None -> all-to-all reference compile."""
    basis = list(proc.basis_gates if proc else ["rz", "sx", "x", "cx"]) + list(extra_basis)
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        coupling_map=proc.coupling() if proc else None,
        basis_gates=basis,
        seed_transpiler=seed,
    )
    return pm.run(qc)


def _count(qc, names):
    ops = qc.count_ops()
    return sum(n for g, n in ops.items() if g in names)


def circuit_metrics(compiled: QuantumCircuit, logical: QuantumCircuit, proc: Processor,
                    optimization_level: int = 1, seed: int = 7) -> dict:
    """Depth, 2Q gates, measured SWAPs, extra CX caused by routing, physical qubits used."""
    # all-to-all compile with identical settings = routing-free reference
    ref = transpile_for(logical, None, optimization_level, seed)
    # same compile but SWAP kept as a basis gate -> counts routing SWAPs explicitly
    with_swap = transpile_for(logical, proc, optimization_level, seed, extra_basis=("swap",))
    active = {compiled.find_bit(q).index for inst in compiled.data
              if inst.operation.name not in ("barrier",) for q in inst.qubits}
    cx = _count(compiled, {"cx"})
    cx_ref = _count(ref, {"cx"})
    return {
        "processor": proc.name,
        "depth": compiled.depth(),
        "depth_ref_all_to_all": ref.depth(),
        "two_qubit_gates": cx,
        "two_qubit_gates_ref": cx_ref,
        "extra_2q_from_routing": cx - cx_ref,
        "swap_count": _count(with_swap, {"swap"}),
        "total_ops": sum(compiled.count_ops().values()),
        "physical_qubits_used": len(active),
        "ancillas": 0,
        "processor_qubits": proc.num_qubits,
    }


def interaction_distance_stats(proc: Processor, n_logical: int, entanglement: str) -> dict:
    """Graph-distance statistics for the logical interaction pairs (best-case identity-ish layout)."""
    g = proc.graph()
    sub = sorted(g.nodes)[:n_logical]
    pairs = entangling_pairs(n_logical, entanglement)
    d = dict(nx.all_pairs_shortest_path_length(g))
    dist = [d[sub[i]][sub[j]] for i, j in pairs]
    return {"pairs": len(pairs), "non_adjacent_pairs": sum(x > 1 for x in dist),
            "mean_distance_trivial_layout": round(sum(dist) / len(dist), 3)}
