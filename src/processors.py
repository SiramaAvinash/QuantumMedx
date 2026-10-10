"""Processor definitions: coupling graph, basis gates and synthetic noise model."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
from qiskit.transpiler import CouplingMap
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from . import PROCESSOR_DIR


@dataclass
class Processor:
    name: str
    num_qubits: int
    coupling_map: list
    basis_gates: list
    noise: dict = field(default_factory=dict)
    description: str = ""
    official_definition: bool = False

    def coupling(self) -> CouplingMap:
        """Undirected coupling map (both CX directions allowed)."""
        edges = [tuple(e) for e in self.coupling_map]
        return CouplingMap(edges + [(b, a) for a, b in edges])

    def graph(self) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(self.num_qubits))
        g.add_edges_from(map(tuple, self.coupling_map))
        return g

    def graph_stats(self) -> dict:
        g = self.graph()
        degs = [d for _, d in g.degree()]
        return {
            "qubits": self.num_qubits,
            "edges": g.number_of_edges(),
            "max_degree": max(degs),
            "mean_degree": round(sum(degs) / len(degs), 3),
            "diameter": nx.diameter(g),
            "avg_shortest_path": round(nx.average_shortest_path_length(g), 3),
            "density": round(nx.density(g), 3),
        }

    def noise_model(self, p_2q: float | None = None) -> NoiseModel:
        """Synthetic depolarizing + readout noise. `p_2q` overrides the CX error (used by sweeps)."""
        n = self.noise
        p2 = p_2q if p_2q is not None else n.get("p_2q", 0.0)
        nm = NoiseModel(basis_gates=list(self.basis_gates))
        if n.get("p_1q", 0) > 0:
            nm.add_all_qubit_quantum_error(depolarizing_error(n["p_1q"], 1), ["sx", "x"])
        if p2 > 0:
            nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
        p01, p10 = n.get("readout_p01", 0), n.get("readout_p10", 0)
        if p01 > 0 or p10 > 0:
            nm.add_all_qubit_readout_error(ReadoutError([[1 - p01, p01], [p10, 1 - p10]]))
        return nm


def load_processor(path) -> Processor:
    path = Path(path)
    if not path.exists():
        path = PROCESSOR_DIR / path
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return Processor(
        name=d["name"], num_qubits=d["num_qubits"], coupling_map=d["coupling_map"],
        basis_gates=d["basis_gates"], noise=d.get("noise", {}),
        description=d.get("description", ""), official_definition=d.get("official_definition", False),
    )


def load_all():
    return (load_processor(PROCESSOR_DIR / "processor_A.json"),
            load_processor(PROCESSOR_DIR / "processor_B.json"),
            load_processor(PROCESSOR_DIR / "control_all_to_all.json"))
