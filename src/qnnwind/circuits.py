"""QNN circuits from the Qiskit circuit library function constructors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes, z_feature_map

from qnnwind.io import write_json


@dataclass(frozen=True)
class QNNCircuit:
    """Feature map, ansatz, and their composition for one configuration."""

    name: str
    entanglement: str
    feature_map: QuantumCircuit
    ansatz: QuantumCircuit
    circuit: QuantumCircuit


def build_circuit(name: str, entanglement: str, qnn_cfg: dict) -> QNNCircuit:
    """``z_feature_map(n, reps)`` composed with ``real_amplitudes(n, reps, entanglement)``.

    All other ``real_amplitudes`` arguments keep their defaults (final rotation layer
    included). Qubit j encodes feature j in the dataset's column order.
    """
    n_qubits = qnn_cfg["n_qubits"]
    feature_map = z_feature_map(n_qubits, reps=qnn_cfg["feature_map_reps"])
    ansatz = real_amplitudes(n_qubits, reps=qnn_cfg["ansatz_reps"], entanglement=entanglement)
    return QNNCircuit(
        name=name,
        entanglement=entanglement,
        feature_map=feature_map,
        ansatz=ansatz,
        circuit=feature_map.compose(ansatz),
    )


def all_circuits(qnn_cfg: dict) -> list[QNNCircuit]:
    """The six configurations QNN-1..QNN-6 in config order."""
    return [build_circuit(name, ent, qnn_cfg) for name, ent in qnn_cfg["entanglement"].items()]


def base_config(qnn_cfg: dict, name: str) -> str:
    """The circuit configuration of a QNN model: itself, or the ``base`` of a readout variant
    (``qnn.variants``, e.g. QNN-1u -> QNN-1)."""
    variants = qnn_cfg.get("variants") or {}
    return variants[name]["base"] if name in variants else name


def circuit_for(qnn_cfg: dict, name: str) -> QNNCircuit:
    """The circuit of any QNN model name, including readout variants (same circuit as the
    base configuration; the QNNCircuit keeps the variant's name)."""
    base = base_config(qnn_cfg, name)
    return build_circuit(name, qnn_cfg["entanglement"][base], qnn_cfg)


def total_gates(qnn_cfg: dict) -> dict[str, int]:
    """Total gate count of every QNN model name of the config (base configs and variants)."""
    base = {qc.name: int(sum(qc.circuit.count_ops().values())) for qc in all_circuits(qnn_cfg)}
    variants = qnn_cfg.get("variants") or {}
    return {**base, **{name: base[v["base"]] for name, v in variants.items()}}


def cnot_blocks(circuit: QuantumCircuit, n_blocks: int) -> list[list[tuple[int, int]]]:
    """The (control, target) pairs of every ``cx`` gate, split into ``n_blocks`` equal blocks."""
    pairs = [
        (circuit.find_bit(inst.qubits[0]).index, circuit.find_bit(inst.qubits[1]).index)
        for inst in circuit.data
        if inst.operation.name == "cx"
    ]
    if len(pairs) % n_blocks:
        raise ValueError(f"{len(pairs)} CNOTs cannot be split into {n_blocks} equal blocks")
    size = len(pairs) // n_blocks
    return [pairs[i * size : (i + 1) * size] for i in range(n_blocks)]


def circuit_summary(qc: QNNCircuit, n_input_params: int) -> dict:
    """Gate counts by type, depth, and parameter counts (for table T2)."""
    ops = dict(qc.circuit.count_ops())
    return {
        "config": qc.name,
        "entanglement": qc.entanglement,
        "gate_counts": {name: int(count) for name, count in sorted(ops.items())},
        "cnots": int(ops.get("cx", 0)),
        "single_qubit_gates": int(sum(c for g, c in ops.items() if g != "cx")),
        "total_gates": int(sum(ops.values())),
        "depth": int(qc.circuit.depth()),
        "input_params": n_input_params,
        "weight_params": int(qc.ansatz.num_parameters),
    }


def entanglement_maps(qnn_cfg: dict) -> dict[str, dict]:
    """CNOT (control, target) sequences per ansatz repetition, for every configuration."""
    maps = {}
    for qc in all_circuits(qnn_cfg):
        blocks = cnot_blocks(qc.ansatz, qnn_cfg["ansatz_reps"])
        maps[qc.name] = {
            "entanglement": qc.entanglement,
            "blocks": [[list(pair) for pair in block] for block in blocks],
        }
    return maps


def save_entanglement_maps(path: Path, qnn_cfg: dict) -> None:
    """Write ``outputs/tables/entanglement_maps.json``."""
    write_json(path, entanglement_maps(qnn_cfg))
