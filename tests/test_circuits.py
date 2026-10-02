"""Circuit structure, parameters, and equivalence with the class constructors."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from qnnwind.circuits import all_circuits, build_circuit, cnot_blocks, save_entanglement_maps
from qnnwind.io import read_json

# config: (entanglement, CNOTs, total gates, depth, block 1, block 2)
EXPECTED = {
    "QNN-1": (
        "full",
        12,
        40,
        16,
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    ),
    "QNN-2": ("linear", 6, 34, 12, [(0, 1), (1, 2), (2, 3)], [(0, 1), (1, 2), (2, 3)]),
    "QNN-3": (
        "circular",
        8,
        36,
        15,
        [(3, 0), (0, 1), (1, 2), (2, 3)],
        [(3, 0), (0, 1), (1, 2), (2, 3)],
    ),
    "QNN-4": (
        "sca",
        8,
        36,
        15,
        [(3, 0), (0, 1), (1, 2), (2, 3)],
        [(3, 2), (0, 3), (1, 0), (2, 1)],
    ),
    "QNN-5": ("reverse_linear", 6, 34, 12, [(2, 3), (1, 2), (0, 1)], [(2, 3), (1, 2), (0, 1)]),
    "QNN-6": ("pairwise", 6, 34, 11, [(0, 1), (2, 3), (1, 2)], [(0, 1), (2, 3), (1, 2)]),
}
CONFIGS = sorted(EXPECTED)


def test_config_entanglements_match_spec(config):
    assert config["qnn"]["entanglement"] == {k: v[0] for k, v in EXPECTED.items()}


@pytest.mark.parametrize("name", CONFIGS)
def test_gate_counts_depth_and_cnot_sequences(config, name):
    ent, n_cx, total, depth, block1, block2 = EXPECTED[name]
    qc = build_circuit(name, ent, config["qnn"])
    ops = dict(qc.circuit.count_ops())
    assert ops == {"h": 8, "p": 8, "ry": 12, "cx": n_cx}
    assert sum(ops.values()) == total
    assert sum(v for k, v in ops.items() if k != "cx") == 28
    assert dict(qc.feature_map.count_ops()) == {"h": 8, "p": 8}
    assert qc.circuit.depth() == depth
    assert cnot_blocks(qc.ansatz, 2) == [block1, block2]
    assert cnot_blocks(qc.circuit, 2) == [block1, block2]


@pytest.mark.parametrize("name", CONFIGS)
def test_parameters(config, name):
    qc = build_circuit(name, EXPECTED[name][0], config["qnn"])
    assert [p.name for p in qc.feature_map.parameters] == [f"x[{i}]" for i in range(4)]
    assert qc.ansatz.num_parameters == 12
    assert qc.circuit.num_parameters == 16
    assert list(qc.circuit.parameters)[:4] == list(qc.feature_map.parameters)
    assert list(qc.circuit.parameters)[4:] == list(qc.ansatz.parameters)


@pytest.mark.parametrize("name", CONFIGS)
def test_equivalent_to_deprecated_class_construction(config, name):
    ent = EXPECTED[name][0]
    qc = build_circuit(name, ent, config["qnn"]).circuit
    # The class constructors are deprecated since Qiskit 2.1; suppress only inside this test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from qiskit.circuit.library import RealAmplitudes, ZFeatureMap

        legacy = ZFeatureMap(4, reps=2).compose(RealAmplitudes(4, reps=2, entanglement=ent))
        legacy = legacy.decompose()
    assert [p.name for p in qc.parameters] == [p.name for p in legacy.parameters]
    rng = np.random.default_rng(2025)
    for _ in range(5):
        values = rng.uniform(-np.pi, np.pi, qc.num_parameters)
        u_new = Operator(qc.assign_parameters(values)).data
        u_old = Operator(legacy.assign_parameters(values)).data
        assert np.max(np.abs(u_new - u_old)) < 1e-12


def test_entanglement_maps_file(config, tmp_path):
    path = tmp_path / "entanglement_maps.json"
    save_entanglement_maps(path, config["qnn"])
    maps = read_json(path)
    assert list(maps) == CONFIGS
    for name, (ent, *_, block1, block2) in EXPECTED.items():
        assert maps[name]["entanglement"] == ent
        assert maps[name]["blocks"] == [[list(p) for p in block1], [list(p) for p in block2]]
    assert b"\r\n" not in path.read_bytes()


def test_full_and_reverse_linear_are_the_same_unitary(config):
    """Documented Qiskit property: QNN-1 (full) and QNN-5 (reverse_linear) are one model.

    No other pair of configurations shares its ansatz unitary.
    """
    circuits = {qc.name: qc.ansatz for qc in all_circuits(config["qnn"])}
    rng = np.random.default_rng(3)
    weights = [rng.uniform(-np.pi, np.pi, 12) for _ in range(5)]

    def same(a, b):
        return all(
            np.max(
                np.abs(
                    Operator(circuits[a].assign_parameters(w)).data
                    - Operator(circuits[b].assign_parameters(w)).data
                )
            )
            < 1e-12
            for w in weights
        )

    identical = [(a, b) for i, a in enumerate(CONFIGS) for b in CONFIGS[i + 1 :] if same(a, b)]
    assert identical == [("QNN-1", "QNN-5")]


def test_all_circuits_order(config):
    assert [qc.name for qc in all_circuits(config["qnn"])] == CONFIGS
