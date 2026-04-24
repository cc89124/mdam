#!/usr/bin/env python3
"""Convert S-gate proxy magic state cultivation circuits to T-gate circuits for Clifft.

Transforms stim S-gate circuits into raw string output containing T/T_DAG gates
(non-Clifford) that Clifft can compile and simulate.

Transformation rules:

  Rule 1: S/S_DAG → T/T_DAG substitution. Every S becomes T, every S_DAG
      becomes T_DAG. This affects both the injection stage and the cultivation
      stage (cat-state verification conjugation). S = diag(1, i) is the
      Clifford proxy for T = diag(1, e^{iπ/4}).

  Rule 2: Errata flips (d=5 only). During the d=5 double cat-state check,
      6 specific qubits need their T/T_DAG inverted due to the cat-state
      verification circuit's Pauli frame.

  Rule 3: Growth stabilizer feedforward (d=5 only). Seven growth-check
      measurements become random with T gates (they were deterministic with
      S). Feedforward CX/CZ gates correct for these.

  Rule 4: Detector healing (d=5 only). The feedforward source measurements
      from Rule 3 are now random after T-injection. Detectors referencing
      these measurements must be healed:

      - Large logical check detector (>10 recs): Only the odd-parity
        measurement at (4,3) Z is stripped (3 CX targets break parity).
        The even-parity feedforward measurements are kept for logical
        parity tracking.

      - Small local detectors: All feedforward source measurements in
        random_m1_indices are stripped. However, if stripping would leave
        a detector completely empty, the original recs are preserved.

      The "if empty, revert" rule reflects the physics of active feedforward:
      Three feedforward measurements -- (4,1), (4,5), (6,2) Z-basis -- are
      deterministically 0 (Bell pair alignment). Their feedforward CX gates
      have even target counts, so the correction preserves stabilizer parity.
      When these appear in a MIXED detector (alongside later-round recs),
      they must be stripped because feedforward severs the temporal
      correlation: if noise flips the Round 1 measurement to 1, feedforward
      actively repairs the state, so Round 2 measures 0 regardless. The
      mixed detector would see 1 XOR 0 = 1 (false positive). But when a
      deterministic measurement is the SOLE rec in a detector, it functions
      as a valuable initialization check (detecting noise on that specific
      measurement operation) and must be kept.

  Rule 5a: Final measurement wrapping (inject+cultivate). The final
      MPP Y_L is wrapped: T → MPP Y_L → T_DAG

Produces 4 circuit variants:
  1. d=3 inject + cultivate only
  2. d=5 inject + cultivate only
  3. d=3 end-to-end (finishes in surface code)
  4. d=5 end-to-end (finishes in surface code)

Additionally, *_expval() variants insert EXP_VAL X_L, Y_L, Z_L probes
before the final MPP for analytical fidelity extraction.

Usage:
  uv run python convert_s_to_t.py                     # noiseless
  uv run python convert_s_to_t.py --noise_strength 0.001  # noisy
"""

import argparse
import re
import stim
import numpy as np

import gen
from cultiv import make_inject_and_cultivate_circuit, make_end2end_cultivation_circuit

# ---------------------------------------------------------------------------
# Errata coordinates for d=5 circuits (Rule 2)
# During d=5 double cat check, these qubits get their S->T substitution flipped
# ---------------------------------------------------------------------------
ERRATA_COORDS = {(0, 0), (3, 0), (4, 0), (4, 2), (7, 0), (8, 0)}

# ---------------------------------------------------------------------------
# Feedforward rules for d=5 growth stabilizers (Rule 3)
# Maps (measurement_coord, is_x_basis) -> [(target_coord, gate)]
# ---------------------------------------------------------------------------
FF_RULES = {
    ((5, 0), True): [((7, 0), "CZ"), ((8, 0), "CZ")],
    ((3, 3), True): [((3, 4), "CZ"), ((4, 6), "CZ")],
    ((6, 0), False): [
        ((7, 0), "CX"),
        ((8, 0), "CX"),
        ((7, 2), "CX"),
        ((8, 0), "CX"),
    ],
    ((6, 2), False): [((7, 0), "CX"), ((8, 0), "CX")],
    ((4, 1), False): [((3, 4), "CX"), ((4, 6), "CX")],
    ((4, 3), False): [((0, 0), "CX"), ((3, 0), "CX"), ((4, 6), "CX")],
    ((4, 5), False): [((3, 4), "CX"), ((4, 6), "CX")],
}


# ===================================================================
# Helpers
# ===================================================================


def parse_coord_map(circuit: stim.Circuit) -> tuple[dict, dict]:
    """Parse QUBIT_COORDS to build coord<->idx mappings.

    Returns:
        coord_to_idx: {(x, y): qubit_index}
        idx_to_coord: {qubit_index: (x, y)}
    """
    coord_to_idx = {}
    idx_to_coord = {}
    for inst in circuit:
        if inst.name == "QUBIT_COORDS":
            args = inst.gate_args_copy()
            idx = inst.targets_copy()[0].value
            coord = (int(args[0]), int(args[1]))
            coord_to_idx[coord] = idx
            idx_to_coord[idx] = coord
    return coord_to_idx, idx_to_coord


def _gate_name(token: str) -> str:
    """Extract bare gate name from a possibly-parameterized token.

    E.g. 'M(0.001)' -> 'M', 'MPP' -> 'MPP', 'DEPOLARIZE1(0.001)' -> 'DEPOLARIZE1'.
    """
    paren = token.find("(")
    return token[:paren] if paren != -1 else token


def count_measurement_records(line: str) -> int:
    """Count how many measurement records a single instruction line produces.

    M, MX, MY, MZ generate one record per target qubit.
    MPP generates one record per Pauli product (space-separated groups).
    Handles noisy variants like M(0.001) correctly.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return 0

    parts = line.split()
    name = _gate_name(parts[0])

    if name in ("M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"):
        # One record per target qubit.
        targets = [p for p in parts[1:] if not p.startswith("(")]
        return len(targets)
    elif name == "MPP":
        # One record per Pauli product (each space-separated token is a product)
        return len(parts) - 1
    return 0


def parse_targets(line: str) -> list[int]:
    """Extract qubit indices from a gate line like 'S_DAG 0 3 7 9'."""
    parts = line.split()
    return [int(p) for p in parts[1:] if not p.startswith("(")]


def emit_t_gate_split(
    targets: list[int], errata_idxs: set, is_t_dag: bool
) -> list[str]:
    """Emit T/T_DAG with errata flips applied.

    For the errata qubits, the gate is flipped (T_DAG<->T).

    Args:
        targets: qubit indices
        errata_idxs: set of qubit indices that get flipped
        is_t_dag: True if the base gate is T_DAG (from S_DAG)
    """
    result = []
    if not errata_idxs:
        gate = "T_DAG" if is_t_dag else "T"
        result.append(f"{gate} {' '.join(map(str, targets))}")
    else:
        normal = [q for q in targets if q not in errata_idxs]
        flipped = [q for q in targets if q in errata_idxs]
        base_gate = "T_DAG" if is_t_dag else "T"
        flip_gate = "T" if is_t_dag else "T_DAG"
        if normal:
            result.append(f"{base_gate} {' '.join(map(str, normal))}")
        if flipped:
            result.append(f"{flip_gate} {' '.join(map(str, flipped))}")
    return result


# ===================================================================
# Core transformation
# ===================================================================


def _decompose_yl_product(yl_product: str) -> tuple[str, str, str]:
    """Decompose Y_L product into X_L, Y_L, Z_L Pauli strings.

    Y_L contains X, Z, and Y terms. Since Y = iXZ:
    - X_L = all X-component qubits (from X terms + Y terms)
    - Z_L = all Z-component qubits (from Z terms + Y terms)

    Returns:
        (x_l_str, y_l_str, z_l_str) as EXP_VAL-compatible Pauli strings.
    """
    x_qubits = [int(m.group(1)) for m in re.finditer(r"X(\d+)", yl_product)]
    z_qubits = [int(m.group(1)) for m in re.finditer(r"Z(\d+)", yl_product)]
    y_qubits = [int(m.group(1)) for m in re.finditer(r"Y(\d+)", yl_product)]

    xl_qubits = sorted(x_qubits + y_qubits)
    zl_qubits = sorted(z_qubits + y_qubits)

    xl_str = "*".join(f"X{q}" for q in xl_qubits)
    zl_str = "*".join(f"Z{q}" for q in zl_qubits)
    return xl_str, yl_product, zl_str


def convert_circuit(
    circuit: stim.Circuit,
    *,
    is_d5: bool,
    is_e2e: bool,
    insert_exp_val: bool = False,
) -> str:
    """Convert an S-gate proxy circuit to a T-gate circuit string.

    Args:
        circuit: The input stim circuit (S-gate proxy, Y-basis).
        is_d5: Whether this is a d=5 circuit (enables errata flips & feedforward).
        is_e2e: Whether this is an end-to-end circuit (surface code ending).
        insert_exp_val: If True (e2e only), insert EXP_VAL X_L, Y_L, Z_L
            probes before the final MPP for analytical fidelity extraction.

    Returns:
        Multi-line string of the transformed circuit.
    """
    flat = circuit.flattened()
    coord_to_idx, idx_to_coord = parse_coord_map(flat)

    # Compute errata qubit indices for d=5
    errata_idxs = set()
    if is_d5:
        for c in ERRATA_COORDS:
            if c in coord_to_idx:
                errata_idxs.add(coord_to_idx[c])

    # Convert circuit to lines
    lines = str(flat).splitlines()

    # ---------------------------------------------------------------
    # Pass 1: Identify S/S_DAG ordinals and the final MPP
    # ---------------------------------------------------------------
    s_dag_lines = []  # line indices with S_DAG
    s_lines = []  # line indices with S
    final_mpp_line = None  # line index of the final MPP

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("S_DAG "):
            s_dag_lines.append(i)
        elif (
            stripped.startswith("S ")
            and not stripped.startswith("SQRT")
            and not stripped.startswith("SHIFT")
        ):
            s_lines.append(i)
        if stripped.startswith("MPP "):
            final_mpp_line = i  # Keep updating; last one wins

    # ---------------------------------------------------------------
    # Pass 2 (d=5 only): Compute feedforward injection data
    # ---------------------------------------------------------------
    feedforward_inject_line = None
    random_m1_indices = set()
    ff_lines_to_inject = []
    odd_parity_idx = None

    if is_d5 and len(s_dag_lines) >= 3:
        third_s_dag_line = s_dag_lines[2]
        search_start = s_lines[0] if len(s_lines) >= 1 else 0

        # Find the first stabilizer round M+MX in the grown code.
        # This is the first M line after the growth check that is part of
        # a full stabilizer round (M followed by MX on the same tick).
        # The feedforward gates go AFTER these measurements.
        first_round_m_line = None
        for i in range(search_start, third_s_dag_line):
            stripped = lines[i].strip()
            line_gate = _gate_name(stripped.split()[0]) if stripped else ""
            if line_gate == "M":
                # Check if next non-noise line is MX (full round pattern)
                next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                next_gate = _gate_name(next_line.split()[0]) if next_line else ""
                if next_gate == "MX":
                    first_round_m_line = i
                    break

        if first_round_m_line is not None:
            # Find the TICK after the first round's detectors (injection point)
            past_detectors = False
            for j in range(first_round_m_line + 1, third_s_dag_line):
                stripped = lines[j].strip()
                if stripped.startswith("DETECTOR"):
                    past_detectors = True
                elif stripped == "TICK" and past_detectors:
                    feedforward_inject_line = j
                    break

            # Count absolute measurements up to the injection point
            abs_meas_at_inject = 0
            for j in range(feedforward_inject_line + 1):
                abs_meas_at_inject += count_measurement_records(lines[j])

            # Build map: (coord, is_x_basis) -> absolute measurement index
            # for ALL measurements up to injection point
            meas_map = {}
            running_abs = 0
            for j in range(feedforward_inject_line + 1):
                stripped = lines[j].strip()
                n_records = count_measurement_records(stripped)
                if n_records > 0:
                    line_gate = _gate_name(stripped.split()[0])
                    if line_gate == "MX":
                        targets = parse_targets(stripped)
                        for offset, qidx in enumerate(targets):
                            coord = idx_to_coord.get(qidx)
                            if coord is not None:
                                meas_map[(coord, True)] = running_abs + offset
                    elif line_gate == "M":
                        targets = parse_targets(stripped)
                        for offset, qidx in enumerate(targets):
                            coord = idx_to_coord.get(qidx)
                            if coord is not None:
                                meas_map[(coord, False)] = running_abs + offset
                    running_abs += n_records

            # Build feedforward lines
            for (meas_coord, is_x_basis), targets in FF_RULES.items():
                key = (meas_coord, is_x_basis)
                if key not in meas_map:
                    raise ValueError(
                        f"Feedforward measurement {key} not found in "
                        f"measurements up to injection point. "
                        f"Available: {sorted(meas_map.keys())}"
                    )
                source_abs = meas_map[key]
                random_m1_indices.add(source_abs)
                rec_offset = abs_meas_at_inject - source_abs
                for target_coord, gate in targets:
                    target_idx = coord_to_idx[target_coord]
                    ff_lines_to_inject.append(f"{gate} rec[-{rec_offset}] {target_idx}")

            odd_parity_idx = meas_map.get(((4, 3), False))

    # ---------------------------------------------------------------
    # Pass 3: Build output with all transformations applied
    # ---------------------------------------------------------------
    output = []
    abs_meas_idx = 0
    s_dag_ordinal = 0
    s_ordinal = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # --- Feedforward injection (Rule 3, d=5 only) ---
        if i == feedforward_inject_line:
            output.append(stripped)  # Emit the TICK
            for ff_line in ff_lines_to_inject:
                output.append(ff_line)
            abs_meas_idx += count_measurement_records(stripped)
            continue

        # --- S_DAG -> T_DAG substitution (Rules 1 & 2) ---
        if stripped.startswith("S_DAG "):
            s_dag_ordinal += 1
            targets = parse_targets(stripped)
            if is_d5 and s_dag_ordinal == 3:
                output.extend(emit_t_gate_split(targets, errata_idxs, is_t_dag=True))
            else:
                output.append(f"T_DAG {' '.join(map(str, targets))}")
            abs_meas_idx += count_measurement_records(stripped)
            continue

        # --- S -> T substitution (Rules 1 & 2) ---
        if (
            stripped.startswith("S ")
            and not stripped.startswith("SQRT")
            and not stripped.startswith("SHIFT")
        ):
            s_ordinal += 1
            targets = parse_targets(stripped)
            if is_d5 and s_ordinal == 2:
                output.extend(emit_t_gate_split(targets, errata_idxs, is_t_dag=False))
            else:
                output.append(f"T {' '.join(map(str, targets))}")
            abs_meas_idx += count_measurement_records(stripped)
            continue

        # --- Detector healing (Rule 4, d=5 only) ---
        if is_d5 and random_m1_indices and stripped.startswith("DETECTOR"):
            match = re.match(r"(DETECTOR\([^)]*\))\s*(.*)", stripped)
            if match:
                det_prefix = match.group(1)
                rec_part = match.group(2).strip()
                rec_targets = re.findall(r"rec\[(-\d+)\]", rec_part)

                is_large_detector = len(rec_targets) > 10

                kept = []
                for rec_str in rec_targets:
                    rel_offset = int(rec_str)
                    abs_idx = abs_meas_idx + rel_offset

                    if is_large_detector:
                        # Large logical check: only strip the odd-parity
                        # measurement; keep all others including the even-
                        # parity feedforward measurements.
                        if abs_idx == odd_parity_idx:
                            continue
                        kept.append(f"rec[{rel_offset}]")
                    else:
                        # Small local detectors: strip feedforward source
                        # measurements that are randomized by T-injection.
                        if abs_idx in random_m1_indices:
                            continue
                        kept.append(f"rec[{rel_offset}]")

                if not kept:
                    # All records were stripped — keep the originals.
                    # These detectors reference only deterministic
                    # feedforward measurements (Bell pair outcomes that
                    # are always 0) and still provide useful error
                    # detection under noise.
                    kept = [f"rec[{int(r)}]" for r in rec_targets]

                output.append(f"{det_prefix} {' '.join(kept)}")

                abs_meas_idx += count_measurement_records(stripped)
                continue

        # --- Final MPP handling (Rule 5) ---
        if i == final_mpp_line and stripped.startswith("MPP "):
            mpp_body = stripped[4:].strip()
            products = mpp_body.split()

            if not is_e2e:
                # Rule 5a: inject+cultivate
                # Split the MPP: Y product first, then stabilizer products.
                # Wrap the Y measurement in T / MPP Y / T_DAG.
                y_product = None
                stab_products = []
                for p in products:
                    if p.startswith("Y"):
                        y_product = p
                    else:
                        stab_products.append(p)

                if y_product:
                    y_qubits = [
                        int(m.group(1)) for m in re.finditer(r"Y(\d+)", y_product)
                    ]

                    # T (with errata flips for d=5) before Y measurement
                    output.extend(
                        emit_t_gate_split(
                            y_qubits, errata_idxs if is_d5 else set(), is_t_dag=False
                        )
                    )
                    output.append(f"MPP {y_product}")
                    # T_DAG (with errata flips for d=5) after Y measurement
                    output.extend(
                        emit_t_gate_split(
                            y_qubits, errata_idxs if is_d5 else set(), is_t_dag=True
                        )
                    )

                if stab_products:
                    output.append(f"MPP {' '.join(stab_products)}")
            else:
                # End-to-end: optionally insert EXP_VAL probes before the
                # final MPP, then pass the MPP through unchanged.
                if insert_exp_val:
                    yl_product = products[0]
                    xl_str, yl_str, zl_str = _decompose_yl_product(yl_product)
                    output.append(f"EXP_VAL {xl_str}")
                    output.append(f"EXP_VAL {yl_str}")
                    output.append(f"EXP_VAL {zl_str}")
                output.append(stripped)

            abs_meas_idx += count_measurement_records(stripped)
            continue

        # --- Default: pass through ---
        output.append(stripped)
        abs_meas_idx += count_measurement_records(stripped)

    return "\n".join(output)


# ===================================================================
# Public API: Generate the 4 circuit variants
# ===================================================================


def _apply_noise(circuit: stim.Circuit, noise_strength: float) -> stim.Circuit:
    """Apply uniform depolarizing noise to a circuit if noise_strength > 0."""
    if noise_strength <= 0:
        return circuit
    noise_model = gen.NoiseModel.uniform_depolarizing(noise_strength)
    return noise_model.noisy_circuit_skipping_mpp_boundaries(circuit)


def make_d3_inject_cultivate(noise_strength: float = 0.0) -> str:
    """Generate d=3 inject + cultivate T-gate circuit."""
    circuit = make_inject_and_cultivate_circuit(
        dcolor=3, inject_style="unitary", basis="Y"
    )
    circuit = _apply_noise(circuit, noise_strength)
    return convert_circuit(circuit, is_d5=False, is_e2e=False)


def make_d5_inject_cultivate(noise_strength: float = 0.0) -> str:
    """Generate d=5 inject + cultivate T-gate circuit."""
    circuit = make_inject_and_cultivate_circuit(
        dcolor=5, inject_style="unitary", basis="Y"
    )
    circuit = _apply_noise(circuit, noise_strength)
    return convert_circuit(circuit, is_d5=True, is_e2e=False)


def make_d3_end2end(noise_strength: float = 0.0) -> str:
    """Generate d=3 end-to-end T-gate circuit."""
    circuit = make_end2end_cultivation_circuit(
        dcolor=3, dsurface=15, basis="Y", r_growing=3, r_end=5, inject_style="unitary"
    )
    circuit = _apply_noise(circuit, noise_strength)
    return convert_circuit(circuit, is_d5=False, is_e2e=True)


def make_d5_end2end(noise_strength: float = 0.0) -> str:
    """Generate d=5 end-to-end T-gate circuit."""
    circuit = make_end2end_cultivation_circuit(
        dcolor=5, dsurface=15, basis="Y", r_growing=5, r_end=5, inject_style="unitary"
    )
    circuit = _apply_noise(circuit, noise_strength)
    return convert_circuit(circuit, is_d5=True, is_e2e=True)


# ===================================================================
# EXP_VAL variants for Phase 2 analytical fidelity
# ===================================================================


def make_d3_end2end_expval(noise_strength: float = 0.0) -> str:
    """Generate d=3 end-to-end T-gate circuit with EXP_VAL probes.

    Inserts EXP_VAL X_L, Y_L, Z_L before the final MPP for analytical
    fidelity extraction. The final MPP is left unchanged to preserve
    detector record indices.
    """
    circuit = make_end2end_cultivation_circuit(
        dcolor=3, dsurface=15, basis="Y", r_growing=3, r_end=5, inject_style="unitary"
    )
    circuit = _apply_noise(circuit, noise_strength)
    return convert_circuit(circuit, is_d5=False, is_e2e=True, insert_exp_val=True)


def make_d5_end2end_expval(noise_strength: float = 0.0) -> str:
    """Generate d=5 end-to-end T-gate circuit with EXP_VAL probes."""
    circuit = make_end2end_cultivation_circuit(
        dcolor=5, dsurface=15, basis="Y", r_growing=5, r_end=5, inject_style="unitary"
    )
    circuit = _apply_noise(circuit, noise_strength)
    return convert_circuit(circuit, is_d5=True, is_e2e=True, insert_exp_val=True)


def _make_sproxy_end2end_expval(dcolor: int, noise_strength: float = 0.0) -> str:
    """Generate S-proxy end-to-end circuit with EXP_VAL probes.

    This is the Clifford S-gate proxy (no S→T substitution). EXP_VAL
    probes are inserted by directly manipulating the stim circuit text,
    since the circuit contains no non-Clifford gates.
    """
    r_growing = dcolor  # r_growing = d1 per step1_make_circuits
    circuit = make_end2end_cultivation_circuit(
        dcolor=dcolor,
        dsurface=15,
        basis="Y",
        r_growing=r_growing,
        r_end=5,
        inject_style="unitary",
    )
    circuit = _apply_noise(circuit, noise_strength)

    # Insert EXP_VAL probes before the final MPP in the flattened circuit
    lines = str(circuit.flattened()).splitlines()
    final_mpp_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("MPP "):
            final_mpp_idx = i

    if final_mpp_idx is None:
        raise ValueError("No MPP found in circuit")

    final_mpp = lines[final_mpp_idx].strip()
    products = final_mpp[4:].split()
    yl_product = products[0]
    xl_str, yl_str, zl_str = _decompose_yl_product(yl_product)

    # Insert EXP_VAL probes just before the final MPP
    output = lines[:final_mpp_idx]
    output.append(f"EXP_VAL {xl_str}")
    output.append(f"EXP_VAL {yl_str}")
    output.append(f"EXP_VAL {zl_str}")
    output.extend(lines[final_mpp_idx:])

    return "\n".join(output)


def make_d3_end2end_expval_sproxy(noise_strength: float = 0.0) -> str:
    """Generate d=3 S-proxy end-to-end circuit with EXP_VAL probes."""
    return _make_sproxy_end2end_expval(3, noise_strength)


def make_d5_end2end_expval_sproxy(noise_strength: float = 0.0) -> str:
    """Generate d=5 S-proxy end-to-end circuit with EXP_VAL probes."""
    return _make_sproxy_end2end_expval(5, noise_strength)


# ===================================================================
# Validation
# ===================================================================


def validate_circuit(circuit_str: str, label: str = ""):
    """Validate a circuit string using Clifft noiseless simulation.

    Uses RemoveNoisePass so that noisy circuit strings can also be
    validated (noise is stripped before the reference shot).
    """
    import clifft

    print(f"Validating {label}...")
    hir_passes = clifft.HirPassManager()
    hir_passes.add(clifft.RemoveNoisePass())
    prog = clifft.compile(
        circuit_str,
        normalize_syndromes=True,
        hir_passes=hir_passes,
        bytecode_passes=clifft.default_bytecode_pass_manager(),
    )
    meas, det, obs = clifft.sample(prog, shots=10000)

    error_rate = np.mean(obs)
    print(f"  Logical Error Rate: {error_rate}")
    assert error_rate == 0.0 or error_rate == 1.0, f"State corrupted! Got {error_rate}"

    assert not np.any(det), "A detector fired in a noiseless simulation!"
    print(f"  SClifftESS: {label} passes noiseless validation.")


# ===================================================================
# Main
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate T-gate magic state cultivation circuits for Clifft."
    )
    parser.add_argument(
        "--noise_strength",
        type=float,
        default=0.0,
        help="Uniform depolarizing noise strength (default: 0.0 = noiseless)",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip noiseless validation (automatically set when noise > 0)",
    )
    args = parser.parse_args()
    p = args.noise_strength
    skip_validation = args.skip_validation or p > 0

    if p > 0:
        noise_tag = f"_p{p}"
        print(f"Generating T-gate circuits with noise p={p}...")
    else:
        noise_tag = ""
        print("Generating T-gate circuits (noiseless)...")
    print()

    variants = [
        ("d3_inject_cultivate", make_d3_inject_cultivate),
        ("d5_inject_cultivate", make_d5_inject_cultivate),
        ("d3_end2end", make_d3_end2end),
        ("d5_end2end", make_d5_end2end),
    ]

    results = {}
    for name, func in variants:
        circuit_str = func(noise_strength=p)
        filename = f"out_{name}{noise_tag}.clifft"
        with open(filename, "w") as f:
            f.write(circuit_str)
        print(f"Written {filename} ({len(circuit_str.splitlines())} lines)")
        results[name] = circuit_str

    if not skip_validation:
        print()
        print("Validating circuits (noiseless)...")
        for name, circuit_str in results.items():
            validate_circuit(circuit_str, name)
    else:
        print()
        print("Skipping noiseless validation (noise > 0 or --skip_validation).")
        print("Detectors are expected to fire under noise.")


if __name__ == "__main__":
    main()
