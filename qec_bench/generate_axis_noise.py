"""Coherent-noise surface-code circuits with the coherent rotation on the X or Y axis
(instead of the default Z).  Same topology/strength as coherent_d{d}_r{r}.stim
(p=1e-3, angle=0.02); only the single-qubit coherent over-rotation axis changes:
  R_Z(0.02)  ->  R_X(0.02)   (coherent_rx_d{d}_r{r}.stim)
  R_Z(0.02)  ->  R_Y(0.02)   (coherent_ry_d{d}_r{r}.stim)
"""
import stim, re
from pathlib import Path

OUT = Path("~/clifft-paper/qec_bench/circuits").expanduser()
CONFIGS = [(3, 1), (3, 3), (5, 1), (5, 5)]    # same set as the per-step report's coherent_*


def make(d, r, axis, p=1e-3, ang=0.02):
    c = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=r, distance=d,
                               after_clifford_depolarization=p,
                               after_reset_flip_probability=p,
                               before_measure_flip_probability=p,
                               before_round_data_depolarization=p)
    out = []
    for line in str(c).split("\n"):
        s = line.strip()
        if s.startswith("DEPOLARIZE1(") or s.startswith("DEPOLARIZE2("):
            out.append(f"R_{axis}({ang}) {s.split(')')[1].strip()}")
        else:
            out.append(line)
    return "\n".join(out)


for axis in ("X", "Y"):
    for d, r in CONFIGS:
        txt = make(d, r, axis)
        name = f"coherent_r{axis.lower()}_d{d}_r{r}.stim"
        (OUT / name).write_text(txt)
        stimtxt = re.sub(rf"R_{axis}\([^)]+\)", "DEPOLARIZE1(0.001)", txt)
        print(f"saved {name}  qubits={stim.Circuit(stimtxt).num_qubits}")
