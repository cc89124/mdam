"""
coherent noise surface code 회로 생성.
d=7 r=1, d=7 r=7 두 가지.

실행:
    python3 generate_circuits.py

출력:
    ~/clifft-paper/qec_bench/circuits/coherent_d7_r1.stim
    ~/clifft-paper/qec_bench/circuits/coherent_d7_r7.stim
"""

import stim
import re
from pathlib import Path

OUTPUT_DIR = Path("~/clifft-paper/qec_bench/circuits").expanduser()


def make_coherent(d: int, r: int, p: float = 1e-3, rz: float = 0.02) -> str:
    """Surface code + R_Z coherent noise 회로 생성."""
    c = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=r,
        distance=d,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )
    lines = []
    for line in str(c).split("\n"):
        s = line.strip()
        if s.startswith("DEPOLARIZE1(") or s.startswith("DEPOLARIZE2("):
            targets = s.split(")")[1].strip()
            lines.append(f"R_Z({rz}) {targets}")
        else:
            lines.append(line)
    return "\n".join(lines)


def main():
    configs = [
        (7, 1),
        (7, 7),
    ]

    for d, r in configs:
        text = make_coherent(d, r)
        out = OUTPUT_DIR / f"coherent_d{d}_r{r}.stim"
        out.write_text(text)

        # qubit 수 확인 (R_Z를 stim이 읽을 수 있는 형태로 임시 변환)
        text_stim = re.sub(r"R_Z\([^)]+\)", "DEPOLARIZE1(0.001)", text)
        c = stim.Circuit(text_stim)
        print(f"saved: {out.name}  (qubits={c.num_qubits})")


if __name__ == "__main__":
    main()