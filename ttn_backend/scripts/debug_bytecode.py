import clifft
from pathlib import Path

circuit_path = Path("~/clifft-paper/qec_bench/circuits/distillation.stim").expanduser()
circuit_text = circuit_path.read_text()

prog = clifft.compile(
    circuit_text,
    hir_passes=clifft.default_hir_pass_manager(),
    bytecode_passes=clifft.default_bytecode_pass_manager(),
)

bytecode = prog.as_dict()["bytecode"]

print("peak_rank =", prog.peak_rank)

for i, inst in enumerate(bytecode):
    op = inst.get("opcode", "")
    if "SWAP" in op or "MEAS" in op or "ARRAY" in op:
        print(f"[{i:03d}] {inst}")