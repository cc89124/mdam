#!/usr/bin/env python
"""Gate L0 -- native VM opcode coverage inventory.

Dry-run every benchmark's clifft-compiled program against the native translate() supported-opcode set
(verify_mdam_oneshot.translate).  For each circuit report: peak_rank, total instructions, whether
translate() would succeed, and a per-opcode count of every UNSUPPORTED opcode (with first index)."""
import os, sys, collections
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "native_vm")))
import clifft
from mdam.frame import frame_layer as fl
from mdam.backend.backend import _opname

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# The exact opcode names translate() emits a native op for (everything else -> "unsupported opcode").
SUPPORTED = {
    "OP_FRAME_H", "OP_FRAME_CNOT", "OP_FRAME_CZ", "OP_FRAME_SWAP", "OP_FRAME_S", "OP_FRAME_S_DAG",
    "OP_APPLY_PAULI", "OP_NOISE", "OP_NOISE_BLOCK", "OP_READOUT_NOISE",
    "OP_MEAS_DORMANT_STATIC", "OP_MEAS_DORMANT_STATIC_FORCED",
    "OP_MEAS_DORMANT_RANDOM", "OP_MEAS_DORMANT_RANDOM_FORCED",
    "OP_ARRAY_CNOT", "OP_ARRAY_CZ", "OP_ARRAY_MULTI_CNOT", "OP_ARRAY_MULTI_CZ",
    "OP_ARRAY_T", "OP_ARRAY_T_DAG", "OP_ARRAY_S", "OP_EXPAND_T", "OP_EXPAND_T_DAG",
    "OP_SWAP_MEAS_INTERFERE", "OP_SWAP_MEAS_INTERFERE_FORCED",
}
IGNORED = set(fl.IGNORE_OPS)   # translate() skips these (continue) -> harmless

BENCHES = ["coherent_d3_r1", "coherent_d3_r3", "coherent_d5_r1", "coherent_d5_r5", "surface_d7_r7",
           "coherent_rx_d3_r1", "coherent_rx_d3_r3", "coherent_rx_d5_r1", "coherent_rx_d5_r5",
           "coherent_ry_d3_r1", "coherent_ry_d3_r3", "coherent_ry_d5_r1", "coherent_ry_d5_r5",
           "cultivation_d3", "cultivation_d5", "distillation"]


def axis_of(c):
    if "rx" in c: return "R_X"
    if "ry" in c: return "R_Y"
    if c.startswith("cultivation") or c.startswith("distill"): return "T"
    return "R_Z"


global_unsup = collections.Counter()
rows = []
print(f"{'circuit':20}{'axis':5}{'k':>3}{'#instr':>8}  translate?  unsupported opcodes (count, first idx)")
print("-" * 100)
for b in BENCHES:
    text = open(f"{ROOT}/qec_bench/circuits/{b}.stim").read()
    prog = clifft.compile(text)
    k = getattr(prog, "peak_rank", 0)
    unsup = collections.Counter(); first = {}
    for s in range(len(prog)):
        nm = _opname(prog[s].opcode)
        if nm in SUPPORTED or nm in IGNORED: continue
        unsup[nm] += 1
        if nm not in first: first[nm] = s
    ok = (len(unsup) == 0)
    global_unsup.update(unsup)
    detail = "OK" if ok else ", ".join(f"{op}({c}@{first[op]})" for op, c in unsup.most_common())
    print(f"{b:20}{axis_of(b):5}{k:>3}{len(prog):>8}  {'YES' if ok else 'NO ':10}  {detail}")
    rows.append((b, axis_of(b), k, len(prog), ok, dict(unsup)))

print("-" * 100)
print("AGGREGATE unsupported opcodes across all benchmarks:")
for op, c in global_unsup.most_common():
    print(f"  {op:24} total={c}")
nsupported = sum(1 for r in rows if r[4])
print(f"\ntranslate() currently supports {nsupported}/{len(BENCHES)} benchmarks.")
print("blocked benchmarks:", [r[0] for r in rows if not r[4]])
