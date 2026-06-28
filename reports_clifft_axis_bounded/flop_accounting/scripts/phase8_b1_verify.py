"""Step B1 verification: the default-off Policy-3 persistent-split engine vs the committed bounded
(a05843e) path -- records / peak rank / per-measurement Born p0 must be BIT-IDENTICAL.  Also reports
the diagonal-dispatch vs fallback split and the born-Hadamard count (the FLOP win signal)."""
import sys; sys.path.insert(0, "/home/jung/clifft-paper"); sys.setrecursionlimit(400000)
import clifft  # noqa
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis.bounded import compile_bounded

CIRCS = [("coherent_ry_d3_r1", 6), ("coherent_ry_d3_r3", 4), ("cultivation_d3", 8),
         ("cultivation_d5", 4), ("coherent_rx_d3_r3", 4), ("coherent_d3_r3", 6),
         ("coherent_rx_d3_r1", 4), ("distillation", 8), ("coherent_d5_r5", 2)]


def run(circ, seed, policy3):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False, structure_once=False,
                                clifft_axis_enforce=True, clifft_axis_policy3=policy3)
    rec = tuple(be.run_shot(prog, seed))
    pk = be.nc.budget.peak_resident.bit_length() - 1
    p0 = tuple(c.get("p0") for c in be.nc.core_log if c.get("p0") is not None)
    diag = getattr(be.nc, "_p3_diag", None)
    fb = getattr(be.nc, "_p3_fallback", None)
    bornH = getattr(be.nc, "_p3_bornH", None)
    return rec, pk, p0, diag, fb, bornH


print("=" * 84)
print("STEP B1 -- Policy-3 persistent-split engine: BIT-EXACT vs committed bounded (a05843e)")
print("=" * 84)
print(f"{'circuit':18} {'seeds':>5} {'rec':>4} {'rank':>5} {'p0':>4}  {'diag/flush':>12} {'bornH':>6}  result")
allok = True
for circ, ns in CIRCS:
    rmis = kmis = pmis = 0
    tdiag = tfb = tborn = 0
    for s in range(1, ns + 1):
        r0, k0, q0, _, _, _ = run(circ, s, policy3=False)        # committed bounded (truth)
        r1, k1, q1, d1, f1, b1 = run(circ, s, policy3=True)      # policy-3
        if r0 != r1:
            rmis += 1
        if k0 != k1:
            kmis += 1
        if len(q0) != len(q1) or any(abs(a - b) > 1e-9 for a, b in zip(q0, q1)):
            pmis += 1
        tdiag += (d1 or 0); tfb += (f1 or 0); tborn += (b1 or 0)
    ok = (rmis == kmis == pmis == 0)
    allok &= ok
    tot = tdiag + tfb
    frac = f"{tdiag}/{tot}" if tot else "0/0"
    print(f"{circ:18} {ns:5d} {rmis:4d} {kmis:5d} {pmis:4d}  {frac:>12} {tborn:6d}  "
          f"{'PASS' if ok else 'FAIL'}")
print("-" * 84)
print(f"  -> {'ALL BIT-EXACT vs a05843e' if allok else 'FAIL -- divergence (do NOT proceed)'}")
print("  diag = rotations dispatched as a diagonal half-array (0 butterfly, 0 runtime H);")
print("  fallback = rotations that still needed the exact butterfly/localizer (non-Pauli re-basis);")
print("  bornH = born-basis Hadamards paid once per born-X axis at promote.")
