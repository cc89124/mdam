"""Step B0 focus checks (Sec.5): the diagonal-dispatch RULE  T X^x Z^z = omega^x X^x Z^z T^{(-1)^x}
verified (a) as a standalone gate-level matrix identity for every residue, and (b) per-T against the
ACTUAL butterfly's effective single-axis action for the focus categories (p_x=p_z=1 / Y residue /
i^p phase / weight 3,5,7 / first-T-after-promote / last-T-before-measure), plus a q5/q14 lifetime
trace.  No engine change; cultivation_d5 source-of-truth = the off-diagonal butterfly."""
import sys; sys.path.insert(0, "/home/jung/clifft-paper"); sys.setrecursionlimit(400000)
import copy
import numpy as np
import clifft  # noqa
import nearclifford_backend.backend as bk
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

sys.path.insert(0, "/home/jung/clifft-paper/reports_clifft_axis_bounded/flop_accounting/scripts")
from phase8_step_b0 import candidate_decompose, candidate_apply, OMEGA

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
Tg = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)   # gate T = diag(1, e^{i pi/4})
Tgd = Tg.conj().T


def pauli2(x, z):
    m = I2
    if x and z:
        m = X @ Z                       # = -i Y
    elif x:
        m = X
    elif z:
        m = Z
    return m


print("=" * 78)
print("STEP B0 / Sec.5 -- diagonal-dispatch RULE verification")
print("=" * 78)

# ---- (a) standalone gate-level identity: T X^x Z^z = omega^x X^x Z^z T^{(-1)^x} ----
print("\n(a) gate identity  T X^x Z^z = omega^x X^x Z^z T^{(-1)^x}   (omega = e^{i pi/4}):")
for x in (0, 1):
    for z in (0, 1):
        P = pauli2(x, z)
        lhs = Tg @ P
        rhs = (OMEGA ** x) * (P @ (Tgd if x else Tg))
        err = float(np.max(np.abs(lhs - rhs)))
        tag = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "XZ(=-iY)"}[(x, z)]
        print(f"   residue {tag:9}  x={x} z={z}: T->T^{'dag' if x else '+ '}   ||LHS-RHS|| = {err:.2e}")

# ---- per-T focus trace for cultivation_d5 ----
print("\n(b) per-T focus trace (cultivation_d5, seed 1): pivot qubit, born basis, residue, rule-vs-actual")
circ = "cultivation_d5"
prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
of1 = C._flush_one
rows = []
ctx = {"last_was_meas": True, "promote_pending": False}
ofc = C.measure_z


def f1(self, x, z, theta, phase=0):
    pre = copy.deepcopy(self)
    k_before = len(self.M)
    r = of1(self, x, z, theta, phase)
    cand = copy.deepcopy(pre); cand.budget.enforce = False
    meta = candidate_decompose(cand, x, z, theta, phase)
    promoted = (len(cand.M) > k_before)
    pivot_q = cand.M[meta["a"]] if meta.get("a") is not None else None
    # per-T 2x2 rule-vs-actual on the collapsed axis:
    #   actual effective single-axis op = exp(-i theta/2 * s * Z) ; rule = gamma * (T if s>0 else Tdag)
    s = meta["s_sign"]; g = meta["gamma"]
    actual = np.array([[np.exp(-1j * theta / 2 * s), 0], [0, np.exp(1j * theta / 2 * s)]], dtype=complex)
    rule = g * (Tg if (s * theta) > 0 else Tgd)
    rule_err = float(np.max(np.abs(actual - rule)))
    rows.append(dict(idx=len(rows), rank=pre.phi.size.bit_length() - 1, pivot_q=pivot_q,
                     born=meta["born"], px=meta["px"], pz=meta["pz"], pp=meta["pp"],
                     s=s, weight=meta["weight"], theta=round(float(theta), 4),
                     promoted=promoted, last_before_meas=False, rule_err=rule_err))
    ctx["last_was_meas"] = False
    return r


def mz(self, q):
    if rows and not ctx["last_was_meas"]:
        rows[-1]["last_before_meas"] = True       # mark the most recent T as last-before-this-measure
    ctx["last_was_meas"] = True
    return ofc(self, q)


C._flush_one = f1; C.measure_z = mz
try:
    be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                structure_once=False, clifft_axis_enforce=True)
    be.run_shot(prog, 1)
finally:
    C._flush_one = of1; C.measure_z = ofc

# census
cats = dict(Yresidue=0, pp_nonzero=0, w3=0, w5=0, w7=0, first_after_promote=0, last_before_meas=0)
for d in rows:
    if d["px"] and d["pz"]:
        cats["Yresidue"] += 1
    if d["pp"] != 0:
        cats["pp_nonzero"] += 1
    if d["weight"] == 3:
        cats["w3"] += 1
    if d["weight"] == 5:
        cats["w5"] += 1
    if d["weight"] == 7:
        cats["w7"] += 1
    if d["promoted"]:
        cats["first_after_promote"] += 1
    if d["last_before_meas"]:
        cats["last_before_meas"] += 1
maxrule = max(d["rule_err"] for d in rows)
print(f"   total T = {len(rows)}   max rule-vs-actual 2x2 error = {maxrule:.2e}")
print(f"   focus-category census: {cats}")
wd = {}
for d in rows:
    wd[d["weight"]] = wd.get(d["weight"], 0) + 1
print(f"   weight distribution: {dict(sorted(wd.items()))}")
ppd = {}
for d in rows:
    ppd[d["pp"]] = ppd.get(d["pp"], 0) + 1
print(f"   i^pp (frame-phase) distribution: {dict(sorted(ppd.items()))}")
sd = {}
for d in rows:
    sd[d["s"]] = sd.get(d["s"], 0) + 1
print(f"   collapse sign (s=+1 -> T, s=-1 -> T^dag) distribution: {sd}")

# representative focus rows
print("\n   representative focus rows (idx rank pivot_q born px pz pp s weight theta promote last_meas rule_err):")
shown = 0
seen = set()
for d in rows:
    key = (d["weight"], d["promoted"], d["last_before_meas"], d["pp"])
    if key in seen and shown >= 12:
        continue
    if d["weight"] in (3, 5, 7) or d["promoted"] or d["last_before_meas"] or d["pp"] != 0 or (d["px"] and d["pz"]):
        seen.add(key)
        print(f"     {d['idx']:3d}  r={d['rank']:2d}  q{str(d['pivot_q']):>3}  {d['born']}  "
              f"px={d['px']} pz={d['pz']} pp={d['pp']}  s={d['s']:+d}  w={d['weight']}  "
              f"th={d['theta']:+.3f}  prom={int(d['promoted'])} lastM={int(d['last_before_meas'])}  "
              f"rule_err={d['rule_err']:.1e}")
        shown += 1
        if shown >= 16:
            break

# ---- q5 / q14 lifetime trace ----
print("\n(c) q5 / q14 lifetime trace (the two qubits Phase-5 flagged as carrying both X- and Z-type):")
for qf in (5, 14):
    seq = [(d["idx"], d["born"], d["px"], d["s"], d["weight"], d["theta"]) for d in rows if d["pivot_q"] == qf]
    print(f"   q{qf}: {len(seq)} T's as pivot  -> " +
          ", ".join(f"[{i}:{b}{'T' if s > 0 else 'Td'}w{w}]" for i, b, px, s, w, th in seq[:18]))
