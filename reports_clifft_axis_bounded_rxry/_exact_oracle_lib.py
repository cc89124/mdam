"""Shared exact-oracle library for the R_Y deterministic Born validation.

Provides:
  Dense          - exact statevector oracle (R_Y in clifft TURNS: exp(-i t*pi Y/2))
  capture_backend(prog, seed) -> per-cidx (p0, engine_bit b, physical outcome m_abs)
                   from the FULL bounded backend path (compiler + frame + engine).
  phys_p0(cap)   - backend physical Born P(outcome=0) = p0 if (b^m_abs)==0 else 1-p0
  parse_stim / build_det_circuit / clean_for_clifft

All comparisons are DETERMINISTIC Born probabilities (no sampling).
"""
import sys, re
import numpy as np
sys.path.insert(0, "/home/jung/clifft-paper")
sys.setrecursionlimit(400000)
import nearclifford_backend.backend as bk
import ttn_backend.frame_layer as fl
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as B

# ---------------------------------------------------------------- dense oracle
I2 = np.eye(2, dtype=complex)
Xm = np.array([[0, 1], [1, 0]], dtype=complex)
Ym = np.array([[0, -1j], [1j, 0]], dtype=complex)
Zm = np.array([[1, 0], [0, -1]], dtype=complex)
Hm = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


class Dense:
    """Exact statevector. Qubits are 0..n-1 (caller remaps original indices)."""
    def __init__(self, n):
        self.n = n
        self.v = np.zeros(1 << n, dtype=complex)
        self.v[0] = 1.0

    def _op1(self, g, q):
        self.v = self.v.reshape([2] * self.n)
        self.v = np.moveaxis(np.tensordot(g, np.moveaxis(self.v, self.n - 1 - q, 0),
                                          axes=(1, 0)), 0, self.n - 1 - q)
        self.v = self.v.reshape(-1)

    def x(self, q):
        self._op1(Xm, q)

    def h(self, q):
        self._op1(Hm, q)

    def ry_turns(self, q, t):
        th = t * np.pi                       # clifft R_Y(t) = exp(-i t*pi Y/2)
        self._op1(np.cos(th / 2) * I2 - 1j * np.sin(th / 2) * Ym, q)

    def rx_turns(self, q, t):
        th = t * np.pi                       # clifft R_X(t) = exp(-i t*pi X/2)
        self._op1(np.cos(th / 2) * I2 - 1j * np.sin(th / 2) * Xm, q)

    def cx(self, c, t):
        d = 1 << self.n
        out = np.empty_like(self.v)
        for i in range(d):
            out[(i ^ (1 << t)) if (i >> c) & 1 else i] = self.v[i]
        self.v = out

    def born_p0(self, q):
        self.v = self.v.reshape([2] * self.n)
        idx = [slice(None)] * self.n
        idx[self.n - 1 - q] = 0
        p = float(np.sum(np.abs(self.v[tuple(idx)]) ** 2))
        self.v = self.v.reshape(-1)
        return p

    def project(self, q, o):
        self.v = self.v.reshape([2] * self.n)
        idx = [slice(None)] * self.n
        idx[self.n - 1 - q] = 1 - o
        self.v[tuple(idx)] = 0.0
        self.v = self.v.reshape(-1)
        nrm = np.linalg.norm(self.v)
        if nrm > 1e-300:
            self.v /= nrm

    def reset0(self, q):
        """force qubit to |0> (after measurement outcome o it is |o>; reset = X if o==1)."""
        p0 = self.born_p0(q)
        if p0 < 0.5:                          # deterministically in |1>
            self.x(q)


# ------------------------------------------------------------- circuit parsing
def flatten_repeat(text):
    """Expand stim REPEAT n { ... } blocks (R_Y prevents using stim's own flattener)."""
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        m = re.match(r"REPEAT\s+(\d+)\s*\{", s)
        if m:
            n = int(m.group(1))
            depth = 1
            body = []
            i += 1
            while i < len(lines) and depth > 0:
                t = lines[i].strip()
                if t.endswith("{") and t.startswith("REPEAT"):
                    depth += 1
                    body.append(lines[i])
                elif t == "}":
                    depth -= 1
                    if depth > 0:
                        body.append(lines[i])
                else:
                    body.append(lines[i])
                i += 1
            block = "\n".join(body)
            for _ in range(n):
                out.append(flatten_repeat(block))
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def parse_stim(path_or_text, is_text=False):
    """Return (qubits_used_sorted, ops). ops is an ordered list of tuples:
       ('R',[q..]) ('XE',idx,[q..]) ('RY',turns,[q..]) ('H',[q..])
       ('CX',[(c,t)..]) ('MR',[q..]) ('M',[q..])
    Annotations (DETECTOR/OBSERVABLE/QUBIT_COORDS/SHIFT/TICK) are dropped."""
    text = path_or_text if is_text else open(path_or_text).read()
    text = flatten_repeat(text)
    used = set()
    ops = []
    xe_idx = 0
    for line in text.split("\n"):
        s = line.strip()
        if not s or s.startswith(("QUBIT_COORDS", "DETECTOR", "OBSERVABLE",
                                  "SHIFT_COORDS", "TICK")):
            continue
        head = s.split()[0]
        if head.startswith("X_ERROR"):
            qs = [int(t) for t in re.findall(r"\d+", s.split(")")[1])]
            ops.append(("XE", xe_idx, qs)); xe_idx += 1; used.update(qs)
        elif head.startswith("R_Y"):
            turns = float(re.findall(r"\(([^)]+)\)", s)[0])
            qs = [int(t) for t in re.findall(r"\d+", s.split(")")[1])]
            ops.append(("RY", turns, qs)); used.update(qs)
        elif head.startswith("R_X"):
            turns = float(re.findall(r"\(([^)]+)\)", s)[0])
            qs = [int(t) for t in re.findall(r"\d+", s.split(")")[1])]
            ops.append(("RX", turns, qs)); used.update(qs)
        elif head == "R":
            qs = [int(t) for t in s.split()[1:]]
            ops.append(("R", qs)); used.update(qs)
        elif head == "H":
            qs = [int(t) for t in s.split()[1:]]
            ops.append(("H", qs)); used.update(qs)
        elif head == "X":
            qs = [int(t) for t in s.split()[1:]]
            ops.append(("X", qs)); used.update(qs)
        elif head == "CX":
            ts = [int(t) for t in s.split()[1:]]
            ops.append(("CX", list(zip(ts[0::2], ts[1::2])))); used.update(ts)
        elif head == "MR":
            qs = [int(t) for t in s.split()[1:]]
            ops.append(("MR", qs)); used.update(qs)
        elif head == "M":
            qs = [int(t) for t in s.split()[1:]]
            ops.append(("M", qs)); used.update(qs)
        else:
            raise ValueError(f"unhandled stim line: {s!r}")
    return sorted(used), ops


def fault_sites(ops):
    """flat list of (xe_idx, qubit) over all X_ERROR targets."""
    sites = []
    for op in ops:
        if op[0] == "XE":
            for q in op[2]:
                sites.append((op[1], q))
    return sites


def measurement_qubits(ops):
    """program-order list of measured qubits (cidx 0..nm-1)."""
    out = []
    for op in ops:
        if op[0] in ("MR", "M"):
            out.extend(op[1])
    return out


def build_det_text(ops, faultset):
    """reconstruct a stim text with X_ERROR replaced by explicit X on faulted targets.
       faultset: set of (xe_idx, qubit) that fire."""
    lines = []
    for op in ops:
        k = op[0]
        if k == "XE":
            fired = [q for q in op[2] if (op[1], q) in faultset]
            if fired:
                lines.append("X " + " ".join(map(str, fired)))
        elif k == "R":
            lines.append("R " + " ".join(map(str, op[1])))
        elif k == "RY":
            lines.append(f"R_Y({op[1]}) " + " ".join(map(str, op[2])))
        elif k == "RX":
            lines.append(f"R_X({op[1]}) " + " ".join(map(str, op[2])))
        elif k == "H":
            lines.append("H " + " ".join(map(str, op[1])))
        elif k == "CX":
            lines.append("CX " + " ".join(f"{c} {t}" for c, t in op[1]))
        elif k == "MR":
            lines.append("MR " + " ".join(map(str, op[1])))
        elif k == "M":
            lines.append("M " + " ".join(map(str, op[1])))
    return "\n".join(lines)


# ------------------------------------------------------- bounded backend capture
def capture_backend(prog, seed):
    """Run ONE bounded-backend shot through the FULL backend path.  Returns
       (seq, record) where:
         seq    = list of (cidx, p0, b) in BACKEND EXECUTION ORDER (one per measure_z),
                  p0 = engine P(bit=0), b = realized engine bit.
         record = dict cidx -> realized record bit r (the stim/clifft record convention).
    Backend conditional Born:  P(record_cidx = 0 | exec-prefix) = p0 if (b^r)==0 else 1-p0."""
    nm = prog.num_measurements
    EV = []                                  # (kind, step, payload) in execution order
    SNAP = {}                                # step -> record copy (before that step)

    o_mz = B.measure_z
    def mz(self, q):
        b = o_mz(self, q)
        p0 = self.core_log[-1]["p0"] if (getattr(self, "log_cores", False)
                                         and self.core_log) else None
        EV.append(("mz", _CUR["be"]._cur_step, (int(b), p0)))
        return b
    B.measure_z = mz

    o_sx = fl.PauliFrame.set_xz
    def sx(self, s, x, z=0):
        EV.append(("sx", _CUR["be"]._cur_step, int(x) & 1))
        return o_sx(self, s, x, z)
    fl.PauliFrame.set_xz = sx

    o_reset = bk.NearCliffordBackend._reset
    def reset(self, prog2):
        o_reset(self, prog2)
        if getattr(self, "clifft_axis_bounded", False):
            self.nc.log_cores = True
            self.nc.core_log = []
    bk.NearCliffordBackend._reset = reset

    be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                structure_once=False, clifft_axis_enforce=True)
    _CUR["be"] = be
    def recorder(step, s):
        SNAP[step] = dict(s.record)
    try:
        be.run_shot(prog, seed, step_recorder=recorder)
    finally:
        B.measure_z = o_mz
        fl.PauliFrame.set_xz = o_sx
        bk.NearCliffordBackend._reset = o_reset

    record = {k: int(v) for k, v in be.record.items() if 0 <= k < nm}
    # active measurements: each 'mz' (engine measure_z) at some step, with (b, p0)
    active = {}                          # step -> (b, p0)
    i = 0
    while i < len(EV):
        if EV[i][0] == "mz":
            assert i + 1 < len(EV) and EV[i + 1][0] == "sx", "mz not followed by sx"
            active[EV[i][1]] = EV[i][2]      # step -> (b, p0)
            i += 2
        else:
            i += 1
    # walk ALL steps in order; any step that added a real cidx is a measurement (active
    # if it had a measure_z, else DORMANT = deterministic frame read).
    steps = sorted(SNAP)
    seq = []                             # exec order: (cidx, p0|None, b|None)
    for idx, step in enumerate(steps):
        before = SNAP[step]
        after = SNAP[steps[idx + 1]] if idx + 1 < len(steps) else dict(be.record)
        newk = [k for k in after if k not in before and 0 <= k < nm]
        for c in newk:
            if step in active:
                b, p0 = active[step]
                seq.append((c, p0, b))
            else:
                seq.append((c, None, None))     # dormant: deterministic outcome = record[c]
    assert len(seq) == nm, f"captured {len(seq)} != {nm} measurements"
    return seq, record


_CUR = {"be": None}


def backend_record_p0(p0, b, r):
    """P(record bit = 0 | exec-prefix) for the bounded backend.
       Dormant (p0 is None): deterministic outcome = r, so P(record=0) = 1.0 iff r==0."""
    if p0 is None:
        return 1.0 if r == 0 else 0.0
    return p0 if (b ^ r) == 0 else (1.0 - p0)


def build_clean_det_text(ops, faultset):
    """Deterministic circuit for clifft.record_probabilities: drop initial R, MR->M
    (ancilla unused after reset => identical measurement statistics), no detectors."""
    lines = []
    for op in ops:
        k = op[0]
        if k == "XE":
            fired = [q for q in op[2] if (op[1], q) in faultset]
            if fired:
                lines.append("X " + " ".join(map(str, fired)))
        elif k == "R":
            continue
        elif k == "RY":
            lines.append(f"R_Y({op[1]}) " + " ".join(map(str, op[2])))
        elif k == "RX":
            lines.append(f"R_X({op[1]}) " + " ".join(map(str, op[2])))
        elif k == "H":
            lines.append("H " + " ".join(map(str, op[1])))
        elif k == "X":
            lines.append("X " + " ".join(map(str, op[1])))
        elif k == "CX":
            lines.append("CX " + " ".join(f"{c} {t}" for c, t in op[1]))
        elif k in ("MR", "M"):
            lines.append("M " + " ".join(map(str, op[1])))
    return "\n".join(lines)
