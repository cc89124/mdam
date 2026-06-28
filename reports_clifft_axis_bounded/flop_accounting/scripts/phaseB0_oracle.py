"""Phase B0/B1/B2 measurement reference oracle (cultivation_d3, n small).

For EACH active OP_MEAS_ACTIVE_DIAGONAL, on the SHARED pre-measurement physical state
(reduced engine, U_C identity => nc.statevector() IS the exact physical state, validated
2.2e-16 in Phase A), force BOTH branches b=0 and b=1 independently and compare the reduced
quantum instrument to the DENSE PROJECTOR oracle (I +- Z_q)/2:

  M_red = E_t^dag Z_q E_t : with U_C identity this is pure-Z on array bit j=M.index(q)
          (x=0 by construction).  We VERIFY x=0 numerically by p0_array == p0_dense.

  B0 operator : p0_array (branch_sqnorm on bit j) == p0_dense (Z_q on dense state) ?
  B1 project  : project array bit_j=b (KEEP axis, no drop, NO frame). dense-compare to
                (I+(-1)^b Z_q)/2 |psi> normalized.  Isolates Born+projection.
  B2 drop     : project + drop axis j (q -> |0>, removed from M) then apply the frame X_q^b
                (the embedding update). dense-compare to the same dense projector branch.
                Isolates drop + frame discipline.

No RNG sharing: each measurement's instrument is checked on its own pre-state, both branches.
"""
import sys, copy; sys.path.insert(0, "/home/jung/clifft-paper"); sys.setrecursionlimit(400000)
import numpy as np, clifft
import nearclifford_backend.backend as bk
from nearclifford_backend.backend import _opname
from ttn_backend import frame_layer as ds_mod
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

# ---------- reduced data-plane overrides (U_C identity probe) ----------
o_h=C.h;o_s=C.s;o_cx=C.cx;o_cz=C.cz;o_birth=bk.NearCliffordBackend._birth;o_rot=bk.NearCliffordBackend._rot
def rh(self,q):
    if q in self.M:self._h_axis(self.M.index(q))
    else:o_h(self,q)
def rs(self,q,dag=False):
    if q in self.M:self._s_axis(self.M.index(q),dag)
    else:o_s(self,q,dag)
def rcx(self,c,t):
    if c in self.M and t in self.M:self._cnot_axes(self.M.index(c),self.M.index(t));self.budget.charge(self.phi.size,0,"reduce:cnot")
    else:o_cx(self,c,t)
def rcz(self,a,b):
    if a in self.M and b in self.M:
        ja=self.M.index(a);jb=self.M.index(b);r=len(self.M);t=self.phi.reshape([2]*r);s=[slice(None)]*r;s[r-1-ja]=1;s[r-1-jb]=1;t[tuple(s)]*=-1.0;self.budget.charge(self.phi.size,0,"reduce:cz")
    else:o_cz(self,a,b)
def rbirth(self,slot):
    q=self._new_q(slot);self.nc._promote(q);self.nc._h_axis(self.nc.M.index(q));return q
def rrot(self,slot,angle):
    q=self.slot2id.get(slot)
    if q is None:return
    if q not in self.nc.M:self.nc._promote(q);self.nc._h_axis(self.nc.M.index(q))
    sign=-1.0 if self.frame.xb(slot) else 1.0;bit=self.nc.M.index(q);v=self.nc.phi.reshape(-1,2,1<<bit);v[:,1,:]*=np.exp(1j*sign*angle);self.nc.budget.charge(self.nc.phi.size,0,"rot:diaghalf");self._track_M()
def setup():
    C.h=rh;C.s=rs;C.cx=rcx;C.cz=rcz;bk.NearCliffordBackend._birth=rbirth;bk.NearCliffordBackend._rot=rrot
def teardown():
    C.h=o_h;C.s=o_s;C.cx=o_cx;C.cz=o_cz;bk.NearCliffordBackend._birth=o_birth;bk.NearCliffordBackend._rot=o_rot

# ---------- robust dense materialisation (NaN-free U_C; d3-scale only) ----------
def _par(u):
    u = u.copy()
    for s in (16, 8, 4, 2, 1):
        u ^= u >> s
    return u & 1

def _pauli_apply(P, v):
    """i^p X^x Z^z applied to state vector v (dim=2^n), O(dim) — exact, no matrix."""
    x, z, p = P
    a = np.arange(v.size)
    b = a ^ x
    phase = (1j ** p) * (1.0 - 2.0 * _par(z & b))
    return phase * v[b]

def robust_statevector(nc):
    """Dense physical state U_C(|0..0> with magic phi), built NaN-free: project a FIXED
    random vector (not |0>) onto the +1 eigenspace of every Zc[i] to get U_C|0>, then
    generate the populated columns by Xc images. Avoids _clifford_matrix's |0>-overlap NaN."""
    for (x, z, p, theta, uid) in list(nc.pending.values()):
        nc._flush_one(x, z, theta, p)
    nc.pending = {}
    n = nc.n; dim = 1 << n
    rngv = np.random.default_rng(12345)
    v = rngv.standard_normal(dim) + 1j * rngv.standard_normal(dim)
    for i in range(n):
        v = 0.5 * (v + _pauli_apply(nc.Zc[i], v))        # project onto +1 of Zc[i]
    u0 = v / np.linalg.norm(v)
    psi = np.zeros(dim, dtype=complex)
    for idx in range(len(nc.phi)):
        amp = nc.phi[idx]
        if abs(amp) < 1e-300:
            continue
        col = u0
        bits = 0
        for j, q in enumerate(nc.M):
            if (idx >> j) & 1:
                bits |= (1 << q)
        bi = bits
        while bi:                                         # U_C|bits> = prod Xc[i] |u0>
            i = (bi & -bi).bit_length() - 1; bi &= bi - 1
            col = _pauli_apply(nc.Xc[i], col)
        psi = psi + amp * col
    return psi

# ---------- oracle helpers ----------
def upg(a, b):
    """up-to-global-phase INFIDELITY 1-|<a^|b^>| (no sqrt amplification; ~1e-16 at machine
    precision). Also penalises a norm mismatch (un-normalised oracle branch carries p_b)."""
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-14 or nb < 1e-14:
        return abs(na - nb)
    return float(1.0 - abs(np.vdot(a/na, b/nb)))

def dense_project(psi, q, b):
    """(I+(-1)^b Z_q)/2 |psi>  (un-normalized) : keep amplitudes with bit_q == b."""
    idx = np.arange(psi.size)
    keep = ((idx >> q) & 1) == b
    return np.where(keep, psi, 0.0)

def apply_X(psi, q):
    idx = np.arange(psi.size)
    return psi[idx ^ (1 << q)]

def dense_proj_pauli(psi, q, b, basis):
    """(I + (-1)^b P_q)/2 |psi> with P=Z (basis 'Z') or P=X (basis 'X')."""
    if basis == "Z":
        return dense_project(psi, q, b)
    Xpsi = apply_X(psi, q)
    return 0.5 * (psi + ((-1)**b) * Xpsi)

def rem_vec(psi, q):
    """factor qubit q OUT (it is product in both reduced & oracle post-measurement states):
    return the 2^(n-1) vector on the other qubits, taking whichever bit_q slice carries the
    amplitude (engine resets q to |keep> via the frame; oracle leaves q in |+-> -- both give
    the same |rem> up to scale/phase)."""
    size = psi.size; idx = np.arange(size)
    low = (1 << q) - 1
    def compress(mask):
        sub = idx[mask]
        y = (sub & low) | ((sub >> (q + 1)) << q)
        out = np.zeros(size >> 1, dtype=complex); out[y] = psi[mask]
        return out
    v0 = compress(((idx >> q) & 1) == 0)
    v1 = compress(((idx >> q) & 1) == 1)
    return v0 if np.linalg.norm(v0) >= np.linalg.norm(v1) else v1

ROWS = []
def oracle_step(step, be, prog):
    if step >= len(prog):
        return
    inst = prog[step]; name = _opname(inst.opcode)
    if name in ("OP_MEAS_ACTIVE_DIAGONAL", "OP_MEAS_ACTIVE_DIAGONAL_FORCED"):
        basis = "Z"
    elif name in ("OP_MEAS_ACTIVE_INTERFERE", "OP_MEAS_ACTIVE_INTERFERE_FORCED"):
        basis = "X"
    else:
        return
    a1 = int(inst.axis_1)
    q = be.slot2id.get(a1)
    # frame parity that XORs into the record: xb for Z(diagonal), zb for X(interfere)
    fpar = int(be.frame.xb(a1)) if basis == "Z" else int(be.frame.zb(a1))
    nc0 = copy.deepcopy(be.nc)
    psi = robust_statevector(nc0)                       # exact physical pre-state (n qubits)
    npsi = np.linalg.norm(psi)
    inM = (q is not None) and (q in nc0.M)
    j = nc0.M.index(q) if inM else None
    p0d = (float(np.sum(np.abs(dense_proj_pauli(psi, q, 0, basis))**2) / (npsi**2))
           if q is not None else None)
    row = dict(meas=len(ROWS), step=step, a1=a1, q=q, j=j, inM=inM, basis=basis, fpar=fpar,
               p0_dense=p0d, p0_array=None, B1=[None, None], B2=[None, None], dp0=None)

    def reduced_branch(b, drop):
        """Replicate the backend's active measurement on a deepcopy, keeping branch b.
        For X(interfere): H-fold (nc.h=_h_axis), Z-branch project; for Z: direct.
        If drop=False, un-fold the H to return to the original basis (project-only B1).
        Returns (p_b, statevector)."""
        nc = copy.deepcopy(nc0)
        if basis == "X":
            nc.h(q)                               # _h_axis on bit j: interference fold
        jb = nc.M.index(q)
        s_b = nc._branch_sqnorm(jb, b)
        v = nc.phi.reshape(-1, 2, 1 << jb); v[:, 1-b, :] = 0.0
        if s_b > 1e-24:
            nc.phi /= s_b**0.5
        if drop:
            nc._drop_localized_core(q, b)         # drop measured axis (q -> |0>)
        elif basis == "X":
            nc.h(q)                               # un-fold: back to original basis (B1 compare)
        return s_b, robust_statevector(nc)

    if inM:
        # Born from the array (same kernels the engine uses)
        ncb = copy.deepcopy(nc0)
        if basis == "X":
            ncb.h(q)
        jb = ncb.M.index(q)
        s0 = ncb._branch_sqnorm(jb, 0); s1 = ncb._branch_sqnorm(jb, 1); tot = s0 + s1
        p0a = (s0/tot) if tot > 1e-300 else 0.5
        row["p0_array"] = p0a
        row["dp0"] = abs(p0a - p0d)
        for b in (0, 1):
            pb = (s0 if b == 0 else s1) / tot if tot > 1e-300 else 0.0
            if pb < 1e-15:                # unreachable branch (deterministic measurement)
                continue
            psib = dense_proj_pauli(psi, q, b, basis)         # dense oracle branch (un-normalized)
            # ---- B1: project only (keep axis, original basis) ----
            _, psR1 = reduced_branch(b, drop=False)
            row["B1"][b] = upg(psR1, psib)
            # ---- B2: project + drop (remaining-qubit state, q product & factored) ----
            _, psR2 = reduced_branch(b, drop=True)
            row["B2"][b] = upg(rem_vec(psR2, q), rem_vec(psib, q))
    ROWS.append(row)

def run(circ, seed):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    setup()
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        be.run_shot(prog, seed, step_recorder=lambda s, b: oracle_step(s, b, prog))
    finally:
        teardown()

if __name__ == "__main__":
    circ = sys.argv[1] if len(sys.argv) > 1 else "cultivation_d3"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    ROWS.clear()
    run(circ, seed)
    print(f"== Phase B oracle: {circ} seed {seed} : {len(ROWS)} active measurements ==")
    print(f"{'m':>3} {'q':>3} {'j':>3} {'inM':>4} {'bas':>3} {'fp':>2} {'p0_dense':>9} {'p0_arr':>9} "
          f"{'|dp0|':>9} {'B1_b0':>9} {'B1_b1':>9} {'B2_b0':>9} {'B2_b1':>9}")
    def f(x): return "   -    " if x is None else f"{x:9.2e}"
    def g(x): return "   -    " if x is None else f"{x:9.5f}"
    worst = dict(dp0=0.0, B1=0.0, B2=0.0, xviol=0, notinM=0)
    for r in ROWS:
        print(f"{r['meas']:>3} {str(r['q']):>3} {str(r['j']):>3} {str(r['inM']):>4} {r['basis']:>3} {r['fpar']:>2} "
              f"{g(r['p0_dense'])} {g(r['p0_array'])} {f(r['dp0'])} "
              f"{f(r['B1'][0])} {f(r['B1'][1])} {f(r['B2'][0])} {f(r['B2'][1])}")
        if not r["inM"]: worst["notinM"] += 1; continue
        worst["dp0"] = max(worst["dp0"], r["dp0"])
        for b in (0, 1):
            if r["B1"][b] is not None: worst["B1"] = max(worst["B1"], r["B1"][b])
            if r["B2"][b] is not None: worst["B2"] = max(worst["B2"], r["B2"][b])
    print(f"\nWORST: |dp0|={worst['dp0']:.2e}  B1={worst['B1']:.2e}  B2={worst['B2']:.2e}  "
          f"meas-not-in-M={worst['notinM']}")
    ok = worst['dp0'] < 1e-12 and worst['B1'] < 1e-12 and worst['B2'] < 1e-12 and worst['notinM'] == 0
    print("RESULT:", "PASS" if ok else "FAIL")
