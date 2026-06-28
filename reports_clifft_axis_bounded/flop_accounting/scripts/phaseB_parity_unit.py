"""Independent unit test of the B2 multi-bit Z-parity contraction (_localize_to_Z), on random
magic states + random parity masks -- the case d5 exercises but d3's dense oracle did not.
Compares the contracted single-pivot Born to the dense parity projector (I +- Z^z)/2, and the
post-projection physical state (via robust dense materialisation) to the dense parity branch."""
import sys, copy; sys.path.insert(0, "/home/jung/clifft-paper"); sys.setrecursionlimit(400000)
import numpy as np
from nearclifford_backend.clifft_axis.bounded import CliftAxisBoundedNearClifford as C

def _par(u):
    u = u.copy()
    for s in (16, 8, 4, 2, 1): u ^= u >> s
    return u & 1
def _pauli_apply(P, v):
    x, z, p = P; a = np.arange(v.size); b = a ^ x
    return ((1j ** p) * (1.0 - 2.0 * _par(z & b))) * v[b]
def upg(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-14 or nb < 1e-14: return abs(na - nb)
    return float(1.0 - abs(np.vdot(a/na, b/nb)))
def robust_statevector(nc):
    for (x, z, p, theta, uid) in list(nc.pending.values()): nc._flush_one(x, z, theta, p)
    nc.pending = {}
    n = nc.n; dim = 1 << n
    rngv = np.random.default_rng(12345)
    v = rngv.standard_normal(dim) + 1j * rngv.standard_normal(dim)
    for i in range(n): v = 0.5 * (v + _pauli_apply(nc.Zc[i], v))
    u0 = v / np.linalg.norm(v)
    psi = np.zeros(dim, dtype=complex)
    for idx in range(len(nc.phi)):
        amp = nc.phi[idx]
        if abs(amp) < 1e-300: continue
        col = u0; bits = 0
        for j, q in enumerate(nc.M):
            if (idx >> j) & 1: bits |= (1 << q)
        bi = bits
        while bi:
            i = (bi & -bi).bit_length() - 1; bi &= bi - 1
            col = _pauli_apply(nc.Xc[i], col)
        psi = psi + amp * col
    return psi

def _parvec(idx, mask):
    v = idx & mask
    for s in (16, 8, 4, 2, 1): v ^= v >> s
    return v & 1

def one(n, rmag, zbits, seed):
    rng = np.random.default_rng(seed)
    nc = C(n)
    for q in range(rmag): nc._promote(q)
    phi0 = rng.standard_normal(1 << rmag) + 1j * rng.standard_normal(1 << rmag)
    phi0 /= np.linalg.norm(phi0)
    nc.phi[:] = phi0
    zmask_q = 0
    for q in zbits: zmask_q |= (1 << q)
    # dense parity Born on phi0 (phi bit j <-> M[j]=j)
    zmask_phi = 0
    for j, q in enumerate(nc.M):
        if (zmask_q >> q) & 1: zmask_phi |= 1 << j
    idx = np.arange(1 << rmag)
    par = _parvec(idx, zmask_phi)
    p0_dense = float(np.sum(np.abs(phi0[par == 0]) ** 2))
    # dense physical pre-state (n qubits) for post-state compare
    psi_pre = robust_statevector(copy.deepcopy(nc))
    # contract: Z^z -> sign*Z_r via FP-free CNOT folds
    r, sign = nc._localize_to_Z(0, zmask_q, 0, prefer=zbits[0])
    jr = nc.M.index(r)
    s0 = nc._branch_sqnorm(jr, 0); s1 = nc._branch_sqnorm(jr, 1); tot = s0 + s1
    p0_loc = (s0 if sign > 0 else s1) / tot
    plus_bit = 0 if sign > 0 else 1
    # project outcome-0 branch (eigenvalue +1) on the localized pivot, normalize
    v = nc.phi.reshape(-1, 2, 1 << jr); v[:, 1 - plus_bit, :] = 0.0
    nc.phi /= np.linalg.norm(nc.phi)
    psi_post = robust_statevector(nc)                       # U_C now carries W (folded) -> exact phys
    # dense oracle: (I + Z^z_physical)/2 |psi_pre|, branch 0, normalized
    zq_phys = 0
    for q in zbits: zq_phys |= (1 << q)
    a = np.arange(psi_pre.size)
    zsign = 1.0 - 2.0 * _parvec(a, zq_phys)
    proj0 = 0.5 * (psi_pre + zsign * psi_pre)               # (I+Z^z)/2
    born_err = abs(p0_dense - p0_loc)
    state_err = upg(psi_post, proj0)
    return born_err, state_err, r, sign

print("B2 multi-bit parity contraction unit test (random magic states):")
worst_b = worst_s = 0.0; ncases = 0
for seed in range(40):
    rng = np.random.default_rng(1000 + seed)
    n = 6; rmag = 5
    nz = int(rng.integers(2, 5))                            # 2..4-bit parity
    zbits = sorted(rng.choice(rmag, size=nz, replace=False).tolist())
    be, se, r, sign = one(n, rmag, zbits, seed)
    worst_b = max(worst_b, be); worst_s = max(worst_s, se); ncases += 1
    if seed < 6:
        print(f"  seed {seed}: Z over qubits {zbits} -> pivot {r} sign {sign:+.0f}  "
              f"|dp0|={be:.2e}  state_err={se:.2e}")
print(f"\n{ncases} random multi-bit parity cases:  worst |dp0|={worst_b:.2e}  worst state_err={worst_s:.2e}")
print("RESULT:", "PASS" if (worst_b < 1e-12 and worst_s < 1e-12) else "FAIL")
