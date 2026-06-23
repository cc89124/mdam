"""Phase 2 inverse-frame: DERIVE + EXHAUSTIVELY VERIFY the incremental update rules BEFORE
integrating (per protocol: do not guess the multiplication direction).

Ground truth inverse images:  Ax[i] = U_C^dag X_i U_C = nc._pullback(1<<i, 0)
                              Az[i] = U_C^dag Z_i U_C = nc._pullback(0, 1<<i)

Candidate rules (to be confirmed):
  LEFT-mult  forward gate G (U_C -> G U_C):   Ax'[i] = _subst(G^dag X_i G)  (subst = pullback via OLD A)
                                              only the gate's qubits' indices change.
  RIGHT-mult fold G (U_C -> U_C G):           Ax'[i] = G^dag . Ax[i] . G   (conjugate every image)
  Pauli fold  (U_C -> U_C X_q):               Ax'[i] = X_q . Ax[i] . X_q

We maintain Ax/Az incrementally and, after EVERY frame mutation, assert they equal the
ground-truth _pullback for all X_i, Z_i AND for random Paulis.  Random Clifford+fold
sequences on n = 1..4 qubits.
"""
import sys
sys.path.insert(0, "/home/jung/clifft-paper")
import random
from nearclifford_backend.simulator import NearClifford, pauli_mul

I = (0, 0, 0)


# ---- pure Pauli conjugations P -> G P G^dag (tuples over all qubits) ----
def conj_h(P, q):
    x, z, p = P
    xq = (x >> q) & 1; zq = (z >> q) & 1
    b = 1 << q
    x2 = (x & ~b) | (zq << q)
    z2 = (z & ~b) | (xq << q)
    return (x2, z2, (p + 2 * (xq & zq)) & 3)            # H Y H = -Y


def conj_s(P, q, dag):
    x, z, p = P
    xq = (x >> q) & 1
    z2 = z ^ (xq << q)                                  # X -> Y
    return (x, z2, (p + xq * (3 if dag else 1)) & 3)    # +i (S) / -i (Sdag) on the X part


def conj_cx(P, c, t):
    x, z, p = P
    xc = (x >> c) & 1; zt = (z >> t) & 1
    bc = 1 << c; bt = 1 << t
    x2 = (x & ~bt) | ((((x >> t) & 1) ^ xc) << t)       # X_c -> X_c X_t
    z2 = (z & ~bc) | ((((z >> c) & 1) ^ zt) << c)       # Z_t -> Z_c Z_t
    return (x2, z2, p)


def conj_x(P, q):
    x, z, p = P
    zq = (z >> q) & 1
    return (x, z, (p + 2 * zq) & 3)                     # X Z X = -Z


def subst(P, Ax, Az):
    """U_C^dag P U_C via X_j->Ax[j], Z_j->Az[j] (pullback through the inverse images)."""
    x, z, p = P
    out = (0, 0, p)
    n = max(Ax.keys()) + 1 if Ax else 0
    for j in range(n):
        if (x >> j) & 1:
            out = pauli_mul(out, Ax[j])
        if (z >> j) & 1:
            out = pauli_mul(out, Az[j])
    return out


class InvFrame:
    def __init__(self, n):
        self.n = n
        self.Ax = {i: (1 << i, 0, 0) for i in range(n)}   # U_C^dag X_i U_C, init U_C=I
        self.Az = {i: (0, 1 << i, 0) for i in range(n)}

    # forward (left-mult) gates: only the gate's qubits' indices change
    def f_h(self, q):
        self.Ax[q], self.Az[q] = self.Az[q], self.Ax[q]   # _subst(H X_q H)=_subst(Z_q)=Az[q] etc.

    def f_s(self, q, dag):
        nAx = subst(conj_s((1 << q, 0, 0), q, not dag), self.Ax, self.Az)   # G^dag X_q G, G=S^(dag)
        self.Ax[q] = nAx                                   # Z_q unchanged under S

    def f_cx(self, c, t):
        nAxc = pauli_mul(self.Ax[c], self.Ax[t])           # _subst(X_c X_t)
        nAzt = pauli_mul(self.Az[c], self.Az[t])           # _subst(Z_c Z_t)
        self.Ax[c] = nAxc; self.Az[t] = nAzt

    # right-mult folds: conjugate EVERY image by G^dag
    def r_h(self, s):
        for i in range(self.n):
            self.Ax[i] = conj_h(self.Ax[i], s); self.Az[i] = conj_h(self.Az[i], s)

    def r_s(self, s, dag):
        for i in range(self.n):
            self.Ax[i] = conj_s(self.Ax[i], s, not dag); self.Az[i] = conj_s(self.Az[i], s, not dag)

    def r_cx(self, c, t):
        for i in range(self.n):
            self.Ax[i] = conj_cx(self.Ax[i], c, t); self.Az[i] = conj_cx(self.Az[i], c, t)

    def fold_x(self, q):
        for i in range(self.n):
            self.Ax[i] = conj_x(self.Ax[i], q); self.Az[i] = conj_x(self.Az[i], q)


def check(nc, inv, tag, trial):
    for i in range(nc.n):
        gx = nc._pullback(1 << i, 0)
        gz = nc._pullback(0, 1 << i)
        if inv.Ax[i] != gx:
            print(f"  MISMATCH Ax[{i}] after {tag} (trial {trial}): inv={inv.Ax[i]} truth={gx}")
            return False
        if inv.Az[i] != gz:
            print(f"  MISMATCH Az[{i}] after {tag} (trial {trial}): inv={inv.Az[i]} truth={gz}")
            return False
    for _ in range(6):                                     # random Paulis
        x = random.randrange(1 << nc.n); z = random.randrange(1 << nc.n)
        truth = nc._pullback(x, z)
        got = subst((x, z, 0), inv.Ax, inv.Az)
        if got != truth:
            print(f"  MISMATCH random P=({x},{z}) after {tag} (trial {trial}): inv={got} truth={truth}")
            return False
    return True


random.seed(1)
ok_all = True
for n in (1, 2, 3, 4):
    for trial in range(200):
        nc = NearClifford(n)
        inv = InvFrame(n)
        for step in range(20):
            op = random.choice(["h", "s", "sd", "cx", "cz", "rh", "rs", "rsd", "rcx", "foldx"])
            q = random.randrange(n)
            c, t = random.sample(range(n), 2) if n >= 2 else (0, 0)
            if op == "h":   nc.h(q); inv.f_h(q)
            elif op == "s": nc.s(q); inv.f_s(q, False)
            elif op == "sd": nc.s(q, dag=True); inv.f_s(q, True)
            elif op == "cx" and n >= 2: nc.cx(c, t); inv.f_cx(c, t)
            elif op == "cz" and n >= 2:
                nc.cz(c, t)                                # cz = h_t cx h_t
                inv.f_h(t); inv.f_cx(c, t); inv.f_h(t)
            elif op == "rh": nc.right_h(q); inv.r_h(q)
            elif op == "rs": nc.right_s(q); inv.r_s(q, False)
            elif op == "rsd": nc.right_s(q, dag=True); inv.r_s(q, True)
            elif op == "rcx" and n >= 2: nc.right_cx(c, t); inv.r_cx(c, t)
            elif op == "foldx":
                zr = nc.Zc[q]; nc.Zc[q] = (zr[0], zr[1], (zr[2] + 2) & 3); nc._frame_ver += 1
                inv.fold_x(q)
            else:
                continue
            if not check(nc, inv, op, trial):
                ok_all = False
                break
        if not ok_all:
            break
    if not ok_all:
        break
    print(f"  n={n}: 200 random 20-op sequences  ALL MATCH")
print("\nALL INVERSE-FRAME RULES VERIFIED" if ok_all else "\nRULE MISMATCH -- fix before integrating")
