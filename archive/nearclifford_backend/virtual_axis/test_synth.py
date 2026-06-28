"""Verify clifford_synth: (a) conjugation rules match dense-matrix conjugation;
(b) random-Clifford synthesis reproduces the target tableau phase-exactly."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from nearclifford_backend.virtual_axis import clifford_synth as cs

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)


def kron_op(r, ops):
    m = np.array([[1]], dtype=complex)
    for q in range(r):                         # q=0 is LSB -> rightmost in kron
        m = np.kron(ops[q], m)
    return m


def pauli_mat(p, r):
    x, z, ph = p
    ops = []
    for q in range(r):
        xq = (x >> q) & 1; zq = (z >> q) & 1
        ops.append((X @ Z) if (xq and zq) else (X if xq else (Z if zq else I2)))
    return (1j ** ph) * kron_op(r, ops)


def gate_mat(g, r):
    if g[0] == 'h':   ops = [H if q == g[1] else I2 for q in range(r)]; return kron_op(r, ops)
    if g[0] == 's':   ops = [S if q == g[1] else I2 for q in range(r)]; return kron_op(r, ops)
    if g[0] == 'sdg': ops = [S.conj().T if q == g[1] else I2 for q in range(r)]; return kron_op(r, ops)
    if g[0] == 'swap':
        m = np.zeros((1 << r, 1 << r), dtype=complex)
        for b in range(1 << r):
            bb = list(format(b, f'0{r}b')[::-1])
            bb[g[1]], bb[g[2]] = bb[g[2]], bb[g[1]]
            m[int(''.join(bb[::-1]), 2), b] = 1
        return m
    if g[0] == 'cx':
        m = np.zeros((1 << r, 1 << r), dtype=complex)
        for b in range(1 << r):
            t = b ^ ((((b >> g[1]) & 1)) << g[2])
            m[t, b] = 1
        return m
    raise ValueError(g)


def test_conjugation(r=3):
    rng = np.random.default_rng(0)
    gates = [('h', q) for q in range(r)] + [('s', q) for q in range(r)] + \
            [('sdg', q) for q in range(r)] + \
            [('cx', c, t) for c in range(r) for t in range(r) if c != t] + \
            [('swap', a, b) for a in range(r) for b in range(a + 1, r)]
    bad = 0
    for g in gates:
        G = gate_mat(g, r)
        for _ in range(40):
            p = (int(rng.integers(0, 1 << r)), int(rng.integers(0, 1 << r)), int(rng.integers(0, 4)))
            lhs = G @ pauli_mat(p, r) @ G.conj().T
            rhs = pauli_mat(cs.apply_gate(g, p), r)
            if not np.allclose(lhs, rhs, atol=1e-9):
                bad += 1
    print(f"(a) conjugation vs dense: {'OK' if bad == 0 else f'{bad} MISMATCH'}")
    return bad == 0


def test_synth(trials=2000):
    rng = np.random.default_rng(1)
    gateset = lambda r: [('h', q) for q in range(r)] + [('s', q) for q in range(r)] + \
        [('cx', c, t) for c in range(r) for t in range(r) if c != t]
    bad = 0
    for _ in range(trials):
        r = int(rng.integers(1, 7))
        gs = gateset(r)
        seq = [gs[int(rng.integers(0, len(gs)))] for _ in range(int(rng.integers(0, 4 * r + 1)))]
        Xt, Zt = cs.conj_tableau(seq, *cs.identity_tableau(r))   # target tableau
        W = cs.synthesize(Xt, Zt, r)
        Xc, Zc = cs.conj_tableau(W, *cs.identity_tableau(r))     # apply synth circuit
        if Xc != Xt or Zc != Zt:
            bad += 1
    print(f"(b) random-Clifford synth (phase-exact, {trials} trials): "
          f"{'OK' if bad == 0 else f'{bad} FAIL'}")
    return bad == 0


if __name__ == "__main__":
    ok = test_conjugation() & test_synth()
    print("ALL PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)
