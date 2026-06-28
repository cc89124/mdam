"""Virtual-axis near-Clifford: OFFLINE localization core.

The physical near-Clifford backend, at a measurement flush, merges the raw physical
support of the core Pauli algebra into a dense block of size 2^B (B = #physical
qubits touched). When that support carries stabilizer / parity redundancy, B exceeds
the true independent rank r, so the block is larger than necessary (and larger than
clifft's active rank).

`localize_to_virtual_axes` computes, OFFLINE, a phase-exact symplectic basis change W
that maps every core Pauli onto its minimal `r` independent VIRTUAL axes, so the
runtime can build a 2^r block FROM THE START (never 2^B). It returns one
`VirtualPauliMask` per input Pauli, already expressed in block-local axis order.

Algorithm (binary symplectic, GF(2)):
  1. independent generating set of V = span{(x,z)} of the input Paulis;
  2. symplectic Gram-Schmidt -> s hyperbolic pairs (a_i,b_i) + c central elements c_j;
     a_i->X_i, b_i->Z_i (axis i),  c_j->Z_{s+j} (axis s+j);  r = s + c;
  3. express each input in that basis (GF(2) solve) -> virtual (x,z) mask, with the
     phase tracked exactly through the basis products (the _pullback phase identity).

Guarantees (asserted by the unit test): r <= B, and the map is a Pauli isomorphism --
every commutation AND product (phase-exact) relation among the inputs is preserved.

NOTE: this pure-symplectic r counts each independent commuting generator as one axis.
A central generator that is a STABILISER of the current state needs no dense axis
(it acts as a sign); removing those (the stabiliser quotient, requires the run-time
stabiliser context) can lower r further -- handled in the compile pass, not here."""
from __future__ import annotations

from dataclasses import dataclass

from nearclifford_backend.simulator import pauli_mul, pauli_commute


# ----------------------------------------------------------------- helpers
def _symp(p, q):
    """Symplectic inner product over GF(2): 1 if p,q anticommute else 0."""
    return 0 if pauli_commute(p, q) else 1


def _bits(mask):
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def _herm(p):
    """Normalise a Pauli to a proper +1/-1 observable (p^2 = +I), i.e. phase
    ph == popcount(x&z) (mod 2). pauli_mul products of observables can land on the
    anti-Hermitian rep (p^2 = -I); mapping THAT onto a +I-squaring virtual generator
    (X_i/Z_i) is not a Clifford and corrupts product phases. Multiplying by i (ph+1)
    restores p^2=+I so the basis->generator map is a genuine Clifford."""
    x, z, ph = p
    if (ph + (x & z).bit_count()) & 1:
        ph = (ph + 1) & 3
    return (x, z, ph)


# ----------------------------------------------------------------- types
@dataclass(frozen=True)
class VirtualPauliMask:
    """Pauli i^phase X^x Z^z over the block-local virtual axes (x,z are masks over
    axis ids 0..n_axes-1, NOT physical qubits)."""
    x: int
    z: int
    phase: int
    n_axes: int

    def commutes(self, other) -> bool:
        return (((self.x & other.z).bit_count()
                 + (self.z & other.x).bit_count()) & 1) == 0

    def mul(self, other) -> "VirtualPauliMask":
        x, z, p = pauli_mul((self.x, self.z, self.phase),
                            (other.x, other.z, other.phase))
        return VirtualPauliMask(x, z, p, self.n_axes)


@dataclass
class LocalizationResult:
    r: int                       # virtual rank = dense block exponent
    physical_B: int              # raw physical support size (diagnostic)
    masks: list                  # VirtualPauliMask per input Pauli (same order)
    axis_order: tuple            # virtual axis ids 0..r-1
    valid: bool
    reason: str | None = None


# ----------------------------------------------------------------- core
def localize_to_virtual_axes(paulis, n, support=None) -> LocalizationResult:
    """`paulis`: list of physical Paulis (x, z, phase), phase in {0,1,2,3} (i^phase),
    masks over n qubits. Returns a LocalizationResult with one VirtualPauliMask per
    input, all supported on the first r <= B virtual axes."""
    if support is None:
        m = 0
        for (x, z, _) in paulis:
            m |= x | z
        support = _bits(m)
    B = len(support)

    # 1. independent generating set of V = span{(x,z)} (phase-tracked Pauli reps)
    gens = []
    red = []                                   # (pivot_bit, xz_vec) for independence
    for P in paulis:
        x, z, _ = P
        cur = x | (z << n)
        for (pb, bv) in red:
            if (cur >> pb) & 1:
                cur ^= bv
        if cur:
            red.append(((cur & -cur).bit_length() - 1, cur))
            gens.append(P)

    # 2. symplectic Gram-Schmidt -> hyperbolic pairs + central elements
    work = list(gens)
    pairs = []
    central = []
    while work:
        a = work.pop(0)
        pi = next((i for i, w in enumerate(work) if _symp(a, w)), None)
        if pi is None:
            central.append(a)
        else:
            b = work.pop(pi)
            cleaned = []
            for w in work:                     # w' = w + <w,b> a + <w,a> b  (commutes a,b)
                sa = _symp(w, a)
                sb = _symp(w, b)
                ww = w
                if sb:
                    ww = pauli_mul(ww, a)
                if sa:
                    ww = pauli_mul(ww, b)
                cleaned.append(ww)
            work = cleaned
            pairs.append((a, b))
    s = len(pairs)
    c = len(central)
    r = s + c

    # 3. ordered basis B_l and its virtual generator (phase-0 X_i / Z_i / Z_{s+j})
    # basis elements normalised to proper observables (p^2=+I) so basis->generator is
    # a genuine Clifford -> phase-exact masks (commutation AND product phase preserved).
    basis = []
    vgen = []
    for i, (a, b) in enumerate(pairs):
        basis.append(_herm(a)); vgen.append((1 << i, 0, 0))            # a_i -> X_i
        basis.append(_herm(b)); vgen.append((0, 1 << i, 0))            # b_i -> Z_i
    for j, cc in enumerate(central):
        basis.append(_herm(cc)); vgen.append((0, 1 << (s + j), 0))     # c_j -> Z_{s+j}

    # reduced form of the basis (x|z) with coefficient tracking (which B_l combine)
    bred = []                                  # (pivot, xz_vec, coeff_mask)
    for idx, Bp in enumerate(basis):
        x, z, _ = Bp
        cur = x | (z << n); cm = 1 << idx
        for (pb, bv, bcm) in bred:
            if (cur >> pb) & 1:
                cur ^= bv; cm ^= bcm
        if cur:
            bred.append(((cur & -cur).bit_length() - 1, cur, cm))

    # 4. each input Pauli -> virtual mask (coords in the basis, phase via _pullback id)
    masks = []
    ok = True
    reason = None
    for P in paulis:
        x, z, _ = P
        cur = x | (z << n); coeff = 0
        for (pb, bv, bcm) in bred:
            if (cur >> pb) & 1:
                cur ^= bv; coeff ^= bcm
        if cur != 0:
            ok = False; reason = "input Pauli not in core span"; masks.append(None)
            continue
        Q = (0, 0, 0)                          # product of physical basis Paulis
        R = (0, 0, 0)                          # product of virtual generators
        for l in _bits(coeff):
            Q = pauli_mul(Q, basis[l])
            R = pauli_mul(R, vgen[l])
        delta = (P[2] - Q[2]) & 3              # P = i^delta * (basis product)
        masks.append(VirtualPauliMask(R[0], R[1], (delta + R[2]) & 3, r))

    return LocalizationResult(r=r, physical_B=B, masks=masks,
                              axis_order=tuple(range(r)), valid=ok, reason=reason)
