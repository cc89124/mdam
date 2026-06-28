"""Fused-core primitive verification: integrate out an ephemeral (ancilla) axis WITHOUT
materialising it. The ancilla starts |0>, the core rotations entangle it with the r system
axes, the measurement projects it. The fused map computes

    |phi_out>_sys = <b|_anc ( prod_i R_{P_i}(theta_i) ) ( |phi_in>_sys (x) |0>_anc )

as a Pauli SUM contracted on the ancilla -- workspace stays 2^r, the (r+1)-axis intermediate
is NEVER built. Verified against a dense reference that DOES materialise 2^(r+1).

This is the kernel of the measurement-core fused virtual-axis backend: the streaming engine's
`peak = r_out + 1` transient is exactly this one ephemeral axis, eliminated by the contraction.
"""
import sys

sys.path.insert(0, "/home/jung/clifft-paper")
import numpy as np

from nearclifford_backend.simulator import pauli_mul
from nearclifford_backend.block_magic import _apply_pauli_local


def pauli_sum(rot_masks, nax):
    """prod_i (cos(th/2) I - i sin(th/2) P_i) as {(x,z): complex coeff} over nax axes,
    phases folded into the coefficients. <= 2^rank distinct Paulis (they collapse mod the
    group), built incrementally -- NOT 2^(#rotations)."""
    s = {(0, 0): 1.0 + 0j}
    for (mx, mz, mph, th) in rot_masks:
        c = np.cos(th / 2.0)
        d = -1j * np.sin(th / 2.0) * (1j ** mph)        # R = c I + d (X^mx Z^mz)
        new = {}
        for (x, z), co in s.items():
            new[(x, z)] = new.get((x, z), 0j) + c * co
            x2, z2, ph2 = pauli_mul((mx, mz, 0), (x, z, 0))   # P on the LEFT: s <- R . s
            new[(x2, z2)] = new.get((x2, z2), 0j) + co * d * (1j ** ph2)
        s = new
    return s


def fused_contract(phi_sys, r, rot_masks, b):
    """|phi_out>_sys = <b|_anc (prod R) (phi_sys (x) |0>_anc), anc = axis r. Workspace 2^r."""
    s = pauli_sum(rot_masks, r + 1)
    out = np.zeros(1 << r, dtype=complex)
    mask_r = (1 << r) - 1
    for (x, z), co in s.items():
        if ((x >> r) & 1) != b:                          # <b| X^xa Z^za |0> = delta(b, xa)
            continue
        out += co * _apply_pauli_local(list(range(r)), phi_sys, x & mask_r, z & mask_r, 0)
    return out


def dense_ref(phi_sys, r, rot_masks, b):
    """Materialise 2^(r+1): apply rotations to phi_sys (x) |0>_anc, project anc = b."""
    full = np.concatenate([phi_sys, np.zeros(1 << r, dtype=complex)])   # anc (bit r) = |0>
    for (mx, mz, mph, th) in rot_masks:
        Pv = _apply_pauli_local(list(range(r + 1)), full, mx, mz, mph)
        full = np.cos(th / 2.0) * full - 1j * np.sin(th / 2.0) * Pv
    return full[:1 << r] if b == 0 else full[1 << r:]    # project ancilla = b


def run(r=4, n_rot=6, trials=200):
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(trials):
        phi = rng.standard_normal(1 << r) + 1j * rng.standard_normal(1 << r)
        phi /= np.linalg.norm(phi)
        # random core rotations over r+1 axes (some touch the ancilla, some don't)
        rot = []
        for _ in range(n_rot):
            mx = int(rng.integers(0, 1 << (r + 1)))
            mz = int(rng.integers(0, 1 << (r + 1)))
            mph = (mx & mz).bit_count() & 1                # Hermitian generator
            th = float(rng.uniform(0.1, 3.0))
            rot.append((mx, mz, mph, th))
        for b in (0, 1):
            f = fused_contract(phi, r, rot, b)
            d = dense_ref(phi, r, rot, b)
            worst = max(worst, np.max(np.abs(f - d)) if len(f) else 0.0)
    return worst


if __name__ == "__main__":
    w = run()
    print(f"fused ancilla-contraction vs dense (2^(r+1)) reference: max|err| = {w:.2e}  "
          f"{'OK' if w < 1e-9 else 'FAIL'}")
    print("workspace: fused builds only 2^r vectors (+ a classical Pauli-sum); the "
          "(r+1)-axis intermediate is never materialised.")
    sys.exit(0 if w < 1e-9 else 1)
