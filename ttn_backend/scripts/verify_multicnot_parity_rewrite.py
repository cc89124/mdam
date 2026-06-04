"""Verify that rewriting OP_ARRAY_MULTI_CNOT as a parity-accumulator / tree-gather
CNOT network is an EXACT identity, and quantify the big-edge crossing reduction.

The claim under test (user proposal):

    MULTI_CNOT(target=t, controls=C) :  t <- t XOR (XOR_{c in C} c)

Instead of crossing a high-chi tree edge once per far-side control, gather the
far-side controls' parity LOCALLY into one accumulator, cross the big edge once
with CNOT(acc, t), then uncompute the local gather. Exact, not approximate.

This script proves three things, all without touching the TTN runtime:

  1. GF(2) linear-map identity   (via clifford_frame.RegionLinearFrame)
       A_rewrite == A_multicnot  for every rewrite variant.
       This also implies the Pauli-frame conjugation is identical, because a
       CNOT (reversible-linear) circuit's symplectic action is fully determined
       by its GF(2) matrix M: X-part -> M, Z-part -> M^{-T}.

  2. Statevector identity        (random complex amplitudes)
       U_rewrite |psi> == U_multicnot |psi>  to ~1e-15.

  3. Pauli-frame symplectic identity (explicit), by conjugating every single-
     qubit X_i and Z_i generator through both circuits and comparing.

  4. Big-edge crossing count: on a synthetic tree it reports
       naive crossings  = 2 * (#far-side controls)
       rewrite crossings = 2
     i.e. m -> 1 (x2 for there-and-back), the work reduction the proposal targets.

Run:
    /home/jung/clifft_env/bin/python -m ttn_backend.scripts.verify_multicnot_parity_rewrite
"""
from __future__ import annotations

import itertools
import numpy as np

from ttn_backend.clifford_frame import RegionLinearFrame


# --------------------------------------------------------------------------
# Rewrite variants. Each returns a list of ("CNOT", control_ident, target_ident)
# whose NET linear action must equal MULTI_CNOT(target, controls).
# --------------------------------------------------------------------------
def seq_naive(target, controls):
    """Baseline: one CNOT(c, target) per control."""
    return [("CNOT", c, target) for c in controls]


def seq_single_accumulator(target, controls, far_set):
    """User's proposal for one big edge.

    `far_set` = controls on the far side of the big edge from the target.
    Fold the far controls into one accumulator (a far control), cross once,
    uncompute. Near controls keep direct CNOTs to the target.
    """
    far = [c for c in controls if c in far_set]
    near = [c for c in controls if c not in far_set]
    ops = []
    if len(far) == 0:
        return [("CNOT", c, target) for c in controls]
    acc = far[0]
    rest = far[1:]
    for c in rest:                      # local gather on the far side
        ops.append(("CNOT", c, acc))
    ops.append(("CNOT", acc, target))   # the single big-edge crossing
    for c in reversed(rest):            # local uncompute (restores acc & rest)
        ops.append(("CNOT", c, acc))
    for c in near:                      # near controls handled normally
        ops.append(("CNOT", c, target))
    return ops


def seq_control_reduction(target, controls):
    """CORRECT general gather: fold CONTROLS-ONLY into one accumulator, cross once.

    This mirrors what the TTN runtime can actually do: it never XORs a parity
    through a non-control "Steiner" ident, because every ident is live data, not
    scratch. Instead it folds controls into one accumulator (control-into-control
    CNOTs, which are exactly the wanted parity bits) and TRANSPORTS that one
    accumulator across the big edge. Here transport is modeled as the single
    CNOT(acc, target); the unfold restores all controls.
    """
    if not controls:
        return []
    acc = controls[0]
    rest = controls[1:]
    fwd = [("CNOT", c, acc) for c in rest]
    return fwd + [("CNOT", acc, target)] + list(reversed(fwd))


def seq_steiner_route_WRONG(target, controls, tree_parent):
    """INTENTIONALLY WRONG: routes parity by XORing through Steiner (non-control)
    idents on the control->target tree path. Because each TTN ident holds its own
    live value (not scratch zero), the Steiner ident's original bit leaks into the
    parity. Included so the verification proves we understand this boundary: this
    variant MUST fail whenever a non-control ident lies on a routing path."""
    depth = {}

    def d(v):
        if v in depth:
            return depth[v]
        n, cur = 0, v
        while tree_parent[cur] is not None:
            cur = tree_parent[cur]
            n += 1
        depth[v] = n
        return n

    active = set()
    for c in controls:
        cur = c
        while cur is not None:
            active.add(cur)
            cur = tree_parent[cur]
    carry = {v: set() for v in active}
    for c in controls:
        carry[c] ^= {c}
    order = sorted((v for v in active if tree_parent[v] is not None), key=lambda v: -d(v))
    fwd = []
    for v in order:
        p = tree_parent[v]
        if carry[v]:
            fwd.append(("CNOT", v, p))
            carry[p] ^= carry[v]
    rewind = [(nm, v, p) for (nm, v, p) in reversed(fwd) if p != target]
    return fwd + rewind


def steiner_nodes_on_paths(controls, tree_parent):
    """Non-control idents that lie strictly between a control and the root."""
    active = set()
    for c in controls:
        cur = c
        while cur is not None:
            active.add(cur)
            cur = tree_parent[cur]
    root = next(v for v, p in tree_parent.items() if p is None)
    return {v for v in active if v not in controls and v != root}


# --------------------------------------------------------------------------
# Exactness checks
# --------------------------------------------------------------------------
def linear_matrix(support, ops):
    """GF(2) matrix A (and affine b) of a CNOT sequence over `support` idents."""
    frame = RegionLinearFrame(support)
    for name, c, t in ops:
        assert name == "CNOT"
        frame.compose_cnot(c, t)
    return frame.A.copy(), frame.b.copy()


def multicnot_matrix(support, target, controls):
    frame = RegionLinearFrame(support)
    frame.compose_multicnot(target, controls)
    return frame.A.copy(), frame.b.copy()


def statevector_apply(n, ops, psi):
    """Apply a CNOT sequence to a 2**n statevector (ident i -> qubit bit i)."""
    out = psi
    for name, c, t in ops:
        out = _cnot_on_vec(n, out, c, t)
    return out


def _cnot_on_vec(n, psi, c, t):
    T = psi.reshape((2,) * n)
    T = np.moveaxis(T, [c, t], [0, 1])
    out = T.copy()
    out[1] = np.flip(out[1], axis=0)          # flip target where control == 1
    out = np.moveaxis(out, [0, 1], [c, t])
    return out.reshape(-1)


def symplectic_of(support, ops):
    """2n x 2n binary symplectic matrix (X|Z blocks) of a CNOT circuit.

    Track how each generator X_i, Z_i conjugates. For CNOT(c,t):
        X_c -> X_c X_t ,  X_t -> X_t ,  Z_c -> Z_c ,  Z_t -> Z_c Z_t .
    Represent a Pauli as (x-bits, z-bits). Conjugation is sign-free for CNOT.
    """
    n = len(support)
    pos = {s: i for i, s in enumerate(support)}
    cols = []
    for i in range(n):                       # X_i generator
        x = np.zeros(n, np.uint8); z = np.zeros(n, np.uint8); x[i] = 1
        cols.append(_conj(ops, pos, x, z))
    for i in range(n):                       # Z_i generator
        x = np.zeros(n, np.uint8); z = np.zeros(n, np.uint8); z[i] = 1
        cols.append(_conj(ops, pos, x, z))
    return np.array([np.concatenate(v) for v in cols], np.uint8).T


def _conj(ops, pos, x, z):
    x = x.copy(); z = z.copy()
    for name, c, t in ops:
        ci, ti = pos[c], pos[t]
        x[ti] ^= x[ci]      # X on control spreads to target
        z[ci] ^= z[ti]      # Z on target spreads to control
    return (x, z)


# --------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)
    # exact variants that MUST always match the reference
    fails = {"single_acc": [0, 0, 0], "control_reduction": [0, 0, 0]}
    # the deliberately-wrong variant: count fails and how many had Steiner routing
    wrong_fail = 0
    wrong_with_steiner = 0
    wrong_no_steiner = 0
    trials = 300

    for _ in range(trials):
        n = int(rng.integers(3, 9))
        support = list(range(n))
        target = int(rng.integers(0, n))
        cand = [i for i in support if i != target]
        m = int(rng.integers(1, len(cand) + 1))
        controls = sorted(rng.choice(cand, size=m, replace=False).tolist())

        A_ref, b_ref = multicnot_matrix(support, target, controls)
        far = set(c for c in controls if rng.integers(0, 2) == 1)
        parent = _random_rooted_tree(rng, support, target)

        exact_variants = {
            "single_acc": seq_single_accumulator(target, controls, far),
            "control_reduction": seq_control_reduction(target, controls),
        }

        psi = (rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n))
        psi /= np.linalg.norm(psi)
        ref_vec = statevector_apply(n, seq_naive(target, controls), psi)
        symp_ref = symplectic_of(support, seq_naive(target, controls))

        for name, ops in exact_variants.items():
            A, b = linear_matrix(support, ops)
            if not (np.array_equal(A, A_ref) and np.array_equal(b, b_ref)):
                fails[name][0] += 1
            if np.linalg.norm(statevector_apply(n, ops, psi) - ref_vec) > 1e-12:
                fails[name][1] += 1
            if not np.array_equal(symplectic_of(support, ops), symp_ref):
                fails[name][2] += 1

        # boundary check: Steiner routing is provably wrong
        wrong = seq_steiner_route_WRONG(target, controls, parent)
        Aw, bw = linear_matrix(support, wrong)
        steiner = steiner_nodes_on_paths(controls, parent)
        if not (np.array_equal(Aw, A_ref) and np.array_equal(bw, b_ref)):
            wrong_fail += 1
            if steiner:
                wrong_with_steiner += 1
            else:
                wrong_no_steiner += 1

    print(f"trials={trials}")
    print("EXACT variants (must be 0/0/0  -> lin / statevec / symplectic):")
    for name, (l, v, s) in fails.items():
        print(f"  {name:18s}: {l} / {v} / {s}")
    print("Steiner-routing variant (boundary, expected to FAIL with leakage):")
    print(f"  total fails={wrong_fail}  with-steiner-node={wrong_with_steiner}  "
          f"no-steiner-node={wrong_no_steiner}")
    ok = (all(f == [0, 0, 0] for f in fails.values()) and wrong_no_steiner == 0)
    print("EXACTNESS:", "PASS" if ok else "FAIL")
    print("  (PASS means: control-only gather is always exact, and the wrong")
    print("   variant fails ONLY when it routes through a live Steiner ident.)")

    # ---- crossing-count demonstration on a deliberately bad layout ----
    print("\n--- big-edge crossing reduction (synthetic) ---")
    for m in (4, 8, 16, 32):
        target = 0
        controls = list(range(1, m + 1))
        far = set(controls)                  # ALL controls on the far side
        naive = seq_naive(target, controls)
        acc = seq_single_accumulator(target, controls, far)
        # crossings of the one big edge = #ops whose (control,target) straddle it
        def big_edge_crossings(ops):
            # big edge separates {target} (near) from far_set; an op crosses it
            # iff exactly one of {c,t} is in far_set.
            return sum(1 for _, c, t in ops
                       if (c in far) != (t in far))
        print(f"m={m:3d}  naive big-edge crossings={big_edge_crossings(naive):3d}  "
              f"rewrite big-edge crossings={big_edge_crossings(acc):3d}  "
              f"extra local CNOTs={len(acc) - 1 - (1 if far else 0) - 0:3d}")


def _random_rooted_tree(rng, nodes, root):
    parent = {root: None}
    attached = [root]
    for v in nodes:
        if v == root:
            continue
        p = int(rng.choice(attached))
        parent[v] = p
        attached.append(v)
    return parent


if __name__ == "__main__":
    main()
