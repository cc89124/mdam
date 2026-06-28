"""Verify the diagonal parity-phase primitive against a dense reference, and
measure the post-apply bond dimensions (to confirm the rank-2 <=x2 bound)."""
from __future__ import annotations
import numpy as np
from ttn_backend.affine_diag import TTN, Bag, apply_parity_phase


def build_random_ttn(bag_neighbors, home, chi=3, seed=0):
    """Random TTN: each bag a random tensor over own idents (dim2) + bonds (dim chi)."""
    rng = np.random.default_rng(seed)
    ttn = TTN(bag_neighbors, home)
    # assign idents to homes
    for ident, b in home.items():
        ttn.bags[b].own_idents.append(ident)
    for b in ttn.bags:
        b.own_idents.sort()
        shape = tuple([2] * len(b.own_idents) + [chi] * len(b.neighbors))
        b.tensor = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    return ttn


def dense_apply_parity(vec, idents, support, theta):
    n = len(idents)
    pos = {ident: k for k, ident in enumerate(idents)}
    e = np.exp(1j * theta)
    out = vec.copy()
    for x in range(len(vec)):
        # bit of ident q in basis index x: idents sorted ascending == axis order;
        # axis 0 is most significant in reshape(-1) of C-order (idents[0]).
        par = 0
        for q in support:
            bit = (x >> (n - 1 - pos[q])) & 1
            par ^= bit
        if par:
            out[x] *= e
    return out


def check(name, bag_neighbors, home, support, theta, chi=3, seed=0):
    ttn = build_random_ttn(bag_neighbors, home, chi=chi, seed=seed)
    vec0, idents = ttn.to_dense()
    ref = dense_apply_parity(vec0, idents, support, theta)
    homes = sorted({home[q] for q in support})
    region = sorted(ttn.steiner(homes)) if len(homes) > 1 else homes
    pre = {(a, b): ttn.edge_chi(a, b) for a in region for b in ttn.bags[a].neighbors
           if b in region and a < b}
    m = apply_parity_phase(ttn, support, theta)
    vec1, idents1 = ttn.to_dense()
    # align (idents order should match)
    assert idents1 == idents, (idents1, idents)
    err = np.max(np.abs(vec1 - ref)) / (np.max(np.abs(ref)) + 1e-15)
    post = {(a, b): ttn.edge_chi(a, b) for a in region for b in ttn.bags[a].neighbors
            if b in region and a < b}
    growth = max([post[e] / pre[e] for e in pre], default=1.0)
    ok = err < 1e-9
    print(f"[{'OK ' if ok else 'FAIL'}] {name:26s} support={support} theta={theta:.3f} "
          f"err={err:.2e}  n_qr={m.get('n_qr',0)}  maxBondGrowth={growth:.2f}x  "
          f"pre={list(pre.values())} post={list(post.values())}")
    return ok


def main():
    allok = True
    # 1) all support in one bag -> no bond change
    allok &= check("local-1bag", [[1], [0]], {0: 0, 1: 0, 2: 1}, [0], 0.3, chi=3)
    allok &= check("local-2q-1bag", [[1], [0]], {0: 0, 1: 0, 2: 1}, [0, 1], 0.7, chi=3)
    # 2) two bags, one support qubit each (weight-2 across one edge)
    allok &= check("2bag-w2", [[1], [0]], {0: 0, 1: 1}, [0, 1], 0.5, chi=4)
    # 3) path of 3 bags, support at the two ends (weight-2, Steiner spans 2 edges)
    allok &= check("path3-ends", [[1], [0, 2], [1]], {0: 0, 1: 1, 2: 2}, [0, 2], 0.9, chi=3)
    # 4) path of 4 bags, support weight-3 across them
    bn = [[1], [0, 2], [1, 3], [2]]
    hm = {0: 0, 1: 1, 2: 2, 3: 3}
    allok &= check("path4-w3", bn, hm, [0, 2, 3], 1.1, chi=3)
    # 5) star: center bag 0 with leaves 1,2,3; support on leaves
    star = [[1, 2, 3], [0], [0], [0]]
    hm5 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0}
    allok &= check("star-w3-leaves", star, hm5, [1, 2, 3], 0.6, chi=3)
    # 6) bigger: 6-bag tree, weight-4
    bn6 = [[1], [0, 2, 4], [1, 3], [2], [1, 5], [4]]
    hm6 = {i: i for i in range(6)}
    allok &= check("tree6-w4", bn6, hm6, [0, 3, 5, 4], 0.8, chi=4)
    print("\nALL", "PASS" if allok else "FAIL")


if __name__ == "__main__":
    main()
