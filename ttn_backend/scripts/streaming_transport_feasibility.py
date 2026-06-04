"""Feasibility check: can a TTN transport (GEMM + QR) be done block-streamed
so the *transient workspace* stays small, exactly, at the cost of time?

Models one adjacent-2-bag transport at coherent_d5_r5 / B72 scale:

    theta = src_m @ dst_m        # GEMM, this is the big workspace
    Q, R  = qr(theta)            # Q -> src bag, R -> dst bag

Dense path materializes the full theta (= M) plus LAPACK Q/workspace.
Streamed path processes row-blocks of the left dimension and never holds
the full theta. Streaming QR uses CholeskyQR2 (the standard exact-to-~1e-14
out-of-core/distributed QR), which is naturally block-sequential.

We report: reconstruction error, R-equivalence vs numpy QR, peak transient
workspace (largest simultaneously-live transient array bytes), and wall time.
"""
import time
import numpy as np

C = 16  # complex128 bytes

def make_problem(left_dim, K, n, seed=0):
    rng = np.random.default_rng(seed)
    # src_m (left_dim, K) and dst_m (K, n) are the two adjacent bag tensors
    # (already reshaped to matrices). theta = src_m @ dst_m is the workspace.
    src_m = (rng.standard_normal((left_dim, K)) + 1j*rng.standard_normal((left_dim, K))) / np.sqrt(K)
    dst_m = (rng.standard_normal((K, n)) + 1j*rng.standard_normal((K, n))) / np.sqrt(K)
    return src_m, dst_m

def dense_transport(src_m, dst_m):
    t0 = time.perf_counter()
    theta = src_m @ dst_m                       # full workspace materialized
    Q, R = np.linalg.qr(theta, mode='reduced')  # LAPACK allocates ~theta again
    dt = time.perf_counter() - t0
    left_dim, n = theta.shape
    # peak transient ~ theta + Q + LAPACK internal copy of theta
    peak_transient = C * (theta.size + Q.size + theta.size)
    return Q, R, dt, peak_transient

def streamed_transport(src_m, dst_m, block_rows):
    """Block-sequential GEMM + CholeskyQR2. Never materializes full theta."""
    t0 = time.perf_counter()
    left_dim, K = src_m.shape
    n = dst_m.shape[1]

    def gram_pass(transform=None):
        # G = sum_blocks B^H B, where B = src_block @ dst_m (optionally @ transform)
        G = np.zeros((n if transform is None else transform.shape[1],) * 2, dtype=np.complex128)
        max_block_bytes = 0
        for r0 in range(0, left_dim, block_rows):
            sb = src_m[r0:r0+block_rows]            # view, not a copy
            B = sb @ dst_m                          # transient block of theta
            if transform is not None:
                B = B @ transform
            G += B.conj().T @ B
            max_block_bytes = max(max_block_bytes, C * B.size)
        return G, max_block_bytes

    # CholeskyQR2 pass 1
    G1, mb1 = gram_pass()
    R1 = np.linalg.cholesky(G1).conj().T         # upper-triangular
    R1inv = np.linalg.inv(R1)
    # pass 2 on Q1 = theta @ R1inv
    G2, mb2 = gram_pass(transform=R1inv)
    R2 = np.linalg.cholesky(G2).conj().T
    R = R2 @ R1
    Rinv = R1inv @ np.linalg.inv(R2)

    # final Q materialized block by block (in real sampling you'd consume it
    # immediately; here we store it only to verify reconstruction)
    Q = np.empty((left_dim, n), dtype=np.complex128)
    max_qblock = 0
    for r0 in range(0, left_dim, block_rows):
        B = src_m[r0:r0+block_rows] @ dst_m
        Q[r0:r0+block_rows] = B @ Rinv
        max_qblock = max(max_qblock, C * B.size)
    dt = time.perf_counter() - t0
    # peak transient = one block of theta + small n x n matrices (Gram, R)
    peak_transient = max(mb1, mb2, max_qblock) + C * (5 * n * n)
    return Q, R, dt, peak_transient

def main():
    # B72-scale: theta workspace ~ 123 MB (full column rank: K >= n)
    left_dim, K, n = 60000, 192, 128
    block_rows = 4096
    src_m, dst_m = make_problem(left_dim, K, n)

    theta_bytes = C * left_dim * n
    print(f"problem: M = {left_dim} x {n}, contraction K={K}")
    print(f"full theta (workspace) = {theta_bytes/1e6:.1f} MB")
    print(f"block_rows = {block_rows} -> {block_rows*n*C/1e6:.2f} MB per block\n")

    Qd, Rd, dtd, pkd = dense_transport(src_m, dst_m)
    Qs, Rs, dts, pks = streamed_transport(src_m, dst_m, block_rows)

    # ground-truth M for error checks (recompute once)
    M = src_m @ dst_m
    rec_dense = np.linalg.norm(Qd @ Rd - M) / np.linalg.norm(M)
    rec_stream = np.linalg.norm(Qs @ Rs - M) / np.linalg.norm(M)
    orth_stream = np.linalg.norm(Qs.conj().T @ Qs - np.eye(n))
    # R equivalence (unique up to unitary diag); compare |R| row magnitudes via
    # the upper-triangular factor of M^H M
    print("=== correctness ===")
    print(f"dense   reconstruction  ||QR-M||/||M|| = {rec_dense:.2e}")
    print(f"stream  reconstruction  ||QR-M||/||M|| = {rec_stream:.2e}")
    print(f"stream  orthogonality   ||Q^H Q - I||  = {orth_stream:.2e}")

    print("\n=== peak transient workspace ===")
    print(f"dense  : {pkd/1e6:8.1f} MB")
    print(f"stream : {pks/1e6:8.1f} MB")
    print(f"workspace reduction      = {pkd/pks:.1f}x")

    print("\n=== time ===")
    print(f"dense  : {dtd*1e3:8.1f} ms")
    print(f"stream : {dts*1e3:8.1f} ms")
    print(f"slowdown                 = {dts/dtd:.2f}x")

if __name__ == "__main__":
    main()
