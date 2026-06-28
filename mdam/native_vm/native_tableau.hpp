// native_tableau.hpp — native port of NearClifford's U_C stabilizer tableau (Xc/Zc) right-folds.
// Xc[i] = U_C X_i U_C^dag, Zc[i] = U_C Z_i U_C^dag (each a PackedPauli over n qubits, mod-4 phase).
// Only the operations the magic-measurement control plane uses are ported: right_h / right_s /
// right_cx (U_C <- U_C G) and the |1>-branch X-fold on Zc[r].  Uses pauli_mul (native_inverse_frame).
#pragma once
#include <vector>
#include "native_pending.hpp"        // PackedPauli
#include "native_inverse_frame.hpp"  // pauli_mul

namespace mdam {

struct NativePackedTableau {
    int n, W;
    std::vector<PackedPauli> Xc, Zc;

    explicit NativePackedTableau(int n_ = 0) { init_identity(n_); }
    void init_identity(int n_) {
        n = n_; W = (n + 63) >> 6; Xc.clear(); Zc.clear(); Xc.reserve(n); Zc.reserve(n);
        for (int i = 0; i < n; i++) {
            PackedPauli X(W), Z(W);
            X.x[PackedPauli::word(i)] = PackedPauli::bit(i);
            Z.z[PackedPauli::word(i)] = PackedPauli::bit(i);
            Xc.push_back(X); Zc.push_back(Z);
        }
    }
    // reset to identity IN PLACE (no realloc; Xc/Zc keep their capacity)
    void reset_identity() {
        for (int i = 0; i < n; i++) {
            for (int w = 0; w < W; w++) { Xc[i].x[w]=0; Xc[i].z[w]=0; Zc[i].x[w]=0; Zc[i].z[w]=0; }
            Xc[i].phase=0; Zc[i].phase=0;
            Xc[i].x[PackedPauli::word(i)] = PackedPauli::bit(i);
            Zc[i].z[PackedPauli::word(i)] = PackedPauli::bit(i);
        }
    }
    // ---- FORWARD gates (U_C <- G U_C): conjugate every stored image by G (== _apply_clifford_to_all) ----
#ifdef FB_COUNT
    #define FB_TAB_COUNT() fbc().tab += 2*n
#else
    #define FB_TAB_COUNT()
#endif
    void fwd_h(int q)  { FB_TAB_COUNT(); for (int i = 0; i < n; i++) { pconj_h(Xc[i], q);  pconj_h(Zc[i], q);  } }
    void fwd_s(int q, bool dag) { FB_TAB_COUNT(); for (int i = 0; i < n; i++) { pconj_s(Xc[i], q, dag); pconj_s(Zc[i], q, dag); } }
    void fwd_cx(int c, int t) { FB_TAB_COUNT(); for (int i = 0; i < n; i++) { pconj_cx(Xc[i], c, t); pconj_cx(Zc[i], c, t); } }

    // right_h(s): H X_s H = Z_s, H Z_s H = X_s -> swap image columns
    void right_h(int s) { std::swap(Xc[s], Zc[s]); }
    // right_s(s,dag): Xc[s] = pauli_mul(Xc[s],Zc[s]); phase += (dag?3:1)
    void right_s(int s, bool dag) {
        PackedPauli m = pauli_mul(Xc[s], Zc[s]);
        m.phase = (uint8_t)((m.phase + (dag ? 3 : 1)) & 3);
        Xc[s] = m;
    }
    // right_cx(c,t): Xc[c] *= Xc[t]; Zc[t] = Zc[c]*Zc[t]
    void right_cx(int c, int t) {
        Xc[c] = pauli_mul(Xc[c], Xc[t]);
        Zc[t] = pauli_mul(Zc[c], Zc[t]);
    }
    // |1>-branch product: fold X_r into the frame -> Zc[r] gains a -1 (phase += 2)
    void fold_x_on_Zc(int r) { Zc[r].phase = (uint8_t)((Zc[r].phase + 2) & 3); }

    // pauli_commute(Zc[i], Pm) where Pm = (0, 1<<q, 0): symplectic inner product even?
    bool Zc_commutes_with_Zq(int i, int q) const {
        // Pm has only z-bit q set; commute = popcount(Zc[i].x & Pm.z) even (Zc[i].z & Pm.x = 0)
        int wq = PackedPauli::word(q);
        int parity = (int)((Zc[i].x[wq] >> (q & 63)) & 1ULL);
        return (parity & 1) == 0;
    }
};

} // namespace mdam
