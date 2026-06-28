// native_inverse_frame.hpp — native port of NearClifford's incremental inverse-frame
// (simulator.py: pauli_mul / _pconj_{h,s,cx,x} / _inv_ax/_inv_az / _inv_fwd_* / _inv_right /
//  _inv_fold_x / _inv_subst).  Maintains Ax[i]=U_C^dag X_i U_C, Az[i]=U_C^dag Z_i U_C so a
// pullback is an O(weight) product instead of an O(n^2) GF(2) recompute.
//
// Multi-word (n>64) packed Paulis with mod-4 phase, reusing PackedPauli from native_pending.hpp.
// Bit-identical to the Python reference over random Clifford sequences (verified).
//
// NOTE: only the INCREMENTAL rules (forward gates h/s/cx + right folds + Pauli fold) are ported.
// The basis-method rebuild (_inv_rebuild after _ag_measure stabilizer projection) is part of the
// dense-engine port and is out of scope here; this header covers the Clifford-sequence pullback.
#pragma once
#include <cstdint>
#include <vector>
#include "native_pending.hpp"   // PackedPauli

namespace mdam {

// ---- Pauli phase-tracking multiply: (xa,za,pa)*(xb,zb,pb), X^x Z^z convention ----
// x=xa^xb, z=za^zb, p=(pa+pb+2*popcount(za & xb)) mod 4.
inline PackedPauli pauli_mul(const PackedPauli& a, const PackedPauli& b) {
    int W = a.W;
    PackedPauli r(W);
    int cross = 0;
    for (int i = 0; i < W; i++) {
        r.x[i] = a.x[i] ^ b.x[i];
        r.z[i] = a.z[i] ^ b.z[i];
        cross += __builtin_popcountll(a.z[i] & b.x[i]);
    }
    r.phase = (uint8_t)((a.phase + b.phase + 2 * cross) & 3);
    return r;
}

// ---- single-qubit-position Pauli conjugations  P -> G P G^dag (phase-tracked) ----
inline void pconj_h(PackedPauli& P, int q) {              // H P H
    int w = PackedPauli::word(q);
    int xq = (int)((P.x[w] >> (q & 63)) & 1ULL), zq = (int)((P.z[w] >> (q & 63)) & 1ULL);
    uint64_t b = PackedPauli::bit(q);
    P.x[w] = (P.x[w] & ~b) | ((uint64_t)zq << (q & 63));
    P.z[w] = (P.z[w] & ~b) | ((uint64_t)xq << (q & 63));
    P.phase = (uint8_t)((P.phase + 2 * (xq & zq)) & 3);
}
inline void pconj_s(PackedPauli& P, int q, bool dag) {    // S^(dag) P S^(dag)†
    int w = PackedPauli::word(q);
    int xq = (int)((P.x[w] >> (q & 63)) & 1ULL);
    P.z[w] ^= ((uint64_t)xq << (q & 63));
    P.phase = (uint8_t)((P.phase + xq * (dag ? 3 : 1)) & 3);
}
inline void pconj_cx(PackedPauli& P, int c, int t) {      // CX(c,t) P CX(c,t)
    int wc = PackedPauli::word(c), wt = PackedPauli::word(t);
    int xc = (int)((P.x[wc] >> (c & 63)) & 1ULL);
    int zt = (int)((P.z[wt] >> (t & 63)) & 1ULL);
    P.x[wt] ^= ((uint64_t)xc << (t & 63));     // X_c -> X_c X_t
    P.z[wc] ^= ((uint64_t)zt << (c & 63));     // Z_t -> Z_c Z_t
}
inline void pconj_x(PackedPauli& P, int q) {              // X_q P X_q : X Z X = -Z
    int w = PackedPauli::word(q);
    int zq = (int)((P.z[w] >> (q & 63)) & 1ULL);
    P.phase = (uint8_t)((P.phase + 2 * zq) & 3);
}

struct NativeInverseFrame {
    int n, W;
    std::vector<PackedPauli> ax, az;   // Ax[i]=U_C^dag X_i U_C, Az[i]=U_C^dag Z_i U_C

    explicit NativeInverseFrame(int n_) : n(n_), W((n_ + 63) >> 6) {
        ax.reserve(n); az.reserve(n);
        for (int i = 0; i < n; i++) {
            PackedPauli X(W), Z(W);
            X.x[PackedPauli::word(i)] = PackedPauli::bit(i);   // X_i
            Z.z[PackedPauli::word(i)] = PackedPauli::bit(i);   // Z_i
            ax.push_back(X); az.push_back(Z);
        }
    }

    // U_C^dag P U_C via X_j->Ax[j], Z_j->Az[j]  (out starts at (0,0,p))
    PackedPauli subst(const PackedPauli& Px, const PackedPauli& Pz, uint8_t p = 0) const {
        PackedPauli out(W); out.phase = p;
        for (int wi = 0; wi < W; wi++) {
            uint64_t xi = Px.x[wi];
            while (xi) { int b = __builtin_ctzll(xi); xi &= xi - 1; out = pauli_mul(out, ax[(wi << 6) + b]); }
        }
        for (int wi = 0; wi < W; wi++) {
            uint64_t zi = Pz.z[wi];
            while (zi) { int b = __builtin_ctzll(zi); zi &= zi - 1; out = pauli_mul(out, az[(wi << 6) + b]); }
        }
        return out;
    }
    // pullback of logical P=(x,z,0) given as a PackedPauli (its x and z fields)
    PackedPauli pullback(const PackedPauli& P) const { return subst(P, P, 0); }

    // ---- forward gates (U_C <- G U_C) ----
    void fwd_h(int q) {
#ifdef FB_COUNT
        fbc().inv++;
#endif
        std::swap(ax[q], az[q]); }
    void fwd_s(int q, bool dag) {
#ifdef FB_COUNT
        fbc().inv++;
#endif
        PackedPauli Xq(W); Xq.x[PackedPauli::word(q)] = PackedPauli::bit(q);   // X_q
        pconj_s(Xq, q, !dag);                                                  // G^dag X_q G, G=S^(dag)
        ax[q] = subst(Xq, Xq, Xq.phase);   // Xq has X on q (and maybe Z); subst uses x-bits via ax, z-bits via az
    }
    void fwd_cx(int c, int t) {
#ifdef FB_COUNT
        fbc().inv++;
#endif
        PackedPauli a = pauli_mul(ax[c], ax[t]);
        PackedPauli b = pauli_mul(az[c], az[t]);
        ax[c] = a; az[t] = b;
    }
    // ---- right folds (U_C <- U_C G): conjugate every image by G^dag ----
    void right_h(int s)  { for (int i = 0; i < n; i++) { pconj_h(ax[i], s);  pconj_h(az[i], s);  } }
    void right_s(int s, bool dag) { for (int i = 0; i < n; i++) { pconj_s(ax[i], s, !dag); pconj_s(az[i], s, !dag); } }
    void right_cx(int c, int t) { for (int i = 0; i < n; i++) { pconj_cx(ax[i], c, t); pconj_cx(az[i], c, t); } }
    void fold_x(int q)   { for (int i = 0; i < n; i++) { pconj_x(ax[i], q); pconj_x(az[i], q); } }
};

} // namespace mdam
