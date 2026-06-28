// S2 frame/ledger conjugation kernel.  Conjugates a set of Paulis (i^p X^x Z^z) by Clifford
// gates IN PLACE.  Each Pauli's x/z mask is stored as W uint64 WORDS (row-major: Pauli i at
// X[i*W .. i*W+W-1], little-endian words), so it supports n>64 qubits (e.g. d5_r5 n=72).  P[i]
// is the phase in {0,1,2,3}.  The per-gate rule P -> G P G^dag is BIT-IDENTICAL to
// nearclifford_backend.simulator NearClifford.{h,s,cx} fn-closures AND lazy._conj_{h,s,cx}
// (verified equal), so ONE kernel serves both the nc tableau (Xc/Zc) and the pending ledger.
// No floats, no RNG.  CZ is decomposed upstream into H,CX,H.
#include <cstdint>

extern "C" {

// gate: 0=H(q1)  1=S(q1)  2=Sdag(q1)  3=CX(c=q1,t=q2)
static inline void conj1(uint64_t* X, uint64_t* Z, int32_t* P, int m, int W,
                         int gate, int q1, int q2) {
    const int w1 = q1 >> 6, b1 = q1 & 63;
    const uint64_t m1 = 1ULL << b1;
    if (gate == 0) {                              // H(q1):  X<->Z on q1, Y->-Y
        for (int i = 0; i < m; ++i) {
            uint64_t* xr = X + (long)i * W; uint64_t* zr = Z + (long)i * W;
            uint64_t xq = (xr[w1] >> b1) & 1ULL, zq = (zr[w1] >> b1) & 1ULL;
            xr[w1] = (xr[w1] & ~m1) | (zq << b1);
            zr[w1] = (zr[w1] & ~m1) | (xq << b1);
            P[i] = (P[i] + 2 * (int32_t)(xq & zq)) & 3;
        }
    } else if (gate == 1 || gate == 2) {          // S / Sdag (q1)
        const int add = (gate == 2) ? 3 : 1;
        for (int i = 0; i < m; ++i) {
            uint64_t* xr = X + (long)i * W; uint64_t* zr = Z + (long)i * W;
            uint64_t xq = (xr[w1] >> b1) & 1ULL;
            zr[w1] ^= (xq << b1);
            P[i] = (P[i] + (int32_t)xq * add) & 3;
        }
    } else {                                      // CX(c=q1, t=q2): X_t^=X_c, Z_c^=Z_t
        const int w2 = q2 >> 6, b2 = q2 & 63;
        const uint64_t m2 = 1ULL << b2;
        for (int i = 0; i < m; ++i) {
            uint64_t* xr = X + (long)i * W; uint64_t* zr = Z + (long)i * W;
            uint64_t xc = (xr[w1] >> b1) & 1ULL, xt = (xr[w2] >> b2) & 1ULL;
            uint64_t zc = (zr[w1] >> b1) & 1ULL, zt = (zr[w2] >> b2) & 1ULL;
            xr[w2] = (xr[w2] & ~m2) | (((xt ^ xc) & 1ULL) << b2);
            zr[w1] = (zr[w1] & ~m1) | (((zc ^ zt) & 1ULL) << b1);
        }
    }
}

void clifford_conj(uint64_t* X, uint64_t* Z, int32_t* P, int m, int W, int gate, int q1, int q2) {
    conj1(X, Z, P, m, W, gate, q1, q2);
}

// a SEQUENCE of gates over the SAME (fixed-size) array -- amortizes the call overhead for a
// run of consecutive Cliffords (e.g. a measurement-to-measurement segment on the tableau).
void clifford_conj_seq(uint64_t* X, uint64_t* Z, int32_t* P, int m, int W,
                       const int32_t* gate, const int32_t* q1, const int32_t* q2, int ng) {
    for (int g = 0; g < ng; ++g) conj1(X, Z, P, m, W, gate[g], q1[g], q2[g]);
}

}  // extern "C"
