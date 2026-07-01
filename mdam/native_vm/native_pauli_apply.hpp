// native_pauli_apply.hpp — THE single MDAM reduced-core Pauli-rotation apply primitive.
//
// MDAM's operator normal form is a *projected factorized Pauli product* on the reduced measurement
// core: the measurement boundary is NEVER materialized as a Pauli-sum K_b = Σ c_u P_u, and no
// 2^r × 2^r matrix is ever formed.  Every deferred/core rotation is applied to ψ_r (the resident
// 2^r amplitude block) by THIS one primitive, which specializes *naturally* (not via a selector):
//
//     R = α·I + β·(i^pp · X^x Z^z)            (β = -i·sin(θ/2)·..., α = cos(θ/2) for a unit rotation)
//
//       x == 0  →  diagonal: in-place Z-phase per amplitude   (Z-only Pauli; x==0,z==0 is the scalar case)
//       x != 0  →  butterfly: update each pair (j, j^x)        (X/Y support)
//
// There is exactly ONE implementation of this math.  The compiled magic kernel (`direct_rot` in
// cpp/mdm_core_executor.cpp) and the general/oracle path (`NativeDenseEngineState::lincomb`) are the
// SAME primitive — not two algorithms and not a P2/P3/diagonal "candidate selector".  The compiled
// kernel is its FLOP-instrumented fast instantiation over a precompiled rotation plan; the oracle path
// is the direct instantiation.  `nvm_selftest_pauli_apply` proves they are bit-identical.
//
// Cost is O(m · 2^r) for m factors — minimal.  r ≪ k ⇒ ψ_r is small ⇒ win; r = k ⇒ degrades to
// ≈O(m · 2^k) (Clifft-level); r = 0 (empty core) ⇒ the apply loop never runs (the coherent maxM=0 case).
#pragma once
#include <complex>
#include <cstdint>

namespace mdam {

static inline std::complex<double> pa_iphase(int pp) {
    switch (pp & 3) { case 0: return {1,0}; case 1: return {0,1}; case 2: return {-1,0}; default: return {0,-1}; }
}

// Apply R = α·I + β·i^pp X^x Z^z in place to v[0:N].  Bit-identical to direct_rot / lincomb.
inline void pauli_rot_apply(std::complex<double>* v, size_t N, uint64_t x, uint64_t z, int pp,
                            std::complex<double> alpha, std::complex<double> beta) {
    std::complex<double> bph = beta * pa_iphase(pp);
    if (x == 0) {                                   // identity (z==0) / Z-only (diagonal)
        std::complex<double> me = alpha + bph, mo = alpha - bph;
        for (size_t j = 0; j < N; j++) v[j] *= ((__builtin_popcountll(j & z) & 1) ? mo : me);
    } else {                                        // X/Y support: butterfly over (j, j^x)
        uint64_t piv = x & (~x + 1ULL);
        for (size_t j = 0; j < N; j++) {
            if (j & piv) continue;
            size_t kk = j ^ x;
            std::complex<double> a = v[j], b = v[kk];
            double sj = (__builtin_popcountll(j & z) & 1) ? -1.0 : 1.0;
            double sk = (__builtin_popcountll(kk & z) & 1) ? -1.0 : 1.0;
            v[j]  = alpha * a + bph * (sk * b);
            v[kk] = alpha * b + bph * (sj * a);
        }
    }
}

} // namespace mdam
