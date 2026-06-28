// native_rng.hpp — bit-exact reproduction of numpy's PCG64 Generator (numpy 2.4.x).
//
// Gate B (Native Batch VM). Verified bit-identical to np.random.default_rng:
//   * next64  = pcg_setseq_128_xsl_rr_64 (step-then-output)
//   * next32  = buffered (low32 returned first, high32 buffered) — numpy next_uint32
//   * dbl     = (next64 >> 11) * 2^-53                            — numpy random_standard_uniform
//   * bounded = Lemire; 32-bit buffered path for rng<=2^32-1, else 64-bit  — numpy integers()
// Seeding (SeedSequence -> initstate/initseq -> srandom) is done ONCE in Python at batch setup
// and the resulting 128-bit (state, inc) is passed in here — so no per-shot Python seeding.
#pragma once
#include <cstdint>
#include <vector>
#include <utility>

namespace mdam {

struct NativeRng {
    __uint128_t state;
    __uint128_t inc;
    uint32_t u32_buf = 0;
    bool has_u32 = false;
    // optional debug draw log: each entry (kind 0=double 1=bounded, value)
    bool dbg_log = false;
    std::vector<std::pair<int,double>> dlog;

    // PCG64 multiplier 0x2360ed051fc65da44385df649fccf645
    static constexpr __uint128_t MULT =
        ((__uint128_t)0x2360ed051fc65da4ULL << 64) | 0x4385df649fccf645ULL;

    // seed from a precomputed numpy PCG64 (state, inc) given as hi/lo 64-bit words
    void seed_from_state(uint64_t state_hi, uint64_t state_lo, uint64_t inc_hi, uint64_t inc_lo) {
        state = ((__uint128_t)state_hi << 64) | state_lo;
        inc   = ((__uint128_t)inc_hi   << 64) | inc_lo;
        has_u32 = false; u32_buf = 0;
    }

    static inline uint64_t rotr64(uint64_t v, unsigned r) {
        r &= 63u;
        return r ? ((v >> r) | (v << (64u - r))) : v;
    }

    inline uint64_t next64() {
        state = state * MULT + inc;                              // step first
        uint64_t hi = (uint64_t)(state >> 64);
        uint64_t lo = (uint64_t)state;
        unsigned rot = (unsigned)(state >> 122);                 // top 6 bits
        return rotr64(hi ^ lo, rot);                             // XSL-RR
    }

    inline uint32_t next32() {
        if (has_u32) { has_u32 = false; return u32_buf; }
        uint64_t n = next64();
        u32_buf = (uint32_t)(n >> 32);                           // buffer HIGH 32
        has_u32 = true;
        return (uint32_t)(n & 0xffffffffu);                      // return LOW 32
    }

    inline double next_double() {
        double d = (double)(next64() >> 11) * (1.0 / 9007199254740992.0);  // 2^-53
        if (dbg_log) dlog.push_back({0, d});
        return d;
    }

    // numpy integers(low, high) == low + bounded(high-low). rng_excl = high - low (# of values).
    inline uint64_t bounded_impl(uint64_t rng_excl) {
        if (rng_excl == 0) return 0;                             // degenerate
        if (rng_excl <= 0x100000000ULL) {                        // rng=rng_excl-1 <= 2^32-1 -> 32-bit Lemire
            uint32_t r = (uint32_t)rng_excl;                     // rng_excl<=2^32 fits when ==2^32? handle below
            if (rng_excl == 0x100000000ULL) {                    // full 32-bit range: plain next32
                return next32();
            }
            uint64_t m = (uint64_t)next32() * (uint64_t)r;
            uint32_t leftover = (uint32_t)m;
            if (leftover < r) {
                uint32_t thr = (uint32_t)((0x100000000ULL - r) % r);
                while (leftover < thr) { m = (uint64_t)next32() * (uint64_t)r; leftover = (uint32_t)m; }
            }
            return (uint32_t)(m >> 32);
        } else {                                                 // 64-bit Lemire
            __uint128_t m = (__uint128_t)next64() * (__uint128_t)rng_excl;
            uint64_t leftover = (uint64_t)m;
            if (leftover < rng_excl) {
                uint64_t thr = (uint64_t)(((__uint128_t)0 - rng_excl) % rng_excl);  // (2^64 - rng) % rng
                while (leftover < thr) { m = (__uint128_t)next64() * (__uint128_t)rng_excl; leftover = (uint64_t)m; }
            }
            return (uint64_t)(m >> 64);
        }
    }
    inline uint64_t bounded(uint64_t rng_excl){ uint64_t v=bounded_impl(rng_excl); if(dbg_log) dlog.push_back({1,(double)v}); return v; }
};

} // namespace mdam
