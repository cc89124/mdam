// native_seed_expand.hpp — native reproduction of numpy SeedSequence -> PCG64 seeding.
//
// Gate D §3.  The authoritative backend.sample(prog, shots, seed) is (backend.py:937):
//     master = np.random.default_rng(seed)
//     for sh in range(shots):
//         sd = int(master.integers(0, 2**63 - 1))     # Lemire-64 on master stream
//         run_shot(prog, sd)  ->  np.random.default_rng(sd)   # SeedSequence(sd) -> PCG64
// The batch is handed the master (state,inc) ONCE; this header reproduces, per shot and
// entirely in C++, SeedSequence(sd) -> PCG64 (state, inc).  The master Lemire-64 draw is
// NativeRng::bounded(2**63-1).  Verified bit-exact vs numpy in seed_expand_ref.py /
// test_native_seed_expand (seed_expand_ref.txt).
#pragma once
#include <cstdint>

namespace mdam {

// numpy SeedSequence constants (numpy/random/bit_generator.pyx)
static constexpr uint32_t SS_INIT_A = 0x43b0d7e5u;
static constexpr uint32_t SS_MULT_A = 0x931e8875u;
static constexpr uint32_t SS_INIT_B = 0x8b51f9ddu;
static constexpr uint32_t SS_MULT_B = 0x58f38dedu;
static constexpr uint32_t SS_MIX_L  = 0xca01f9ddu;
static constexpr uint32_t SS_MIX_R  = 0x4973f715u;
static constexpr uint32_t SS_XSHIFT = 16u;
static constexpr int      SS_POOL   = 4;

struct SeedExpand {
    // PCG64 (state, inc) 128-bit pair from an integer seed (no spawn key) ==
    //   np.random.default_rng(sd).bit_generator.state['state'] {state, inc}.
    static void seedseq_pcg64(uint64_t sd, __uint128_t& out_state, __uint128_t& out_inc) {
        // assemble entropy: little-endian uint32 words of sd (sd==0 -> one 0 word).  sd<2^64 -> nent<=2.
        uint32_t ent[2]; int nent;
        if (sd == 0) { ent[0] = 0u; nent = 1; }
        else { nent = 0; uint64_t t = sd; while (t) { ent[nent++] = (uint32_t)(t & 0xffffffffu); t >>= 32; } }

        // mix_entropy -> pool[4]
        uint32_t pool[SS_POOL];
        uint32_t hash_const = SS_INIT_A;
        auto hashmix = [&](uint32_t value) -> uint32_t {
            value ^= hash_const;
            hash_const *= SS_MULT_A;
            value *= hash_const;
            value ^= value >> SS_XSHIFT;
            return value;
        };
        auto mix = [&](uint32_t x, uint32_t y) -> uint32_t {
            uint32_t result = (uint32_t)(SS_MIX_L * x) - (uint32_t)(SS_MIX_R * y);
            result ^= result >> SS_XSHIFT;
            return result;
        };
        for (int i = 0; i < SS_POOL; i++) pool[i] = hashmix(i < nent ? ent[i] : 0u);
        for (int i_src = 0; i_src < SS_POOL; i_src++)
            for (int i_dst = 0; i_dst < SS_POOL; i_dst++)
                if (i_src != i_dst) pool[i_dst] = mix(pool[i_dst], hashmix(pool[i_src]));
        for (int i_src = SS_POOL; i_src < nent; i_src++)      // unreachable for sd<2^64 (nent<=2)
            for (int i_dst = 0; i_dst < SS_POOL; i_dst++)
                pool[i_dst] = mix(pool[i_dst], hashmix(ent[i_src]));

        // generate_state(8 uint32) -> 4 uint64 (little-endian pairs)
        uint32_t st32[8];
        hash_const = SS_INIT_B;
        for (int i = 0; i < 8; i++) {
            uint32_t data_val = pool[i % SS_POOL];
            data_val ^= hash_const;
            hash_const *= SS_MULT_B;
            data_val *= hash_const;
            data_val ^= data_val >> SS_XSHIFT;
            st32[i] = data_val;
        }
        uint64_t v[4];
        for (int i = 0; i < 4; i++) v[i] = (uint64_t)st32[2 * i] | ((uint64_t)st32[2 * i + 1] << 32);
        __uint128_t initstate = ((__uint128_t)v[0] << 64) | v[1];
        __uint128_t initseq   = ((__uint128_t)v[2] << 64) | v[3];

        // pcg64 srandom: inc = (initseq<<1)|1; state=0; step; state+=initstate; step.
        static const __uint128_t MULT =
            ((__uint128_t)0x2360ed051fc65da4ULL << 64) | 0x4385df649fccf645ULL;
        __uint128_t inc = (initseq << 1) | (__uint128_t)1;
        __uint128_t state = 0;
        state = state * MULT + inc;
        state += initstate;
        state = state * MULT + inc;
        out_state = state;
        out_inc = inc;
    }
};

} // namespace mdam
