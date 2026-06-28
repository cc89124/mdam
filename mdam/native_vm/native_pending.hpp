// native_pending.hpp — native pending ledger + packed multi-word Pauli conjugation.
// Ports lazy._conj_h/_conj_s/_conj_cx/_commute_xz to W-word (n>64) packed Paulis and provides a
// UID-indexed ledger with generation-based reset (no per-shot reallocation).  Bit-identical to the
// Python primitives (verified on random multi-word Paulis).
#pragma once
#include <cstdint>
#include <vector>
#include "native_instr.hpp"

namespace mdam {

#ifdef FB_COUNT
// Gate F-B/F5 §12: runtime symbolic operation counters (counting build only; release = no code).
//   tab/pend/inv = forward conjugations (F-B);  *_r = commit folds (F5)
struct FbCounters { long tab=0, pend=0, inv=0;
    long tab_right=0, inv_right=0, foldx=0, consume=0, mupd=0, dropscan=0; };
inline FbCounters& fbc(){ static FbCounters c; return c; }
#endif

// A Pauli over n qubits packed into W = ceil(n/64) uint64 words for X and Z, plus mod-4 phase.
// §5 hot-loop allocation: x/z are FIXED inline arrays (no heap), so the many short-lived
// PackedPaulis created by pauli_mul / tableau & inverse-frame conjugation cost ZERO allocations.
// MAXW=2 covers up to 128 qubits; the native VM (cultivation_d3, n<=15) uses W=1.  Member names
// x/z and x[w]/z[w] indexing are unchanged; only `.x.size()` is replaced by `.W`.
struct PackedPauli {
    static constexpr int MAXW = 2;
    uint64_t x[MAXW], z[MAXW];    // W (<= MAXW) words each, inline (no heap)
    int W = 1;                    // active word count
    uint8_t phase = 0;            // mod 4
    explicit PackedPauli(int W_ = 1) : W(W_) { for (int i = 0; i < MAXW; i++) { x[i] = 0; z[i] = 0; } }
    static inline uint64_t bit(int q) { return 1ULL << (q & 63); }
    static inline int word(int q) { return q >> 6; }
    inline int getx(int q) const { return (int)((x[word(q)] >> (q & 63)) & 1ULL); }
    inline int getz(int q) const { return (int)((z[word(q)] >> (q & 63)) & 1ULL); }
};

// H on qubit q:  x_q,z_q swap; phase += 2*(x_q & z_q)
inline void conj_h(PackedPauli& P, int q) {
    int w = PackedPauli::word(q); uint64_t b = PackedPauli::bit(q);
    int xq = (int)((P.x[w] >> (q & 63)) & 1ULL), zq = (int)((P.z[w] >> (q & 63)) & 1ULL);
    P.x[w] = (P.x[w] & ~b) | ((uint64_t)zq << (q & 63));
    P.z[w] = (P.z[w] & ~b) | ((uint64_t)xq << (q & 63));
    P.phase = (uint8_t)((P.phase + 2 * (xq & zq)) & 3);
}
// S (dag) on qubit q:  z_q ^= x_q; phase += x_q*(dag?3:1)
inline void conj_s(PackedPauli& P, int q, bool dag) {
    int w = PackedPauli::word(q);
    int xq = (int)((P.x[w] >> (q & 63)) & 1ULL);
    P.z[w] ^= ((uint64_t)xq << (q & 63));
    P.phase = (uint8_t)((P.phase + xq * (dag ? 3 : 1)) & 3);
}
// CX(c,t):  x_t ^= x_c ; z_c ^= z_t   (phase unchanged)
inline void conj_cx(PackedPauli& P, int c, int t) {
    int wc = PackedPauli::word(c), wt = PackedPauli::word(t);
    int xc = (int)((P.x[wc] >> (c & 63)) & 1ULL);
    int zt = (int)((P.z[wt] >> (t & 63)) & 1ULL);
    P.x[wt] ^= ((uint64_t)xc << (t & 63));
    P.z[wc] ^= ((uint64_t)zt << (c & 63));
}
// commute(A,B): (popcount(Ax&Bz) + popcount(Az&Bx)) even ?  (true = commute)
inline bool commute(const PackedPauli& A, const PackedPauli& B) {
    int parity = 0;
    for (int i = 0; i < A.W; i++)
        parity += __builtin_popcountll(A.x[i] & B.z[i]) + __builtin_popcountll(A.z[i] & B.x[i]);
    return (parity & 1) == 0;
}

// Pending entry: a physical-basis Pauli rotation, identified by a monotonic uid.
struct PendingEntry {
    PackedPauli p;
    double theta = 0.0;
    uint32_t uid = 0;
    uint32_t generation = 0;   // live iff generation == ledger.gen
};

// UID-indexed ledger.  Entries are appended in increasing-uid order; a generation counter gives
// O(1) shot reset without clearing storage.  consume() marks an entry dead (removes from live).
struct PendingLedger {
    std::vector<PendingEntry> slots;   // indexed by compact uid
    uint32_t gen = 1;
    int W = 1;

    void reset() { gen++; }            // no realloc: all prior entries become stale (generation < gen)

    void create(uint32_t uid, const PackedPauli& p, double theta) {
        if (uid >= slots.size()) slots.resize(uid + 1, PendingEntry{PackedPauli(W), 0.0, 0, 0});
        slots[uid] = PendingEntry{p, theta, uid, gen};
    }
    bool live(uint32_t uid) const { return uid < slots.size() && slots[uid].generation == gen; }
    void consume(uint32_t uid) { if (uid < slots.size()) slots[uid].generation = 0; }  // execute-prefix consume

    // apply a Clifford gate to ALL live pending entries (the per-gate conjugation)
    template <class F> void for_live(F f) { for (auto& e : slots) if (e.generation == gen) { f(e);
#ifdef FB_COUNT
        fbc().pend++;
#endif
    } }
};

} // namespace mdam
