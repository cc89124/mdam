#pragma once

// =============================================================================
// CostMeter — read-only algorithmic-FLOP / work accounting for the SVM dense
// kernels.  ADDED FOR BENCHMARKING ONLY.  It never changes any numeric result:
// each kernel, when the meter is enabled, records ONE event at its serial entry
// point (before the OpenMP region) capturing the pre-op active rank and the
// primitive-op counts it is about to perform.  When disabled (default) the only
// cost is one not-taken branch per kernel call.
//
// Primitive-op counts are stored raw; the FLOP convention (complex*complex = 6,
// real*complex = 2, complex add/sub = 2, |z|^2 = 4, conj-mul-acc/vdot = 8) is
// applied in Python so it is IDENTICAL to the bounded backend's convention.
// =============================================================================

#include <array>
#include <cstdint>

namespace clifft {

enum class CostKernel : int {
    ARRAY_CNOT = 0,
    ARRAY_CZ,
    ARRAY_SWAP,
    ARRAY_MULTI_CNOT,
    ARRAY_MULTI_CZ,
    ARRAY_H,
    ARRAY_S,
    ARRAY_S_DAG,
    ARRAY_T,
    ARRAY_T_DAG,
    ARRAY_ROT,
    ARRAY_U2,
    ARRAY_U4,
    EXPAND,
    EXPAND_T,
    EXPAND_T_DAG,
    EXPAND_ROT,
    MEAS_DIAGONAL,
    MEAS_INTERFERE,
    SWAP_MEAS_INTERFERE,
    EXP_VAL,
    COUNT
};

struct KernelStat {
    uint64_t calls = 0;
    uint64_t sum_pow2k = 0;   // sum over calls of 2^(active_k) at entry
    uint64_t processed = 0;   // amplitudes touched (read or written)
    // primitive-op totals (shared FLOP convention applied in Python)
    uint64_t cmul = 0;        // complex * complex
    uint64_t rcmul = 0;       // real   * complex (scale / negate)
    uint64_t cadd = 0;        // complex add/sub
    uint64_t sqmag = 0;       // |z|^2
    uint64_t vdot = 0;        // conj-multiply-accumulate term
    uint64_t rank_sum = 0;    // sum of active_k (for mean)
    uint32_t rank_max = 0;    // peak active_k observed
};

struct CostMeter {
    bool enabled = false;
    std::array<KernelStat, static_cast<int>(CostKernel::COUNT)> stats{};
    // Global rank histogram (§5.4): per active-rank event count, sum 2^k, and FLOP.
    static constexpr int kMaxRank = 40;
    std::array<uint64_t, kMaxRank> events_by_rank{};
    std::array<uint64_t, kMaxRank> pow2k_by_rank{};
    std::array<uint64_t, kMaxRank> flop_by_rank{};

    void reset() {
        for (auto& s : stats)
            s = KernelStat{};
        events_by_rank.fill(0);
        pow2k_by_rank.fill(0);
        flop_by_rank.fill(0);
    }

    inline void record(CostKernel kk, uint32_t k, uint64_t processed, uint64_t cmul,
                        uint64_t rcmul, uint64_t cadd, uint64_t sqmag, uint64_t vdot) {
        KernelStat& s = stats[static_cast<int>(kk)];
        s.calls += 1;
        s.sum_pow2k += (uint64_t{1} << k);
        s.processed += processed;
        s.cmul += cmul;
        s.rcmul += rcmul;
        s.cadd += cadd;
        s.sqmag += sqmag;
        s.vdot += vdot;
        s.rank_sum += k;
        if (k > s.rank_max)
            s.rank_max = k;
        if (k < kMaxRank) {
            events_by_rank[k] += 1;
            pow2k_by_rank[k] += (uint64_t{1} << k);
            flop_by_rank[k] += 6 * cmul + 2 * rcmul + 2 * cadd + 4 * sqmag + 8 * vdot;
        }
    }
};

// Single process-global meter (NOT thread_local): every per-ISA translation
// unit and the bindings share this one instance.  record() is only ever called
// from the serial shot loop (kernels parallelise INTERNALLY, after recording),
// so a plain global is race-free for single-threaded benchmark runs.
CostMeter& cost_meter();

// Stable kernel name for the i-th CostKernel (for the Python snapshot dict).
const char* cost_kernel_name(int i);

}  // namespace clifft

// One-line recorder for kernel bodies. Compiles to a single not-taken branch
// when the meter is disabled.
#define CLIFFT_COST(kk, k, processed, cmul, rcmul, cadd, sqmag, vdot)                    \
    do {                                                                                 \
        if (::clifft::cost_meter().enabled)                                              \
            ::clifft::cost_meter().record((kk), (k), (processed), (cmul), (rcmul),       \
                                          (cadd), (sqmag), (vdot));                       \
    } while (0)

// Amplitude-count helpers in terms of the pre-op active rank k.
#define CLIFFT_POW2K(k) (uint64_t{1} << (k))                                  // 2^k
#define CLIFFT_HALF(k) ((k) ? (uint64_t{1} << ((k) - 1)) : uint64_t{0})       // 2^(k-1)
#define CLIFFT_QUARTER(k) ((k) >= 2 ? (uint64_t{1} << ((k) - 2)) : uint64_t{0})  // 2^(k-2)
