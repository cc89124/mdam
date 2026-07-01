// native_dense.hpp — native dense-state buffer for the MDAM measurement core (Gate B-4).
//
// This is the buffer-management layer around the ALREADY-VERIFIED numerical kernel
// mdm_execute_core (cpp/mdm_core_executor.cpp).  It owns three complex128 buffers preallocated
// ONCE to 2^max_work_rank (resident / joint-scratch / survivor) and NEVER reallocates in the shot
// hot loop; a measurement core reads the resident as phi_in and the kernel writes the survivor,
// after which resident<->survivor are SWAPPED (double buffer) so the survivor becomes the next
// resident with zero copy.  Rank grow (fresh |0> axis embedded by the kernel's joint build),
// branch construction, Born, normalization, measured-axis drop and survivor commit all happen
// inside the verified kernel; this layer only sizes, feeds, and commits.
//
//   ranks:  r_in (resident) -> r_mat (joint, holds the measured axis) -> r_out = r_mat-1 (survivor)
//   r_mat = r_in + (#fresh magic axes the core introduces) ; r_mat > r_in => grow, then drop the
//   measured axis => net rank change r_out - r_in in {-1, 0, +1, ...}.  (r_out+1 == r_mat work.)
#pragma once
#include <complex>
#include <cstdint>
#include <vector>

namespace mdam {

using cd = std::complex<double>;

// Dense FLOP accounting (default-ON, ~free), SAME convention as clifft_flop_from_schedule
// (offdiag=12, diag=6, perm=0, meas=12 per 2^r).  Namespace-level so BOTH the compiled kernel
// (execute_core) and the oracle path (lincomb) charge the SAME counters.  Split: CORE = the
// NECESSARY dense work (rotations + measurement collapse, which Clifft also pays); LOC = the
// localizer Cliffords on the block (an implementation choice; a direct projector would avoid them).
static inline uint64_t& dense_flop_rot(){      static uint64_t c=0; return c; }   // core rotation factors
static inline uint64_t& dense_flop_collapse(){ static uint64_t c=0; return c; }   // Born+project+norm (measurement)
static inline uint64_t& dense_flop_loc(){      static uint64_t c=0; return c; }   // localizer Cliffords on the block
static inline int&      dense_peak_r(){        static int p=0;      return p; }   // peak materialized rank (== max_M)
// CORE (necessary, == Clifft pays) = rotations + collapse.  LOC (localizer) is the separable implementation extra.
static inline uint64_t  dense_flop_core(){ return dense_flop_rot() + dense_flop_collapse(); }

// the verified kernel (linked from cpp/mdm_core_executor.cpp)
extern "C" int mdm_execute_core(const cd* phi_in, cd* joint, cd* survivor,
    int r_in, int r_mat,
    const uint64_t* rot_x, const uint64_t* rot_z, const int* rot_pp,
    const double* rot_c, const double* rot_s, int nrot,
    const int* lm_type, const int* lm_a, const int* lm_b, int nlm,
    int m_bit, double sign, int mode, double rand_val,
    double* p0_out, double* p1_out, double* norm_out, int* survivor_rank_out);

enum CoreMode { FORCE_ZERO = 0, FORCE_ONE = 1, USE_RANDOM = 2 };

struct CoreResult { int outcome; double p0; double norm2; int r_out; };

struct NativeDenseState {
    std::vector<cd> resident, joint, survivor;   // sized to 2^cap_rank (lazy-grow, monotone)
    int r = 0;                                   // current resident rank
    int max_work = 0;                            // soft cap from make_prog (peak_rank+2); informational
    int cap_rank = -1;                           // log2(capacity) actually allocated so far
    static constexpr int INIT_FLOOR = 4;         // start at 2^4 = 16 cd to avoid tiny-realloc churn

    // Grow the three buffers to hold rank `need` (== 2^need amplitudes) if not already.  MONOTONE:
    // capacity only ever increases and PERSISTS across shots (reset() never shrinks), so after the
    // first shot reaches the circuit's peak materialized rank (maxM, the near-Clifford-localized core
    // rank — NOT 2^max_work) every subsequent shot reuses the buffers with ZERO realloc in the hot
    // loop (§11.10 invariant holds in steady state).  resident is PRESERVED across grow (resize keeps
    // its first 2^r amplitudes); joint/survivor are kernel scratch so they are just re-sized to cap.
    void ensure_cap(int need) {
        if (need <= cap_rank) return;
        size_t cap = (size_t)1 << need;
        resident.resize(cap, cd(0, 0));          // preserve existing amplitudes (state lives here)
        joint.resize(cap, cd(0, 0));             // scratch (kernel writes before read)
        survivor.resize(cap, cd(0, 0));          // scratch (kernel writes before read)
        cap_rank = need;
    }

    void init(int max_work_rank) {
        max_work = max_work_rank;
        cap_rank = -1;
        resident.clear(); joint.clear(); survivor.clear();
        ensure_cap(INIT_FLOOR < max_work ? INIT_FLOOR : (max_work > 0 ? max_work : 0));
        reset();
    }
    void reset() { r = 0; resident[0] = cd(1, 0); }   // scalar-1 initial state, no realloc/shrink

    // load a resident state of rank `rank_` from amplitudes (for snapshot replay / engine resume)
    void set_state(int rank_, const cd* amps) {
        ensure_cap(rank_);
        r = rank_; size_t N = (size_t)1 << r;
        for (size_t i = 0; i < N; i++) resident[i] = amps[i];
    }

    // Execute ONE measurement core against the resident state and commit the survivor.
    //   rot_*: per-rotation Pauli (over the r_mat bit layout) + cos/sin(theta/2)
    //   lm_*:  measurement-axis localization plan (type 0=H 1=S 2=Sdag 3=CNOT)
    // r_in is taken from the current rank; the caller supplies r_mat, m_bit, sign.
    CoreResult execute_core(int r_mat,
                            const uint64_t* rot_x, const uint64_t* rot_z, const int* rot_pp,
                            const double* rot_c, const double* rot_s, int nrot,
                            const int* lm_type, const int* lm_a, const int* lm_b, int nlm,
                            int m_bit, double sign, int mode, double rand_val) {
        ensure_cap(r_mat);                       // lazy-grow: joint holds the 2^r_mat materialized core
        // Dense FLOP (same convention as the oracle path + Clifft): each rotation 12(butterfly)/6(diag) x 2^r_mat,
        // one measurement collapse 12 x 2^r_mat -> CORE; localizer h/s 12/6 x 2^r_mat -> LOC (cnot = perm = 0).
        for (int i = 0; i < nrot; i++) dense_flop_rot() += (uint64_t)(rot_x[i] ? 12 : 6) << r_mat;
        dense_flop_collapse() += (uint64_t)12 << r_mat;
        if (r_mat > dense_peak_r()) dense_peak_r() = r_mat;
        for (int i = 0; i < nlm; i++) { int ty = lm_type[i]; if (ty==0) dense_flop_loc() += (uint64_t)12 << r_mat;
                                        else if (ty==1) dense_flop_loc() += (uint64_t)6 << r_mat; }
        double p0, p1, nrm; int srk;
        int outcome = mdm_execute_core(resident.data(), joint.data(), survivor.data(),
                                       r, r_mat, rot_x, rot_z, rot_pp, rot_c, rot_s, nrot,
                                       lm_type, lm_a, lm_b, nlm, m_bit, sign, mode, rand_val,
                                       &p0, &p1, &nrm, &srk);
        std::swap(resident, survivor);   // double-buffer commit: survivor -> resident (zero copy)
        r = srk;
        return CoreResult{outcome, p0, nrm, srk};
    }

    cd amp(size_t k) const { return resident[k]; }   // read a survivor amplitude
};

} // namespace mdam
