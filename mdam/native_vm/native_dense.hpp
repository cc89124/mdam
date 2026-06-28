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
    std::vector<cd> resident, joint, survivor;   // all 2^max_work, allocated once
    int r = 0;                                   // current resident rank
    int max_work = 0;

    void init(int max_work_rank) {
        max_work = max_work_rank;
        size_t cap = (size_t)1 << max_work;
        resident.assign(cap, cd(0, 0));
        joint.assign(cap, cd(0, 0));
        survivor.assign(cap, cd(0, 0));
        reset();
    }
    void reset() { r = 0; resident[0] = cd(1, 0); }   // scalar-1 initial state, no realloc

    // load a resident state of rank `rank_` from amplitudes (for snapshot replay / engine resume)
    void set_state(int rank_, const cd* amps) {
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
