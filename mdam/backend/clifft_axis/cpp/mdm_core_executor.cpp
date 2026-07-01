// mdm_core_executor.cpp -- compiled numerical hot path for ONE MDAM measurement core.
//
// Executes a whole measurement core in a SINGLE call: build the branch-pair joint from phi_in,
// apply the ordered core rotations with the DIRECT general-Pauli kernel (NO per-rotation
// localization L-R), apply the measurement-axis localization plan (L-M), compute Born, select the
// outcome from a predetermined uniform value, normalize the selected branch, drop the measured
// axis, write the survivor.  No symbolic terms, no per-rotation allocation, one dense apply per
// rotation UID.  Working set = the joint 2^{r_work} (= 2^{r_out+1} for off-diagonal cores) + the
// 2^{r_out} survivor, allocated by the caller and reused.
//
// Build (timing):  g++ -O3 -march=native -DNDEBUG -shared -fPIC -o mdm_core.so mdm_core_executor.cpp
// Build (profile): add -DMDAM_COST_PROFILE=1   (exact primitive counters; excluded from timing build)
//
// Phase convention (MUST match the Python oracle / verified branch-pair prototype):
//   (i^pp X^x Z^z)|j>  ->  amplitude routing  out[k] = i^pp * (-1)^popcount((k^x)&z) * v[k^x]
//   R_P(theta) = cos(theta) I - i sin(theta) P
#include <complex>
#include <cstdint>
#include <cstring>
#include <cmath>

using cd = std::complex<double>;

#ifdef MDAM_COST_PROFILE
#define CNT(field, n) (g_counters.field += (uint64_t)(n))
#else
#define CNT(field, n) ((void)0)
#endif

struct CostCounters {
    uint64_t rotation_count;
    uint64_t diagonal_rotation_calls;
    uint64_t butterfly_rotation_calls;
    uint64_t complex_adds;
    uint64_t complex_multiplies;
    uint64_t real_adds;
    uint64_t real_multiplies;
    uint64_t amplitude_reads;
    uint64_t amplitude_writes;
    uint64_t amplitude_pairs_updated;
    uint64_t bytes_read;
    uint64_t bytes_written;
    uint64_t h_passes;
    uint64_t cnot_passes;
    uint64_t measurement_passes;
    uint64_t norm_passes;
    uint64_t normalization_passes;
    uint64_t survivor_writes;
    uint64_t allocations_in_hot_loop;
};

#ifdef MDAM_COST_PROFILE
static CostCounters g_counters;
#endif

static inline int popc(uint64_t v) { return __builtin_popcountll(v); }
static inline cd iphase(int pp) {
    switch (pp & 3) { case 0: return cd(1,0); case 1: return cd(0,1); case 2: return cd(-1,0); default: return cd(0,-1); }
}

// ----- direct general-Pauli rotation R = cI - i s (i^pp X^x Z^z), applied to v[0:N] in place -----
static void direct_rot(cd* v, long N, uint64_t x, uint64_t z, int pp, double c, double s) {
    cd ph = iphase(pp);
    cd bph = cd(0,-s) * ph;          // -i s i^pp
    if (x == 0) {                    // diagonal: parity phase per amplitude
        CNT(diagonal_rotation_calls, 1);
        cd m_even = cd(c,0) + bph;    // parity 0
        cd m_odd  = cd(c,0) - bph;    // parity 1
        for (long j = 0; j < N; ++j) {
            v[j] *= ((popc((uint64_t)j & z) & 1) ? m_odd : m_even);
            CNT(complex_multiplies, 1); CNT(amplitude_reads, 1); CNT(amplitude_writes, 1);
        }
        CNT(bytes_read, (uint64_t)N*16); CNT(bytes_written, (uint64_t)N*16);
    } else {                         // off-diagonal: butterfly over pairs (j, j^x), j canonical
        CNT(butterfly_rotation_calls, 1);
        uint64_t piv = x & (~x + 1ULL);
        for (long j = 0; j < N; ++j) {
            if ((uint64_t)j & piv) continue;
            long kk = j ^ (long)x;
            cd a = v[j], b = v[kk];
            double sj = (popc((uint64_t)j & z) & 1) ? -1.0 : 1.0;
            double sk = (popc((uint64_t)kk & z) & 1) ? -1.0 : 1.0;
            v[j]  = cd(c,0)*a + bph*(sk*b);
            v[kk] = cd(c,0)*b + bph*(sj*a);
            CNT(complex_multiplies, 4); CNT(complex_adds, 2);
            CNT(amplitude_pairs_updated, 1); CNT(amplitude_reads, 2); CNT(amplitude_writes, 2);
        }
        CNT(bytes_read, (uint64_t)N*16); CNT(bytes_written, (uint64_t)N*16);
    }
}

// ----- L-M Clifford passes on the joint (localize measured Pauli to Z_m) -----
static void h_pass(cd* v, long N, int axis) {
    const double INV = 0.70710678118654752440;
    uint64_t bit = 1ULL << axis;
    for (long j = 0; j < N; ++j) {
        if ((uint64_t)j & bit) continue;
        long k = j | (long)bit;
        cd a = v[j], b = v[k];
        v[j] = (a + b) * INV; v[k] = (a - b) * INV;
        CNT(complex_adds, 2); CNT(complex_multiplies, 2);
        CNT(amplitude_reads, 2); CNT(amplitude_writes, 2);
    }
    CNT(h_passes, 1);
}
static void s_pass(cd* v, long N, int axis, int dag) {
    uint64_t bit = 1ULL << axis; cd m = dag ? cd(0,-1) : cd(0,1);
    for (long j = 0; j < N; ++j) if ((uint64_t)j & bit) { v[j] *= m; CNT(complex_multiplies,1); }
}
static void cnot_pass(cd* v, long N, int ctrl, int tgt) {
    uint64_t cb = 1ULL << ctrl, tb = 1ULL << tgt;
    for (long j = 0; j < N; ++j) {
        if (((uint64_t)j & cb) && !((uint64_t)j & tb)) {
            long k = j ^ (long)tb; cd t = v[j]; v[j] = v[k]; v[k] = t;
            CNT(amplitude_reads, 2); CNT(amplitude_writes, 2);
        }
    }
    CNT(cnot_passes, 1);
}

// L-M op encoding: op_type 0=H(a), 1=S(a), 2=Sdag(a), 3=CNOT(a,b)
struct LMOp { int type; int a; int b; };

// Gate G profiling-only kernel modes (NOT correctness paths; for lower-bound timing experiments):
//   0 = normal general-Pauli butterfly
//  11 = rotation-zero  : skip the core rotation loop (keep build + L-M + Born + project)
//  12 = synthetic-localized : replace each butterfly with ONE Clifft-like diagonal O(N) pass
// Modes 11/12 keep build+L-M+Born+project so the survivor stays a valid normalized state (safe to run
// in-shot; physically wrong but does not corrupt rank/structure).  Mode 0 is the only correctness path.
static int g_kernel_mode = 0;
extern "C" void mdm_set_kernel_mode(int m) { g_kernel_mode = m; }

// Self-test hook: expose the compiled kernel's per-rotation primitive so the VM can prove it is
// bit-identical to native_pauli_apply.hpp's canonical pauli_rot_apply (i.e. the compiled kernel and the
// oracle/general path are ONE normal-form primitive, not two algorithms).  Not on any hot path.
extern "C" void mdm_direct_rot_test(double* vr, long N, unsigned long long x, unsigned long long z,
                                    int pp, double c, double s) {
    direct_rot(reinterpret_cast<cd*>(vr), N, (uint64_t)x, (uint64_t)z, pp, c, s);
}

// Clifft-like synthetic cheap rotation: one diagonal-style O(N) multiply pass (single memory pass,
// no butterfly pairing) at the SAME state size.  Math is meaningless; used only to estimate the cost
// floor "if MDAM applied each rotation with a Clifft-localized single-axis kernel".
static void synthetic_local_rot(cd* v, long N, uint64_t z, double c, double s) {
    cd m_even = cd(c, -s), m_odd = cd(c, s);
    for (long j = 0; j < N; ++j) v[j] *= ((popc((uint64_t)j & z) & 1) ? m_odd : m_even);
}

extern "C" {

// Reset/read profile counters (no-op fields if not profile build).
void mdm_reset_counters() {
#ifdef MDAM_COST_PROFILE
    std::memset(&g_counters, 0, sizeof(g_counters));
#endif
}
void mdm_get_counters(CostCounters* out) {
#ifdef MDAM_COST_PROFILE
    *out = g_counters;
#else
    std::memset(out, 0, sizeof(*out));
#endif
}
int mdm_is_profile_build() {
#ifdef MDAM_COST_PROFILE
    return 1;
#else
    return 0;
#endif
}

// Execute one measurement core.  Arrays are caller-allocated and reused.
//   phi_in        : 2^r_in complex amplitudes (resident state at measurement entry)
//   joint         : scratch, capacity >= 2^r_work (built here; not pre-zeroed needed)
//   survivor      : output, capacity >= 2^r_out
//   r_in,r_mat    : ranks; r_out = r_mat-1 ; r_work = r_mat (joint holds the measured axis)
//   rot_x/z/pp    : per-rotation Pauli over the r_mat bit layout; rot_c/s = cos/sin(theta/2)
//   nrot          : number of core rotations
//   lm_*          : measurement-axis localization plan (type,a,b) length nlm
//   m_bit         : measured (pivot) bit index in the r_mat layout (post L-M, M'=sign*Z_m)
//   sign          : +1 or -1 (M' -> sign*Z_m)
//   mode          : 0=FORCE_ZERO 1=FORCE_ONE 2=USE_RANDOM_VALUE
//   rand_val      : predetermined uniform in [0,1) used iff mode==2
// Returns outcome (0/1); writes p0,p1,norm,survivor_rank via out-params.
int mdm_execute_core(const cd* phi_in, cd* joint, cd* survivor,
                     int r_in, int r_mat,
                     const uint64_t* rot_x, const uint64_t* rot_z, const int* rot_pp,
                     const double* rot_c, const double* rot_s, int nrot,
                     const int* lm_type, const int* lm_a, const int* lm_b, int nlm,
                     int m_bit, double sign, int mode, double rand_val,
                     double* p0_out, double* p1_out, double* norm_out, int* survivor_rank_out) {
    long Nin = 1L << r_in;
    long Njoint = 1L << r_mat;
    // Gate G dense-zero (mode 13, profiling-only): skip the ENTIRE kernel; emit a valid uniform survivor
    // (both branches of every axis populated -> drop_residual stays a no-op -> dominant M-variant kept),
    // outcome from rand_val, rank correct.  Measures the control-plane floor (dense kernel cost = 0).
    if (g_kernel_mode == 13) {
        int r_out = r_mat - 1; long Nout = 1L << r_out;
        double a = (Nout > 0) ? 1.0 / std::sqrt((double)Nout) : 1.0;
        for (long w = 0; w < Nout; ++w) survivor[w] = cd(a, 0);
        int outcome = (mode == 0) ? 0 : (mode == 1) ? 1 : ((rand_val < 0.5) ? 0 : 1);
        *p0_out = 0.5; *p1_out = 0.5; *norm_out = 0.5; *survivor_rank_out = r_out;
        return outcome;
    }
    // build joint = phi_in (low block) tensor |0>_new (high blocks zero).  ONE fill, reused buffer.
    std::memcpy(joint, phi_in, sizeof(cd) * (size_t)Nin);
    std::memset(joint + Nin, 0, sizeof(cd) * (size_t)(Njoint - Nin));
    CNT(allocations_in_hot_loop, 0);
    // ordered core rotations -- each UID exactly once, direct general-Pauli
    if (g_kernel_mode == 11) {
        // rotation-zero: skip the loop entirely (lower-bound profiling)
    } else if (g_kernel_mode == 12) {
        for (int i = 0; i < nrot; ++i) synthetic_local_rot(joint, Njoint, rot_z[i], rot_c[i], rot_s[i]);
    } else {
        for (int i = 0; i < nrot; ++i) {
            direct_rot(joint, Njoint, rot_x[i], rot_z[i], rot_pp[i], rot_c[i], rot_s[i]);
            CNT(rotation_count, 1);
        }
    }
    // measurement-axis localization (L-M): localize M' -> sign*Z_{m_bit}
    for (int i = 0; i < nlm; ++i) {
        switch (lm_type[i]) {
            case 0: h_pass(joint, Njoint, lm_a[i]); break;
            case 1: s_pass(joint, Njoint, lm_a[i], 0); break;
            case 2: s_pass(joint, Njoint, lm_a[i], 1); break;
            case 3: cnot_pass(joint, Njoint, lm_a[i], lm_b[i]); break;
        }
    }
    CNT(measurement_passes, nlm);
    // Born from the two m-branches
    uint64_t mbit = 1ULL << m_bit;
    double s0 = 0.0, s1 = 0.0;
    for (long j = 0; j < Njoint; ++j) {
        double w = std::norm(joint[j]);
        if ((uint64_t)j & mbit) s1 += w; else s0 += w;
        CNT(real_multiplies, 2); CNT(real_adds, 2); CNT(amplitude_reads, 1);
    }
    CNT(norm_passes, 1);
    double tot = s0 + s1;
    // plus-eigenspace (outcome 0) branch:  sign>0 -> m=0 ; sign<0 -> m=1
    int plus_bit = (sign > 0) ? 0 : 1;
    double p0 = (tot > 1e-300) ? ((plus_bit == 0 ? s0 : s1) / tot) : 0.5;
    if (p0 < 0) p0 = 0; if (p0 > 1) p0 = 1;
    int outcome;
    if (mode == 0) outcome = 0; else if (mode == 1) outcome = 1;
    else outcome = (rand_val < p0) ? 0 : 1;
    int keepbit = (outcome == 0) ? plus_bit : (1 - plus_bit);   // which m-block survives
    double keep_norm2 = (keepbit == 0) ? s0 : s1;
    double inv = (keep_norm2 > 1e-300) ? 1.0 / std::sqrt(keep_norm2) : 0.0;
    // drop measured axis: gather the keepbit block (m removed) into survivor 2^{r_out}, normalized
    int r_out = r_mat - 1;
    long Nout = 1L << r_out;
    uint64_t low = mbit - 1ULL;
    for (long w = 0; w < Nout; ++w) {
        long lo = w & (long)low;
        long hi = (w >> m_bit) << (m_bit + 1);
        long full = lo | ((long)keepbit << m_bit) | hi;
        survivor[w] = joint[full] * inv;
        CNT(complex_multiplies, 1); CNT(survivor_writes, 1); CNT(amplitude_writes, 1);
    }
    CNT(normalization_passes, 1);
    *p0_out = p0; *p1_out = 1.0 - p0; *norm_out = keep_norm2; *survivor_rank_out = r_out;
    return outcome;
}

}  // extern "C"
