// native_magic_state.hpp — C2-A: native composite near-Clifford dense-engine state.
// Combines the individually-verified native structures into ONE state with no Python references:
//   tableau (Xc/Zc) + inverse_frame (Ax/Az) + pending ledger + dense buffer (phi) + M + record.
// right_h/s/cx update BOTH the tableau and the inverse frame together (mirrors NearClifford.right_*).
#pragma once
#include <vector>
#include <complex>
#include <cmath>
#include "native_pending.hpp"
#include "native_inverse_frame.hpp"
#include "native_tableau.hpp"
#include "native_dense.hpp"
#include "native_record.hpp"

namespace mdam {

struct NativeDenseEngineState {
    int n = 0, W = 0;
    NativePackedTableau tableau;
    NativeInverseFrame inverse_frame{0};
    PendingLedger pending;
    NativeDenseState dense;
    std::vector<int> M;              // ordered magic axes (rank = M.size())
    NativeRecordBuffer record;
    uint32_t rot_uid = 0;            // monotonic rotation uid (== lazy._rot_uid)
    // Gate J Phase-2A+ Step 2: optional commit-foldx capture.  fold_x is the ONLY inverse-frame
    // phase mutation in a commit other than the (static) right-folds; logging the folded qubit q
    // (a byproduct of the existing keepbit/drop decision — NO recomputation) lets the compiled
    // phase_pack reproduce the commit as pp += right_fold_delta(static) + Σ_q 2·z_q(post-mask).
    std::vector<int>* foldx_log = nullptr;   // default null = zero cost
    long ag_fired = 0;                        // Phase-2A+ Step 2: count ag_measure (stabilizer-branch)
                                              // rebuilds — those commits rebuild the inverse phases from
                                              // the tableau (NOT right-folds+foldx), a separate case.
    // §4 scratch for pullback_via_basis (stabilizer-branch GF(2) rebuild) — 0 alloc after first use
    struct BasisEnt { int pb; uint64_t bv, bcm; };
    mutable std::vector<uint64_t> _basis_cvec;
    mutable std::vector<BasisEnt> _basis_bas;

    void init(int n_, int max_work_rank, uint32_t num_measurements) {
        n = n_; W = (n + 63) >> 6;
        tableau.init_identity(n);
        inverse_frame = NativeInverseFrame(n);
        pending.W = W;
        dense.init(max_work_rank);
        record.init(num_measurements);
        M.clear(); M.reserve(n + 4);
        _basis_cvec.reserve(2 * n); _basis_bas.reserve(2 * n);   // §4: stabilizer-rebuild scratch
        rot_uid = 0;
    }

    // reset all state IN PLACE for a new shot (no realloc; dense buffers + tableau/inverse reused)
    void reset_state() {
        tableau.reset_identity();
        for (int i = 0; i < n; i++) {
            for (int w = 0; w < W; w++) { inverse_frame.ax[i].x[w]=0; inverse_frame.ax[i].z[w]=0;
                                          inverse_frame.az[i].x[w]=0; inverse_frame.az[i].z[w]=0; }
            inverse_frame.ax[i].phase=0; inverse_frame.az[i].phase=0;
            inverse_frame.ax[i].x[PackedPauli::word(i)] = PackedPauli::bit(i);
            inverse_frame.az[i].z[PackedPauli::word(i)] = PackedPauli::bit(i);
        }
        pending.reset();
        dense.reset();
        record.reset();
        M.clear();
        rot_uid = 0;
    }

    // ---- forward active Clifford gates (tableau + inverse frame + pending conjugation) ----
    void h(int q) {
        tableau.fwd_h(q); inverse_frame.fwd_h(q);
        pending.for_live([&](PendingEntry& e){ conj_h(e.p, q); });
    }
    void s(int q, bool dag) {
        tableau.fwd_s(q, dag); inverse_frame.fwd_s(q, dag);
        pending.for_live([&](PendingEntry& e){ conj_s(e.p, q, dag); });
    }
    void cx(int c, int t) {
        tableau.fwd_cx(c, t); inverse_frame.fwd_cx(c, t);
        pending.for_live([&](PendingEntry& e){ conj_cx(e.p, c, t); });
    }
    void cz(int a, int b) { h(b); cx(a, b); h(b); }   // == engine.cz

    // ---- Gate F-B: inverse-frame-ONLY forward (the cheap, row-mixing part kept at runtime; the
    // expensive tableau + pending conjugation is region-snapshotted instead).  Same inverse ops/order
    // as the full h/s/cx/cz, so the inverse frame evolves identically. ----
    void h_inv(int q)            { inverse_frame.fwd_h(q); }
    void s_inv(int q, bool dag)  { inverse_frame.fwd_s(q, dag); }
    void cx_inv(int c, int t)    { inverse_frame.fwd_cx(c, t); }
    void cz_inv(int a, int b)    { inverse_frame.fwd_h(b); inverse_frame.fwd_cx(a, b); inverse_frame.fwd_h(b); }

#ifdef FB_COUNT
    #define FB_RIGHT() do{ fbc().tab_right++; fbc().inv_right++; }while(0)
    #define FB_FOLDX() do{ fbc().foldx++; }while(0)
#else
    #define FB_RIGHT()
    #define FB_FOLDX()
#endif
    // ---- Gate F5: inverse-ONLY commit right-folds (the tableau-side mask fold is discarded by the
    // F-B snapshot at the next boundary; only the inverse frame is live and must be folded). ----
    int fb_commit_mode = 0;     // 0 = full commit (tableau+inverse); 1 = inverse-only + skip consume
    // Gate J 2D: when set, the commit right-folds + fold_x skip the inverse frame (it is NOT maintained;
    // phase_pack is updated by the caller via the compiled rfd + foldx_log).  Tableau is still folded
    // (the oracle's ag_measure reads carried tableau phase).  Default false = unchanged.
    bool inverse_off = false;
    void right_h_inv(int s)      { FB_RIGHT(); inverse_frame.right_h(s); }
    void right_s_inv(int s, bool dag) { FB_RIGHT(); inverse_frame.right_s(s, dag); }
    void right_cx_inv(int c, int t)   { FB_RIGHT(); inverse_frame.right_cx(c, t); }
    void fold_x_inv(int r)       { FB_FOLDX(); inverse_frame.fold_x(r); }

    // ---- defer a rotation (physical generator (x,z), phase 0) -> pending (== lazy.apply_rotation) ----
    void apply_rotation(int q, double theta) {
        uint32_t uid = rot_uid++;
        PackedPauli p(W); p.z[PackedPauli::word(q)] = PackedPauli::bit(q);   // Z_q generator (x=0)
        pending.create(uid, p, theta);
    }

    // ---- pullback P=(x,z,0) through the inverse frame (O(weight)) ----
    mutable long pullback_calls = 0;   // Gate J 2C: live-pullback counter (target 0 on compiled path)
    PackedPauli pullback(const PackedPauli& P) const { pullback_calls++; return inverse_frame.pullback(P); }

    // ---- Gate J 2C: no-inverse forward gates (tableau + pending only; inverse NOT forwarded). The
    // inverse frame is instead reconstructed from phase_pack + static masks at each measure boundary. ----
    void h_noinv(int q)            { tableau.fwd_h(q); pending.for_live([&](PendingEntry& e){ conj_h(e.p, q); }); }
    void s_noinv(int q, bool dag)  { tableau.fwd_s(q, dag); pending.for_live([&](PendingEntry& e){ conj_s(e.p, q, dag); }); }
    void cx_noinv(int c, int t)    { tableau.fwd_cx(c, t); pending.for_live([&](PendingEntry& e){ conj_cx(e.p, c, t); }); }
    void cz_noinv(int a, int b)    { h_noinv(b); cx_noinv(a, b); h_noinv(b); }

    // ---- Gate J 2C: reconstruct the live inverse frame from static masks + carried phase_pack, and
    // read it back (phase_pack = (ax[i].phase, az[i].phase)).  Proves phase_pack is the carried state. ----
    void reconstruct_inverse(const std::vector<PackedPauli>& ax_m, const std::vector<PackedPauli>& az_m,
                             const uint8_t* phase_pack) {
        for (int i = 0; i < n; i++) {
            inverse_frame.ax[i] = ax_m[i]; inverse_frame.ax[i].phase = (uint8_t)(phase_pack[i] & 3);
            inverse_frame.az[i] = az_m[i]; inverse_frame.az[i].phase = (uint8_t)(phase_pack[n + i] & 3);
        }
    }
    void read_phase_pack(uint8_t* out) const {
        for (int i = 0; i < n; i++) { out[i] = inverse_frame.ax[i].phase & 3; out[n + i] = inverse_frame.az[i].phase & 3; }
    }
    // Gate J 2D: set ONLY the inverse-frame phases from phase_pack (masks untouched).  Enough for the
    // Imem key on the compiled-magic path (masks are not read when Imem injects rpp/sign) — avoids the
    // full reconstruct.  inverse_off=true then keeps the commit from mutating the (stale-mask) inverse.
    void set_inverse_phases(const uint8_t* phase_pack) {
        for (int i = 0; i < n; i++) { inverse_frame.ax[i].phase = (uint8_t)(phase_pack[i] & 3);
                                      inverse_frame.az[i].phase = (uint8_t)(phase_pack[n + i] & 3); }
    }

    // ---- composite right-folds: tableau + inverse frame together (== eng.right_*) ----
    // Gate J 2D: inverse_off skips the inverse-frame part (phase_pack carries it instead).
    void right_h(int s)  { FB_RIGHT(); tableau.right_h(s);  if(!inverse_off) inverse_frame.right_h(s); }
    void right_s(int s, bool dag) { FB_RIGHT(); tableau.right_s(s, dag); if(!inverse_off) inverse_frame.right_s(s, dag); }
    void right_cx(int c, int t) { FB_RIGHT(); tableau.right_cx(c, t); if(!inverse_off) inverse_frame.right_cx(c, t); }
    void fold_x(int r)   { FB_FOLDX(); if(foldx_log) foldx_log->push_back(r); tableau.fold_x_on_Zc(r); if(!inverse_off) inverse_frame.fold_x(r); }

    // ---- branch sqnorm over the current resident phi (rank = dense.r), bit j, branch ----
    double branch_sqnorm(int j, int branch) const {
        size_t N = (size_t)1 << dense.r; double acc = 0.0;
        for (size_t s = 0; s < N; s++)
            if ((int)((s >> j) & 1) == branch) acc += std::norm(dense.resident[s]);
        return acc;
    }
    // ---- _support_bits over resident phi: OR / AND of indices with |amp|>1e-10 ----
    void support_bits(uint64_t& or_bits, uint64_t& and_bits, bool& any_nz) const {
        size_t N = (size_t)1 << dense.r; or_bits = 0; and_bits = ~0ULL; any_nz = false;
        for (size_t s = 0; s < N; s++)
            if (std::abs(dense.resident[s]) > 1e-10) { or_bits |= s; and_bits &= s; any_nz = true; }
        if (!any_nz) { or_bits = 0; and_bits = 0; }
    }
    // ---- drop one localized product axis a (keep branch survives); mirrors _drop_localized_core ----
    void drop_localized_core(int a /*M index*/, int keep) {
        int r = dense.r;            // current rank (= bit_length(_sz)-1)
        int msb = r - 1;
        if (a != msb) {             // swap axis a to MSB: swap bit a and bit msb across all amps
            size_t N = (size_t)1 << r;
            for (size_t s = 0; s < N; s++) {
                int ba = (int)((s >> a) & 1), bm = (int)((s >> msb) & 1);
                if (ba != bm) {
                    size_t t = s ^ ((1ULL << a) | (1ULL << msb));
                    if (t > s) std::swap(dense.resident[s], dense.resident[t]);
                }
            }
            std::swap(M[a], M[msb]);
        }
        size_t half = (size_t)1 << (r - 1);
        int q = M[msb];
        if (keep == 1)              // kept high half -> move down
            for (size_t i = 0; i < half; i++) dense.resident[i] = dense.resident[half + i];
        dense.r = r - 1;
        M.pop_back();
        if (keep == 1) fold_x(q);   // |1> product -> fold X_q into frame
    }
    // ---- _drop_residual_products: drop every remaining product Z-eigenstate axis ----
    void drop_residual_products() {
        while (!M.empty()) {
#ifdef FB_COUNT
            fbc().dropscan++;
#endif
            if (dense.r < 1) break;
            uint64_t orb, andb; bool any;
            support_bits(orb, andb, any);
            int target_a = -1, keep = -1;
            for (int a = 0; a < (int)M.size(); a++) {
                uint64_t bit = 1ULL << a; int empty;
                if (!(orb & bit)) empty = 1;           // branch_1 empty -> keep 0
                else if (andb & bit) empty = 0;        // branch_0 empty -> keep 1
                else continue;
                if (branch_sqnorm(a, empty) < 1e-20) { target_a = a; keep = 1 - empty; break; }
            }
            if (target_a < 0) break;
            drop_localized_core(target_a, keep);
        }
    }

    int rank() const { return dense.r; }

    // ===== dense-apply primitives on the resident phi (for the ORACLE measurement path) =====
    static inline cd iphase(int pp){ switch(pp&3){case 0:return cd(1,0);case 1:return cd(0,1);case 2:return cd(-1,0);default:return cd(0,-1);} }

    // promote qubit q into M: append axis, new MSB |0> (high block zero), rank++ (== _promote)
    void promote(int q) {
        for (int m : M) if (m == q) return;
        size_t old = (size_t)1 << dense.r;
        for (size_t i = old; i < (old << 1); i++) dense.resident[i] = cd(0,0);
        M.push_back(q); dense.r += 1;
    }
    // phi <- alpha*phi + beta*(P phi), P = i^pp X^mx Z^mz over the M layout (== _pauli_lincomb_inplace,
    // full formula == kernel direct_rot; no diaghalf global-phase shortcut -> may differ by a global
    // phase from Python's diaghalf path, which is record/Born invariant).
    void lincomb(uint64_t mx, uint64_t mz, int pp, cd alpha, cd beta) {
        size_t N = (size_t)1 << dense.r; cd* v = dense.resident.data(); cd bph = beta * iphase(pp);
        if (mx == 0) {
            cd me = alpha + bph, mo = alpha - bph;
            for (size_t j = 0; j < N; j++) v[j] *= ((__builtin_popcountll(j & mz) & 1) ? mo : me);
        } else {
            uint64_t piv = mx & (~mx + 1ULL);
            for (size_t j = 0; j < N; j++) { if (j & piv) continue; size_t kk = j ^ mx;
                cd a = v[j], b = v[kk];
                double sj = (__builtin_popcountll(j & mz) & 1) ? -1.0 : 1.0;
                double sk = (__builtin_popcountll(kk & mz) & 1) ? -1.0 : 1.0;
                v[j] = alpha*a + bph*(sk*b); v[kk] = alpha*b + bph*(sj*a); }
        }
    }
    // single-axis Clifford ops on resident (== _h_axis/_s_axis/_cnot_axes); bit j == axis j
    void h_axis(int j) {
        size_t N = (size_t)1 << dense.r; uint64_t bit = 1ULL << j; const double INV = 0.70710678118654752440;
        cd* v = dense.resident.data();
        for (size_t s = 0; s < N; s++) if (!(s & bit)) { size_t k = s | bit; cd a = v[s], b = v[k]; v[s]=(a+b)*INV; v[k]=(a-b)*INV; }
    }
    void s_axis(int j, bool dag) {
        size_t N = (size_t)1 << dense.r; uint64_t bit = 1ULL << j; cd m = dag ? cd(0,-1) : cd(0,1);
        cd* v = dense.resident.data();
        for (size_t s = 0; s < N; s++) if (s & bit) v[s] *= m;
    }
    void cnot_axes(int jc, int jt) {
        size_t N = (size_t)1 << dense.r; uint64_t cb = 1ULL<<jc, tb = 1ULL<<jt; cd* v = dense.resident.data();
        for (size_t s = 0; s < N; s++) if ((s & cb) && !(s & tb)) { size_t k = s ^ tb; std::swap(v[s], v[k]); }
    }
    // guarded no-op reduce (cultivation_d3: _reduce_full/_find_z_stab never fire); returns false if a
    // parity-slaved qubit WOULD be peeled (so the caller errors instead of silently diverging).
    // ===== Gottesman-Knill stabilizer branch (_ag_measure) + inverse-frame basis rebuild =====
    // _pullback_via_basis(x,z): GF(2) decompose logical P=(x,z) over the frame columns Xc/Zc, then
    // phase-exact image/computational product.  n<=31 (2n-bit column packs into uint64).
    PackedPauli pullback_via_basis(uint64_t x, uint64_t z) const {
        int N = n; uint64_t fmask = (N < 64) ? ((1ULL << N) - 1) : ~0ULL;
        std::vector<uint64_t>& cvec = _basis_cvec; cvec.assign(2 * N, 0);   // scratch, no per-call alloc
        for (int i = 0; i < N; i++) cvec[i]     = (tableau.Xc[i].x[0] & fmask) | ((tableau.Xc[i].z[0] & fmask) << N);
        for (int i = 0; i < N; i++) cvec[N + i] = (tableau.Zc[i].x[0] & fmask) | ((tableau.Zc[i].z[0] & fmask) << N);
        using B = BasisEnt; std::vector<B>& bas = _basis_bas; bas.clear();
        for (int j = 0; j < 2 * N; j++) { uint64_t cur = cvec[j], cm = 1ULL << j;
            for (auto& b : bas) if ((cur >> b.pb) & 1) { cur ^= b.bv; cm ^= b.bcm; }
            if (cur) { int pb = __builtin_ctzll(cur); bas.push_back({pb, cur, cm}); } }
        uint64_t curt = (x & fmask) | ((z & fmask) << N), coeff = 0;
        for (auto& b : bas) if ((curt >> b.pb) & 1) { curt ^= b.bv; coeff ^= b.bcm; }
        PackedPauli Q(W), R(W);
        for (int j = 0; j < 2 * N; j++) if ((coeff >> j) & 1) {
            if (j < N) { Q = pauli_mul(Q, tableau.Xc[j]); PackedPauli Xj(W); Xj.x[0] = 1ULL << j; R = pauli_mul(R, Xj); }
            else { Q = pauli_mul(Q, tableau.Zc[j - N]); PackedPauli Zj(W); Zj.z[0] = 1ULL << (j - N); R = pauli_mul(R, Zj); }
        }
        PackedPauli res(W); res.x[0] = R.x[0]; res.z[0] = R.z[0];
        res.phase = (uint8_t)(((int)R.phase - (int)Q.phase) & 3);
        return res;
    }
    void rebuild_inverse_frame() {
        for (int i = 0; i < n; i++) { inverse_frame.ax[i] = pullback_via_basis(1ULL << i, 0);
                                      inverse_frame.az[i] = pullback_via_basis(0, 1ULL << i); }
    }
    // _ag_measure(Pm, anti_s): project the stabilizer tableau; magic register untouched; rebuild inverse.
    void ag_measure(const PackedPauli& Pm, int p, int out) {
        ag_fired++;
        PackedPauli Sp = tableau.Zc[p];
        for (int i = 0; i < n; i++) {
            if (i != p && !commute(tableau.Zc[i], Pm)) tableau.Zc[i] = pauli_mul(tableau.Zc[i], Sp);
            if (!commute(tableau.Xc[i], Pm)) tableau.Xc[i] = pauli_mul(tableau.Xc[i], Sp);
        }
        tableau.Xc[p] = Sp;
        PackedPauli nz(W); nz.x[0] = Pm.x[0]; nz.z[0] = Pm.z[0]; nz.phase = (uint8_t)((Pm.phase + 2 * out) & 3);
        tableau.Zc[p] = nz;
        rebuild_inverse_frame();      // AG projection has no incremental rule -> rebuild from basis
    }

    bool reduce_full_is_noop() {
        // _find_z_stab finds a qubit q whose Z is a Z-parity of OTHER magic qubits (a product).
        // Detect any axis that is a pure Z-product across the state -> would be peeled. Conservative:
        // if every single-axis branch has BOTH branches populated, nothing is a product -> no-op.
        for (size_t a = 0; a < M.size(); a++) {
            double s0 = branch_sqnorm((int)a, 0), s1 = branch_sqnorm((int)a, 1);
            if (s0 < 1e-20 || s1 < 1e-20) return false;   // a product axis exists -> NOT a no-op
        }
        return true;
    }
};

} // namespace mdam
