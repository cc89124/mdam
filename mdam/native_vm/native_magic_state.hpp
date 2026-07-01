// native_magic_state.hpp — C2-A: native composite near-Clifford dense-engine state.
// Combines the individually-verified native structures into ONE state with no Python references:
//   tableau (Xc/Zc) + inverse_frame (Ax/Az) + pending ledger + dense buffer (phi) + M + record.
// right_h/s/cx update BOTH the tableau and the inverse frame together (mirrors NearClifford.right_*).
#pragma once
#include <vector>
#include <complex>
#include <cmath>
#include <map>
#include <unordered_map>
#include "native_pending.hpp"
#include "native_inverse_frame.hpp"
#include "native_tableau.hpp"
#include "native_dense.hpp"
#include "native_record.hpp"
#include "native_pauli_apply.hpp"
#include "native_invframe_static.hpp"   // clean-room inverse-frame rebuild StaticPlan (rb_*; default OFF)

namespace mdam {

// ===== Step 1: pullback mask-invariance checker (StaticPlan premise test, default OFF) =====
// For each (boundary mp, request kind, input physical Pauli), is the pulled-back (x,z) MASK shot-static
// (only sign/phase dynamic)?  Aggregated in C++ (no per-call log).  kinds: 0 PLAN_Pm, 1 oracle_Pm,
// 2 flush_pullback, 3 PLAN_rot.  pb_mp set per boundary in measure_z; pb_kind set at each call site.
struct PbKey { uint64_t tag, ix0, ix1, iz0, iz1;
    bool operator<(const PbKey& o) const {
        if(tag!=o.tag)return tag<o.tag; if(ix0!=o.ix0)return ix0<o.ix0; if(ix1!=o.ix1)return ix1<o.ix1;
        if(iz0!=o.iz0)return iz0<o.iz0; return iz1<o.iz1; } };
struct PbVal { uint64_t ox0=0,ox1=0,oz0=0,oz1=0; int phase0=0; bool phase_varies=false; long calls=0, mask_viol=0;
    int presid0=0; bool presid_set=false; long phase_affine_viol=0; };   // Phase B: residual = out_phase - Σ ax/az phase (mod4)
inline bool& pb_cap_on(){ static bool v=false; return v; }
inline int&  pb_mp(){ static int v=0; return v; }
inline int&  pb_kind(){ static int v=0; return v; }
inline std::map<PbKey,PbVal>& pb_map(){ static std::map<PbKey,PbVal> m; return m; }

// ===== Phase B: pullback StaticPlan (default OFF; premise PROVEN — Step1 mask + Phase-B affine, 0 viol) =====
// For key (mp,kind,in_x,in_z): out mask is shot-static, and
//   out_phase = (c_static + Σ_{q∈Sx} ax[q].phase + Σ_{q∈Sz} az[q].phase) mod 4.
// Fast path replaces inverse_frame.pullback(P)'s O(weight·W) subst with map-lookup + O(weight) phase sum.
struct PbStaticEnt {
    uint64_t ox[2]={0,0}, oz[2]={0,0};       // shot-static output mask
    uint8_t  c_static=0;                      // phase residual base (mod 4)
    std::vector<uint16_t> sx, sz;             // input-Pauli support = affine-phase coefficient sets
    long calls=0, shadow_mask_viol=0, shadow_phase_viol=0;
    bool built=false;
};
inline int&  pb_static_on(){ static int v=0; return v; }       // --mdam-static-pullback: use fast path
inline int&  pb_static_shadow(){ static int v=0; return v; }   // --mdam-static-pullback-shadow: verify vs live every call
inline int&  pb_static_phase(){ static int v=1; return v; }    // --mdam-static-pullback-phase: 1=affine(Phase B), 0=live phase(Phase A)
struct PbKeyHash { size_t operator()(const PbKey& k) const {
    uint64_t h=k.tag; auto mix=[&](uint64_t x){ x^=h; x*=0x9E3779B97F4A7C15ULL; x^=x>>29; h=x; };
    mix(k.ix0); mix(k.ix1); mix(k.iz0); mix(k.iz1); return (size_t)h; } };
struct PbKeyEq { bool operator()(const PbKey&a,const PbKey&b) const {
    return a.tag==b.tag&&a.ix0==b.ix0&&a.ix1==b.ix1&&a.iz0==b.iz0&&a.iz1==b.iz1; } };
inline std::unordered_map<PbKey,PbStaticEnt,PbKeyHash,PbKeyEq>& pb_static_map(){
    static std::unordered_map<PbKey,PbStaticEnt,PbKeyHash,PbKeyEq> m; return m; }
// Direct-mapped cache in front of the map: the per-call hash+bucket+node-chase (~97cyc) was the
// bottleneck (> the subst it replaces).  unordered_map element pointers are stable across rehash
// (node-based; we never erase), so caching PbStaticEnt* is safe.  Hit path = 1 hash + 1 key compare.
static constexpr int PBCACHE_BITS=12, PBCACHE_SZ=1<<PBCACHE_BITS, PBCACHE_MASK=PBCACHE_SZ-1;
struct PbCacheSlot { PbKey key{}; PbStaticEnt* ent=nullptr; bool valid=false; };
inline std::vector<PbCacheSlot>& pb_cache(){ static std::vector<PbCacheSlot> c(PBCACHE_SZ); return c; }
inline PbStaticEnt& pb_lookup(const PbKey& k){
    size_t h = PbKeyHash{}(k) & PBCACHE_MASK;
    PbCacheSlot& sl = pb_cache()[h];
    if(sl.valid && PbKeyEq{}(sl.key,k)) return *sl.ent;
    PbStaticEnt& e = pb_static_map()[k];     // miss: resolve + fill slot (pointer stable across rehash)
    sl.key=k; sl.ent=&e; sl.valid=true; return e;
}
struct PbShadowFail { bool hit=false; int kind=0,mp=0;
    uint64_t ix0=0,ix1=0,iz0=0,iz1=0, sox0=0,sox1=0,soz0=0,soz1=0, lox0=0,lox1=0,loz0=0,loz1=0;
    int sphase=0,lphase=0; };
inline PbShadowFail& pb_shadow_fail(){ static PbShadowFail f; return f; }
// rebuild-vs-substitution cycle split (default OFF) — explains the pullback region composition
inline bool&     pb_time_on(){ static bool v=false; return v; }
inline uint64_t& pb_rebuild_cyc(){ static uint64_t v=0; return v; }
inline uint64_t& pb_rebuild_cnt(){ static uint64_t v=0; return v; }
inline uint64_t& pb_subst_cyc(){ static uint64_t v=0; return v; }
inline uint64_t& pb_subst_cnt(){ static uint64_t v=0; return v; }
inline uint64_t& pb_lookup_cyc(){ static uint64_t v=0; return v; }   // static path: map lookup
inline uint64_t& pb_affine_cyc(){ static uint64_t v=0; return v; }   // static path: mask copy + affine phase sum


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
    static constexpr int GFW = 4;   // 2N-bit GF(2) words (PackedPauli MAXW=2 -> n<=128 -> 2N<=256 -> 4 words)
    struct BasisEnt { int pb; uint64_t bv[GFW], bcm[GFW]; };
    mutable std::vector<uint64_t> _basis_cvec;
    mutable std::vector<BasisEnt> _basis_bas;
    mutable uint64_t _last_coeff[GFW] = {0};   // rb checker: GF(2) basis-decomp support of last pullback_from_basis

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
        inv_dirty = false; basis_valid = false;   // fresh identity inverse frame this shot
        if(rb_cap_on()){ rb_count_hist()[rb_epoch()]++; rb_epoch()=0; }   // rb checker: log rebuilds/shot, reset epoch
    }

    // ---- forward active Clifford gates (tableau + inverse frame + pending conjugation) ----
    void h(int q) {
        tableau.fwd_h(q); if(lazy_inverse){ inv_dirty=true; basis_valid=false; } else inverse_frame.fwd_h(q);
        pending.for_live([&](PendingEntry& e){ conj_h(e.p, q); });
    }
    void s(int q, bool dag) {
        tableau.fwd_s(q, dag); if(lazy_inverse){ inv_dirty=true; basis_valid=false; } else inverse_frame.fwd_s(q, dag);
        pending.for_live([&](PendingEntry& e){ conj_s(e.p, q, dag); });
    }
    void cx(int c, int t) {
        tableau.fwd_cx(c, t); if(lazy_inverse){ inv_dirty=true; basis_valid=false; } else inverse_frame.fwd_cx(c, t);
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
    // ---- Lazy inverse frame (authoritative run() only) ----
    // The inverse frame is a pure function of the tableau (rebuild_inverse_frame reconstructs it).  The
    // eager maintenance (incremental fwd in h/s/cx + full rebuild in ag_measure) materializes ALL 2n
    // entries, but the oracle/magic_plan READS only a few via pullback().  When lazy_inverse is set
    // (authoritative path), tableau-mutating ops only mark the frame dirty; pullback() then computes the
    // needed Pauli on-demand from the tableau basis (build once, back-substitute per read).  For maxM=0
    // circuits the frame is never read -> 0 rebuilds (was 1 O(n^2) rebuild per measurement).  The FAST
    // paths leave lazy_inverse=false -> behaviour byte-identical to before.
    bool lazy_inverse = false;
    bool magic_ever = false;            // high-water: any magic axis materialized (promote)? -> adaptive lazy policy
    mutable bool inv_dirty = false;     // frame deferred: rebuild/pullback from the live tableau on read
    mutable bool basis_valid = false;   // _basis_bas matches the current tableau (cache across pullbacks)
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

    // ---- general single-qubit Pauli rotation, generator P = X^px Z^pz on qubit q, phase 0.
    // Mirrors nc.apply_rotation(px<<q, pz<<q, theta) used by the fused-U2/U4 ZXZ application
    // (Rz: px=0,pz=1 ; Rx: px=1,pz=0).  Same deferred-pending path as apply_rotation; the only new
    // case vs the diagonal T/ROT path is the X-generator (px=1) which the kernel/pullback already
    // handle (rot_x masks). ----
    void apply_rotation_pauli(int q, int px, int pz, double theta) {
        uint32_t uid = rot_uid++;
        PackedPauli p(W);
        if (px) p.x[PackedPauli::word(q)] = PackedPauli::bit(q);
        if (pz) p.z[PackedPauli::word(q)] = PackedPauli::bit(q);
        pending.create(uid, p, theta);
    }

    // ---- pullback P=(x,z,0) through the inverse frame (O(weight)) ----
    mutable long pullback_calls = 0;   // Gate J 2C: live-pullback counter (target 0 on compiled path)
    PackedPauli pullback(const PackedPauli& P) const { pullback_calls++;
        // Lazy: on the FIRST read after a deferred mutation, materialize the full frame from the live
        // tableau (== what eager ag_measure/fwd would have produced), then use the O(weight) frame product.
        // This is bit-identical to eager (NOT the cheaper direct pullback_from_basis, whose multi-bit
        // multiply order gives a different phase than the frame's single-bit subst).  When the frame is
        // never read (maxM=0 / empty core) the rebuild never happens -> the whole win, with no divergence.
        if(inv_dirty) const_cast<NativeDenseEngineState*>(this)->rebuild_inverse_frame();

        // ---- Phase B: StaticPlan fast path (default OFF) ----
        if(pb_static_on()){
            int Wf = inverse_frame.W;
            uint64_t _tl0 = pb_time_on()? __builtin_ia32_rdtsc() : 0;
            PbKey k{ ((uint64_t)(uint32_t)pb_mp()<<8) | (uint64_t)(uint8_t)pb_kind(), P.x[0],P.x[1],P.z[0],P.z[1] };
            PbStaticEnt& e = pb_lookup(k);
            if(pb_time_on()){ pb_lookup_cyc()+=__builtin_ia32_rdtsc()-_tl0; }
            bool live_needed = (!e.built) || pb_static_shadow() || !pb_static_phase();
            PackedPauli Rlive(Wf);
            if(live_needed) Rlive = inverse_frame.pullback(P);
            if(!e.built){                                   // warm capture (first call per key)
                e.ox[0]=Rlive.x[0]; e.ox[1]=Rlive.x[1]; e.oz[0]=Rlive.z[0]; e.oz[1]=Rlive.z[1];
                int pred=0;
                for(int wi=0;wi<Wf;wi++){ uint64_t xi=P.x[wi]; while(xi){int bb=__builtin_ctzll(xi); xi&=xi-1; int idx=(wi<<6)+bb; e.sx.push_back((uint16_t)idx); pred+=inverse_frame.ax[idx].phase;} }
                for(int wi=0;wi<Wf;wi++){ uint64_t zi=P.z[wi]; while(zi){int bb=__builtin_ctzll(zi); zi&=zi-1; int idx=(wi<<6)+bb; e.sz.push_back((uint16_t)idx); pred+=inverse_frame.az[idx].phase;} }
                e.c_static=(uint8_t)((((int)Rlive.phase) - pred)&3);
                e.built=true;
            }
            uint64_t _ta0 = pb_time_on()? __builtin_ia32_rdtsc() : 0;
            PackedPauli R(Wf);
            R.x[0]=e.ox[0]; R.x[1]=e.ox[1]; R.z[0]=e.oz[0]; R.z[1]=e.oz[1];
            if(pb_static_phase()){                          // Phase B: affine phase, no live subst
                int s=e.c_static;
                for(uint16_t q : e.sx) s+=inverse_frame.ax[q].phase;
                for(uint16_t q : e.sz) s+=inverse_frame.az[q].phase;
                R.phase=(uint8_t)(s&3);
            } else R.phase = Rlive.phase;                   // Phase A: static mask, live phase
            if(pb_time_on()){ pb_affine_cyc()+=__builtin_ia32_rdtsc()-_ta0; }
            if(pb_static_shadow() && live_needed){          // shadow verify static vs live
                bool mok=(R.x[0]==Rlive.x[0]&&R.x[1]==Rlive.x[1]&&R.z[0]==Rlive.z[0]&&R.z[1]==Rlive.z[1]);
                bool pok=(R.phase==Rlive.phase);
                if(!mok) e.shadow_mask_viol++;
                if(!pok) e.shadow_phase_viol++;
                if((!mok||!pok) && !pb_shadow_fail().hit){
                    PbShadowFail& f=pb_shadow_fail(); f.hit=true; f.kind=pb_kind(); f.mp=pb_mp();
                    f.ix0=P.x[0]; f.ix1=P.x[1]; f.iz0=P.z[0]; f.iz1=P.z[1];
                    f.sox0=R.x[0]; f.sox1=R.x[1]; f.soz0=R.z[0]; f.soz1=R.z[1];
                    f.lox0=Rlive.x[0]; f.lox1=Rlive.x[1]; f.loz0=Rlive.z[0]; f.loz1=Rlive.z[1];
                    f.sphase=R.phase; f.lphase=Rlive.phase;
                }
            }
            e.calls++;
            return R;
        }

        uint64_t _ts0 = pb_time_on()? __builtin_ia32_rdtsc() : 0;
        PackedPauli R = inverse_frame.pullback(P);
        if(pb_time_on()){ pb_subst_cyc()+=__builtin_ia32_rdtsc()-_ts0; pb_subst_cnt()++; }
        if(pb_cap_on()){
            PbKey k{ ((uint64_t)(uint32_t)pb_mp()<<8) | (uint64_t)(uint8_t)pb_kind(), P.x[0],P.x[1],P.z[0],P.z[1] };
            PbVal& v = pb_map()[k];
            // Phase B premise: out_phase = (Σ involved ax/az phase) + static_cross_const (mod 4).  Test that the
            // residual (out_phase - Σ ax/az phase mod 4) is CONSTANT per key -> phase is an EXACT affine fn of
            // the inverse-frame phase vector (coefficients = input support, no fitting needed).
            int pred=0;
            for(int wi=0;wi<inverse_frame.W;wi++){ uint64_t xi=P.x[wi]; while(xi){int bb=__builtin_ctzll(xi); xi&=xi-1; pred+=inverse_frame.ax[(wi<<6)+bb].phase;} }
            for(int wi=0;wi<inverse_frame.W;wi++){ uint64_t zi=P.z[wi]; while(zi){int bb=__builtin_ctzll(zi); zi&=zi-1; pred+=inverse_frame.az[(wi<<6)+bb].phase;} }
            int resid=((int)R.phase - pred)&3;
            if(v.calls==0){ v.ox0=R.x[0]; v.ox1=R.x[1]; v.oz0=R.z[0]; v.oz1=R.z[1]; v.phase0=R.phase; v.presid0=resid; v.presid_set=true; }
            else { if(v.ox0!=R.x[0]||v.ox1!=R.x[1]||v.oz0!=R.z[0]||v.oz1!=R.z[1]) v.mask_viol++;
                   if((int)R.phase!=v.phase0) v.phase_varies=true;
                   if(resid!=v.presid0) v.phase_affine_viol++; }
            v.calls++;
        }
        return R; }

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
        dense.ensure_cap(dense.r + 1);                    // lazy-grow: rank++ writes the high block
        size_t old = (size_t)1 << dense.r;
        for (size_t i = old; i < (old << 1); i++) dense.resident[i] = cd(0,0);
        M.push_back(q); dense.r += 1; magic_ever = true;   // adaptive lazy policy: this circuit materializes magic
    }
    // phi <- alpha*phi + beta*(P phi), P = i^pp X^mx Z^mz over the M layout (== _pauli_lincomb_inplace,
    // full formula == kernel direct_rot; no diaghalf global-phase shortcut -> may differ by a global
    // phase from Python's diaghalf path, which is record/Born invariant).
    // Oracle/general instantiation of THE single reduced-core Pauli-apply primitive (native_pauli_apply.hpp).
    // Identical math to the compiled kernel's direct_rot; diagonal(mx==0)/butterfly(mx!=0) are its natural
    // branches, not a selector.  core_apply_calls counts factors actually swept (==0 for an empty core, the
    // coherent maxM=0 / r=0 case).
    static inline uint64_t& core_apply_count(){ static uint64_t c=0; return c; }
    // dense_flop_core/loc are namespace-level (native_dense.hpp) so the compiled kernel + oracle share them.
    void lincomb(uint64_t mx, uint64_t mz, int pp, cd alpha, cd beta) {
        core_apply_count()++;
        dense_flop_rot() += (uint64_t)(mx ? 12 : 6) << dense.r;
        if (dense.r > dense_peak_r()) dense_peak_r() = dense.r;
        pauli_rot_apply(dense.resident.data(), (size_t)1 << dense.r, mx, mz, pp, alpha, beta);
    }
    // single-axis Clifford ops on resident (== _h_axis/_s_axis/_cnot_axes); bit j == axis j
    void h_axis(int j) {
        dense_flop_loc() += (uint64_t)12 << dense.r;
        size_t N = (size_t)1 << dense.r; uint64_t bit = 1ULL << j; const double INV = 0.70710678118654752440;
        cd* v = dense.resident.data();
        for (size_t s = 0; s < N; s++) if (!(s & bit)) { size_t k = s | bit; cd a = v[s], b = v[k]; v[s]=(a+b)*INV; v[k]=(a-b)*INV; }
    }
    void s_axis(int j, bool dag) {
        dense_flop_loc() += (uint64_t)6 << dense.r;
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
    // Build the GF(2) echelon basis of the 2N tableau columns into _basis_bas.  Depends ONLY on the
    // tableau (Xc/Zc) -> shot-constant across all pullbacks between two tableau mutations, so build it
    // ONCE per rebuild instead of once per pulled Pauli.  (rebuild_inverse_frame was O(n^3): 2n calls
    // each re-running this O(n^2) elimination.  Now O(n^2): one elimination + 2n O(n) back-substitutions.)
    // MULTIWORD (n>32 safe): the 2N-bit GF(2) column/coeff vectors span nw2=ceil(2N/64) words (was a
    // single uint64 -> overflowed/truncated for n>32, e.g. d5_r5 n=72 -> 2N=144).  n<=32 -> nw2=1 (loops
    // one word, identical cost to the old single-uint64 path -> no regression for the small benches).
    void build_inverse_basis() const {
        int N = n, nw2 = (2 * N + 63) >> 6;
        std::vector<uint64_t>& cvec = _basis_cvec; cvec.assign((size_t)2 * N * nw2, 0);   // flat: col j @ j*nw2
        auto sb = [](uint64_t* w, int b){ w[b >> 6] |= (1ULL << (b & 63)); };
        for (int i = 0; i < N; i++) { uint64_t* col = &cvec[(size_t)i * nw2];
            for (int q = 0; q < N; q++) { if (tableau.Xc[i].getx(q)) sb(col, q); if (tableau.Xc[i].getz(q)) sb(col, N + q); } }
        for (int i = 0; i < N; i++) { uint64_t* col = &cvec[(size_t)(N + i) * nw2];
            for (int q = 0; q < N; q++) { if (tableau.Zc[i].getx(q)) sb(col, q); if (tableau.Zc[i].getz(q)) sb(col, N + q); } }
        std::vector<BasisEnt>& bas = _basis_bas; bas.clear();
        for (int j = 0; j < 2 * N; j++) {
            uint64_t cur[GFW] = {0}, cm[GFW] = {0};
            for (int w = 0; w < nw2; w++) cur[w] = cvec[(size_t)j * nw2 + w];
            cm[j >> 6] = (1ULL << (j & 63));
            for (auto& b : bas) if ((cur[b.pb >> 6] >> (b.pb & 63)) & 1) { for (int w = 0; w < nw2; w++) { cur[w] ^= b.bv[w]; cm[w] ^= b.bcm[w]; } }
            int pb = -1; for (int w = 0; w < nw2; w++) if (cur[w]) { pb = w * 64 + __builtin_ctzll(cur[w]); break; }
            if (pb >= 0) { BasisEnt e; e.pb = pb; for (int w = 0; w < GFW; w++) { e.bv[w] = cur[w]; e.bcm[w] = cm[w]; } bas.push_back(e); }
        }
    }
    // Pull back logical P (X^x Z^z) using the PREBUILT _basis_bas (caller must build_inverse_basis() first).
    PackedPauli pullback_from_basis(const PackedPauli& P) const {
        int N = n, nw2 = (2 * N + 63) >> 6;
        uint64_t curt[GFW] = {0}, coeff[GFW] = {0};
        for (int q = 0; q < N; q++) { if (P.getx(q)) curt[q >> 6] |= 1ULL << (q & 63);
                                      if (P.getz(q)) curt[(N + q) >> 6] |= 1ULL << ((N + q) & 63); }
        for (auto& b : _basis_bas) if ((curt[b.pb >> 6] >> (b.pb & 63)) & 1) { for (int w = 0; w < nw2; w++) { curt[w] ^= b.bv[w]; coeff[w] ^= b.bcm[w]; } }
        PackedPauli Q(W), R(W);
        for (int j = 0; j < 2 * N; j++) if ((coeff[j >> 6] >> (j & 63)) & 1) {
            if (j < N) { Q = pauli_mul(Q, tableau.Xc[j]); PackedPauli Xj(W); Xj.x[PackedPauli::word(j)] = PackedPauli::bit(j); R = pauli_mul(R, Xj); }
            else { int jj = j - N; Q = pauli_mul(Q, tableau.Zc[jj]); PackedPauli Zj(W); Zj.z[PackedPauli::word(jj)] = PackedPauli::bit(jj); R = pauli_mul(R, Zj); }
        }
        PackedPauli res(W); for (int w = 0; w < W; w++) { res.x[w] = R.x[w]; res.z[w] = R.z[w]; }
        res.phase = (uint8_t)(((int)R.phase - (int)Q.phase) & 3);
        if(rb_cap_on()||rb_static_on()) for(int w=0;w<GFW;w++) _last_coeff[w]=coeff[w];   // rb checker/plan: capture decomp support
        return res;
    }
    // rb checker: after a generator is pulled, record its mask + phase-residual under key (epoch,g,az).
    // residual = (out_phase + Σ_{j∈coeff} tableau_phase[j]) mod 4 — constant per key proves phase is an
    // affine fn of the tableau generator phases (from res.phase = R.phase - Q.phase, Q carries the phases).
    // signature of the tableau MASKS only (Xc/Zc x,z bits; phases excluded) — FNV-1a over W words/gen.
    uint64_t rb_mask_sig() const {
        uint64_t h=1469598103934665603ULL;
        auto mix=[&](uint64_t v){ h^=v; h*=1099511628211ULL; };
        for(int i=0;i<n;i++){ for(int w=0;w<W;w++){ mix(tableau.Xc[i].x[w]); mix(tableau.Xc[i].z[w]);
                                                    mix(tableau.Zc[i].x[w]); mix(tableau.Zc[i].z[w]); } }
        return h;
    }
    void rb_record(uint64_t sig, int g, int az, const PackedPauli& out) const {
        int N=n, nw2=(2*N+63)>>6, psum=0;
        for(int w=0; w<nw2; w++){ uint64_t cw=_last_coeff[w]; while(cw){ int b=__builtin_ctzll(cw); cw&=cw-1; int j=(w<<6)+b;
            if(j<N) psum += tableau.Xc[j].phase; else psum += tableau.Zc[j-N].phase; } }
        int resid = ((int)out.phase + psum) & 3;
        RbKey k{ sig, (uint16_t)g, (uint8_t)az };
        RbVal& v = rb_map()[k];
        if(v.calls==0){ v.x0=out.x[0]; v.x1=out.x[1]; v.z0=out.z[0]; v.z1=out.z[1]; v.phase0=out.phase;
                        v.resid0=resid; v.resid_set=true; }
        else { if(v.x0!=out.x[0]||v.x1!=out.x[1]||v.z0!=out.z[0]||v.z1!=out.z[1]) v.mask_viol++;
               if((int)out.phase!=v.phase0) v.phase_varies=true;
               if(resid!=v.resid0) v.phase_affine_viol++; }
        v.calls++;
    }
    // fast path: fill an inverse generator from a cached plan (static mask + affine mod-4 phase).
    void rb_apply_gen(PackedPauli& out, const RbGenPlan& g) const {
        out.x[0]=g.mx[0]; out.x[1]=g.mx[1]; out.z[0]=g.mz[0]; out.z[1]=g.mz[1];
        int N=n, psum=0;
        for(uint16_t j : g.coeff){ if(j<N) psum+=tableau.Xc[j].phase; else psum+=tableau.Zc[j-N].phase; }
        out.phase=(uint8_t)(((int)g.c_static - psum) & 3);
    }
    // miss: capture a generator's static mask + coeff support + c_static (= residual) from _last_coeff.
    void rb_capture_gen(RbGenPlan& g, const PackedPauli& out) const {
        g.mx[0]=out.x[0]; g.mx[1]=out.x[1]; g.mz[0]=out.z[0]; g.mz[1]=out.z[1];
        int N=n, nw2=(2*N+63)>>6, psum=0; g.coeff.clear();
        for(int w=0; w<nw2; w++){ uint64_t cw=_last_coeff[w]; while(cw){ int b=__builtin_ctzll(cw); cw&=cw-1; int j=(w<<6)+b;
            g.coeff.push_back((uint16_t)j); if(j<N) psum+=tableau.Xc[j].phase; else psum+=tableau.Zc[j-N].phase; } }
        g.c_static=(uint8_t)(((int)out.phase + psum) & 3);
    }
    // shadow: compare the plan-applied generator vs the live rebuild result.
    void rb_shadow_check_gen(const RbGenPlan& g, const PackedPauli& live, int gi, int az, uint64_t sig) const {
        PackedPauli t(W); rb_apply_gen(t, g);
        bool mok=(t.x[0]==live.x[0]&&t.x[1]==live.x[1]&&t.z[0]==live.z[0]&&t.z[1]==live.z[1]);
        bool pok=(t.phase==live.phase);
        if((!mok||!pok) && !rb_shadow_fail().hit){ RbShadowFail& f=rb_shadow_fail(); f.hit=true;
            f.g=gi; f.az=az; f.sig=sig; f.sx0=t.x[0]; f.sx1=t.x[1]; f.sz0=t.z[0]; f.sz1=t.z[1];
            f.lx0=live.x[0]; f.lx1=live.x[1]; f.lz0=live.z[0]; f.lz1=live.z[1]; f.sphase=t.phase; f.lphase=live.phase; }
    }
    void rebuild_inverse_frame() {
        uint64_t _t0 = pb_time_on()? __builtin_ia32_rdtsc() : 0;
        // ---- StaticInverseFramePlan fast path (default OFF) ----
        if(rb_static_on()){
            uint64_t sig = rb_mask_sig();
            RbPlan& pl = rb_plan_map()[sig];
            if(pl.built && !rb_static_shadow()){       // HIT: skip basis build + all pullbacks
                for(int i=0;i<n;i++){ rb_apply_gen(inverse_frame.ax[i], pl.ax[i]); rb_apply_gen(inverse_frame.az[i], pl.az[i]); }
                pl.hits++; rb_static_hits()++; inv_dirty=false; basis_valid=false;
                if(pb_time_on()){ pb_rebuild_cyc()+=__builtin_ia32_rdtsc()-_t0; pb_rebuild_cnt()++; }
                return;
            }
            // MISS (capture) or SHADOW (verify): run the live rebuild
            build_inverse_basis(); basis_valid=true;
            bool capture = !pl.built;
            if(capture){ pl.ax.resize(n); pl.az.resize(n); }
            for(int i=0;i<n;i++){
                PackedPauli Xi(W); Xi.x[PackedPauli::word(i)] = PackedPauli::bit(i);
                PackedPauli Zi(W); Zi.z[PackedPauli::word(i)] = PackedPauli::bit(i);
                inverse_frame.ax[i] = pullback_from_basis(Xi);
                if(capture) rb_capture_gen(pl.ax[i], inverse_frame.ax[i]);
                else rb_shadow_check_gen(pl.ax[i], inverse_frame.ax[i], i, 0, sig);
                inverse_frame.az[i] = pullback_from_basis(Zi);
                if(capture) rb_capture_gen(pl.az[i], inverse_frame.az[i]);
                else rb_shadow_check_gen(pl.az[i], inverse_frame.az[i], i, 1, sig);
            }
            if(capture){ pl.built=true; rb_static_misses()++; } else { pl.hits++; rb_static_hits()++; }
            inv_dirty=false;
            if(pb_time_on()){ pb_rebuild_cyc()+=__builtin_ia32_rdtsc()-_t0; pb_rebuild_cnt()++; }
            return;
        }
        // ---- live rebuild (+ optional de-risk checker) ----
        build_inverse_basis(); basis_valid=true;     // ONCE per rebuild (the O(n^2) elimination)
        bool cap = rb_cap_on();
        uint64_t sig = cap ? rb_mask_sig() : 0;       // content key: tableau-mask signature (masks only)
        for (int i = 0; i < n; i++) {
            PackedPauli Xi(W); Xi.x[PackedPauli::word(i)] = PackedPauli::bit(i);
            PackedPauli Zi(W); Zi.z[PackedPauli::word(i)] = PackedPauli::bit(i);
            inverse_frame.ax[i] = pullback_from_basis(Xi);
            if(cap) rb_record(sig, i, 0, inverse_frame.ax[i]);
            inverse_frame.az[i] = pullback_from_basis(Zi);
            if(cap) rb_record(sig, i, 1, inverse_frame.az[i]);
        }
        inv_dirty=false;                             // frame now fully materialized
        if(cap) rb_epoch()++;
        if(pb_time_on()){ pb_rebuild_cyc()+=__builtin_ia32_rdtsc()-_t0; pb_rebuild_cnt()++; }
    }
    // Materialize the full inverse frame if it was left dirty by the lazy path (for direct .ax/.az readers).
    void materialize_inverse_frame() { if(inv_dirty) rebuild_inverse_frame(); }
    // _ag_measure(Pm, anti_s): project the stabilizer tableau; magic register untouched; rebuild inverse.
    void ag_measure(const PackedPauli& Pm, int p, int out) {
        ag_fired++;
        PackedPauli Sp = tableau.Zc[p];
        for (int i = 0; i < n; i++) {
            if (i != p && !commute(tableau.Zc[i], Pm)) tableau.Zc[i] = pauli_mul(tableau.Zc[i], Sp);
            if (!commute(tableau.Xc[i], Pm)) tableau.Xc[i] = pauli_mul(tableau.Xc[i], Sp);
        }
        tableau.Xc[p] = Sp;
        PackedPauli nz = Pm; nz.phase = (uint8_t)((Pm.phase + 2 * out) & 3);   // copy ALL W words (n>64 spans word 1)
        tableau.Zc[p] = nz;
        // AG projection has no incremental rule -> rebuild from basis.  Lazy (authoritative): defer to the
        // next pullback (computed on-demand from the live tableau); for maxM=0 it is never read -> no rebuild.
        if(lazy_inverse){ inv_dirty=true; basis_valid=false; } else rebuild_inverse_frame();
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
