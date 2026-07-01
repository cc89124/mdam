// native_invframe_static.hpp — CLEAN-ROOM inverse-frame rebuild StaticPlan (de-risk checker + fast path).
//
// Motivation (calibrated rdtsc, pullback_staticplan_phaseB.md): rebuild_inverse_frame is 30.2% of
// cult_d3 wall and 14.5% of distillation wall — the real control-plane bottleneck (the pullback
// substitution Phase B optimized was only ~1% of wall; the "26%" was a PROFILE ISKIP artifact).
//
// Premise (from the rebuild math): inverse generator OUTPUT MASKS depend only on the tableau MASKS
// (build_inverse_basis / pullback_from_basis use getx/getz), and the OUTPUT PHASE is
//     out_phase = c_static[g] - Σ_{j∈coeff(g)} tableau_phase[j]   (mod 4)
// i.e. affine in the tableau generator phases with a shot-static coefficient set = coeff(g).
//
// ⚠ This file shares NOTHING with F4/imem/old plan_cache (the retracted false-positive source).
//   Separate state, separate flags (rb_*), separate checker, default OFF.  rebuild_inverse_frame and
//   the authoritative path are unchanged unless rb_cap_on / rb_static_on is explicitly set.
#pragma once
#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

namespace mdam {

// ---- de-risk checker (Step 1 mask invariance + Step 2 phase-affine), aggregated in C++, default OFF ----
// KEY = content signature of the tableau MASKS (Xc/Zc x,z bits; NOT phases — those drive the affine model).
// Same signature -> identical rebuild output masks BY DETERMINISM (output mask is a pure fn of tableau masks);
// the checker verifies key-sufficiency (mask viol 0) + phase-affine (resid const) + saturation (distinct sigs).
inline bool& rb_cap_on(){ static bool v=false; return v; }
inline int&  rb_epoch(){ static int v=0; return v; }   // per-shot rebuild index (kept for count histogram)

struct RbKey { uint64_t sig; uint16_t g; uint8_t az;   // (tableau-mask signature, generator index, ax=0/az=1)
    bool operator<(const RbKey& o) const {
        if(sig!=o.sig) return sig<o.sig; if(g!=o.g) return g<o.g; return az<o.az; } };
struct RbVal { uint64_t x0=0,x1=0,z0=0,z1=0; int phase0=0; bool phase_varies=false;
    long calls=0, mask_viol=0; int resid0=0; bool resid_set=false; long phase_affine_viol=0; };
inline std::map<RbKey,RbVal>& rb_map(){ static std::map<RbKey,RbVal> m; return m; }
// per-shot rebuild-count histogram (count -> #shots): proves the rebuild SEQUENCE length is shot-static
inline std::map<int,long>& rb_count_hist(){ static std::map<int,long> m; return m; }

// ===== fast path: StaticInverseFramePlan (default OFF) =====
// On a rebuild, key by the tableau-mask signature.  HIT -> skip build_inverse_basis (O(n²)) + the 2n
// pullback_from_basis; fill each inverse generator from the cached static mask + affine phase
//   phase = (c_static[g] - Σ_{j∈coeff[g]} tableau_phase[j]) mod 4.
// MISS -> run the live rebuild and capture the plan.  unordered_map values are pointer-stable (≤tens of
// keys, never erased).  Correctness gated by shadow verification (compute live too, compare every gen).
struct RbGenPlan {                       // one per (inverse generator i, ax/az)
    uint64_t mx[2]={0,0}, mz[2]={0,0};   // static output mask
    uint8_t  c_static=0;                 // affine phase base
    std::vector<uint16_t> coeff;         // tableau-generator indices j (j<N -> Xc[j].phase, else Zc[j-N].phase)
};
struct RbPlan { std::vector<RbGenPlan> ax, az; bool built=false; long hits=0; };
inline int& rb_static_on(){ static int v=0; return v; }       // --mdam-static-invframe
inline int& rb_static_shadow(){ static int v=0; return v; }   // --mdam-static-invframe-shadow
inline std::unordered_map<uint64_t,RbPlan>& rb_plan_map(){ static std::unordered_map<uint64_t,RbPlan> m; return m; }
inline long& rb_static_hits(){ static long v=0; return v; }
inline long& rb_static_misses(){ static long v=0; return v; }
struct RbShadowFail { bool hit=false; int g=0,az=0; uint64_t sig=0;
    uint64_t sx0=0,sx1=0,sz0=0,sz1=0, lx0=0,lx1=0,lz0=0,lz1=0; int sphase=0,lphase=0; };
inline RbShadowFail& rb_shadow_fail(){ static RbShadowFail f; return f; }

} // namespace mdam
