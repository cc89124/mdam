// native_instr.hpp — Gate I control-plane dissection instrumentation (profiling-only, -DMDAM_INSTR).
// Default build (no MDAM_INSTR): ALL macros are transparent passthrough / no-ops -> zero overhead,
// the release .so is byte-for-byte unchanged.  The instrumented build adds:
//   (1) a leave-one-out SKIP mask (g_iskip): zero out one cleanly-skippable control component at a
//       time and re-time the whole batch; the delta == that component's ns/shot.  Valid ONLY in
//       dense-zero kernel mode (mode 13), where outcome = (rand_val<0.5) is independent of the
//       skipped state, so RNG draws / branch structure / M evolution are all preserved.
//   (2) rdtsc accumulators (g_itime[]) + call counts (g_icnt[]) for the few-call-per-shot regions
//       that are branch-coupled (region-load, drop, oracle, noise) and cannot be cleanly skipped.
#pragma once
#include <cstdint>
#ifdef MDAM_INSTR
#include <x86intrin.h>
#endif

namespace mdam {

// ---- leave-one-out skip bits (cleanly skippable in dense-zero) ----
enum {
    ISK_FRAME       = 1<<0,   // NativeFrame h/cnot/cz/swap/s + active-gate frame ops + set_xz
    ISK_NOISE_APPLY = 1<<1,   // noise frame-mask XOR (apply_x/apply_z); RNG draw kept
    ISK_INV_FWD     = 1<<2,   // live inverse-frame forward (cx_inv/cz_inv/s_inv/h_inv)
    ISK_PULLBACK    = 1<<3,   // magic_plan per-rotation pullback (rpp/theta/cos/sin)
    ISK_SIGN        = 1<<4,   // magic_plan measurement-sign pullback + Wout pconj
    ISK_RIGHTFOLD   = 1<<5,   // magic_execute commit right_h/s/cx
    ISK_FOLDX       = 1<<6,   // magic_execute fold_x
    ISK_PLANCOPY    = 1<<7,   // skeleton vector copies Wout/lt/la/lb/rx/rz (NOT M_mat)
    ISK_CORERESOLVE = 1<<8,   // dynamic_core_scr cache resolve
};

// ---- rdtsc time slots ----
enum {
    IT_SHOT=0,    // whole run_fb (cyc<->ns calibration)
    IT_REGION=1,  // fb_load_boundary
    IT_DROP=2,    // drop_residual_products
    IT_ORACLE=3,  // oracle path
    IT_NOISE=4,   // apply_site (noise total: RNG + searchsorted + channel + apply)
    IT_PLAN=5,    // magic_plan total
    IT_EXEC=6,    // magic_execute total
    IT_NSLOT=8
};

#ifdef MDAM_INSTR
struct InstrState { int skip=0; uint64_t tcyc[IT_NSLOT]={0}; uint64_t tcnt[IT_NSLOT]={0}; };
inline InstrState& instr(){ static InstrState s; return s; }
static inline uint64_t irdtsc(){ return __rdtsc(); }
#define ISKIP(bit, stmt)   do{ if(!(mdam::instr().skip & (bit))) { stmt; } }while(0)
#define ITIME_BEG(slot)    uint64_t _it_##slot = mdam::irdtsc()
#define ITIME_END(slot)    do{ mdam::instr().tcyc[slot]+=mdam::irdtsc()-_it_##slot; mdam::instr().tcnt[slot]++; }while(0)
#else
#define ISKIP(bit, stmt)   do{ stmt; }while(0)
#define ITIME_BEG(slot)
#define ITIME_END(slot)
#endif

} // namespace mdam
