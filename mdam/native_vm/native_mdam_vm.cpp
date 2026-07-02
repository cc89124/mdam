// native_mdam_vm.cpp — ctypes C API for the full cultivation_d3 native one-shot (native_mdam_shot).
#include <cstring>
#include <unordered_set>
#include <cmath>
#include "native_mdam_shot.hpp"
#include "native_compiled_region.hpp"   // Gate J Phase-2: compiled sampler + shadow
using namespace mdam;

extern "C" {

// ---- Gate J Phase-2A: compile the region program + shadow-evaluate against the authoritative VM ----
void* nvm_jcompile(void* prog){ return new CompiledMdamProgram(compile_jprogram(*reinterpret_cast<MdamProgram*>(prog))); }
void nvm_jcompile_free(void* cp){ delete reinterpret_cast<CompiledMdamProgram*>(cp); }
void nvm_jcompile_info(void* cp, int* out){ auto&c=*reinterpret_cast<CompiledMdamProgram*>(cp);
    out[0]=c.dyn.ndyn; out[1]=c.nrot; out[2]=c.nmagic; out[3]=c.dyn.n_noise; out[4]=c.record_cap; }
// oneshot shadow: run one seed, write the concrete record (for faithfulness vs nvm_mdam_run) + stats.
int nvm_jshadow_run(void* prog, void* cpv, void* vm, uint64_t shi,uint64_t slo,uint64_t ihi,uint64_t ilo,
                    uint8_t* out_record, long* stats){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    auto& cp=*reinterpret_cast<CompiledMdamProgram*>(cpv);
    s.reset_shot(p,shi,slo,ihi,ilo); JShadowStats st; run_jshadow(s,p,cp,st);
    std::memcpy(out_record, s.record.bits.data(), (size_t)p.num_measurements);
    stats[0]=st.theta_checks; stats[1]=st.theta_mismatch; stats[2]=st.rec_checks; stats[3]=st.rec_mismatch;
    stats[4]=st.opcode_dispatch; stats[5]=st.frame_fwd; stats[6]=st.first_bad_rot; stats[7]=st.first_bad_rec;
    return s.err?1:0;
}
// batch shadow over master seeds (same seed expansion as sample_batch); accumulate stats.
int nvm_jshadow_batch(void* prog, void* cpv, void* vm, uint64_t num_shots,
                      uint64_t mshi,uint64_t mslo,uint64_t mihi,uint64_t milo, long* stats){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    auto& cp=*reinterpret_cast<CompiledMdamProgram*>(cpv);
    NativeRng master; master.seed_from_state(mshi,mslo,mihi,milo);
    const uint64_t RNG_EXCL=((uint64_t)1<<63)-1; JShadowStats st;
    for(uint64_t sh=0; sh<num_shots; sh++){
        uint64_t sd=master.bounded(RNG_EXCL); __uint128_t stt,inc; SeedExpand::seedseq_pcg64(sd,stt,inc);
        s.reset_shot(p,(uint64_t)(stt>>64),(uint64_t)stt,(uint64_t)(inc>>64),(uint64_t)inc);
        run_jshadow(s,p,cp,st);
        if(s.err) break;
    }
    stats[0]=st.theta_checks; stats[1]=st.theta_mismatch; stats[2]=st.rec_checks; stats[3]=st.rec_mismatch;
    stats[4]=st.opcode_dispatch; stats[5]=st.frame_fwd; stats[6]=st.first_bad_rot; stats[7]=st.first_bad_rec;
    return 0;
}

// DEBUG: first opcode where symbolic frame diverges from authoritative NativeFrame (one shot).
// out[0]=opno out[1]=slot out[2]=kind out[3]=is_z out[4]=found.
void nvm_frame_divergence(void* prog, void* cpv, void* vm, uint64_t shi,uint64_t slo,uint64_t ihi,uint64_t ilo, long* out){
    auto& p=*reinterpret_cast<mdam::MdamProgram*>(prog); auto& s=*reinterpret_cast<mdam::MdamShot*>(vm);
    auto& cp=*reinterpret_cast<mdam::CompiledMdamProgram*>(cpv);
    s.reset_shot(p,shi,slo,ihi,ilo); mdam::frame_first_divergence(s,p,cp,out);
}
void nvm_jfast_dbg(int d){ mdam::jfast_dbg()=d; }   // timing bisect: 1=skip fire-handling
// ---- Gate J Phase-2C-A: NativeInverseFrame-off fast path (phase_pack + reconstruct at measure) ----
// requires the vm to have imem_mode=2 (set via nvm_mdam_vm_set_imem) and the Imem table prepopulated.
int nvm_jfast2c_run(void* prog, void* cpv, void* jpv, void* vm, uint64_t shi,uint64_t slo,uint64_t ihi,uint64_t ilo,
                    uint8_t* out_record, long* stats){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    auto& cp=*reinterpret_cast<CompiledMdamProgram*>(cpv); auto& jp=*reinterpret_cast<JPhaseCompiled*>(jpv);
    s.reset_shot(p,shi,slo,ihi,ilo); JFast2CStats st; int rc=run_jfast_2c(s,p,cp,jp,st);
    std::memcpy(out_record, s.record.bits.data(), (size_t)p.num_measurements);
    stats[0]=st.opcode_dispatch; stats[1]=st.reconstructs; stats[2]=st.pullback_calls;
    stats[3]=st.imem_miss; stats[4]=st.oracle_count; stats[5]=st.phase_mismatch;
    return rc;
}
int nvm_jfast2c_batch(void* prog, void* cpv, void* jpv, void* vm, uint64_t num_shots,
                      uint64_t mshi,uint64_t mslo,uint64_t mihi,uint64_t milo,
                      uint8_t* out_record, int reuse_buf, long* stats){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    auto& cp=*reinterpret_cast<CompiledMdamProgram*>(cpv); auto& jp=*reinterpret_cast<JPhaseCompiled*>(jpv);
    NativeRng master; master.seed_from_state(mshi,mslo,mihi,milo);
    const uint64_t RNG_EXCL=((uint64_t)1<<63)-1; const size_t nm=(size_t)p.num_measurements;
    JFast2CStats st; int rc=0;
    for(uint64_t sh=0; sh<num_shots; sh++){
        uint64_t sd=master.bounded(RNG_EXCL); __uint128_t stt,inc; SeedExpand::seedseq_pcg64(sd,stt,inc);
        s.reset_shot(p,(uint64_t)(stt>>64),(uint64_t)stt,(uint64_t)(inc>>64),(uint64_t)inc);
        rc=run_jfast_2c(s,p,cp,jp,st);
        std::memcpy(out_record + (reuse_buf?0:(size_t)sh*nm), s.record.bits.data(), nm);
        if(s.err){ rc=1; break; }
    }
    if(stats){ stats[0]=st.opcode_dispatch; stats[1]=st.reconstructs; stats[2]=st.pullback_calls;
        stats[3]=st.imem_miss; stats[4]=st.oracle_count; stats[5]=st.phase_mismatch; }
    return rc;
}
// ---- Gate J Phase-2D-3: timing breakdown toggles (A/B skip bitmask + rdtsc measure block) ----
void nvm_j2d_dbg(int d){ mdam::j2d_dbg()=d; }
void nvm_j2d_time(int t){ mdam::j2d_time()=t; }
void nvm_j2d_cyc_reset(){ uint64_t* c=mdam::j2d_cyc(); c[0]=c[1]=c[2]=c[3]=0; }
void nvm_j2d_cyc_get(uint64_t* out){ uint64_t* c=mdam::j2d_cyc(); out[0]=c[0]; out[1]=c[1]; out[2]=c[2]; out[3]=c[3]; }
// 2E component-breakdown instrumentation (default OFF).  cyc: [0]=shot, [1]=measure block, [2]=boundary prep.
void nvm_j2e_dbg(int d){ mdam::j2e_dbg()=d; }
void nvm_j2e_time(int t){ mdam::j2e_time()=t; }
void nvm_j2e_noise_mode(int m){ mdam::j2e_noise_mode()=m; }   // 0 full / 1 draw-only / 2 off (coarse noise wall-delta, timing-only)
void nvm_j2e_noise_skip(int s){ mdam::j2e_noise_skip()=s; }   // 1 skip-to-next-fire / 0 per-site loop (EXACT, correctness-preserving)
void nvm_j2e_cyc_reset(){ uint64_t* c=mdam::j2e_cyc(); for(int i=0;i<16;i++) c[i]=0; }
void nvm_j2e_cyc_get(uint64_t* out){ uint64_t* c=mdam::j2e_cyc(); for(int i=0;i<16;i++) out[i]=c[i]; }   // caller MUST pass uint64_t[16]: [0]whole [1]measure [2]bnd [4]noise_sample [5]noise_apply [8]site_calls [9]draws [10]fires
// 2F-M dense-vs-commit split inside magic_compiled_fast (default OFF).  cyc: [0]=dense kernel, [1]=commit.
void nvm_mcf_time(int t){ mdam::mcf_time()=t; }
void nvm_mcf_cyc_reset(){ uint64_t* c=mdam::mcf_cyc(); c[0]=c[1]=0; }
void nvm_mcf_cyc_get(uint64_t* out){ uint64_t* c=mdam::mcf_cyc(); out[0]=c[0]; out[1]=c[1]; }
// 2C+ oracle-path dissection (default OFF).  cyc[8]: [0]reconstruct [1]flush [2]anti_s [3]pullback+localize
//   [4]branch+project+norm [5]drop+reduce [6]read_phase_pack [7]measure_z(oracle) total.
void nvm_orc_time(int t){ mdam::orc_time()=t; }
void nvm_orc_cyc_reset(){ uint64_t* c=mdam::orc_cyc(); for(int i=0;i<12;i++) c[i]=0; }
void nvm_orc_cyc_get(uint64_t* out){ uint64_t* c=mdam::orc_cyc(); for(int i=0;i<12;i++) out[i]=c[i]; }
// ---- Gate J Phase-2D-1: compiled magic WITHOUT reconstruct (phase-only + Imem inject + rfd commit) ----
int nvm_jfast2d_run(void* prog, void* cpv, void* jpv, void* vm, uint64_t shi,uint64_t slo,uint64_t ihi,uint64_t ilo,
                    uint8_t* out_record, long* stats){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    auto& cp=*reinterpret_cast<CompiledMdamProgram*>(cpv); auto& jp=*reinterpret_cast<JPhaseCompiled*>(jpv);
    s.reset_shot(p,shi,slo,ihi,ilo); JFast2DStats st; int rc=run_jfast_2d(s,p,cp,jp,st);
    std::memcpy(out_record, s.record.bits.data(), (size_t)p.num_measurements);
    stats[0]=st.opcode_dispatch; stats[1]=st.compiled_fast; stats[2]=st.reconstructs; stats[3]=st.pullback_calls;
    stats[4]=st.imem_miss; stats[5]=st.oracle_count; stats[6]=st.cold_fallback;
    return rc;
}
int nvm_jfast2d_batch(void* prog, void* cpv, void* jpv, void* vm, uint64_t num_shots,
                      uint64_t mshi,uint64_t mslo,uint64_t mihi,uint64_t milo,
                      uint8_t* out_record, int reuse_buf, long* stats){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    auto& cp=*reinterpret_cast<CompiledMdamProgram*>(cpv); auto& jp=*reinterpret_cast<JPhaseCompiled*>(jpv);
    NativeRng master; master.seed_from_state(mshi,mslo,mihi,milo);
    const uint64_t RNG_EXCL=((uint64_t)1<<63)-1; const size_t nm=(size_t)p.num_measurements;
    JFast2DStats st; int rc=0;
    for(uint64_t sh=0; sh<num_shots; sh++){
        uint64_t sd=master.bounded(RNG_EXCL); __uint128_t stt,inc; SeedExpand::seedseq_pcg64(sd,stt,inc);
        s.reset_shot(p,(uint64_t)(stt>>64),(uint64_t)stt,(uint64_t)(inc>>64),(uint64_t)inc);
        rc=run_jfast_2d(s,p,cp,jp,st);
        std::memcpy(out_record + (reuse_buf?0:(size_t)sh*nm), s.record.bits.data(), nm);
        if(s.err){ rc=1; break; }
    }
    if(stats){ stats[0]=st.opcode_dispatch; stats[1]=st.compiled_fast; stats[2]=st.reconstructs; stats[3]=st.pullback_calls;
        stats[4]=st.imem_miss; stats[5]=st.oracle_count; stats[6]=st.cold_fallback; }
    return rc;
}
// ---- Gate J Phase-2E: Gate-J compiled control + Gate-F-B region snapshot, MERGED.  shadow!=0 keeps the
// live _noinv tableau forward and verifies the snapshot at every boundary (measure uses live); shadow==0
// (FAST) skips the tableau/pending forward and loads the snapshot.  REQUIRES the F-B snapshot already
// built (run nvm_mdam_sample_batch with fb_mode=COMPILE once) + jp/cp/imem warmed.  stats (13):
// [dispatch, compiled_fast, reconstructs, pullback, imem_miss, oracle, cold, tableau_conj,
//  pending_create_rot, cap_theta, boundary_loads, boundary_pending, fb_mismatch].
static inline void e_pack(long* st, const JFast2EStats& v){ if(!st) return;
    st[0]=v.opcode_dispatch; st[1]=v.compiled_fast; st[2]=v.reconstructs; st[3]=v.pullback_calls;
    st[4]=v.imem_miss; st[5]=v.oracle_count; st[6]=v.cold_fallback; st[7]=v.tableau_conj;
    st[8]=v.pending_create_rot; st[9]=v.cap_theta_count; st[10]=v.boundary_loads; st[11]=v.boundary_pending;
    st[12]=v.fb_mismatch; st[13]=v.dense_only_calls; st[14]=v.phasepack_updates; st[15]=v.generic_measure_calls;
    st[16]=v.bplan_resolve; st[17]=v.bplan_build; st[18]=v.imem_keybuild; st[19]=v.imem_probe; }
// cmode: 0=SHADOW, 1=FAST-2E (compiled magic via measure_z), 2=FAST-2F (compiled magic via magic_compiled_fast).
int nvm_jfast2e_run(void* prog, void* cpv, void* jpv, void* vm, uint64_t shi,uint64_t slo,uint64_t ihi,uint64_t ilo,
                    uint8_t* out_record, int cmode, long* stats){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    auto& cp=*reinterpret_cast<CompiledMdamProgram*>(cpv); auto& jp=*reinterpret_cast<JPhaseCompiled*>(jpv);
    s.fb_mismatch=0; s.reset_shot(p,shi,slo,ihi,ilo); JFast2EStats st; int rc=run_jfast_2e(s,p,cp,jp,st,cmode);
    std::memcpy(out_record, s.record.bits.data(), (size_t)p.num_measurements); e_pack(stats,st);
    return rc;
}
int nvm_jfast2e_batch(void* prog, void* cpv, void* jpv, void* vm, uint64_t num_shots,
                      uint64_t mshi,uint64_t mslo,uint64_t mihi,uint64_t milo,
                      uint8_t* out_record, int reuse_buf, int cmode, long* stats){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    auto& cp=*reinterpret_cast<CompiledMdamProgram*>(cpv); auto& jp=*reinterpret_cast<JPhaseCompiled*>(jpv);
    NativeRng master; master.seed_from_state(mshi,mslo,mihi,milo);
    const uint64_t RNG_EXCL=((uint64_t)1<<63)-1; const size_t nm=(size_t)p.num_measurements;
    s.fb_mismatch=0; JFast2EStats st; int rc=0;
    for(uint64_t sh=0; sh<num_shots; sh++){
        uint64_t sd=master.bounded(RNG_EXCL); __uint128_t stt,inc; SeedExpand::seedseq_pcg64(sd,stt,inc);
        s.reset_shot(p,(uint64_t)(stt>>64),(uint64_t)stt,(uint64_t)(inc>>64),(uint64_t)inc);
        rc=run_jfast_2e(s,p,cp,jp,st,cmode);
        std::memcpy(out_record + (reuse_buf?0:(size_t)sh*nm), s.record.bits.data(), nm);
        if(s.err){ rc=1; break; }
    }
    e_pack(stats,st);
    return rc;
}
// ---- Gate J Phase-2F-M: compiled-magic dense-only fast path (cmode=2).  The 4 compiled magics bypass
// the generic measure_z plan/commit; the dense kernel is fed directly from StaticPlan + Imem + fb_theta.
// Same warmup as 2E (run a 2E batch first to fill plan_cache/core_cache/imem/rfd); oracle stays generic.
int nvm_jfast2f_run(void* prog, void* cpv, void* jpv, void* vm, uint64_t shi,uint64_t slo,uint64_t ihi,uint64_t ilo,
                    uint8_t* out_record, long* stats){
    return nvm_jfast2e_run(prog,cpv,jpv,vm,shi,slo,ihi,ilo,out_record,2,stats);
}
int nvm_jfast2f_batch(void* prog, void* cpv, void* jpv, void* vm, uint64_t num_shots,
                      uint64_t mshi,uint64_t mslo,uint64_t mihi,uint64_t milo,
                      uint8_t* out_record, int reuse_buf, long* stats){
    return nvm_jfast2e_batch(prog,cpv,jpv,vm,num_shots,mshi,mslo,mihi,milo,out_record,reuse_buf,2,stats);
}
// ---- Gate J Phase-2G: 2F-M + BoundaryPlan memoization (cmode=3).  Per-(mag,M-variant) O(1) dispatch:
// no Mkey heap copy, no plan/commit linear scan over M_key vectors, imem key built ONCE + probed ONCE
// per compiled boundary (was 2x in 2F).  Same warmup as 2F (run a 2F/2E batch first).  stats: e_pack 20.
int nvm_jfast2g_run(void* prog, void* cpv, void* jpv, void* vm, uint64_t shi,uint64_t slo,uint64_t ihi,uint64_t ilo,
                    uint8_t* out_record, long* stats){
    return nvm_jfast2e_run(prog,cpv,jpv,vm,shi,slo,ihi,ilo,out_record,3,stats);
}
int nvm_jfast2g_batch(void* prog, void* cpv, void* jpv, void* vm, uint64_t num_shots,
                      uint64_t mshi,uint64_t mslo,uint64_t mihi,uint64_t milo,
                      uint8_t* out_record, int reuse_buf, long* stats){
    return nvm_jfast2e_batch(prog,cpv,jpv,vm,num_shots,mshi,mslo,mihi,milo,out_record,reuse_buf,3,stats);
}
// ---- Gate K Step-2: boundary-edge SHADOW cache (cmode=4 = 2G dispatch + per-boundary verify/store, NO
// live skip).  Warm like 2G (run a 2F/2G batch first).  kcache verifies a cached edge reproduces the live
// boundary bit-exact; nvm_jkcache_stats reads the hit/mismatch/collision counters.
int nvm_jfast4_run(void* prog, void* cpv, void* jpv, void* vm, uint64_t shi,uint64_t slo,uint64_t ihi,uint64_t ilo,
                   uint8_t* out_record, long* stats){
    return nvm_jfast2e_run(prog,cpv,jpv,vm,shi,slo,ihi,ilo,out_record,4,stats);
}
int nvm_jfast4_batch(void* prog, void* cpv, void* jpv, void* vm, uint64_t num_shots,
                     uint64_t mshi,uint64_t mslo,uint64_t mihi,uint64_t milo,
                     uint8_t* out_record, int reuse_buf, long* stats){
    return nvm_jfast2e_batch(prog,cpv,jpv,vm,num_shots,mshi,mslo,mihi,milo,out_record,reuse_buf,4,stats);
}
void nvm_jkcache_reset(void* vm){ auto& s=*reinterpret_cast<MdamShot*>(vm);
    s.kcache.clear(); s.k_lookup=s.k_hit=s.k_miss=s.k_mismatch=s.k_collision=0;
    s.k_lookup_o=s.k_hit_o=s.k_miss_o=s.k_mismatch_o=0;
    s.k_full_hit=s.k_partial=s.k_miss5=s.k_fwdmap=0; s.k_materialize=s.k_antis_live=0; }
void nvm_jkcache_stats(void* vm, long* out){   // [0..9]=lookup,hit,miss,mismatch,collision,lookup_o,hit_o,miss_o,mismatch_o,distinct_keys
    auto& s=*reinterpret_cast<MdamShot*>(vm);   // [10..15]=full_hit,partial,miss5,fwdmap,materialize,antis_live
    out[0]=s.k_lookup; out[1]=s.k_hit; out[2]=s.k_miss; out[3]=s.k_mismatch; out[4]=s.k_collision;
    out[5]=s.k_lookup_o; out[6]=s.k_hit_o; out[7]=s.k_miss_o; out[8]=s.k_mismatch_o;
    long dk=0; for(auto& mm : s.kcache) dk+=(long)mm.size(); out[9]=dk;
    out[10]=s.k_full_hit; out[11]=s.k_partial; out[12]=s.k_miss5; out[13]=s.k_fwdmap;
    out[14]=s.k_materialize; out[15]=s.k_antis_live; }
// Gate K Step-4A: FAST (cmode=5) — full edge hit skips the live boundary.  Warm like 2G + a kshadow/fast
// pass so the cache (incl. both outcome branches + oracle core uids) is populated.
int nvm_jfast5_run(void* prog, void* cpv, void* jpv, void* vm, uint64_t shi,uint64_t slo,uint64_t ihi,uint64_t ilo,
                   uint8_t* out_record, long* stats){
    return nvm_jfast2e_run(prog,cpv,jpv,vm,shi,slo,ihi,ilo,out_record,5,stats);
}
// DEBUG: cmode5 with magic-outcome logging (per-boundary outcome + p0, FAST hits + live boundaries in order).
int nvm_jfast5_run_magiclog(void* prog, void* cpv, void* jpv, void* vm, uint64_t shi,uint64_t slo,uint64_t ihi,uint64_t ilo,
                            uint8_t* out_record, int* oc, double* p0, int maxn, int cmode){
    auto& p=*reinterpret_cast<mdam::MdamProgram*>(prog); auto& s=*reinterpret_cast<mdam::MdamShot*>(vm);
    auto& cp=*reinterpret_cast<mdam::CompiledMdamProgram*>(cpv); auto& jp=*reinterpret_cast<mdam::JPhaseCompiled*>(jpv);
    s.fb_mismatch=0; s.reset_shot(p,shi,slo,ihi,ilo); s.magic_log_on=true; s.magic_log.clear();
    mdam::JFast2EStats st; mdam::run_jfast_2e(s,p,cp,jp,st,cmode);
    std::memcpy(out_record, s.record.bits.data(), (size_t)p.num_measurements);
    int n=(int)s.magic_log.size(); if(n>maxn)n=maxn;
    for(int i=0;i<n;i++){ oc[i]=s.magic_log[i].outcome; p0[i]=s.magic_log[i].p0; }
    s.magic_log_on=false; return (int)s.magic_log.size();
}
int nvm_jfast5_batch(void* prog, void* cpv, void* jpv, void* vm, uint64_t num_shots,
                     uint64_t mshi,uint64_t mslo,uint64_t mihi,uint64_t milo,
                     uint8_t* out_record, int reuse_buf, long* stats){
    return nvm_jfast2e_batch(prog,cpv,jpv,vm,num_shots,mshi,mslo,mihi,milo,out_record,reuse_buf,5,stats);
}
// ---- Gate J Phase-2B: NativeFrame-off fast path (event-driven accumulation) ----
// oneshot: run with NO frame; record from rec_sig.  Returns record + stats[6]=
// [opcode_dispatch, fires, accum_xor, rotations, frame_fwd(=0), frame_read(=0)].
int nvm_jfast_run(void* prog, void* cpv, void* vm, uint64_t shi,uint64_t slo,uint64_t ihi,uint64_t ilo,
                  uint8_t* out_record, long* stats){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    auto& cp=*reinterpret_cast<CompiledMdamProgram*>(cpv);
    s.reset_shot(p,shi,slo,ihi,ilo); JFastStats st; int rc=run_jfast(s,p,cp,st);
    std::memcpy(out_record, s.record.bits.data(), (size_t)p.num_measurements);
    stats[0]=st.opcode_dispatch; stats[1]=st.fires; stats[2]=st.accum_xor; stats[3]=st.rotations;
    stats[4]=st.frame_fwd; stats[5]=st.frame_read;
    return rc;
}
// debug: single-shot run_jfast WITH rotation logging (slot, xb=theta_sig bit, base angle, signed theta).
// Mirrors nvm_mdam_run_rotlog so authoritative vs FAST rotation-sign sequences can be diffed.
int nvm_jfast_rotlog(void* prog, void* cpv, void* vm, uint64_t shi,uint64_t slo,uint64_t ihi,uint64_t ilo,
                     uint8_t* out_record, double* slot, double* xb, double* angle, double* theta, int maxn){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    auto& cp=*reinterpret_cast<CompiledMdamProgram*>(cpv);
    s.reset_shot(p,shi,slo,ihi,ilo); s.rot_log_on=true; s.rot_log.clear(); JFastStats st; run_jfast(s,p,cp,st);
    std::memcpy(out_record, s.record.bits.data(), (size_t)p.num_measurements);
    int n=(int)s.rot_log.size(); if(n>maxn)n=maxn;
    for(int i=0;i<n;i++){ slot[i]=s.rot_log[i][0]; xb[i]=s.rot_log[i][1]; angle[i]=s.rot_log[i][2]; theta[i]=s.rot_log[i][3]; }
    s.rot_log_on=false; return (int)s.rot_log.size();
}
// batch: seed-expand like sample_batch; write [num_shots × num_meas] records (out_record may be the
// same small buffer reused for timing).  reuse_buf!=0 overwrites one row each shot (pure timing).
int nvm_jfast_batch(void* prog, void* cpv, void* vm, uint64_t num_shots,
                    uint64_t mshi,uint64_t mslo,uint64_t mihi,uint64_t milo,
                    uint8_t* out_record, int reuse_buf, long* stats){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    auto& cp=*reinterpret_cast<CompiledMdamProgram*>(cpv);
    NativeRng master; master.seed_from_state(mshi,mslo,mihi,milo);
    const uint64_t RNG_EXCL=((uint64_t)1<<63)-1; const size_t nm=(size_t)p.num_measurements;
    JFastStats st; int rc=0;
    for(uint64_t sh=0; sh<num_shots; sh++){
        uint64_t sd=master.bounded(RNG_EXCL); __uint128_t stt,inc; SeedExpand::seedseq_pcg64(sd,stt,inc);
        s.reset_shot(p,(uint64_t)(stt>>64),(uint64_t)stt,(uint64_t)(inc>>64),(uint64_t)inc);
        rc=run_jfast(s,p,cp,st);
        std::memcpy(out_record + (reuse_buf?0:(size_t)sh*nm), s.record.bits.data(), nm);
        if(s.err){ rc=1; break; }
    }
    if(stats){ stats[0]=st.opcode_dispatch; stats[1]=st.fires; stats[2]=st.accum_xor;
        stats[3]=st.rotations; stats[4]=st.frame_fwd; stats[5]=st.frame_read; }
    return rc;
}

// ---- Gate J Phase-2A+: magic-side phase_pack compiler + shadow (inverse-frame phases) ----
// compile: run ONE seed building the per-region Z4-affine forward map + region-start ref masks.
void* nvm_jphase_compile(void* prog, void* vm, uint64_t shi,uint64_t slo,uint64_t ihi,uint64_t ilo){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    int nmagic=0; for(uint8_t k:p.kind) if(k==MO_SWAP_MEAS_INTERFERE
        ||k==MO_MEAS_ACTIVE_DIAGONAL||k==MO_MEAS_ACTIVE_INTERFERE) nmagic++;   // Gate L: coherent boundaries
    auto* cp=new JPhaseCompiled(); cp->maps.resize(nmagic);
    s.reset_shot(p,shi,slo,ihi,ilo); JPhaseStats st; jphase_run(s,p,*cp,st,true);
    return cp;
}
void nvm_jphase_free(void* cp){ delete reinterpret_cast<JPhaseCompiled*>(cp); }
void nvm_jphase_info(void* cp, int* out){ auto&c=*reinterpret_cast<JPhaseCompiled*>(cp);
    out[0]=c.n; out[1]=c.twoN; out[2]=c.nmagic; out[3]=c.built?1:0; }
// shadow batch: run N seeds, re-sync phase_pack at region start, apply compiled map, compare at boundary.
int nvm_jphase_shadow_batch(void* prog, void* cpv, void* vm, uint64_t num_shots,
                            uint64_t mshi,uint64_t mslo,uint64_t mihi,uint64_t milo, long* stats){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    auto& cp=*reinterpret_cast<JPhaseCompiled*>(cpv);
    NativeRng master; master.seed_from_state(mshi,mslo,mihi,milo);
    const uint64_t RNG_EXCL=((uint64_t)1<<63)-1; JPhaseStats st;
    for(uint64_t sh=0; sh<num_shots; sh++){
        uint64_t sd=master.bounded(RNG_EXCL); __uint128_t stt,inc; SeedExpand::seedseq_pcg64(sd,stt,inc);
        s.reset_shot(p,(uint64_t)(stt>>64),(uint64_t)stt,(uint64_t)(inc>>64),(uint64_t)inc);
        jphase_run(s,p,cp,st,false);
        if(s.err) break;
    }
    stats[0]=st.regions_total; stats[1]=st.regions_match; stats[2]=st.phase_checks;
    stats[3]=st.phase_mismatch; stats[4]=st.regions_variant;
    stats[5]=st.first_bad_region; stats[6]=st.first_bad_slot;
    stats[7]=st.commit_checks; stats[8]=st.commit_mismatch; stats[9]=st.commit_new_variant;
    stats[10]=st.first_bad_commit_region; stats[11]=st.commit_maskbad; stats[12]=st.commit_rebuild;
    return 0;
}
// oneshot phase shadow (faithfulness: returns record so caller can check vs nvm_mdam_run).
int nvm_jphase_shadow_run(void* prog, void* cpv, void* vm, uint64_t shi,uint64_t slo,uint64_t ihi,uint64_t ilo,
                          uint8_t* out_record, long* stats){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    auto& cp=*reinterpret_cast<JPhaseCompiled*>(cpv);
    s.reset_shot(p,shi,slo,ihi,ilo); JPhaseStats st; jphase_run(s,p,cp,st,false);
    std::memcpy(out_record, s.record.bits.data(), (size_t)p.num_measurements);
    stats[0]=st.regions_total; stats[1]=st.regions_match; stats[2]=st.phase_checks;
    stats[3]=st.phase_mismatch; stats[4]=st.regions_variant;
    stats[5]=st.first_bad_region; stats[6]=st.first_bad_slot;
    stats[7]=st.commit_checks; stats[8]=st.commit_mismatch; stats[9]=st.commit_new_variant;
    stats[10]=st.first_bad_commit_region; stats[11]=st.commit_maskbad; stats[12]=st.commit_rebuild;
    return s.err?1:0;
}

void* nvm_mdam_create(int nops, const uint8_t* kind, const int32_t* a1, const int32_t* a2,
                      const int32_t* i0, const int32_t* i1, const double* dval,
                      const uint64_t* mmask, int nmmask,
                      const double* hazards, int nhaz,
                      const int* site_nchan, const double* ch_prob, const uint64_t* ch_x, const uint64_t* ch_z, int nsites,
                      const uint64_t* cp_x, const uint64_t* cp_z, int ncp,
                      int num_qubits, int num_meas, int engine_n, int max_work, int record_cap) {
    auto* p = new MdamProgram();
    p->kind.assign(kind, kind+nops); p->a1.assign(a1,a1+nops); p->a2.assign(a2,a2+nops);
    p->i0.assign(i0,i0+nops); p->i1.assign(i1,i1+nops); p->dval.assign(dval,dval+nops);
    if(nmmask>0) p->mmask.assign(mmask, mmask+nmmask);
    if(nhaz>0) p->hazards.assign(hazards, hazards+nhaz);
    // Noise + conditional-Pauli masks are MULTIWORD: cp_x/cp_z/ch_x/ch_z are flat (count x MW) arrays,
    // MW=ceil(num_qubits/64) words per mask (matches Python make_prog).  Single-uint64 truncated feedback
    // Paulis on qubits >=64 (e.g. coherent_d7_* at n=118 -> frame divergence -> record flip).
    int MW = (num_qubits + 63) / 64; if(MW < 1) MW = 1;
    p->noise_sites.resize(nsites);
    { int off=0; for(int s=0;s<nsites;s++){ for(int j=0;j<site_nchan[s];j++){ NoiseChannel c; c.prob=ch_prob[off];
        c.x_words.assign(ch_x + (size_t)off*MW, ch_x + (size_t)off*MW + MW);
        c.z_words.assign(ch_z + (size_t)off*MW, ch_z + (size_t)off*MW + MW);
        p->noise_sites[s].channels.push_back(c); off++; } } }
    p->cp_masks.resize(ncp);
    for(int s=0;s<ncp;s++){ NoiseChannel c; c.prob=1.0;
        c.x_words.assign(cp_x + (size_t)s*MW, cp_x + (size_t)s*MW + MW);
        c.z_words.assign(cp_z + (size_t)s*MW, cp_z + (size_t)s*MW + MW);
        p->cp_masks[s].channels.push_back(c); }
    p->num_qubits=num_qubits; p->num_measurements=num_meas; p->engine_n=engine_n; p->max_work=max_work; p->record_cap=record_cap;
    return p;
}
void nvm_mdam_free(void* prog){ delete reinterpret_cast<MdamProgram*>(prog); }

// Gate L Tier-3 DIRECT: attach the precomputed frame-keyed fused-unitary decomposition tables.
// U2: bcd = nnodes*4*3 doubles (per node, 4 in_states, ZXZ angles b,c,d); outs = nnodes*4.
void nvm_mdam_set_u2(void* prog, int nnodes, const double* bcd, const uint8_t* outs){
    auto& p=*reinterpret_cast<MdamProgram*>(prog);
    p.n_u2=nnodes;
    p.u2_bcd.assign(bcd, bcd+(size_t)nnodes*4*3);
    p.u2_out.assign(outs, outs+(size_t)nnodes*4);
}
// U4: per node 16 in_states.  start/cnt = nnodes*16 each (op index range into ops); ops = nops_total*5
// doubles (type,which,px,pz,theta); outs = nnodes*16.
void nvm_mdam_set_u4(void* prog, int nnodes, const int32_t* start, const int32_t* cnt,
                     const double* ops, int nops_total, const uint8_t* outs){
    auto& p=*reinterpret_cast<MdamProgram*>(prog);
    p.n_u4=nnodes;
    p.u4_start.assign(start, start+(size_t)nnodes*16);
    p.u4_cnt.assign(cnt, cnt+(size_t)nnodes*16);
    p.u4_ops.assign(ops, ops+(size_t)nops_total*5);
    p.u4_out.assign(outs, outs+(size_t)nnodes*16);
}

void* nvm_mdam_vm_create(void* prog){ auto* s=new MdamShot(); s->init(*reinterpret_cast<MdamProgram*>(prog)); return s; }
void nvm_mdam_vm_free(void* vm){ delete reinterpret_cast<MdamShot*>(vm); }

// Gate F: toggle the structural caches (core_cache = E-C, plan_cache = F4).  Resets cache state so
// the next shot rebuilds.  Used for the F-ladder (E0 = both on but plan rebuilt-each, etc.).
void nvm_mdam_vm_set_cache(void* vm, int core_on, int plan_on){
    auto& s=*reinterpret_cast<MdamShot*>(vm);
    s.core_cache_on=(core_on!=0); s.plan_cache_on=(plan_on!=0);
    for(auto& v: s.core_cache) v.clear();
    for(auto& vlist: s.plan_cache) vlist.clear();
}

// Gate F-B: select region-compiler mode (0=OFF, 1=COMPILE, 2=SHADOW, 3=FAST) and reset its state so
// the next shot recompiles snapshots.  COMPILE auto-runs on the first shot when mode>=SHADOW anyway.
void nvm_mdam_vm_set_fb(void* vm, int mode){
    auto& s=*reinterpret_cast<MdamShot*>(vm);
    s.fb_mode=mode; s.fb_compiled=false; s.fb_snap.clear();
    s.fb_mismatch=0; s.fb_hits=0; s.fb_misses=0;
    s.fb_bad_boundary=-1; s.fb_bad_idx=-1; s.fb_bad_field=nullptr;
}
// Gate F5: enable inverse-only commit folds + skip discarded pending consume (FAST only). 0=off,1=on.
void nvm_mdam_vm_set_f5(void* vm, int mode){ reinterpret_cast<MdamShot*>(vm)->f5_mode=mode; }

// Gate I (Imem): compiled-control memo mode (0=off,1=shadow,2=fast); resets the memo+counters.
void nvm_mdam_vm_set_imem(void* vm, int mode){
    auto& s=*reinterpret_cast<MdamShot*>(vm);
    s.imem_mode=mode; s.imem.clear(); s.imem_hist.clear();
    s.imem_hits=0; s.imem_misses=0; s.imem_mismatch=0;
}
void nvm_mdam_imem_stats(void* vm, long* out){   // [hits, misses, mismatch, table_size]
    auto& s=*reinterpret_cast<MdamShot*>(vm);
    out[0]=s.imem_hits; out[1]=s.imem_misses; out[2]=s.imem_mismatch; out[3]=(long)s.imem.size();
}
// Gate I-D: affine-feasibility capture toggle + readback (per-shot icap, default off).
void nvm_mdam_set_icap(void* vm, int on){ reinterpret_cast<MdamShot*>(vm)->icap_on=(on!=0); }
int nvm_mdam_get_icap(void* vm, long* out, int maxn){
    auto& s=*reinterpret_cast<MdamShot*>(vm); int n=(int)s.icap.size(); if(n>maxn) n=maxn;
    for(int i=0;i<n;i++) out[i]=s.icap[i]; return (int)s.icap.size();
}
// Decision-graph feasibility: dsig capture toggle + readback (default off).  4 uint64/magic measurement:
// [mp, keyhash, statehash, flag(0=compiled full-key, 1=oracle state-only)].  dsig accumulates across the
// whole batch (NOT cleared per shot); the harness resets, runs sample_batch (authoritative), reads back.
void nvm_mdam_dsig_set(void* vm, int on){ reinterpret_cast<MdamShot*>(vm)->dsig_on=(on!=0); }
void nvm_mdam_dsig_reset(void* vm){ auto& s=*reinterpret_cast<MdamShot*>(vm); s.dsig.clear(); s.dsig_over=0; }
long nvm_mdam_dsig_count(void* vm){ return (long)(reinterpret_cast<MdamShot*>(vm)->dsig.size()/4); }
long nvm_mdam_dsig_get(void* vm, uint64_t* out, long maxrec){
    auto& s=*reinterpret_cast<MdamShot*>(vm); long tot=(long)(s.dsig.size()/4); long n=tot>maxrec?maxrec:tot;
    for(long i=0;i<n*4;i++) out[i]=s.dsig[i]; return tot;
}

// Gate G: run ONE shot with core capture; returns #cores.  Then nvm_mdam_core_get(i, ...) reads core i.
int nvm_mdam_capture(void* prog, void* vm, uint64_t shi,uint64_t slo,uint64_t ihi,uint64_t ilo){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    s.core_capture=true; s.core_caps.clear(); s.reset_shot(p,shi,slo,ihi,ilo); s.run(p); s.core_capture=false;
    return (int)s.core_caps.size();
}
// scalar header of core i: out=[r_in,r_mat,nrot,nlm,m_bit, sign(as int*1e0? -> separate)]
void nvm_mdam_core_hdr(void* vm,int i,int* hdr,double* sign){
    auto& cc=reinterpret_cast<MdamShot*>(vm)->core_caps[i];
    hdr[0]=cc.r_in; hdr[1]=cc.r_mat; hdr[2]=cc.nrot; hdr[3]=cc.nlm; hdr[4]=cc.m_bit; *sign=cc.sign;
}
void nvm_mdam_core_data(void* vm,int i,uint64_t* rx,uint64_t* rz,int* rpp,double* rc,double* rs,
                        int* lt,int* la,int* lb,double* phi_re,double* phi_im){
    auto& cc=reinterpret_cast<MdamShot*>(vm)->core_caps[i];
    for(int k=0;k<cc.nrot;k++){ rx[k]=cc.rx[k]; rz[k]=cc.rz[k]; rpp[k]=cc.rpp[k]; rc[k]=cc.rc[k]; rs[k]=cc.rs[k]; }
    for(int k=0;k<cc.nlm;k++){ lt[k]=cc.lt[k]; la[k]=cc.la[k]; lb[k]=cc.lb[k]; }
    size_t N=cc.phi_in.size(); for(size_t k=0;k<N;k++){ phi_re[k]=cc.phi_in[k].real(); phi_im[k]=cc.phi_in[k].imag(); }
}
// out[0]=fb_mismatch, [1]=fb_hits, [2]=fb_misses, [3]=compiled, [4]=bad_boundary, [5]=bad_idx
#ifdef FB_COUNT
void nvm_fb_count_reset(){ mdam::fbc()=mdam::FbCounters{}; }
void nvm_fb_count_get(long* out){ auto&c=mdam::fbc();
    out[0]=c.tab; out[1]=c.pend; out[2]=c.inv; out[3]=c.tab_right; out[4]=c.inv_right;
    out[5]=c.foldx; out[6]=c.consume; out[7]=c.mupd; out[8]=c.dropscan; }
#endif
#ifdef MDAM_INSTR
// Gate I control-plane dissection: skip mask + rdtsc accumulators (profiling-only build).
void nvm_instr_set_skip(int mask){ mdam::instr().skip = mask; }
void nvm_instr_reset(){ for(int i=0;i<mdam::IT_NSLOT;i++){ mdam::instr().tcyc[i]=0; mdam::instr().tcnt[i]=0; } }
void nvm_instr_get(uint64_t* cyc, uint64_t* cnt){ for(int i=0;i<mdam::IT_NSLOT;i++){ cyc[i]=mdam::instr().tcyc[i]; cnt[i]=mdam::instr().tcnt[i]; } }
#endif
// ---- one-normal-form self-test + core-apply counter ----
// Proves the compiled kernel's direct_rot and the canonical pauli_rot_apply are the SAME primitive:
// applies a random Pauli rotation R=α I+β i^pp X^x Z^z via both to identical states; returns max abs
// componentwise difference (0.0 == bit-identical -> one normal form, two instantiations).
void mdm_direct_rot_test(double*, long, unsigned long long, unsigned long long, int, double, double);
double nvm_selftest_pauli_apply(uint64_t seed){
    const long N=256; std::vector<mdam::cd> a(N), b(N);
    uint64_t st = seed ? seed : 0x9E3779B97F4A7C15ULL;
    auto nx=[&](){ st^=st<<13; st^=st>>7; st^=st<<17; return st; };
    for(long i=0;i<N;i++){ double re=((double)(nx()>>11))*(1.0/9007199254740992.0)*2-1,
                                  im=((double)(nx()>>11))*(1.0/9007199254740992.0)*2-1; a[i]=b[i]=mdam::cd(re,im); }
    uint64_t x=nx()&0xFF, z=nx()&0xFF; int pp=(int)(nx()&3);
    double theta=((double)(nx()>>11))*(1.0/9007199254740992.0)*6.283185307179586;
    double c=std::cos(theta/2.0), s=std::sin(theta/2.0);
    mdm_direct_rot_test(reinterpret_cast<double*>(a.data()), N, (unsigned long long)x, (unsigned long long)z, pp, c, s);
    mdam::pauli_rot_apply(b.data(), (size_t)N, x, z, pp, mdam::cd(c,0), mdam::cd(0,-s));
    double mx=0; for(long i=0;i<N;i++){ double d=std::abs(a[i]-b[i]); if(d>mx) mx=d; }
    return mx;
}
unsigned long long nvm_core_apply_count(){ return (unsigned long long)mdam::NativeDenseEngineState::core_apply_count(); }
void nvm_core_apply_reset(){ mdam::NativeDenseEngineState::core_apply_count()=0; }
unsigned long long nvm_dense_flop_core(){ return (unsigned long long)mdam::dense_flop_core(); }
unsigned long long nvm_dense_flop_rot(){ return (unsigned long long)mdam::dense_flop_rot(); }
unsigned long long nvm_dense_flop_collapse(){ return (unsigned long long)mdam::dense_flop_collapse(); }
unsigned long long nvm_dense_flop_loc(){ return (unsigned long long)mdam::dense_flop_loc(); }
int nvm_dense_peak_r(){ return mdam::dense_peak_r(); }
// Step 1 pullback mask-invariance.  stats: out[kind*4 + {calls, unique_keys, mask_violations, phase_varies}], kinds 0..3.
void nvm_pb_cap(int on){ mdam::pb_cap_on()=(on!=0); }
void nvm_pb_reset(){ mdam::pb_map().clear(); }
void nvm_pb_stats(long* out){ for(int i=0;i<20;i++) out[i]=0;   // 5 fields x 4 kinds
    for(auto& kv : mdam::pb_map()){ int kind=(int)(kv.first.tag & 0xff); if(kind<0||kind>3) continue;
        out[kind*5+0]+=kv.second.calls; out[kind*5+1]+=1; out[kind*5+2]+=kv.second.mask_viol;
        out[kind*5+3]+=(kv.second.phase_varies?1:0); out[kind*5+4]+=kv.second.phase_affine_viol; } }
// ---- Phase B: pullback StaticPlan fast path (default OFF) ----
void nvm_pb_static(int on){ mdam::pb_static_on()=on; }
void nvm_pb_static_shadow(int on){ mdam::pb_static_shadow()=on; }
void nvm_pb_static_phase(int on){ mdam::pb_static_phase()=on; }
void nvm_pb_static_reset(){ mdam::pb_static_map().clear(); mdam::pb_shadow_fail()=mdam::PbShadowFail{};
    for(auto& s : mdam::pb_cache()){ s.valid=false; s.ent=nullptr; } }
// stats: out[0]=keys, [1]=total_calls, [2]=shadow_mask_viol, [3]=shadow_phase_viol, [4]=shadow_fail_hit
void nvm_pb_static_stats(long* out){ for(int i=0;i<8;i++) out[i]=0;
    for(auto& kv : mdam::pb_static_map()){ out[0]+=1; out[1]+=kv.second.calls;
        out[2]+=kv.second.shadow_mask_viol; out[3]+=kv.second.shadow_phase_viol; }
    out[4]=mdam::pb_shadow_fail().hit?1:0; }
// first shadow mismatch dump: out[0]=hit,[1]=kind,[2]=mp,[3..6]=in(x0,x1,z0,z1),[7..10]=static out,
//   [11..14]=live out,[15]=static phase,[16]=live phase
void nvm_pb_shadow_fail(long* out){ mdam::PbShadowFail& f=mdam::pb_shadow_fail();
    out[0]=f.hit?1:0; out[1]=f.kind; out[2]=f.mp;
    out[3]=(long)f.ix0; out[4]=(long)f.ix1; out[5]=(long)f.iz0; out[6]=(long)f.iz1;
    out[7]=(long)f.sox0; out[8]=(long)f.sox1; out[9]=(long)f.soz0; out[10]=(long)f.soz1;
    out[11]=(long)f.lox0; out[12]=(long)f.lox1; out[13]=(long)f.loz0; out[14]=(long)f.loz1;
    out[15]=f.sphase; out[16]=f.lphase; }
// rebuild-vs-substitution cycle split: out[0]=rebuild_cyc,[1]=rebuild_cnt,[2]=subst_cyc,[3]=subst_cnt
unsigned long long nvm_rdtsc(){ return (unsigned long long)__builtin_ia32_rdtsc(); }
// ---- clean-room inverse-frame rebuild StaticPlan: de-risk checker (default OFF) ----
void nvm_rb_cap(int on){ mdam::rb_cap_on()=(on!=0); }
void nvm_rb_reset(){ mdam::rb_map().clear(); mdam::rb_count_hist().clear(); mdam::rb_epoch()=0; }
// out[0]=gen_keys,[1]=calls,[2]=mask_viol,[3]=phase_varies_keys,[4]=phase_affine_viol,[5]=distinct_sigs,[6]=count_hist_distinct
void nvm_rb_stats(long* out){ for(int i=0;i<8;i++) out[i]=0; long nsig=0; uint64_t last=~0ULL; bool first=true;  // map sorted by sig
    for(auto& kv : mdam::rb_map()){ out[0]+=1; out[1]+=kv.second.calls; out[2]+=kv.second.mask_viol;
        out[3]+=(kv.second.phase_varies?1:0); out[4]+=kv.second.phase_affine_viol;
        if(first||kv.first.sig!=last){ nsig++; last=kv.first.sig; first=false; } }
    out[5]=nsig; out[6]=(long)mdam::rb_count_hist().size(); }
// dump rebuild-count histogram: pairs (count, #shots); returns #pairs
int nvm_rb_count_hist(long* out, int maxn){ int i=0; for(auto& kv : mdam::rb_count_hist()){ if(i>=maxn) break;
    out[2*i]=kv.first; out[2*i+1]=kv.second; i++; } return i; }
// ---- inverse-frame rebuild StaticPlan FAST PATH (default OFF) ----
void nvm_rb_static(int on){ mdam::rb_static_on()=on; }
void nvm_rb_static_shadow(int on){ mdam::rb_static_shadow()=on; }
void nvm_rb_static_reset(){ mdam::rb_plan_map().clear(); mdam::rb_static_hits()=0; mdam::rb_static_misses()=0;
    mdam::rb_shadow_fail()=mdam::RbShadowFail{}; }
// out[0]=plans(distinct sigs),[1]=hits,[2]=misses,[3]=shadow_fail_hit
void nvm_rb_static_stats(long* out){ out[0]=(long)mdam::rb_plan_map().size(); out[1]=mdam::rb_static_hits();
    out[2]=mdam::rb_static_misses(); out[3]=mdam::rb_shadow_fail().hit?1:0; }
// shadow mismatch dump: [0]=hit,[1]=g,[2]=az,[3]=sig,[4..7]=static out,[8..11]=live out,[12]=sphase,[13]=lphase
void nvm_rb_shadow_fail(long* out){ mdam::RbShadowFail& f=mdam::rb_shadow_fail();
    out[0]=f.hit?1:0; out[1]=f.g; out[2]=f.az; out[3]=(long)f.sig;
    out[4]=(long)f.sx0; out[5]=(long)f.sx1; out[6]=(long)f.sz0; out[7]=(long)f.sz1;
    out[8]=(long)f.lx0; out[9]=(long)f.lx1; out[10]=(long)f.lz0; out[11]=(long)f.lz1;
    out[12]=f.sphase; out[13]=f.lphase; }
void nvm_pb_time(int on){ mdam::pb_time_on()=(on!=0); }
void nvm_pb_time_reset(){ mdam::pb_rebuild_cyc()=0; mdam::pb_rebuild_cnt()=0; mdam::pb_subst_cyc()=0; mdam::pb_subst_cnt()=0; }
void nvm_pb_time_get(unsigned long long* out){ out[0]=mdam::pb_rebuild_cyc(); out[1]=mdam::pb_rebuild_cnt();
    out[2]=mdam::pb_subst_cyc(); out[3]=mdam::pb_subst_cnt(); out[4]=mdam::pb_lookup_cyc(); out[5]=mdam::pb_affine_cyc(); }
void nvm_pb_time_reset2(){ mdam::pb_lookup_cyc()=0; mdam::pb_affine_cyc()=0; }
void nvm_dense_flop_reset(){ mdam::dense_flop_rot()=0; mdam::dense_flop_collapse()=0; mdam::dense_flop_loc()=0; mdam::dense_peak_r()=0; }
unsigned long long nvm_vm_rng_draws(void* vm){ return (unsigned long long)reinterpret_cast<mdam::MdamShot*>(vm)->rng_draws; }
void nvm_mdam_fb_stats(void* vm, long* out, char* badfield, int flen){
    auto& s=*reinterpret_cast<MdamShot*>(vm);
    out[0]=s.fb_mismatch; out[1]=s.fb_hits; out[2]=s.fb_misses; out[3]=s.fb_compiled?1:0;
    out[4]=s.fb_bad_boundary; out[5]=s.fb_bad_idx;
    if(badfield&&flen>0){ const char* f=s.fb_bad_field?s.fb_bad_field:""; std::strncpy(badfield,f,flen-1); badfield[flen-1]=0; }
}

// run one full shot; returns 0 on success, 1 on internal error (out_err set).
int nvm_mdam_run(void* prog, void* vm, uint64_t shi, uint64_t slo, uint64_t ihi, uint64_t ilo,
                 uint8_t* out_record, unsigned long long* out_draws, int* out_compiled, int* out_oracle,
                 char* out_err, int errlen) {
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    s.reset_shot(p, shi,slo,ihi,ilo);
    s.run(p);
    std::memcpy(out_record, s.record.bits.data(), (size_t)p.num_measurements);
    *out_draws=s.rng_draws; *out_compiled=s.magic_compiled; *out_oracle=s.magic_oracle;
    if(s.err){ std::strncpy(out_err, s.err, errlen-1); out_err[errlen-1]=0; return 1; }
    out_err[0]=0; return 0;
}

// ===== Phase-0/1/2 lightweight-semantic-key boundary-edge capture =========================================
// Run ONE authoritative shot (bit-exact vs Python) with per-boundary capture into s.bcap.  Caller pulls via
// nvm_bcap_n / nvm_bcap_get then continues to the next seed (the interner pool persists -> sid ids are stable
// across the whole run).  Edges come from the AUTHORITATIVE measure_z, so NO F4/imem/plan/bplan involvement.
int nvm_mdam_run_bcap(void* prog, void* vm, uint64_t shi, uint64_t slo, uint64_t ihi, uint64_t ilo,
                      uint8_t* out_record, char* out_err, int errlen){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    s.reset_shot(p, shi,slo,ihi,ilo);
    s.bcap.clear(); s.bcap_on=true; s.run(p); s.bcap_on=false;
    if(out_record) std::memcpy(out_record, s.record.bits.data(), (size_t)p.num_measurements);
    if(s.err){ std::strncpy(out_err,s.err,errlen-1); out_err[errlen-1]=0; return 1; }
    out_err[0]=0; return 0;
}
long nvm_bcap_n(void* vm){ return (long)reinterpret_cast<MdamShot*>(vm)->bcap.size(); }
// out_i: row-major [n,13] int64 = {mp,sid_in,inv_sig,pend_sig,m_sig,xb,zb,i1,kind,oracle,outcome,sid_out,rec};
// out_p0: [n] double.  Caller sizes both to >= nvm_bcap_n().
void nvm_bcap_get(void* vm, long long* out_i, double* out_p0){
    auto& s=*reinterpret_cast<MdamShot*>(vm); size_t n=s.bcap.size();
    for(size_t k=0;k<n;k++){ auto& r=s.bcap[k]; long long* o=out_i+13*(long)k;
        o[0]=r.mp; o[1]=r.sid_in; o[2]=r.inv_sig; o[3]=r.pend_sig; o[4]=r.m_sig;
        o[5]=r.xb; o[6]=r.zb; o[7]=r.i1; o[8]=r.kind; o[9]=r.oracle; o[10]=r.outcome; o[11]=r.sid_out; o[12]=r.rec;
        out_p0[k]=r.p0; } }
long nvm_bcap_distinct_states(void* vm){ return (long)reinterpret_cast<MdamShot*>(vm)->bcap_amp.size(); }

// ===== Phase-3 authoritative-edge cache (run_mcache) ====================================================
void nvm_mcache_set_mode(void* vm, int mode){ reinterpret_cast<MdamShot*>(vm)->mc_mode=mode; }
void nvm_mcache_reset(void* vm){ reinterpret_cast<MdamShot*>(vm)->mc_reset(); }
void nvm_mcache_set_time(void* vm, int t){ reinterpret_cast<MdamShot*>(vm)->mc_time=(t!=0); }
void nvm_mcache_cyc_get(void* vm, uint64_t* out){ auto& s=*reinterpret_cast<MdamShot*>(vm); for(int i=0;i<8;i++) out[i]=s.mc_cyc[i]; }
void nvm_mcache_set_skip(void* vm, int mask){ reinterpret_cast<MdamShot*>(vm)->mc_skip=mask; }
void nvm_mcache_set_optime(void* vm, int t){ reinterpret_cast<MdamShot*>(vm)->mc_optime=(t!=0); }
void nvm_mcache_set_fblock(void* vm, int t){ reinterpret_cast<MdamShot*>(vm)->mc_fblock=(t!=0); }
void nvm_mcache_opcyc_get(void* vm, uint64_t* out){ auto& s=*reinterpret_cast<MdamShot*>(vm); for(int i=0;i<8;i++) out[i]=s.mc_opcyc[i]; }
int nvm_mdam_run_mcache(void* prog, void* vm, uint64_t shi, uint64_t slo, uint64_t ihi, uint64_t ilo,
                        uint8_t* out_record, char* out_err, int errlen){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    s.reset_shot(p, shi,slo,ihi,ilo);
    s.run_mcache(p);
    if(out_record) std::memcpy(out_record, s.record.bits.data(), (size_t)p.num_measurements);
    if(s.err){ std::strncpy(out_err,s.err,errlen-1); out_err[errlen-1]=0; return 1; }
    out_err[0]=0; return 0;
}
int nvm_mcache_batch(void* prog, void* vm, uint64_t num_shots,
                     uint64_t mshi, uint64_t mslo, uint64_t mihi, uint64_t milo,
                     uint8_t* out_record, char* out_err, int errlen){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    return s.run_mcache_batch(p, num_shots, mshi,mslo,mihi,milo, out_record, out_err, errlen);
}
// precise cache footprint (vector capacities): out[0]=pool_bytes out[1]=sid-pool(bcap_amp)_bytes
// out[2]=edge_bytes out[3]=total.  Measures the ACTUAL memory the edge cache + state pool hold.
void nvm_mcache_membytes(void* vm, uint64_t* out){
    auto& s=*reinterpret_cast<MdamShot*>(vm);
    uint64_t pool=0;
    for(auto& e:s.mc_pool){ pool += (uint64_t)e.dense.capacity()*sizeof(std::complex<double>);
        pool += (uint64_t)(e.ax.capacity()+e.az.capacity()+e.Xc.capacity()+e.Zc.capacity())*sizeof(mdam::PackedPauli);
        pool += (uint64_t)e.pend.capacity()*sizeof(mdam::PendingEntry) + (uint64_t)e.M.capacity()*sizeof(int); }
    uint64_t sidp=0; for(auto& a:s.bcap_amp) sidp += (uint64_t)a.capacity()*sizeof(std::complex<double>);
    uint64_t edges=0; for(auto& m:s.mc_edges) edges += (uint64_t)m.size()*(sizeof(uint64_t)+sizeof(MdamShot::MEdge)+32);  // +node/bucket overhead approx
    out[0]=pool; out[1]=sidp; out[2]=edges; out[3]=pool+sidp+edges;
}
// stats[0..9] = hit, miss, partial, antis, verify, mismatch, restore, pool_size, edge_count, distinct_states
void nvm_mcache_stats(void* vm, long* out){
    auto& s=*reinterpret_cast<MdamShot*>(vm);
    long ec=0; for(auto& m:s.mc_edges) ec+=(long)m.size();
    out[0]=s.mc_hit; out[1]=s.mc_miss; out[2]=s.mc_partial; out[3]=s.mc_antis; out[4]=s.mc_verify;
    out[5]=s.mc_mismatch; out[6]=s.mc_restore; out[7]=(long)s.mc_pool.size(); out[8]=ec; out[9]=(long)s.bcap_amp.size();
}
// ===== path-3 clean-room segment/automaton SEPARABILITY shadow (default OFF) =====
void nvm_sg_shadow(void* vm, int on){ reinterpret_cast<MdamShot*>(vm)->sg_shadow=on; }
void nvm_sg_signs(void* vm, int on){ reinterpret_cast<MdamShot*>(vm)->sg_signs=on; }
void nvm_sg_reset(void* vm){ reinterpret_cast<MdamShot*>(vm)->sg_reset(); }
// path-3 reduced-execution lean walk: run_lean_batch (uses the pre-warmed sg table; out_incomplete optional)
int nvm_run_lean_batch(void* prog, void* vm, uint64_t num_shots,
                       uint64_t mshi, uint64_t mslo, uint64_t mihi, uint64_t milo,
                       uint8_t* out_record, uint8_t* out_incomplete, char* out_err, int errlen){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    return s.run_lean_batch(p, num_shots, mshi,mslo,mihi,milo, out_record, out_incomplete, out_err, errlen);
}
// DIAGNOSTIC: why doesn't the automaton saturate?  Count distinct dense blocks (bcap_amp) under
// out[0]=exact (baseline = bcap_amp size), out[1]=rounded(1e-9), out[2]=phase-canonical+rounded,
// out[3]=|amp|^2 (modulus) rounded.  A big drop under phase-canonical => global phase is the culprit.
void nvm_diag_compress(void* vm, long* out){
    auto& s=*reinterpret_cast<MdamShot*>(vm);
    const double g=1e-9; auto rnd=[&](double x){ return (long long)std::llround(x/g); };
    std::unordered_set<uint64_t> se, sr, sp, sm;
    auto fnv=[&](uint64_t h,long long v){ h^=(uint64_t)v; h*=1099511628211ULL; return h; };
    for(auto& blk : s.bcap_amp){ size_t N=blk.size();
        uint64_t he=1469598103934665603ULL, hr=he, hp=he, hm=he;
        // exact + rounded + modulus
        for(size_t j=0;j<N;j++){ he=fnv(he,(long long)(blk[j].real()*0)); // exact handled below
            hr=fnv(fnv(hr,rnd(blk[j].real())),rnd(blk[j].imag()));
            hm=fnv(hm,rnd(std::norm(blk[j]))); }
        // exact: hash raw bits
        he=1469598103934665603ULL; for(size_t j=0;j<N;j++){ double re=blk[j].real(),im=blk[j].imag();
            uint64_t br,bi; std::memcpy(&br,&re,8); std::memcpy(&bi,&im,8); he=fnv(fnv(he,(long long)br),(long long)bi); }
        // phase-canonical: divide by phase of first significant amp, then round
        size_t j0=0; while(j0<N && std::norm(blk[j0])<1e-18) j0++;
        std::complex<double> ph = (j0<N)? blk[j0]/std::abs(blk[j0]) : std::complex<double>(1,0);
        for(size_t j=0;j<N;j++){ std::complex<double> z = blk[j]/ph;
            hp=fnv(fnv(hp,rnd(z.real())),rnd(z.imag())); }
        se.insert(he); sr.insert(hr); sp.insert(hp); sm.insert(hm);
    }
    out[0]=(long)se.size(); out[1]=(long)sr.size(); out[2]=(long)sp.size(); out[3]=(long)sm.size();
}
void nvm_lean_reset_counts(void* vm){ auto& s=*reinterpret_cast<MdamShot*>(vm); s.ln_incomplete_shots=0; s.ln_miss=0; s.ln_fb_count=0; }
void nvm_lean_stats(void* vm, long* out){ auto& s=*reinterpret_cast<MdamShot*>(vm);
    out[0]=s.ln_incomplete_shots; out[1]=s.ln_miss; out[2]=s.ln_fb_count; }
int nvm_run_lean_fb_batch(void* prog, void* vm, uint64_t num_shots,
                          uint64_t mshi, uint64_t mslo, uint64_t mihi, uint64_t milo,
                          uint8_t* out_record, char* out_err, int errlen){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    return s.run_lean_fb_batch(p, num_shots, mshi,mslo,mihi,milo, out_record, out_err, errlen);
}
// Adaptive bounded-regret executor: lean optimistic start + conservative sticky SLOW_ONLY demote.
// Output is bit-identical to nvm_run_lean_fb_batch (policy changes speed, never per-shot record).
int nvm_run_lean_adapt_batch(void* prog, void* vm, uint64_t num_shots,
                             uint64_t mshi, uint64_t mslo, uint64_t mihi, uint64_t milo,
                             uint8_t* out_record, char* out_err, int errlen){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    return s.run_lean_adapt_batch(p, num_shots, mshi,mslo,mihi,milo, out_record, out_err, errlen);
}
// config: window, node_cap, edge_cap, mem_cap(bytes), horizon(shots), node_floor, cost_margin, bad_needed
void nvm_adapt_config(void* vm, long window, long node_cap, long edge_cap, long mem_cap, long horizon,
                      double node_floor, double cost_margin, int bad_needed){
    auto& s=*reinterpret_cast<MdamShot*>(vm);
    if(window>0)     s.ad_window=window;      if(node_cap>0) s.ad_node_cap=node_cap;
    if(edge_cap>0)   s.ad_edge_cap=edge_cap;  if(mem_cap>0)  s.ad_mem_cap=mem_cap;
    if(horizon>=0)   s.ad_horizon=horizon;    if(node_floor>=0) s.ad_node_floor=node_floor;
    if(cost_margin>0)s.ad_cost_margin=cost_margin; if(bad_needed>0) s.ad_bad_needed=bad_needed;
}
// out: 0 final_policy(0=LEAN,1=SLOW_ONLY), 1 demote_shot, 2 windows, 3 slow_shots, 4 node_rate_init,
//      5 node_rate_last, 6 lean_ns_last, 7 slow_ns_last, 8 fb_rate_last, 9 nodes, 10 edges, 11 mem_est_bytes
void nvm_adapt_stats(void* vm, double* out){ auto& s=*reinterpret_cast<MdamShot*>(vm);
    out[0]=s.ad_final_policy; out[1]=(double)s.ad_demote_shot; out[2]=(double)s.ad_windows;
    out[3]=(double)s.ad_slow_shots; out[4]=s.ad_node_rate_init; out[5]=s.ad_node_rate_last;
    out[6]=s.ad_lean_ns_last; out[7]=s.ad_slow_ns_last; out[8]=s.ad_fb_rate_last;
    out[9]=(double)s.ln_id.size(); out[10]=(double)s.ln_edge.size(); out[11]=(double)s.ad_mem_est();
    out[12]=(double)s.mc_pool.size(); out[13]=(double)s.mc_pool_bytes(); }
// out: 0 distinct_edges, 1 edge_checks, 2 edge_viol, 3 boundaries, 4 p0_checks, 5 p0_viol,
//      6 antis_checks, 7 antis_viol, 8 distinct_nodes(p0 map size)
void nvm_sg_stats(void* vm, long* out){ auto& s=*reinterpret_cast<MdamShot*>(vm);
    out[0]=s.sg_edges; out[1]=s.sg_checks; out[2]=s.sg_viol; out[3]=s.sg_bounds;
    out[4]=s.sg_p0_checks; out[5]=s.sg_p0_viol; out[6]=s.sg_antis_checks; out[7]=s.sg_antis_viol;
    out[8]=(long)s.sg_p0.size(); }

// Gate D: run a whole shot batch with ONE Python->C++ call.  Python hands the master PCG64
// (state,inc) once (extracted from np.random.default_rng(seed)); native expands per-shot seeds.
// out_record is a preallocated [num_shots, num_measurements] uint8 contiguous buffer.
// stats_out (optional, 4 x uint64): {total_draws, total_compiled, total_oracle, first_error_shot}.
int nvm_mdam_sample_batch(void* prog, void* vm, uint64_t num_shots,
                          uint64_t mshi, uint64_t mslo, uint64_t mihi, uint64_t milo,
                          uint8_t* out_record, uint64_t* stats_out, char* out_err, int errlen) {
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    NativeBatchStats st;
    int rc = s.run_batch(p, num_shots, mshi, mslo, mihi, milo, out_record, &st, out_err, errlen);
    if (stats_out) { stats_out[0]=st.total_draws; stats_out[1]=st.total_compiled;
                     stats_out[2]=st.total_oracle; stats_out[3]=(uint64_t)st.first_error_shot;
                     stats_out[4]=st.m_state_hi; stats_out[5]=st.m_state_lo;
                     stats_out[6]=st.m_inc_hi;   stats_out[7]=st.m_inc_lo; }
    return rc;
}

// §2/§10 PROFILE: per-op-category breakdown of the authoritative run() (PROFILE build only).
// out[0..16] = prof[] (ns total over num_shots): SEED,RESET,RUN,MAGIC_PLAN,MAGIC_KERNEL,MAGIC_COMMIT,
// ORACLE,OUTPUT,OP_FRAME,OP_ACTIVEGATE,OP_ROT,OP_NOISE,OP_DORMANT,OP_OTHER,PLAN_CORE,PULLBACK,LOCALIZER.
int nvm_mdam_run_batch_prof(void* prog, void* vm, uint64_t num_shots,
                            uint64_t mshi, uint64_t mslo, uint64_t mihi, uint64_t milo,
                            uint8_t* out_record, double* out_prof){
#ifdef MDAM_PROFILE
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    s.run_batch_prof(p, num_shots, mshi, mslo, mihi, milo, out_record);
    if(out_prof) for(int i=0;i<17;i++) out_prof[i]=s.prof[i];
    return 0;
#else
    (void)prog;(void)vm;(void)num_shots;(void)mshi;(void)mslo;(void)mihi;(void)milo;(void)out_record;
    if(out_prof) for(int i=0;i<17;i++) out_prof[i]=0;
    return 1;   // not a PROFILE build
#endif
}

int nvm_mdam_run_framelog(void* prog, void* vm, uint64_t shi, uint64_t slo, uint64_t ihi, uint64_t ilo,
                          uint8_t* out_record, uint64_t* fx, uint64_t* fz, int maxn){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    s.reset_shot(p, shi,slo,ihi,ilo); s.frame_log_on=true; s.frame_log.clear(); s.run(p);
    std::memcpy(out_record, s.record.bits.data(), (size_t)p.num_measurements);
    int n=(int)s.frame_log.size(); if(n>maxn)n=maxn;
    for(int i=0;i<n;i++){ fx[i]=s.frame_log[i][0]; fz[i]=s.frame_log[i][1]; }
    s.frame_log_on=false; return (int)s.frame_log.size();
}

int nvm_mdam_run_rotlog(void* prog, void* vm, uint64_t shi, uint64_t slo, uint64_t ihi, uint64_t ilo,
                        uint8_t* out_record, double* slot, double* xb, double* angle, double* theta, int maxn){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    s.reset_shot(p, shi,slo,ihi,ilo); s.rot_log_on=true; s.rot_log.clear(); s.run(p);
    std::memcpy(out_record, s.record.bits.data(), (size_t)p.num_measurements);
    int n=(int)s.rot_log.size(); if(n>maxn)n=maxn;
    for(int i=0;i<n;i++){ slot[i]=s.rot_log[i][0]; xb[i]=s.rot_log[i][1]; angle[i]=s.rot_log[i][2]; theta[i]=s.rot_log[i][3]; }
    s.rot_log_on=false; return (int)s.rot_log.size();
}

// debug: run, dump engine state before magic index k (W=1 assumed).  Returns pending count.
// Fills M (nM), pending px/pz/pp/pth, and Xc/Zc x/z/phase (engine_n each).
int nvm_mdam_dump(void* prog, void* vm, uint64_t shi, uint64_t slo, uint64_t ihi, uint64_t ilo, int k,
                  int* nM_out, int* Mbuf, uint64_t* px, uint64_t* pz, int* pp, double* pth, int* nP_out,
                  uint64_t* xcx, uint64_t* xcz, int* xcp, uint64_t* zcx, uint64_t* zcz, int* zcp, int maxn) {
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    s.reset_shot(p, shi,slo,ihi,ilo); s.dump_before_magic=k; s.dumped=false; s.magic_seen=0;
    s.run(p);
    *nM_out=(int)s.dM.size(); for(size_t i=0;i<s.dM.size();i++) Mbuf[i]=s.dM[i];
    int nP=(int)s.dPth.size(); if(nP>maxn)nP=maxn;
    for(int i=0;i<nP;i++){ px[i]=s.dPx[i]; pz[i]=s.dPz[i]; pp[i]=s.dPp[i]; pth[i]=s.dPth[i]; }
    *nP_out=(int)s.dPth.size();
    for(int i=0;i<(int)s.dXcp.size();i++){ xcx[i]=s.dXcx[i]; xcz[i]=s.dXcz[i]; xcp[i]=s.dXcp[i]; zcx[i]=s.dZcx[i]; zcz[i]=s.dZcz[i]; zcp[i]=s.dZcp[i]; }
    return (int)s.dXcp.size();
}

// debug: run with magic-measurement logging; returns count, fills per-magic (q,rin,rmat,nrot,nlm,feasible,outcome,p0)
int nvm_mdam_run_magiclog(void* prog, void* vm, uint64_t shi, uint64_t slo, uint64_t ihi, uint64_t ilo,
                          uint8_t* out_record, int* qa, int* rin, int* rmat, int* nrot, int* nlm,
                          int* feas, int* oc, double* p0, int maxn) {
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    s.reset_shot(p, shi,slo,ihi,ilo); s.magic_log_on=true; s.magic_log.clear();
    s.run(p);
    std::memcpy(out_record, s.record.bits.data(), (size_t)p.num_measurements);
    int n=(int)s.magic_log.size(); if(n>maxn)n=maxn;
    for(int i=0;i<n;i++){ auto&m=s.magic_log[i]; qa[i]=m.q; rin[i]=m.rin; rmat[i]=m.rmat; nrot[i]=m.nrot;
        nlm[i]=m.nlm; feas[i]=m.feasible; oc[i]=m.outcome; p0[i]=m.p0; }
    s.magic_log_on=false; return (int)s.magic_log.size();
}

// debug: run with rng draw logging; fills kinds/vals (0=double,1=bounded) up to maxn, returns count
int nvm_mdam_run_logged(void* prog, void* vm, uint64_t shi, uint64_t slo, uint64_t ihi, uint64_t ilo,
                        uint8_t* out_record, int* kinds, double* vals, int maxn) {
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    s.reset_shot(p, shi,slo,ihi,ilo);
    s.rng.dbg_log = true; s.rng.dlog.clear();
    // reset_shot re-seeded AND the sampler.init already drew 1 with dbg_log off; redo with logging:
    s.rng.seed_from_state(shi,slo,ihi,ilo); s.rng.dlog.clear();
    s.sampler.init(p.hazards, &s.rng);
    s.run(p);
    std::memcpy(out_record, s.record.bits.data(), (size_t)p.num_measurements);
    int n = (int)s.rng.dlog.size(); if (n > maxn) n = maxn;
    for (int i=0;i<n;i++){ kinds[i]=s.rng.dlog[i].first; vals[i]=s.rng.dlog[i].second; }
    s.rng.dbg_log = false;
    return (int)s.rng.dlog.size();
}

// §0 internal-trace: run with noise-fire logging; fills (site,xword,zword) per fired site.
int nvm_mdam_run_noiselog(void* prog, void* vm, uint64_t shi, uint64_t slo, uint64_t ihi, uint64_t ilo,
                          uint8_t* out_record, uint64_t* site, uint64_t* xw, uint64_t* zw, int maxn) {
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    s.reset_shot(p, shi,slo,ihi,ilo);
    s.sampler.log_on=true; s.sampler.fire_log.clear();
    s.run(p);
    std::memcpy(out_record, s.record.bits.data(), (size_t)p.num_measurements);
    int n=(int)s.sampler.fire_log.size(); if(n>maxn)n=maxn;
    for(int i=0;i<n;i++){ site[i]=s.sampler.fire_log[i][0]; xw[i]=s.sampler.fire_log[i][1]; zw[i]=s.sampler.fire_log[i][2]; }
    s.sampler.log_on=false; return (int)s.sampler.fire_log.size();
}

void nvm_mdam_run_verbose(void* prog, void* vm, uint64_t shi, uint64_t slo, uint64_t ihi, uint64_t ilo, uint8_t* out_record){
    auto& p=*reinterpret_cast<MdamProgram*>(prog); auto& s=*reinterpret_cast<MdamShot*>(vm);
    s.reset_shot(p, shi,slo,ihi,ilo); s.verbose=true; s.run(p); s.verbose=false;
    std::memcpy(out_record, s.record.bits.data(), (size_t)p.num_measurements);
}
}  // extern "C"
