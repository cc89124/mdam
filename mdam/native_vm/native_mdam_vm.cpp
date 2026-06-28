// native_mdam_vm.cpp — ctypes C API for the full cultivation_d3 native one-shot (native_mdam_shot).
#include <cstring>
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
void nvm_orc_cyc_reset(){ uint64_t* c=mdam::orc_cyc(); for(int i=0;i<8;i++) c[i]=0; }
void nvm_orc_cyc_get(uint64_t* out){ uint64_t* c=mdam::orc_cyc(); for(int i=0;i<8;i++) out[i]=c[i]; }
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
    int nmagic=0; for(uint8_t k:p.kind) if(k==MO_SWAP_MEAS_INTERFERE) nmagic++;
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
    p->noise_sites.resize(nsites);
    { int off=0; for(int s=0;s<nsites;s++){ for(int j=0;j<site_nchan[s];j++){ NoiseChannel c; c.prob=ch_prob[off]; c.x_words={ch_x[off]}; c.z_words={ch_z[off]}; p->noise_sites[s].channels.push_back(c); off++; } } }
    p->cp_masks.resize(ncp);
    for(int s=0;s<ncp;s++){ NoiseChannel c; c.prob=1.0; c.x_words={cp_x[s]}; c.z_words={cp_z[s]};
        p->cp_masks[s].channels.push_back(c); }
    p->num_qubits=num_qubits; p->num_measurements=num_meas; p->engine_n=engine_n; p->max_work=max_work; p->record_cap=record_cap;
    return p;
}
void nvm_mdam_free(void* prog){ delete reinterpret_cast<MdamProgram*>(prog); }

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
