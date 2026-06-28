// native_mdam_shot.hpp — C2-C4: the FULL cultivation_d3 noisy magic-core one-shot in C++.
// Integrates every verified component into ONE opcode loop (0 Python callbacks per shot):
//   native RNG + backend PauliFrame + slot2id + active-gate engine ops + native ClifftNoiseSampler
//   + dormant measurements + conditional Pauli + readout noise + magic measurement (compiled plan/
//   execute + oracle fallback) + measurement wrappers + record.
#pragma once
#include <vector>
#include <array>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <unordered_map>
#include "native_rng.hpp"
#include "native_seed_expand.hpp"
#include "native_frame.hpp"
#include "native_record.hpp"
#include "native_noise.hpp"
#include "native_magic_state.hpp"
#include "native_magic_measure.hpp"
#include "native_oracle_measure.hpp"

namespace mdam {

static const double NV_T_ANGLE = 0.78539816339744830961;   // pi/4

// Gate D batch stats (optional out-param; accumulated across the whole batch).
struct NativeBatchStats {
    uint64_t total_draws=0, total_compiled=0, total_oracle=0;
    int64_t  first_error_shot=-1;
    // master PCG64 continuation state after the batch (for §4.4 batch-splitting equivalence)
    uint64_t m_state_hi=0, m_state_lo=0, m_inc_hi=0, m_inc_lo=0;
};

enum MdamOp : uint8_t {
    MO_FRAME_H=0, MO_FRAME_CNOT, MO_FRAME_CZ, MO_FRAME_SWAP, MO_FRAME_S,
    MO_APPLY_PAULI, MO_NOISE, MO_NOISE_BLOCK, MO_READOUT_NOISE,
    MO_MEAS_DORM_STATIC, MO_MEAS_DORM_RANDOM,
    MO_ARRAY_CNOT, MO_ARRAY_CZ, MO_MULTI_CNOT, MO_MULTI_CZ,
    MO_ARRAY_T, MO_ARRAY_T_DAG, MO_ARRAY_S, MO_EXPAND_T, MO_EXPAND_T_DAG,
    MO_SWAP_MEAS_INTERFERE, MO_END
};

// immutable compiled program (POD), built once in Python
struct MdamProgram {
    std::vector<uint8_t> kind; std::vector<int32_t> a1, a2, i0, i1; std::vector<double> dval;
    std::vector<uint64_t> mmask;                 // multi-cnot/cz target masks (indexed by i0 for those ops)
    std::vector<double> hazards;                 // noise hazards (static)
    std::vector<NoiseSite> noise_sites;          // per-site channels
    std::vector<NoiseSite> cp_masks;             // conditional-Pauli masks (as 1-channel sites, prob ignored)
    int32_t num_qubits=0, num_measurements=0, engine_n=0, max_work=0;
    int32_t record_cap=0;        // record buffer capacity: covers ALL classical indices incl. feedback
                                 // conditions (which can EXCEED num_measurements; Python's record is a dict)
};

// Gate F-B: per-measurement-boundary structural region snapshot.  The audit proves the tableau /
// inverse-frame / pending SUPPORT MASKS are fully shot-static (the only structural variant, M, lives
// entirely inside the magic measurement and is handled by F4).  So the non-magic active-gate forward
// evolution between two boundaries is captured ONCE as: static masks + a per-row phase delta
// (region_const; forward conjugation adds mask-determined constants WITHOUT row-mixing) + the static
// live-pending set (uid/mask/phase).  rotation THETA is dynamic (= frame.xb) and captured at runtime.
struct RegionSnap {
    bool valid=false;
    std::vector<PackedPauli> Xc, Zc, ax, az;   // tableau + inverse masks (phase field unused, =0)
    std::vector<uint8_t> rconst;               // 4n region phase delta (Xc|Zc|ax|az), mod 4
    std::vector<uint32_t> puid;                // live pending uids at this boundary (static)
    std::vector<PackedPauli> pp;               // their masks + static Pauli phase (theta filled at runtime)
};

struct MdamShot {
    NativeRng rng; NativeFrame frame; NativeRecordBuffer record;
    NativeNoiseSampler sampler; NativeDenseEngineState engine;
    MagicScratch magic_scratch;          // §4: persistent magic-plan scratch (0 per-shot heap)
    // Gate F-B region compiler.  fb_mode: 0=OFF, 1=COMPILE, 2=SHADOW, 3=FAST.  COMPILE is auto-run on
    // the first shot to populate fb_snap; SHADOW runs the full replay AND verifies snapshot+transition
    // at every boundary; FAST skips the active-gate engine work and loads the snapshot.
    enum { FB_OFF=0, FB_COMPILE=1, FB_SHADOW=2, FB_FAST=3 };
    int fb_mode = FB_OFF; bool fb_compiled=false;
    int f5_mode = 0;   // Gate F5: 0=off; 1=inverse-only commit folds + skip discarded pending consume (FAST only)
    // Gate G: capture each compiled core's kernel descriptor + phi_in for a standalone microbench.
    struct CoreCap { int r_in,r_mat,nrot,nlm,m_bit; double sign;
        std::vector<uint64_t> rx,rz; std::vector<int> rpp,lt,la,lb; std::vector<double> rc,rs; std::vector<cd> phi_in; };
    std::vector<CoreCap> core_caps; bool core_capture=false;
    std::vector<RegionSnap> fb_snap;
    std::vector<double> fb_theta; uint32_t fb_rot_uid=0;   // runtime rotation theta capture (uid-indexed)
    std::vector<uint8_t> fb_phase_prev;                    // post-commit phase of previous boundary
    int fb_region=0;
    long fb_mismatch=0, fb_hits=0, fb_misses=0;            // shadow mismatch / fast hits / misses
    // first shadow mismatch detail (§12 BLOCKED report)
    int fb_bad_boundary=-1, fb_bad_idx=-1; const char* fb_bad_field=nullptr;
    // §5 (E-C): per-measurement-point cache of the (proven shot-static) dynamic-core uid list,
    // indexed by magic-point (0-based order within a shot).  Filled on the first shot, reused after.
    std::vector<std::vector<uint32_t>> core_cache; bool core_cache_on=true;
    // Gate F (F4): per-magic-point StaticPlan skeleton cache, M-keyed.  plan_cache[mp] is a SMALL list
    // of StaticPlan (one per observed st.M variant); st.M fully determines the skeleton (gate_f_audit).
    std::vector<std::vector<StaticPlan>> plan_cache; bool plan_cache_on=true;
    // Gate J Phase-2G (BoundaryPlan): per-(magic-point, M-variant) memoized dispatch.  Resolved O(1) by
    // the packed M key (the same mpack as the Imem key), so the hot path does NO std::vector<int> Mkey
    // heap copy, NO plan_compiled/commit_find linear scan over M_key vectors.  The Imem ENTRY is NOT
    // cached here (it depends on the per-shot phase_pack) — only the static plan/commit indices are.
    struct BoundaryVariant { uint64_t mpack; bool built=false; bool compiled=false; int plan_idx=-1, commit_idx=-1; };
    std::vector<std::vector<BoundaryVariant>> boundary_cache;   // [mag][variant], scanned by mpack (<=4)
    std::vector<int> slot2id; int next_q=0;
    uint64_t rng_draws=0; int magic_compiled=0, magic_oracle=0; const char* err=nullptr; bool verbose=false;
    bool magic_log_on=false;
    struct MagicLog { int q, rin, rmat, nrot, nlm, feasible, outcome; double p0; };
    std::vector<MagicLog> magic_log;
    // Gate I-D affine-feasibility capture (default OFF, zero cost when off).  Per magic measurement,
    // appends [mp, Mpack, signbit(-1=oracle), outcome, nr, rpp0..rpp{nr-1}] so Python can test whether
    // rpp (the live-inverse pullback phase) is an affine function of the prior-outcome history.
    bool icap_on=false; std::vector<long> icap;
    // Decision-graph feasibility capture (default OFF, zero cost when off).  Per magic measurement appends
    // [mp, keyhash, statehash, flag] where statehash = FNV(entry resident state) and keyhash = FNV(state,
    // rpp, rc, rs, sign) for compiled (flag 0) / = statehash for the oracle (flag 1; entry-state only).  Lets
    // a harness count, per measurement-point mp, how many DISTINCT dense-kernel inputs occur + the repeat
    // rate = the decision-graph / memoization-cache key cardinality.  Bit-exact FNV (pessimistic: -0/+0
    // distinct, so it OVER-counts distinct states -> a safe upper bound, conservative against feasibility).
    bool dsig_on=false; std::vector<uint64_t> dsig; long dsig_over=0;
    static inline uint64_t dfnv(uint64_t h,const void* p,size_t n){
        const unsigned char* b=(const unsigned char*)p; for(size_t i=0;i<n;i++){ h^=b[i]; h*=1099511628211ULL; } return h; }
    // Gate K Step-2: boundary-edge SHADOW cache (cmode 4).  Verifies a cached edge reproduces the LIVE 2G
    // boundary bit-exact WITHOUT skipping live.  key = FNV(mag, M_in, resident_in, rpp, sign, thetas); per
    // (key,outcome): p0 + survivor dense bytes + M_out + phase_pack_out.  Raw inputs stored for FNV-collision
    // defense (compare originals on a key match).  Counters split compiled vs oracle (oracle key is the
    // optimistic (state,M) -> a mismatch there is the "oracle key insufficient" diagnosis, not a bug).
    struct KEdge { int sid_in=-1; std::vector<int> M_in, rpp_in; double sign_in=0; std::vector<double> th_in;
        std::vector<uint8_t> pp_in;        // Step-4B-2: key+collision on the carried PRE-fwd_map phase_pack (was bnd=fwd_map(pp)); edge encodes fwd_map∘boundary so the hit path needs no fwd_map
        double p0=-2.0; bool oracle=false; bool antis=false; bool has[2]={false,false};   // Step-4B-3: antis = oracle took the stabilizer ag_measure branch (idraw2+out, NOT Born) -> NOT fast-eligible, keep live
        std::vector<cd> surv[2]; std::vector<int> Mout[2]; std::vector<uint8_t> ppout[2];
        std::vector<uint8_t> tphase[2];    // carried tableau phase (2n; the ONLY carried tableau quantity — masks are static-reloaded; inverse_frame NOT carried — oracle rebuilds it from bnd)
        int rout[2]={-1,-1}; int next_sid[2]={-1,-1}; };   // survivor rank + Step-4B-1 output state id per outcome
    std::vector<std::unordered_map<uint64_t,KEdge>> kcache;   // [mag] -> key -> edge (persists across shots)
    long k_lookup=0,k_hit=0,k_miss=0,k_mismatch=0,k_collision=0;
    long k_lookup_o=0,k_hit_o=0,k_miss_o=0,k_mismatch_o=0;    // oracle split
    long k_full_hit=0,k_partial=0,k_miss5=0,k_fwdmap=0;       // Gate K Step-4A FAST (cmode 5): full_hit / partial(drawn branch absent) / miss / fwd_map count
    double magic_last_p0=0.0;                                  // p0 of the last oracle measure_z (for the edge cache)
    double kfast_inj_rv=0.0; bool kfast_use_inj=false;        // Step-4A: pre-drawn Born rv injected into the live compiled path (avoids double-draw when the drawn branch isn't cached)
    // Gate K Step-4B-1: state interning — each distinct resident state gets an integer id, so the edge key
    // uses cur_sid (4 bytes) instead of FNV-hashing the 256-byte resident EVERY boundary.  Collision-safe
    // (per-fingerprint candidate chain + amplitude compare).  Interning hashes the survivor ONLY when a new
    // state is produced (miss/oracle ~1/shot); hits carry next_sid (no hash).  Persists across shots.
    std::unordered_map<uint64_t,std::vector<int>> state_intern;   // fingerprint -> candidate ids (collision chain)
    std::vector<std::vector<cd>> state_amp;                       // id -> amplitudes
    int cur_sid=0;                                                // carried id of the CURRENT resident state (AUTHORITATIVE identity)
    int intern_state(const cd* a, int rank){
        size_t N=(size_t)1<<rank; uint64_t fp=dfnv(1469598103934665603ULL, a, sizeof(cd)*N);
        auto& cand=state_intern[fp];
        for(int id : cand){ if(state_amp[id].size()==N){ bool eq=true;
            for(size_t j=0;j<N;j++) if(state_amp[id][j]!=a[j]){ eq=false; break; } if(eq) return id; } }
        int id=(int)state_amp.size(); state_amp.emplace_back(a, a+N); cand.push_back(id); return id; }
    // Step-4B-4 lazy survivor carry: cur_sid is the authoritative identity; the dense engine's resident bytes
    // are an OPTIONAL payload.  A FAST hit carries only cur_sid (no set_state copy), so engine.dense goes
    // STALE; dense_sid tracks which sid engine.dense currently holds.  A live boundary (miss/oracle/antis)
    // re-materializes engine.dense from state_amp[cur_sid] ONLY when it's stale.
    int dense_sid=-1;            // sid that engine.dense currently holds (-1 = unknown/stale)
    long k_materialize=0;        // materialize_on_miss count (proves resident_materialize_on_hit = 0)
    long k_antis_live=0;         // anti_s oracle boundaries kept live (stabilizer ag_measure, not Born)
    inline void materialize_dense(int sid){      // write state_amp[sid] back into the live dense engine
        const std::vector<cd>& a=state_amp[sid]; int rk=(int)__builtin_ctzll(a.size());   // size = 2^rank (always a power of 2)
        engine.dense.set_state(rk, a.data()); }
    // Gate I (Imem): compiled-control memo replacing the live-inverse pullback in the magic plan.
    // Gate I-D PROVED rpp (per-rotation pullback phase) + measurement sign are a DETERMINISTIC (Z4-affine)
    // function of (mp, Mpack, keepbit-history).  So we cache (sign, rpp-vector) keyed on that and skip the
    // live pullback on hits (rc/rs from theta + masks from the F4 skeleton are still computed).  Exact by
    // construction (stores the live result); SHADOW mode verifies memo==live before FAST is enabled.
    struct ImemEntry { bool valid=false; double sign=1.0; std::vector<int> rpp; };
    std::unordered_map<uint64_t, ImemEntry> imem;
    std::vector<int> imem_hist;                 // keepbits of prior measurements this shot
    int imem_mode=0;                            // 0=off, 1=shadow(live+verify), 2=fast(inject on hit)
    long imem_hits=0, imem_misses=0, imem_mismatch=0;
    int dump_before_magic=-1; int magic_seen=0;     // dump engine state before this magic index
    int magic_point=0;                               // §5: per-SHOT magic-measurement index (cache key)
    std::vector<uint64_t> dXcx,dXcz,dZcx,dZcz,dAx_x,dAx_z,dAz_x,dAz_z; std::vector<uint8_t> dXcp,dZcp,dAxp,dAzp;
    std::vector<int> dM; std::vector<uint64_t> dPx,dPz; std::vector<uint8_t> dPp; std::vector<double> dPth; bool dumped=false;
    void dump_engine(){ dM=engine.M; dumped=true;
        auto P=[&](std::vector<uint64_t>&xs,std::vector<uint64_t>&zs,std::vector<uint8_t>&ps,std::vector<PackedPauli>&V){
            xs.clear();zs.clear();ps.clear(); for(auto&p:V){ for(int w=0;w<engine.W;w++){xs.push_back(p.x[w]);zs.push_back(p.z[w]);} ps.push_back(p.phase);} };
        P(dXcx,dXcz,dXcp,engine.tableau.Xc); P(dZcx,dZcz,dZcp,engine.tableau.Zc);
        P(dAx_x,dAx_z,dAxp,engine.inverse_frame.ax); P(dAz_x,dAz_z,dAzp,engine.inverse_frame.az);
        dPx.clear();dPz.clear();dPp.clear();dPth.clear();
        for(auto&e:engine.pending.slots) if(e.generation==engine.pending.gen){ for(int w=0;w<engine.W;w++){dPx.push_back(e.p.x[w]);dPz.push_back(e.p.z[w]);} dPp.push_back(e.p.phase); dPth.push_back(e.theta);} }

    // §5 allocation profile: fingerprint of the persistent (exponential/O(n)) buffers' data
    // pointers + capacities.  Unchanged across shots <=> 0 per-shot reallocation of those buffers.
    uint64_t buf_fingerprint() const {
        uint64_t h = 1469598103934665603ULL;
        auto mix = [&](const void* ptr, size_t cap) {
            h = (h ^ (uint64_t)(uintptr_t)ptr) * 1099511628211ULL;
            h = (h ^ (uint64_t)cap) * 1099511628211ULL;
        };
        mix(engine.dense.resident.data(), engine.dense.resident.capacity());
        mix(engine.dense.joint.data(),    engine.dense.joint.capacity());
        mix(engine.dense.survivor.data(), engine.dense.survivor.capacity());
        mix(engine.tableau.Xc.data(),     engine.tableau.Xc.capacity());
        mix(engine.tableau.Zc.data(),     engine.tableau.Zc.capacity());
        mix(engine.inverse_frame.ax.data(), engine.inverse_frame.ax.capacity());
        mix(engine.inverse_frame.az.data(), engine.inverse_frame.az.capacity());
        mix(record.bits.data(),  record.bits.capacity());
        mix(slot2id.data(),      slot2id.capacity());
        mix(frame.x.data(),      frame.x.capacity());
        mix(frame.z.data(),      frame.z.capacity());
        return h;
    }
    // EXPONENTIAL (2^max_work) buffers only — §11.10 is specifically about these.  SWAP-INVARIANT:
    // execute_core does std::swap(resident,survivor) (zero-copy double-buffer commit), so the role
    // assignment permutes WITHOUT reallocation.  We hash the SORTED (ptr,cap) multiset so only an
    // actual reallocation (a pointer/capacity entering/leaving the set) changes the fingerprint.
    uint64_t buf_fp_dense() const {
        uint64_t a = (uint64_t)(uintptr_t)engine.dense.resident.data();
        uint64_t b = (uint64_t)(uintptr_t)engine.dense.joint.data();
        uint64_t c = (uint64_t)(uintptr_t)engine.dense.survivor.data();
        uint64_t ca = engine.dense.resident.capacity(), cb = engine.dense.joint.capacity(), cc = engine.dense.survivor.capacity();
        // sort the three (ptr) with their caps moved alongside (only 3 elements)
        auto sw=[&](uint64_t&x,uint64_t&y,uint64_t&px,uint64_t&py){ if(x>y){ std::swap(x,y); std::swap(px,py);} };
        sw(a,b,ca,cb); sw(b,c,cb,cc); sw(a,b,ca,cb);
        uint64_t h = 1469598103934665603ULL;
        for (uint64_t v : {a, ca, b, cb, c, cc}) h = (h ^ v) * 1099511628211ULL;
        return h;
    }
    // O(n) buffers (tableau/inverse/record/slot2id/frame) — may grow if a shot births > engine_n qubits.
    uint64_t buf_fp_small() const {
        uint64_t h = 1469598103934665603ULL;
        auto mix = [&](const void* ptr, size_t cap) {
            h = (h ^ (uint64_t)(uintptr_t)ptr) * 1099511628211ULL; h = (h ^ (uint64_t)cap) * 1099511628211ULL; };
        mix(engine.tableau.Xc.data(), engine.tableau.Xc.capacity());
        mix(engine.tableau.Zc.data(), engine.tableau.Zc.capacity());
        mix(engine.inverse_frame.ax.data(), engine.inverse_frame.ax.capacity());
        mix(engine.inverse_frame.az.data(), engine.inverse_frame.az.capacity());
        mix(record.bits.data(), record.bits.capacity());
        mix(slot2id.data(), slot2id.capacity());
        mix(frame.x.data(), frame.x.capacity());
        mix(frame.z.data(), frame.z.capacity());
        return h;
    }

    void init(const MdamProgram& p) {
        frame = NativeFrame((size_t)std::max(p.num_qubits, 256));
        record.init(p.record_cap > p.num_measurements ? p.record_cap : p.num_measurements);
        engine.init(p.engine_n, p.max_work, p.num_measurements);
        slot2id.assign(p.num_qubits, -1);
        // §4: size the magic scratch ONCE.  Live-pending upper bound = # rotation-creating ops
        // (ARRAY_T/T_DAG + EXPAND_T/T_DAG); a rotation is consumed only by a magic measurement, so the
        // number live at any measurement <= total rotations created.  +engine_n margin for axes.
        int nrot=0; for (uint8_t k : p.kind) if (k==MO_ARRAY_T||k==MO_ARRAY_T_DAG||k==MO_EXPAND_T||k==MO_EXPAND_T_DAG) nrot++;
        magic_scratch.reserve_for(p.engine_n, nrot + p.engine_n + 8);
        // §5: pre-size the core cache (1 magic measure_z per SWAP_MEAS op) so shot 0 does no resize.
        int nmagic=0; for (uint8_t k : p.kind) if (k==MO_SWAP_MEAS_INTERFERE) nmagic++;
        core_cache.resize(nmagic + 4); for (auto& v : core_cache) v.reserve(p.engine_n + 4);
        plan_cache.assign(nmagic + 4, {});             // F4: per-magic-point M-keyed skeleton variant list
        for (auto& v : plan_cache) v.reserve(8);       // <=4 variants/boundary observed; reserve avoids realloc
    }
    void reset_shot(const MdamProgram& p, uint64_t shi,uint64_t slo,uint64_t ihi,uint64_t ilo){
        rng.seed_from_state(shi,slo,ihi,ilo);
        std::fill(frame.x.begin(),frame.x.end(),0); std::fill(frame.z.begin(),frame.z.end(),0);
        record.reset();
        if((int)record.cap < (p.record_cap>p.num_measurements?p.record_cap:p.num_measurements)) record.init(p.record_cap>p.num_measurements?p.record_cap:p.num_measurements);
        engine.reset_state();   // in-place reset; dense/tableau/inverse buffers reused (no per-shot realloc)
        std::fill(slot2id.begin(), slot2id.end(), -1); next_q=0;
        rng_draws=0; magic_compiled=magic_oracle=0; err=nullptr;
        magic_point=0;                   // §5: per-shot magic-point counter reset (core_cache persists)
        if(icap_on) icap.clear();        // Gate I-D affine-feasibility capture (per-shot)
        imem_hist.clear();               // Gate I (Imem) per-shot keepbit history
        kfast_use_inj=false;             // Gate K Step-4A: clear any stale injected Born rv
        sampler.init(p.hazards, &rng);   // 1 init draw if hazards nonempty
    }
    inline double udraw(){ if(kfast_use_inj){ kfast_use_inj=false; return kfast_inj_rv; } rng_draws++; return rng.next_double(); }   // Step-4B-3: oracle-fast injects the early-out's pre-drawn Born rv (no double-draw, no extra stream advance)
    inline uint64_t idraw2(){ rng_draws++; return rng.bounded(2); }   // integers(0,2)
    inline int newq(int slot){ int q=next_q++; slot2id[slot]=q; return q; }

    // §2/§10 internal-breakdown profiling (PROFILE build only: -DMDAM_PROFILE.  Release = zero code).
    bool prof_on=false;
    // 0 SEED,1 RESET,2 RUN(non-magic opcode loop),3 MAGIC_PLAN,4 MAGIC_KERNEL,5 MAGIC_COMMIT,
    // 6 ORACLE,7 OUTPUT,8 OP_FRAME,9 OP_ACTIVEGATE,10 OP_ROT,11 OP_NOISE,12 OP_DORMANT,13 OP_OTHER
    double prof[17]={0};
    enum { PROF_SEED=0, PROF_RESET=1, PROF_RUN=2, PROF_MAGIC_PLAN=3, PROF_MAGIC_KERNEL=4,
           PROF_MAGIC_COMMIT=5, PROF_ORACLE=6, PROF_OUTPUT=7, PROF_OP_FRAME=8, PROF_OP_ACTIVEGATE=9,
           PROF_OP_ROT=10, PROF_OP_NOISE=11, PROF_OP_DORMANT=12, PROF_OP_OTHER=13,
           PROF_PLAN_CORE=14, PROF_PLAN_PULLBACK=15, PROF_PLAN_LOCALIZER=16 };
    static inline double now_ns() {
        return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // measure_z dispatch: compiled plan/execute (1 Born draw) else oracle (1 Born draw)
    int measure_z(int q) {
        if (magic_seen == dump_before_magic && !dumped) dump_engine();
        magic_seen++;
        int mp = magic_point++;                       // per-shot magic-point index (cache key)
        long _icap_mpack=0; uint64_t _ip=0;
        if(icap_on||imem_mode){ _icap_mpack=(long)engine.M.size(); for(size_t _k=0;_k<engine.M.size();_k++) _icap_mpack|=((long)(engine.M[_k]&15))<<(4*(_k+1));
            int _n=engine.n; if(_n<=8) for(int _i=0;_i<_n;_i++){ _ip|=((uint64_t)(engine.inverse_frame.ax[_i].phase&3))<<(4*_i);
                                                                 _ip|=((uint64_t)(engine.inverse_frame.az[_i].phase&3))<<(4*_i+2); } }
        // Gate I (Imem): compiled (mp, Mpack, inverse-frame-phase) -> (rpp,sign) lookup; inject on FAST
        // hit.  rpp = pullback(e->p).phase and sign = pullback(Z_q)+Wout are an EXACT function of the
        // inverse-frame phases (the inverse MASKS are shot-static; fold_x mutates only phase), so the
        // packed inverse-phase vector is a COMPLETE key — it encodes the full history incl. rare
        // drop/oracle fold_x events that a keepbit-history would miss.  Perfect-packed (no hash) for n<=8.
        ImemEntry* _ms=nullptr; const std::vector<int>* _inj_rpp=nullptr; const double* _inj_sign=nullptr;
        uint64_t _imem_key=0;
        if(imem_mode && engine.n<=8){
            _imem_key=(uint64_t)mp | (_ip<<4) | ((uint64_t)_icap_mpack<<(4+4*engine.n));
            auto _it=imem.find(_imem_key); if(_it!=imem.end()) _ms=&_it->second;
            if(imem_mode==2 && _ms && _ms->valid){ _inj_rpp=&_ms->rpp; _inj_sign=&_ms->sign; } }
        std::vector<uint32_t>* cslot = nullptr;
        if (core_cache_on) { if ((int)core_cache.size() <= mp) core_cache.resize(mp + 1); cslot = &core_cache[mp]; }
        StaticPlan* pslot = nullptr;
        if (plan_cache_on) {                                  // F4: M-keyed skeleton lookup (exact fallback)
            if ((int)plan_cache.size() <= mp) plan_cache.resize(mp + 1);
            auto& variants = plan_cache[mp];
            for (auto& sp : variants) if (sp.M_key == engine.M) { pslot = &sp; break; }
            if (!pslot) { variants.emplace_back(); pslot = &variants.back(); pslot->M_key = engine.M; }
        }
#ifdef MDAM_PROFILE
        double _prof_mz_t0 = prof_on ? now_ns() : 0.0;
        MagicPlan pl = magic_plan(engine, q, magic_scratch, prof_on ? &prof[PROF_PLAN_CORE] : nullptr, cslot, pslot);
#else
        ITIME_BEG(IT_PLAN); MagicPlan pl = magic_plan(engine, q, magic_scratch, nullptr, cslot, pslot, _inj_rpp, _inj_sign); ITIME_END(IT_PLAN);
#endif
#ifdef MDAM_PROFILE
        if (prof_on) { double t1=now_ns(); prof[PROF_MAGIC_PLAN]+=t1-_prof_mz_t0;
            if (pl.feasible) { double rv=udraw(); magic_compiled++;
                double k=0.0; double te0=now_ns();
                int oc; if(magic_log_on){ NativeMagicTrace tr; oc=magic_execute(engine,pl,rv,&tr,&k,nullptr);
                    magic_log.push_back({q,pl.rin,pl.rmat,(int)pl.rx().size(),(int)pl.lt().size(),1,oc,tr.p0}); }
                else oc=magic_execute(engine,pl,rv,nullptr,&k,nullptr);
                double te1=now_ns(); prof[PROF_MAGIC_KERNEL]+=k; prof[PROF_MAGIC_COMMIT]+=(te1-te0)-k;
                return oc; }
            int rb=(int)engine.M.size(); double to0=now_ns(); OracleResult R=oracle_measure_magic_counted(q);
            prof[PROF_ORACLE]+=now_ns()-to0; magic_oracle++;
            if(magic_log_on) magic_log.push_back({q,rb,-1,-1,-1,0,R.outcome,R.p0});
            if(!R.ok){err=R.err;return 0;} return R.outcome; }
#endif
        if (pl.feasible) { double rv = udraw(); magic_compiled++;
            if (core_capture) {                  // Gate G: snapshot this core's kernel inputs
                CoreCap cc; cc.r_in=engine.dense.r; cc.r_mat=pl.rmat; cc.nrot=(int)pl.rx().size();
                cc.nlm=(int)pl.lt().size(); cc.m_bit=pl.m_idx; cc.sign=pl.sign;
                cc.rx=pl.rx(); cc.rz=pl.rz(); cc.rpp=pl.rpp(); cc.rc=pl.rc(); cc.rs=pl.rs();
                cc.lt=pl.lt(); cc.la=pl.la(); cc.lb=pl.lb();
                size_t Nin=(size_t)1<<engine.dense.r; cc.phi_in.assign(engine.dense.resident.begin(), engine.dense.resident.begin()+Nin);
                core_caps.push_back(std::move(cc));
            }
            if (dsig_on) {                       // decision-graph key/state signature (entry state, pre-execute)
                size_t Nin=(size_t)1<<engine.dense.r;
                uint64_t sh=dfnv(1469598103934665603ULL, engine.dense.resident.data(), sizeof(cd)*Nin);
                uint64_t kh=sh;
                if(!engine.M.empty()) kh=dfnv(kh, engine.M.data(), sizeof(int)*engine.M.size());  // M -> StaticPlan id (rx/rz masks)
                if(!pl.rpp().empty()) kh=dfnv(kh, pl.rpp().data(), sizeof(int)*pl.rpp().size());
                if(!pl.rc().empty())  kh=dfnv(kh, pl.rc().data(),  sizeof(double)*pl.rc().size());
                if(!pl.rs().empty())  kh=dfnv(kh, pl.rs().data(),  sizeof(double)*pl.rs().size());
                double _sg=pl.sign; kh=dfnv(kh,&_sg,sizeof(double));
                dsig.push_back((uint64_t)mp); dsig.push_back(kh); dsig.push_back(sh); dsig.push_back(0ULL);
            }
            if (magic_log_on) { NativeMagicTrace tr; int oc = magic_execute(engine, pl, rv, &tr);
                magic_log.push_back({q, pl.rin, pl.rmat, (int)pl.rx().size(), (int)pl.lt().size(), 1, oc, tr.p0}); return oc; }
            { ITIME_BEG(IT_EXEC); int _oc=magic_execute(engine, pl, rv, nullptr); ITIME_END(IT_EXEC);
              if(icap_on){ icap.push_back(mp); icap.push_back(_icap_mpack); icap.push_back((long)_ip); icap.push_back(pl.sign>0?0:1); icap.push_back(_oc);
                int _nr=(int)magic_scratch.rpp.size(); icap.push_back(_nr); for(int _i=0;_i<_nr;_i++) icap.push_back(magic_scratch.rpp[_i]); }
              if(imem_mode && engine.n<=8){ imem_store_verify(_imem_key,_ms,pl.sign); }
              return _oc; }  // hot path: NO trace -> 0 allocation
        }
        // oracle draws its own Born internally -> route udraw through it
        int rin_before = (int)engine.M.size();
        uint64_t _dsh=0; if(dsig_on){ size_t Nin=(size_t)1<<engine.dense.r; _dsh=dfnv(1469598103934665603ULL, engine.dense.resident.data(), sizeof(cd)*Nin); }
        ITIME_BEG(IT_ORACLE); OracleResult R = oracle_measure_magic_counted(q); ITIME_END(IT_ORACLE);
        magic_oracle++; magic_last_p0=R.p0;     // Gate K shadow: stash oracle Born p0 for the edge-cache verify
        if(dsig_on){ uint64_t _kh=_dsh; if(!engine.M.empty()) _kh=dfnv(_kh, engine.M.data(), sizeof(int)*engine.M.size());  // oracle key = (state, M) approx (missing its core phases -> optimistic)
            dsig.push_back((uint64_t)mp); dsig.push_back(_kh); dsig.push_back(_dsh); dsig.push_back(1ULL); }
        if (magic_log_on) magic_log.push_back({q, rin_before, -1, -1, -1, 0, R.outcome, R.p0});
        if (!R.ok) { err = R.err; return 0; }
        if(icap_on){ icap.push_back(mp); icap.push_back(_icap_mpack); icap.push_back((long)_ip); icap.push_back(-1); icap.push_back(R.outcome); icap.push_back(0); }
        return R.outcome;
    }
    // Gate I (Imem): on shadow verify memo==live (else mismatch++); on fast-miss / first-shadow store live.
    inline void imem_store_verify(uint64_t key, ImemEntry* ms, double sign){
        if(imem_mode==2 && ms && ms->valid){ imem_hits++; return; }   // fast hit (already injected)
        if(ms && ms->valid){                                          // shadow: compare memo vs live
            bool ok=(ms->sign==sign) && (ms->rpp.size()==magic_scratch.rpp.size());
            if(ok) for(size_t j=0;j<ms->rpp.size();j++) if(ms->rpp[j]!=magic_scratch.rpp[j]){ ok=false; break; }
            if(!ok) imem_mismatch++;
            return;
        }
        if(imem_mode==2) imem_misses++;                              // fast miss -> store
        ImemEntry e; e.valid=true; e.sign=sign;
        e.rpp.assign(magic_scratch.rpp.begin(), magic_scratch.rpp.end());
        imem[key]=std::move(e);
    }
    // wrapper so the oracle's single rng.next_double() is counted in rng_draws
    OracleResult oracle_measure_magic_counted(int q){
        // replicate oracle_measure_magic but count the draw.  ORC_T(slot,...) = rdtsc dissection (default off).
        ORC_T(1, oracle_flush_core(engine, q, magic_scratch));
        // anti_s: non-magic qubits whose Zc anticommutes with physical Z_q -> stabilizer branch
        std::vector<int>& anti_s = magic_scratch.anti_s; anti_s.clear();
        ORC_T(2, { for (int i=0;i<engine.n;i++){ bool inM=false; for(int m:engine.M) if(m==i) inM=true;
            if(!inM && !engine.tableau.Zc_commutes_with_Zq(i,q)) anti_s.push_back(i); } });
        if (!anti_s.empty()) {
            PackedPauli Pmphys(engine.W); Pmphys.z[PackedPauli::word(q)]=PackedPauli::bit(q);  // physical Z_q
            int out = (int)idraw2();                  // _ag_measure: integers(0,2)
            engine.ag_measure(Pmphys, anti_s[0], out);
            if(!engine.reduce_full_is_noop()) return {-1,0.0,false,"reduce_full would fire"};
            return {out, 0.0, true, nullptr};
        }
        double sign; int r;
        ORC_T(3, { PackedPauli Pm(engine.W); Pm.z[PackedPauli::word(q)]=PackedPauli::bit(q);
                   PackedPauli pm = engine.pullback(Pm);
                   r = oracle_localize(engine, pm, q, sign, magic_scratch); });
        double p0; int outcome;
        if (r<0){ p0=std::max(0.0,std::min(1.0,(1.0+sign)/2.0)); outcome=(udraw()<p0)?0:1;
                  if(!engine.reduce_full_is_noop()) return {-1,0.0,false,"reduce_full would fire"};
                  return {outcome,p0,true,nullptr}; }
        int jr=-1; for(size_t i=0;i<engine.M.size();i++) if(engine.M[i]==r) jr=(int)i;
        double s0,s1,nrm2; int keepbit;
        ORC_T(4, { s0=engine.branch_sqnorm(jr,0); s1=engine.branch_sqnorm(jr,1); double tot=s0+s1;
                   p0=(tot>1e-300)?((sign>0?s0:s1)/tot):0.5; p0=std::max(0.0,std::min(1.0,p0));
                   outcome=(udraw()<p0)?0:1;
                   int plus_bit=(sign>0)?0:1; keepbit=(outcome==0)?plus_bit:(1-plus_bit);
                   size_t N=(size_t)1<<engine.dense.r;
                   for(size_t s=0;s<N;s++) if((int)((s>>jr)&1)==(1-keepbit)) engine.dense.resident[s]=cd(0,0);
                   nrm2=(keepbit==0)?s0:s1;
                   if(nrm2>1e-24){ double inv=1.0/std::sqrt(nrm2); for(size_t s=0;s<N;s++) engine.dense.resident[s]*=inv; } });
        ORC_T(5, { engine.drop_localized_core(jr,keepbit); engine.drop_residual_products(); });
        if(!engine.reduce_full_is_noop()) return {-1,0.0,false,"reduce_full would fire"};
        return {outcome,p0,true,nullptr};
    }

    void apply_mask(const NoiseSite& m){
        if(m.channels.empty()) return; const NoiseChannel& c=m.channels[0];
        for(size_t wi=0;wi<c.x_words.size();wi++){ uint64_t w=c.x_words[wi]; while(w){ int b=__builtin_ctzll(w); w&=w-1; frame.apply_x((uint32_t)(wi*64+b)); } }
        for(size_t wi=0;wi<c.z_words.size();wi++){ uint64_t w=c.z_words[wi]; while(w){ int b=__builtin_ctzll(w); w&=w-1; frame.apply_z((uint32_t)(wi*64+b)); } }
    }
    std::vector<std::array<double,4>> rot_log; bool rot_log_on=false;
    void rot(const MdamProgram& p, int slot, double angle){
        int q = (slot<(int)slot2id.size())?slot2id[slot]:-1; if(q<0) return;
        int xb = frame.xb(slot);
        double theta = xb ? -angle : angle;
        if (rot_log_on) rot_log.push_back({(double)slot,(double)xb,angle,theta});
        engine.apply_rotation(q, theta);
    }

    bool frame_log_on=false; std::vector<std::array<uint64_t,2>> frame_log;
    void run(const MdamProgram& p){
        size_t N=p.kind.size();
        for(size_t i=0;i<N && !err;i++){
            if(frame_log_on){ uint64_t fx=0,fz=0; for(int s=0;s<p.num_qubits&&s<64;s++){ if(frame.xb(s))fx|=1ULL<<s; if(frame.zb(s))fz|=1ULL<<s; } frame_log.push_back({fx,fz}); }
            int a1=p.a1[i], a2=p.a2[i], i0=p.i0[i], i1=p.i1[i]; double dv=p.dval[i];
            uint8_t _k=p.kind[i];
#ifdef MDAM_PROFILE
            double _op_t0=0.0, _op_msum0=0.0;
            if(prof_on){ _op_t0=now_ns(); _op_msum0=prof[PROF_MAGIC_PLAN]+prof[PROF_MAGIC_KERNEL]+prof[PROF_MAGIC_COMMIT]+prof[PROF_ORACLE]; }
#endif
            switch((MdamOp)_k){
                case MO_FRAME_H: frame.h(a1); break;
                case MO_FRAME_CNOT: frame.cnot(a1,a2); break;
                case MO_FRAME_CZ: frame.cz(a1,a2); break;
                case MO_FRAME_SWAP: frame.swap(a1,a2); break;
                case MO_FRAME_S: frame.s_gate(a1); break;
                case MO_APPLY_PAULI: { int rc=record.get((uint32_t)i0); if(verbose) fprintf(stderr,"  op%zu APPLY_PAULI cond=%d rec[cond]=%d mask=%d\n",i,i0,rc,i1); if(rc==1) apply_mask(p.cp_masks[i1]); } break;
                case MO_NOISE: sampler.apply_site(i0, p.noise_sites[i0], frame); break;
                case MO_NOISE_BLOCK: for(int s=i0;s<i0+i1;s++) sampler.apply_site(s, p.noise_sites[s], frame); break;
                case MO_READOUT_NOISE: if(udraw()<dv) record.flip((uint32_t)i0); break;
                case MO_MEAS_DORM_STATIC: record.set((uint32_t)i0, frame.xb(a1)^i1); break;
                case MO_MEAS_DORM_RANDOM: { int m=(int)idraw2(); record.set((uint32_t)i0, m^i1); frame.set_xz(a1,(uint8_t)m,0);
                    if(verbose) fprintf(stderr,"  op%zu DORM_RANDOM a1=%d cidx=%d m=%d sign=%d rec=%d\n",i,a1,i0,m,i1,m^i1); } break;
                case MO_ARRAY_CNOT: { int u=slot2id[a1], v=slot2id[a2]; if(u>=0&&v>=0) engine.cx(u,v); frame.cnot(a1,a2); } break;
                case MO_ARRAY_CZ: { int u=slot2id[a1], v=slot2id[a2]; if(u>=0&&v>=0) engine.cz(u,v); frame.cz(a1,a2); } break;
                case MO_MULTI_CNOT: { int tgt=a1, t=slot2id[tgt]; uint64_t mask=p.mmask[i0];
                    while(mask){ int ctrl=__builtin_ctzll(mask); mask&=mask-1; if(ctrl==tgt) continue;
                        int c=slot2id[ctrl]; if(t>=0&&c>=0) engine.cx(c,t); frame.cnot(ctrl,tgt); } } break;
                case MO_MULTI_CZ: { uint64_t mask=p.mmask[i0];
                    while(mask){ int tgt=__builtin_ctzll(mask); mask&=mask-1; if(tgt==a1) continue;
                        int u=slot2id[a1], v=slot2id[tgt]; if(u>=0&&v>=0) engine.cz(u,v); frame.cz(a1,tgt); } } break;
                case MO_ARRAY_T: rot(p,a1,NV_T_ANGLE); break;
                case MO_ARRAY_T_DAG: rot(p,a1,-NV_T_ANGLE); break;
                case MO_ARRAY_S: { int q=slot2id[a1]; if(q>=0) engine.s(q,false); frame.s_gate(a1); } break;
                case MO_EXPAND_T: { newq(a1); engine.h(slot2id[a1]); rot(p,a1,NV_T_ANGLE); } break;
                case MO_EXPAND_T_DAG: { newq(a1); engine.h(slot2id[a1]); rot(p,a1,-NV_T_ANGLE); } break;
                case MO_SWAP_MEAS_INTERFERE: {
                    // _swap_slots(a1,a2)
                    int i_1=slot2id[a1], i_2=slot2id[a2];
                    slot2id[a1]=-1; slot2id[a2]=-1;
                    if(i_1>=0) slot2id[a2]=i_1; if(i_2>=0) slot2id[a1]=i_2;
                    frame.swap(a1,a2);
                    int q=slot2id[a2]; if(q<0) break;
                    engine.h(q);
                    int b=measure_z(q);
                    slot2id[a2]=-1;
                    int m_abs = b ^ frame.zb(a2);
                    record.set((uint32_t)i0, m_abs^i1);
                    frame.set_xz(a2,(uint8_t)m_abs,0);
                } break;
                case MO_END: default: break;
            }
#ifdef MDAM_PROFILE
            if(prof_on){
                double dt=now_ns()-_op_t0;
                double mdt=(prof[PROF_MAGIC_PLAN]+prof[PROF_MAGIC_KERNEL]+prof[PROF_MAGIC_COMMIT]+prof[PROF_ORACLE])-_op_msum0;
                double cdt=dt-mdt;   // non-magic part of this op
                switch((MdamOp)_k){
                    case MO_FRAME_H: case MO_FRAME_CNOT: case MO_FRAME_CZ: case MO_FRAME_SWAP: case MO_FRAME_S:
                        prof[PROF_OP_FRAME]+=cdt; break;
                    case MO_ARRAY_CNOT: case MO_ARRAY_CZ: case MO_MULTI_CNOT: case MO_MULTI_CZ: case MO_ARRAY_S:
                        prof[PROF_OP_ACTIVEGATE]+=cdt; break;
                    case MO_ARRAY_T: case MO_ARRAY_T_DAG: case MO_EXPAND_T: case MO_EXPAND_T_DAG:
                        prof[PROF_OP_ROT]+=cdt; break;
                    case MO_NOISE: case MO_NOISE_BLOCK: prof[PROF_OP_NOISE]+=cdt; break;
                    case MO_MEAS_DORM_STATIC: case MO_MEAS_DORM_RANDOM: prof[PROF_OP_DORMANT]+=cdt; break;
                    default: prof[PROF_OP_OTHER]+=cdt; break;
                }
            }
#endif
        }
    }

    // ===== Gate F-B region compiler =====================================================
    inline void fb_capture_phase(std::vector<uint8_t>& out) const {
        int n=engine.n; out.resize(4*n);
        for(int i=0;i<n;i++){ out[i]=engine.tableau.Xc[i].phase; out[n+i]=engine.tableau.Zc[i].phase;
            out[2*n+i]=engine.inverse_frame.ax[i].phase; out[3*n+i]=engine.inverse_frame.az[i].phase; }
    }
    // record the static snapshot (masks) + region phase delta (cur - prev) at boundary b
    void fb_record_boundary(int b, const std::vector<uint8_t>& prev) {
        if((int)fb_snap.size()<=b) fb_snap.resize(b+1);
        RegionSnap& s=fb_snap[b]; int n=engine.n;
        s.Xc.resize(n); s.Zc.resize(n); s.ax.resize(n); s.az.resize(n); s.rconst.resize(4*n);
        std::vector<uint8_t> cur; fb_capture_phase(cur);
        for(int i=0;i<n;i++){
            s.Xc[i]=engine.tableau.Xc[i]; s.Xc[i].phase=0;
            s.Zc[i]=engine.tableau.Zc[i]; s.Zc[i].phase=0;
            s.ax[i]=engine.inverse_frame.ax[i]; s.ax[i].phase=0;
            s.az[i]=engine.inverse_frame.az[i]; s.az[i].phase=0;
            s.rconst[i]=(uint8_t)((cur[i]-prev[i])&3);
            s.rconst[n+i]=(uint8_t)((cur[n+i]-prev[n+i])&3);
            s.rconst[2*n+i]=(uint8_t)((cur[2*n+i]-prev[2*n+i])&3);
            s.rconst[3*n+i]=(uint8_t)((cur[3*n+i]-prev[3*n+i])&3);
        }
        s.puid.clear(); s.pp.clear();
        for(auto&e:engine.pending.slots) if(e.generation==engine.pending.gen){ s.puid.push_back(e.uid); s.pp.push_back(e.p); }
        s.valid=true;
    }
    inline void fb_flag(int b,int idx,const char* f){ if(fb_mismatch==0){fb_bad_boundary=b;fb_bad_idx=idx;fb_bad_field=f;} fb_mismatch++; }
    // SHADOW: verify snapshot masks + (prev + region_const) phase + pending against the live engine
    void fb_shadow_boundary(int b, const std::vector<uint8_t>& prev) {
        if(b>=(int)fb_snap.size()||!fb_snap[b].valid){ fb_flag(b,-1,"no-snap"); return; }
        RegionSnap& s=fb_snap[b]; int n=engine.n, W=engine.W;
        std::vector<uint8_t> cur; fb_capture_phase(cur);
        // Only the tableau (Xc/Zc) is region-snapshotted; the inverse frame is kept live (cheap,
        // row-mixing) and is not compared here.  Pending is snapshotted (masks/phase static).
        for(int i=0;i<n;i++){
            for(int w=0;w<W;w++){
                if(engine.tableau.Xc[i].x[w]!=s.Xc[i].x[w]||engine.tableau.Xc[i].z[w]!=s.Xc[i].z[w]) fb_flag(b,i,"Xc.mask");
                if(engine.tableau.Zc[i].x[w]!=s.Zc[i].x[w]||engine.tableau.Zc[i].z[w]!=s.Zc[i].z[w]) fb_flag(b,i,"Zc.mask");
            }
            if(cur[i]!=((prev[i]+s.rconst[i])&3)) fb_flag(b,i,"Xc.phase");
            if(cur[n+i]!=((prev[n+i]+s.rconst[n+i])&3)) fb_flag(b,i,"Zc.phase");
        }
        // pending: live engine set vs snapshot (uid/mask/phase) + theta
        std::vector<PendingEntry*> live;
        for(auto&e:engine.pending.slots) if(e.generation==engine.pending.gen) live.push_back(&e);
        if(live.size()!=s.puid.size()){ fb_flag(b,(int)live.size(),"pend.count"); }
        else for(size_t j=0;j<live.size();j++){
            if(live[j]->uid!=s.puid[j]) fb_flag(b,(int)j,"pend.uid");
            for(int w=0;w<W;w++) if(live[j]->p.x[w]!=s.pp[j].x[w]||live[j]->p.z[w]!=s.pp[j].z[w]) fb_flag(b,(int)j,"pend.mask");
            if(live[j]->p.phase!=s.pp[j].phase) fb_flag(b,(int)j,"pend.phase");
            uint32_t uid=s.puid[j]; double th=(uid<fb_theta.size())?fb_theta[uid]:0.0;
            if(live[j]->theta!=th) fb_flag(b,(int)j,"pend.theta");
        }
    }
    // FAST: load snapshot masks, add region_const to the carried phase, rebuild live pending
    void fb_load_boundary(int b) {
        if(b>=(int)fb_snap.size()||!fb_snap[b].valid){ fb_misses++; return; }
        fb_hits++;
        RegionSnap& s=fb_snap[b]; int n=engine.n, W=engine.W;
        // tableau only: load static masks + add the region phase delta to the carried phase.  The
        // inverse frame is left untouched (it is maintained live via cx_inv/cz_inv/s_inv/h_inv).
        for(int i=0;i<n;i++){
            for(int w=0;w<W;w++){
                engine.tableau.Xc[i].x[w]=s.Xc[i].x[w]; engine.tableau.Xc[i].z[w]=s.Xc[i].z[w];
                engine.tableau.Zc[i].x[w]=s.Zc[i].x[w]; engine.tableau.Zc[i].z[w]=s.Zc[i].z[w];
            }
            engine.tableau.Xc[i].phase=(uint8_t)((engine.tableau.Xc[i].phase+s.rconst[i])&3);
            engine.tableau.Zc[i].phase=(uint8_t)((engine.tableau.Zc[i].phase+s.rconst[n+i])&3);
        }
        engine.pending.gen++;   // invalidate all prior live entries; rebuild exactly the snapshot set
        for(size_t j=0;j<s.puid.size();j++){
            uint32_t uid=s.puid[j]; double th=(uid<fb_theta.size())?fb_theta[uid]:0.0;
            engine.pending.create(uid, s.pp[j], th);
        }
    }

    // Gate F-B shot loop.  COMPILE (first shot) = full replay + record snapshots.  SHADOW = full
    // replay + verify each boundary.  FAST = skip active-gate engine work, load snapshots at boundaries.
    void run_fb(const MdamProgram& p) {
        bool compiling = !fb_compiled;
        bool fast   = (!compiling && fb_mode==FB_FAST);
        bool shadow = (!compiling && fb_mode==FB_SHADOW);
        engine.fb_commit_mode = (fast && f5_mode) ? 1 : 0;   // F5: inverse-only commit folds in FAST
        fb_region=0; fb_rot_uid=0;
        fb_phase_prev.assign(4*engine.n, 0);
        auto cap_theta=[&](int slot,double angle)->int{      // returns q (or -1); records theta @ fb_rot_uid
            int q=(slot<(int)slot2id.size())?slot2id[slot]:-1; if(q<0) return -1;
            int xb=frame.xb(slot); double theta=xb?-angle:angle;
            if((int)fb_theta.size()<=(int)fb_rot_uid) fb_theta.resize(fb_rot_uid+1);
            fb_theta[fb_rot_uid]=theta;
            if(!fast) engine.apply_rotation(q,theta);          // creates pending uid==fb_rot_uid
            // FAST: skip pending create (rebuilt from snapshot at boundary); inverse frame is unaffected
            // by apply_rotation (it only adds a pending Z_q generator), so nothing inverse to do here.
            fb_rot_uid++; return q; };
        size_t N=p.kind.size();
        ITIME_BEG(IT_SHOT);
        for(size_t i=0;i<N && !err;i++){
            int a1=p.a1[i], a2=p.a2[i], i0=p.i0[i], i1=p.i1[i]; double dv=p.dval[i];
            switch((MdamOp)p.kind[i]){
                case MO_FRAME_H: ISKIP(ISK_FRAME, frame.h(a1)); break;
                case MO_FRAME_CNOT: ISKIP(ISK_FRAME, frame.cnot(a1,a2)); break;
                case MO_FRAME_CZ: ISKIP(ISK_FRAME, frame.cz(a1,a2)); break;
                case MO_FRAME_SWAP: ISKIP(ISK_FRAME, frame.swap(a1,a2)); break;
                case MO_FRAME_S: ISKIP(ISK_FRAME, frame.s_gate(a1)); break;
                case MO_APPLY_PAULI: { int rc=record.get((uint32_t)i0); if(rc==1) apply_mask(p.cp_masks[i1]); } break;
                case MO_NOISE: { ITIME_BEG(IT_NOISE); sampler.apply_site(i0, p.noise_sites[i0], frame); ITIME_END(IT_NOISE); } break;
                case MO_NOISE_BLOCK: { ITIME_BEG(IT_NOISE); for(int s=i0;s<i0+i1;s++) sampler.apply_site(s, p.noise_sites[s], frame); ITIME_END(IT_NOISE); } break;
                case MO_READOUT_NOISE: if(udraw()<dv) record.flip((uint32_t)i0); break;
                case MO_MEAS_DORM_STATIC: record.set((uint32_t)i0, frame.xb(a1)^i1); break;
                case MO_MEAS_DORM_RANDOM: { int m=(int)idraw2(); record.set((uint32_t)i0, m^i1); ISKIP(ISK_FRAME, frame.set_xz(a1,(uint8_t)m,0)); } break;
                case MO_ARRAY_CNOT: { int u=slot2id[a1],v=slot2id[a2]; if(u>=0&&v>=0){ if(fast){ ISKIP(ISK_INV_FWD, engine.cx_inv(u,v)); } else engine.cx(u,v);} ISKIP(ISK_FRAME, frame.cnot(a1,a2)); } break;
                case MO_ARRAY_CZ: { int u=slot2id[a1],v=slot2id[a2]; if(u>=0&&v>=0){ if(fast){ ISKIP(ISK_INV_FWD, engine.cz_inv(u,v)); } else engine.cz(u,v);} ISKIP(ISK_FRAME, frame.cz(a1,a2)); } break;
                case MO_MULTI_CNOT: { int tgt=a1, t=slot2id[tgt]; uint64_t mask=p.mmask[i0];
                    while(mask){ int ctrl=__builtin_ctzll(mask); mask&=mask-1; if(ctrl==tgt) continue;
                        int c=slot2id[ctrl]; if(t>=0&&c>=0){ if(fast){ ISKIP(ISK_INV_FWD, engine.cx_inv(c,t)); } else engine.cx(c,t);} ISKIP(ISK_FRAME, frame.cnot(ctrl,tgt)); } } break;
                case MO_MULTI_CZ: { uint64_t mask=p.mmask[i0];
                    while(mask){ int tgt=__builtin_ctzll(mask); mask&=mask-1; if(tgt==a1) continue;
                        int u=slot2id[a1],v=slot2id[tgt]; if(u>=0&&v>=0){ if(fast){ ISKIP(ISK_INV_FWD, engine.cz_inv(u,v)); } else engine.cz(u,v);} ISKIP(ISK_FRAME, frame.cz(a1,tgt)); } } break;
                case MO_ARRAY_T: cap_theta(a1,NV_T_ANGLE); break;
                case MO_ARRAY_T_DAG: cap_theta(a1,-NV_T_ANGLE); break;
                case MO_ARRAY_S: { int q=slot2id[a1]; if(q>=0){ if(fast){ ISKIP(ISK_INV_FWD, engine.s_inv(q,false)); } else engine.s(q,false);} ISKIP(ISK_FRAME, frame.s_gate(a1)); } break;
                case MO_EXPAND_T: { newq(a1); int q2=slot2id[a1]; if(fast){ ISKIP(ISK_INV_FWD, engine.h_inv(q2)); } else engine.h(q2); cap_theta(a1,NV_T_ANGLE); } break;
                case MO_EXPAND_T_DAG: { newq(a1); int q2=slot2id[a1]; if(fast){ ISKIP(ISK_INV_FWD, engine.h_inv(q2)); } else engine.h(q2); cap_theta(a1,-NV_T_ANGLE); } break;
                case MO_SWAP_MEAS_INTERFERE: {
                    int i_1=slot2id[a1], i_2=slot2id[a2];
                    slot2id[a1]=-1; slot2id[a2]=-1;
                    if(i_1>=0) slot2id[a2]=i_1; if(i_2>=0) slot2id[a1]=i_2;
                    ISKIP(ISK_FRAME, frame.swap(a1,a2));
                    int q=slot2id[a2]; if(q<0) break;
                    if(fast){ ISKIP(ISK_INV_FWD, engine.h_inv(q)); } else engine.h(q);   // tableau via snapshot; inverse kept live
                    int b=fb_region;
                    if(compiling) fb_record_boundary(b, fb_phase_prev);
                    else if(shadow) fb_shadow_boundary(b, fb_phase_prev);
                    else if(fast) { ITIME_BEG(IT_REGION); fb_load_boundary(b); ITIME_END(IT_REGION); }
                    int bit=measure_z(q);
                    if(!fast) fb_capture_phase(fb_phase_prev);   // post-commit phase for next region
                    fb_region++;
                    slot2id[a2]=-1;
                    int m_abs = bit ^ frame.zb(a2);
                    record.set((uint32_t)i0, m_abs^i1);
                    ISKIP(ISK_FRAME, frame.set_xz(a2,(uint8_t)m_abs,0));
                } break;
                case MO_END: default: break;
            }
        }
        ITIME_END(IT_SHOT);
        if(compiling) fb_compiled=true;
    }

    // ---- Gate D: full native batch.  One Python->C++ entry expands the master (state,inc)
    // into per-shot seeds in C++ (master.integers(0,2**63-1) Lemire-64 -> SeedSequence -> PCG64),
    // runs each shot in-place, and writes num_shots rows of num_measurements bytes into out_record.
    // 0 Python loop, 0 callbacks, 0 per-shot Python objects.  Matches backend.sample() semantics.
    int run_batch(const MdamProgram& p, uint64_t num_shots,
                  uint64_t mshi, uint64_t mslo, uint64_t mihi, uint64_t milo,
                  uint8_t* out_record, NativeBatchStats* stats, char* out_err, int errlen) {
        NativeRng master; master.seed_from_state(mshi, mslo, mihi, milo);
        const uint64_t RNG_EXCL = ((uint64_t)1 << 63) - 1;     // integers(0, 2**63-1) -> #values
        if (stats) *stats = NativeBatchStats{};
        const size_t nm = (size_t)p.num_measurements;
        for (uint64_t sh = 0; sh < num_shots; sh++) {
            uint64_t sd = master.bounded(RNG_EXCL);            // per-shot seed from master stream
            __uint128_t st, inc; SeedExpand::seedseq_pcg64(sd, st, inc);
            reset_shot(p, (uint64_t)(st >> 64), (uint64_t)st, (uint64_t)(inc >> 64), (uint64_t)inc);
            if (fb_mode != FB_OFF) run_fb(p); else run(p);
            std::memcpy(out_record + (size_t)sh * nm, record.bits.data(), nm);
            if (stats) { stats->total_draws += rng_draws; stats->total_compiled += magic_compiled;
                         stats->total_oracle += magic_oracle; }
            if (err) {
                if (stats) { stats->first_error_shot = (int64_t)sh;
                    stats->m_state_hi=(uint64_t)(master.state>>64); stats->m_state_lo=(uint64_t)master.state;
                    stats->m_inc_hi=(uint64_t)(master.inc>>64); stats->m_inc_lo=(uint64_t)master.inc; }
                if (out_err) { std::strncpy(out_err, err, errlen - 1); out_err[errlen - 1] = 0; }
                return 1;
            }
        }
        if (stats) { stats->m_state_hi=(uint64_t)(master.state>>64); stats->m_state_lo=(uint64_t)master.state;
            stats->m_inc_hi=(uint64_t)(master.inc>>64); stats->m_inc_lo=(uint64_t)master.inc; }
        if (out_err) out_err[0] = 0;
        return 0;
    }

#ifdef MDAM_PROFILE
    // §2/§10 PROFILE batch: per-phase timers.  prof[] filled: SEED,RESET,RUN(total run() wall),OUTPUT
    // outer; OP_FRAME/ACTIVEGATE/ROT/NOISE/DORMANT/OTHER + MAGIC_PLAN/KERNEL/COMMIT/ORACLE inner.
    // Inner categories sum ~= PROF_RUN (residual = per-op timer overhead / unknown).
    void run_batch_prof(const MdamProgram& p, uint64_t num_shots,
                        uint64_t mshi, uint64_t mslo, uint64_t mihi, uint64_t milo, uint8_t* out_record) {
        NativeRng master; master.seed_from_state(mshi, mslo, mihi, milo);
        const uint64_t RNG_EXCL = ((uint64_t)1 << 63) - 1;
        for (int i=0;i<17;i++) prof[i]=0;
        prof_on = true;
        const size_t nm = (size_t)p.num_measurements;
        for (uint64_t sh = 0; sh < num_shots; sh++) {
            double t = now_ns();
            uint64_t sd = master.bounded(RNG_EXCL);
            __uint128_t st, inc; SeedExpand::seedseq_pcg64(sd, st, inc);
            double t1 = now_ns(); prof[PROF_SEED] += t1 - t;
            reset_shot(p, (uint64_t)(st >> 64), (uint64_t)st, (uint64_t)(inc >> 64), (uint64_t)inc);
            double t2 = now_ns(); prof[PROF_RESET] += t2 - t1;
            run(p);
            double t3 = now_ns(); prof[PROF_RUN] += t3 - t2;
            std::memcpy(out_record, record.bits.data(), nm);
            prof[PROF_OUTPUT] += now_ns() - t3;
        }
        prof_on = false;
    }
#endif
};

} // namespace mdam
