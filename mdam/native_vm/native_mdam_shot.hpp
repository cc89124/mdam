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
#include <cstdlib>
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
    MO_SWAP_MEAS_INTERFERE,
    // Gate L1: coherent-circuit opcodes (arbitrary-theta diagonal rotation + active-register meas/swap).
    // dval[i] carries the rotation angle theta = arg(weight_re + i*weight_im) for ROT/EXPAND_ROT.
    MO_ARRAY_ROT, MO_EXPAND_ROT, MO_ARRAY_SWAP, MO_MEAS_ACTIVE_DIAGONAL, MO_MEAS_ACTIVE_INTERFERE,
    // Gate L Tier-3 (de-fused bytecode_passes=None dialect): birth |+> + active Hadamard (no U2/U4).
    MO_EXPAND, MO_ARRAY_H,
    // Gate L Tier-3 DIRECT (fused dialect): frame-keyed fused 1q/2q unitaries.  These PRESERVE the
    // measurement-core localization (maxM) — de-fusing them de-localizes (d5_r5 12->31).  i0 = cp_idx
    // (node index); the per-in-state decomposition (ZXZ angles / 2q op-list) is precomputed in Python.
    MO_ARRAY_U2, MO_ARRAY_U4,
    MO_END
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
    // Gate L Tier-3 DIRECT: precomputed frame-keyed fused-unitary decompositions (compile-time static;
    // the matrices are constants, only the frame-selected in_state is runtime).  Indexed by cp_idx (i0).
    // U2: per node, 4 in_states (= 2*zb|xb) -> ZXZ angles (b,c,d) + out frame.  apply Rz(d)Rx(c)Rz(b).
    int32_t n_u2=0;
    std::vector<double>  u2_bcd;    // [(cp_idx*4 + in_state)*3 + {0:b,1:c,2:d}]
    std::vector<uint8_t> u2_out;    // [cp_idx*4 + in_state] -> out frame (bit0=x, bit1=z)
    // U4: per node, 16 in_states (= 8*zb_hi|4*xb_hi|2*zb_lo|xb_lo) -> op-list + out frame (4 bits: lo x/z, hi x/z).
    // each op = 5 doubles (type, which, px, pz, theta); type 0=cx(lo->hi) 1=cz 2=rot1(on which: 0=lo 1=hi).
    int32_t n_u4=0;
    std::vector<int32_t> u4_start, u4_cnt;   // [cp_idx*16 + in_state] -> first op index / op count
    std::vector<double>  u4_ops;             // flattened, 5 doubles/op
    std::vector<uint8_t> u4_out;             // [cp_idx*16 + in_state] -> out frame
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

    // ===== Phase-0/1/2 boundary-edge capture (lightweight semantic-key SUFFICIENCY proof) ==============
    // Per measurement boundary on the AUTHORITATIVE path (bit-exact vs Python), capture a RICH key
    // {sid_in, inv_sig, pend_sig, m_sig, frame parity xb/zb, i1, kind, oracle/antis} and the resulting EDGE
    // {p0, outcome, sid_out, rec}.  A Python harness proves: same (mp,key) -> same p0, and same
    // (mp,key,outcome) -> same (sid_out, rec).  A mismatch == "key too small" (names the missing field).
    // Uses ONLY authoritative measure_z, so edges are correct by construction (NO F4/imem/plan/bplan cache).
    bool bcap_on=false; double bcap_p0=0.0; bool bcap_antis=false;
    struct BCapRec { int mp; uint32_t sid_in, inv_sig, pend_sig, m_sig; uint8_t xb, zb, i1, kind, oracle, outcome; uint32_t sid_out; uint8_t rec; double p0; };
    std::vector<BCapRec> bcap;
    // canonical interner (separate pool; -0.0 -> +0.0 so signed-zero-only differences don't split a state into
    // two ids, which would read as a spurious "key insufficient").  Persists across shots -> stable global ids.
    std::unordered_map<uint64_t,std::vector<int>> bcap_intern; std::vector<std::vector<cd>> bcap_amp;
    std::vector<cd> bcap_buf;
    int bcap_sid(const cd* a, int rank){
        size_t N=(size_t)1<<rank; bcap_buf.resize(N);
        for(size_t j=0;j<N;j++) bcap_buf[j]=cd(a[j].real()+0.0, a[j].imag()+0.0);   // canonicalize signed zero
        uint64_t fp=dfnv(1469598103934665603ULL, bcap_buf.data(), sizeof(cd)*N);
        auto& cand=bcap_intern[fp];
        for(int id:cand){ if(bcap_amp[id].size()==N){ bool eq=true; for(size_t j=0;j<N;j++) if(bcap_amp[id][j]!=bcap_buf[j]){eq=false;break;} if(eq) return id; } }
        int id=(int)bcap_amp.size(); bcap_amp.emplace_back(bcap_buf); cand.push_back(id); return id; }
    uint32_t bcap_inv_sig(){ uint64_t h=1469598103934665603ULL; int n=engine.n;
        for(int i=0;i<n;i++){ uint8_t a=engine.inverse_frame.ax[i].phase, z=engine.inverse_frame.az[i].phase; h=dfnv(h,&a,1); h=dfnv(h,&z,1);} return (uint32_t)(h^(h>>32)); }
    uint32_t bcap_pend_sig(){ uint64_t h=1469598103934665603ULL;
        for(auto&e:engine.pending.slots) if(e.generation==engine.pending.gen){ for(int w=0;w<engine.W;w++){ h=dfnv(h,&e.p.x[w],8); h=dfnv(h,&e.p.z[w],8);} h=dfnv(h,&e.p.phase,1); double th=e.theta; h=dfnv(h,&th,8);} return (uint32_t)(h^(h>>32)); }
    uint32_t bcap_m_sig(){ uint64_t h=1469598103934665603ULL; for(int m:engine.M){ int mm=m; h=dfnv(h,&mm,sizeof(int)); } return (uint32_t)(h^(h>>32)); }

    // ===== Phase-3: authoritative-edge cache (run_mcache) =============================================
    // Proven (bcap) design: a lightweight semantic key K=(mp,kind,sid_in,inv_sig,pend_sig,m_sig) determines
    // the boundary EDGE.  MISS -> run the AUTHORITATIVE measure_z (NO F4/imem), store the edge + snapshot the
    // post-boundary engine into a DEDUPED pool (one EngSnap per distinct post-state).  HIT -> draw Born rv,
    // pick outcome, RESTORE the pool snapshot, set record via XOR rule -> measure_z SKIPPED.  Eager inverse so
    // the inverse frame is always materialized (snapshot-able).  anti_s boundaries (idraw2 coin) stay LIVE.
    struct EngSnap { int r=0; std::vector<cd> dense; std::vector<int> M;
        std::vector<PackedPauli> ax, az, Xc, Zc; std::vector<PendingEntry> pend; uint32_t rot_uid=0; };
    std::vector<EngSnap> mc_pool;
    size_t mc_pool_bytes_live=0;      // running sum of pooled snapshot bytes -> O(1) memory-budget estimate
    std::unordered_map<uint64_t,std::vector<int>> mc_pool_idx;          // exact-dedup: fingerprint -> pool ids (collision chain)
    struct MEdge { double p0=-2.0; uint8_t antis=0; bool has[2]={false,false}; int pool[2]={-1,-1}; uint8_t disp[2]={0,0};
                   int sid_out[2]={-1,-1}; };   // Phase-4 carry: dense-block id per outcome (so the next key needs NO dense re-hash)
    std::vector<std::unordered_map<uint64_t,MEdge>> mc_edges;            // [mp] -> key -> edge (persists across shots)
    long mc_hit=0, mc_miss=0, mc_partial=0, mc_antis=0, mc_verify=0, mc_mismatch=0, mc_restore=0;
    int mc_mode=0;          // 0 off, 1 SHADOW (build+verify, no skip), 2 FAST-snapshot, 3 FAST-carry (carried sid, no dense re-hash)
    int mc_cur_sid=-1;      // Phase-4: carried dense-block id (sid_in(B+1)==sid_out(B); dense only changes at a measure)
    // Step 3-1 hit-cost decomposition (rdtsc, default OFF, zero cost when off).  cyc[]:
    // 0 key-hash, 1 lookup-find, 2 eng_restore (hit), 3 measure_z (live), 4 pool_intern (miss dedup),
    // 5 hit-total, 6 live-total, 7 (reserved)
    bool mc_time=false; uint64_t mc_cyc[8]={0,0,0,0,0,0,0,0};
    // Control-plane decomposition (rdtsc per opcode category, default OFF).  opcyc[]:
    // 0 OUTER_FRAME, 1 ENGINE_GATE(tableau/inverse/pending conj), 2 ROT(pending create), 3 NOISE,
    // 4 DORMANT/record-cond, 5 BOUNDARY(measure: hit-path / miss measure_z + record/frame), 6 OTHER, 7 whole-loop
    bool mc_optime=false; uint64_t mc_opcyc[8]={0,0,0,0,0,0,0,0};
    // Ablation (TIMING-ONLY; output is WRONG when set) — skip a category's WORK to measure its cost by the
    // clean-wall delta (no per-op timer overhead).  bits: 1 OUTER_FRAME, 2 ENGINE_GATE, 4 ROT, 8 NOISE,
    // 16 DORMANT/readout, 32 BOUNDARY(measure).  The per-op branch is present in full + ablated runs -> cancels.
    int mc_skip=0;
    static int mc_catof(uint8_t k){
        switch((MdamOp)k){
            case MO_FRAME_H: case MO_FRAME_CNOT: case MO_FRAME_CZ: case MO_FRAME_SWAP: case MO_FRAME_S: return 0;
            case MO_ARRAY_CNOT: case MO_ARRAY_CZ: case MO_MULTI_CNOT: case MO_MULTI_CZ: case MO_ARRAY_S:
            case MO_EXPAND: case MO_ARRAY_H: case MO_ARRAY_U2: case MO_ARRAY_U4: return 1;
            case MO_ARRAY_T: case MO_ARRAY_T_DAG: case MO_EXPAND_T: case MO_EXPAND_T_DAG:
            case MO_ARRAY_ROT: case MO_EXPAND_ROT: return 2;
            case MO_NOISE: case MO_NOISE_BLOCK: case MO_READOUT_NOISE: return 3;
            case MO_MEAS_DORM_STATIC: case MO_MEAS_DORM_RANDOM: case MO_APPLY_PAULI: return 4;
            case MO_SWAP_MEAS_INTERFERE: case MO_MEAS_ACTIVE_DIAGONAL: case MO_MEAS_ACTIVE_INTERFERE: return 5;
            default: return 6;
        }
    }
    void mc_reset(){ mc_pool.clear(); mc_pool_idx.clear(); mc_edges.clear(); mc_pool_bytes_live=0;
        mc_hit=mc_miss=mc_partial=mc_antis=mc_verify=mc_mismatch=mc_restore=0;
        for(int i=0;i<8;i++){ mc_cyc[i]=0; mc_opcyc[i]=0; }
        bcap_intern.clear(); bcap_amp.clear(); }
    // Actual heavy memory of the mcache: each pooled EngSnap holds a 2^r dense core (dominant term, 16B/amp).
    // A non-saturating circuit grows mc_pool one snapshot per distinct post-boundary state -> unbounded.  Used
    // by the adaptive memory guard so a runaway probation demotes to AUTH BEFORE OOM (not just ln_id count).
    size_t mc_pool_bytes() const {
        size_t b = mc_pool.size()*sizeof(EngSnap);
        for(const auto& s: mc_pool){
            b += s.dense.capacity()*sizeof(cd) + s.M.capacity()*sizeof(int)
               + (s.ax.capacity()+s.az.capacity()+s.Xc.capacity()+s.Zc.capacity())*sizeof(PackedPauli)
               + s.pend.capacity()*sizeof(PendingEntry);
        }
        size_t ec=0; for(const auto& m: mc_edges) ec+=m.size();
        return b + ec*(sizeof(MEdge)+16) + mc_pool_idx.size()*24;
    }
    // Free the mcache (dense-core pool + edges).  Safe on demote: AUTH (run()) never reads it.
    void mc_pool_free(){ { std::vector<EngSnap> a; mc_pool.swap(a); }
        mc_pool_idx.clear(); mc_edges.clear(); mc_pool_bytes_live=0; }
    // ===== Clean-room SEGMENT/automaton SEPARABILITY shadow (path-3, default OFF, authoritative untouched) ==
    // Load-bearing premise of the lean boundary-walk: the boundary SIGNATURE sequence is a deterministic
    // automaton driven ONLY by Born outcomes.  The inter-boundary gate-walk (100% symbolic F2 on
    // tableau/inverse-frame/pending — NO dense; verified in code) is a pure function of (prev boundary key,
    // outcome), INDEPENDENT of the per-shot noise/feedback (which only drives the SEPARATE outer-frame/record
    // layer — sampler.apply_site + apply_mask touch ONLY the outer NativeFrame).  The one possible coupling is
    // rot() reading frame.xb(slot) for the rotation sign; this shadow tests whether that coupling ever changes
    // the automaton.  Runs the authoritative gate-walk and verifies key(mp+1) == f(mp, key(mp), outcome(mp))
    // across shots with different noise.  viol>0 names noise->automaton coupling (=> key needs frame parity).
    int  sg_shadow=0;                                    // 0 off, 1 on
    int  sg_signs=0;                                     // 1 = fold in-segment rotation-sign bits into the source key
    std::unordered_map<uint64_t,uint64_t> sg_trans;      // source-key -> next_key (persists across shots)
    long sg_checks=0, sg_viol=0, sg_edges=0, sg_bounds=0;
    uint64_t sg_prev_key=0; int sg_prev_out=-1, sg_prev_mp=-1; bool sg_have_prev=false;
    uint64_t sg_seg_signs=1469598103934665603ULL;        // order-hash of xb bits rot() read since the last boundary
    inline void sg_sign_acc(int slot,int xb){ sg_seg_signs = (sg_seg_signs*1099511628211ULL) ^ (uint64_t)((slot<<1)|(xb&1)); }
    inline void sg_rot_sign(int slot,int xb){ if(sg_shadow||ln_active) sg_sign_acc(slot,xb); }
    // shared edge-key (source of a boundary edge): must be byte-identical between the sg shadow (build) and
    // run_lean (walk).  Includes the in-segment rotation signs (sg_signs must be ON for the lean table).
    inline uint64_t sg_edge_key(int prev_mp, uint64_t prev_node, int prev_out) const {
        uint64_t tk=1469598103934665603ULL;
        tk=dfnv(tk,&prev_mp,4); tk=dfnv(tk,&prev_node,8); tk=dfnv(tk,&prev_out,4);
        tk=dfnv(tk,&sg_seg_signs,8);
        return tk;
    }
    inline void sg_note_boundary(int mp, uint64_t K, int b){
        if(!sg_shadow) return;
        sg_bounds++;
        int cur_id = ln_id[K];                           // interned by sg_note_p0 (called just before)
        if(sg_have_prev){
            uint64_t tk = sg_signs ? sg_edge_key(sg_prev_mp, sg_prev_key, sg_prev_out)
                                   : [&]{ uint64_t t=1469598103934665603ULL; t=dfnv(t,&sg_prev_mp,4);
                                          t=dfnv(t,&sg_prev_key,8); t=dfnv(t,&sg_prev_out,4); return t; }();
            auto it=sg_trans.find(tk);
            if(it==sg_trans.end()){ sg_trans.emplace(tk,K); sg_edges++; }
            else { sg_checks++; if(it->second!=K) sg_viol++; }
            // int-keyed edge for the lean walk (determinism proven by the sg_trans check above)
            if(sg_signs) ln_edge.emplace(sg_edge_key_id(sg_prev_id, sg_prev_out), cur_id);
        }
        sg_prev_key=K; sg_prev_out=b; sg_prev_mp=mp; sg_prev_id=cur_id; sg_have_prev=true;
        sg_seg_signs=1469598103934665603ULL;             // reset for the NEXT segment
    }
    // premise-2 (Mealy completeness): the Born threshold p0 must be a deterministic function of the node key,
    // so the lean walk can draw the outcome from a cached p0 WITHOUT the engine measure_z.  Also verify the
    // antis-ness (stabilizer coin vs Born) is node-consistent (antis nodes stay LIVE in the lean walk).
    std::unordered_map<uint64_t,double> sg_p0;           // node key -> p0
    std::unordered_map<uint64_t,uint8_t> sg_antis;       // node key -> antis flag
    long sg_p0_checks=0, sg_p0_viol=0, sg_antis_checks=0, sg_antis_viol=0;
    inline void sg_note_p0(uint64_t K, double p0, uint8_t antis){
        if(!sg_shadow) return;
        { auto it=sg_antis.find(K); if(it==sg_antis.end()) sg_antis.emplace(K,antis);
          else { sg_antis_checks++; if(it->second!=antis) sg_antis_viol++; } }
        // intern node -> dense id + p0/antis arrays (int-keyed lean walk)
        auto r=ln_id.emplace(K,(int)ln_p0v.size());
        if(r.second){ ln_p0v.push_back(p0); ln_antisv.push_back(antis); }
        if(antis) return;                                // antis -> no Born p0 (kept live)
        auto it=sg_p0.find(K); if(it==sg_p0.end()) sg_p0.emplace(K,p0);
        else { sg_p0_checks++; if(it->second!=p0) sg_p0_viol++; }
    }
    void sg_reset(){ sg_trans.clear(); sg_checks=sg_viol=sg_edges=sg_bounds=0; sg_have_prev=false;
        sg_seg_signs=1469598103934665603ULL; sg_prev_id=-1;
        sg_p0.clear(); sg_antis.clear(); sg_p0_checks=sg_p0_viol=sg_antis_checks=sg_antis_viol=0;
        ln_id.clear(); ln_edge.clear(); ln_p0v.clear(); ln_antisv.clear(); }
    // ===== path-3 REDUCED-EXECUTION lean walk (run_lean): frame-layer + noise + the automaton table only;
    // SKIPS the entire engine gate-walk (tableau/inverse-frame/pending/dense/measure_z = the 85% opcode_loop).
    // Correctness rests on: frame⊥engine (separate objects), engine layer RNG-free (verified), automaton
    // complete (node->p0/antis/next all deterministic, 0 viol).  Uses the warm sg table (sg_signs=1).  A shot
    // that hits an uncached edge/node aborts to ln_incomplete (excluded from the record comparison, counted).
    bool ln_active=false;
    int ln_prev_out=0, ln_cur_id=-1;
    bool ln_incomplete=false; long ln_incomplete_shots=0, ln_miss=0;
    // int-node-id automaton (built during the warm sg run): node key -> dense id; dense p0/antis arrays;
    // edge = FNV(prev_id, prev_out, in-segment signs) -> next id.  ONE hash lookup + 2 array reads per boundary
    // (vs 1 FNV + 3 unordered_map.find on the 64-bit key path).  id encodes mp (K=mc_key includes mp).
    std::unordered_map<uint64_t,int> ln_id, ln_edge;
    std::vector<double> ln_p0v; std::vector<uint8_t> ln_antisv;
    int sg_prev_id=-1;                                         // interned id of the previous boundary (edge build); -1 = virtual start
    inline uint64_t sg_edge_key_id(int prev_id, int prev_out) const {
        uint64_t tk=1469598103934665603ULL; tk=dfnv(tk,&prev_id,4); tk=dfnv(tk,&prev_out,4); tk=dfnv(tk,&sg_seg_signs,8); return tk; }
    inline void lean_rot(int slot){ int q=(slot<(int)slot2id.size())?slot2id[slot]:-1; if(q<0) return;
        int xb=frame.xb(slot); sg_sign_acc(slot,xb); }        // == rot()'s sign capture (q<0 guard); pending SKIPPED
    int lean_measure(){
        uint64_t ek=sg_edge_key_id(ln_cur_id, ln_prev_out);
        auto it=ln_edge.find(ek);
        if(it==ln_edge.end()){ ln_incomplete=true; ln_miss++; return -1; }   // uncached trajectory
        int id=it->second, b;
        if(ln_antisv[id]){ b=(int)idraw2(); }                 // stabilizer coin (1 idraw2, matches authoritative antis)
        else { double rv=udraw(); b=(rv<ln_p0v[id])?0:1; }    // Born (1 udraw, matches authoritative)
        ln_cur_id=id; ln_prev_out=b;
        sg_seg_signs=1469598103934665603ULL;                  // reset segment signs (== sg_note_boundary)
        return b;
    }
    // ===== Gate N: frame-block superinstruction (default OFF) =========================
    // distillation is 81% pure MO_FRAME_* opcodes (1625/1995) in 90 runs (mean 18, max 52).
    // Each costs one big-switch dispatch (6 array loads + jump) for an ~8-cyc body.  Batch each
    // maximal run into ONE dispatch + a tight inner loop with grow() hoisted out.  Executes the
    // identical ops in identical order -> bit-exact by construction.
    bool mc_fblock=false;
    bool fb_blk_built=false; size_t fb_blk_N=0;
    std::vector<int>     fbi_at;                 // pc -> block id (run start) or -1
    std::vector<int>     fb_off, fb_len, fb_maxslot;
    std::vector<uint8_t> fb_sub;                 // flat sub-op kinds (MO_FRAME_H..MO_FRAME_S)
    std::vector<int32_t> fb_s1, fb_s2;
    void fb_build_blocks(const MdamProgram& p){
        size_t N=p.kind.size();
        fbi_at.assign(N,-1); fb_off.clear(); fb_len.clear(); fb_maxslot.clear();
        fb_sub.clear(); fb_s1.clear(); fb_s2.clear();
        size_t i=0;
        while(i<N){
            if(p.kind[i]<=MO_FRAME_S){                       // 0..4 = MO_FRAME_{H,CNOT,CZ,SWAP,S}
                size_t j=i; int ms=0;
                while(j<N && p.kind[j]<=MO_FRAME_S){
                    if(p.a1[j]>ms) ms=p.a1[j]; if(p.a2[j]>ms) ms=p.a2[j]; j++;
                }
                if(j-i>=3){                                  // worth batching
                    fbi_at[i]=(int)fb_len.size(); fb_off.push_back((int)fb_sub.size());
                    fb_len.push_back((int)(j-i)); fb_maxslot.push_back(ms);
                    for(size_t t=i;t<j;t++){ fb_sub.push_back(p.kind[t]); fb_s1.push_back(p.a1[t]); fb_s2.push_back(p.a2[t]); }
                    i=j; continue;
                }
            }
            i++;
        }
        fb_blk_built=true; fb_blk_N=N;
    }
    inline void fb_exec(int b){
        frame.grow((size_t)fb_maxslot[b]);                   // hoist grow out of the per-op path
        uint8_t* X=frame.x.data(); uint8_t* Z=frame.z.data();
        int off=fb_off[b], L=fb_len[b];
        for(int j=0;j<L;j++){ int s=off+j; int a1=fb_s1[s], a2=fb_s2[s]; uint8_t t;
            switch(fb_sub[s]){
                case MO_FRAME_H:    t=X[a1]; X[a1]=Z[a1]; Z[a1]=t; break;
                case MO_FRAME_CNOT: X[a2]^=X[a1]; Z[a1]^=Z[a2]; break;
                case MO_FRAME_CZ:   Z[a1]^=X[a2]; Z[a2]^=X[a1]; break;
                case MO_FRAME_SWAP: t=X[a1];X[a1]=X[a2];X[a2]=t; t=Z[a1];Z[a1]=Z[a2];Z[a2]=t; break;
                case MO_FRAME_S:    Z[a1]^=X[a1]; break;
            }
        }
    }
    // ===== LEAN frame-block: in run_lean, MO_ARRAY_{CNOT,CZ,S,H} act as PURE frame ops (engine SKIPPED) ==
    // identical to MO_FRAME_{CNOT,CZ,S,H}.  So the lean block builder batches BOTH families into longer runs
    // -> fewer big-switch dispatches (attacks the dispatch residual).  Separate arrays -> shared fb_* (used by
    // run_mcache, where ARRAY_* also do engine work) is UNTOUCHED.  Action codes: 0=H 1=CNOT 2=CZ 3=SWAP 4=S.
    bool lfb_built=false; size_t lfb_N=0;
    std::vector<int> lfb_at, lfb_off, lfb_len, lfb_maxslot;
    std::vector<uint8_t> lfb_act; std::vector<int32_t> lfb_s1, lfb_s2;
    static inline int lfb_action(uint8_t k){
        switch((MdamOp)k){ case MO_FRAME_H: case MO_ARRAY_H: return 0;
            case MO_FRAME_CNOT: case MO_ARRAY_CNOT: return 1; case MO_FRAME_CZ: case MO_ARRAY_CZ: return 2;
            case MO_FRAME_SWAP: return 3; case MO_FRAME_S: case MO_ARRAY_S: return 4; default: return -1; } }
    void lfb_build(const MdamProgram& p){
        size_t N=p.kind.size(); lfb_at.assign(N,-1); lfb_off.clear(); lfb_len.clear(); lfb_maxslot.clear();
        lfb_act.clear(); lfb_s1.clear(); lfb_s2.clear(); size_t i=0;
        while(i<N){ if(lfb_action(p.kind[i])>=0){ size_t j=i; int ms=0;
                while(j<N && lfb_action(p.kind[j])>=0){ if(p.a1[j]>ms) ms=p.a1[j]; if(p.a2[j]>ms) ms=p.a2[j]; j++; }
                if(j-i>=3){ lfb_at[i]=(int)lfb_len.size(); lfb_off.push_back((int)lfb_act.size());
                    lfb_len.push_back((int)(j-i)); lfb_maxslot.push_back(ms);
                    for(size_t t=i;t<j;t++){ lfb_act.push_back((uint8_t)lfb_action(p.kind[t])); lfb_s1.push_back(p.a1[t]); lfb_s2.push_back(p.a2[t]); }
                    i=j; continue; } }
            i++; }
        lfb_built=true; lfb_N=N;
    }
    inline void lfb_exec(int b){
        frame.grow((size_t)lfb_maxslot[b]); uint8_t* X=frame.x.data(); uint8_t* Z=frame.z.data();
        int off=lfb_off[b], L=lfb_len[b];
        for(int j=0;j<L;j++){ int s=off+j; int a1=lfb_s1[s], a2=lfb_s2[s]; uint8_t t;
            switch(lfb_act[s]){
                case 0: t=X[a1]; X[a1]=Z[a1]; Z[a1]=t; break;
                case 1: X[a2]^=X[a1]; Z[a1]^=Z[a2]; break;
                case 2: Z[a1]^=X[a2]; Z[a2]^=X[a1]; break;
                case 3: t=X[a1];X[a1]=X[a2];X[a2]=t; t=Z[a1];Z[a1]=Z[a2];Z[a2]=t; break;
                case 4: Z[a1]^=X[a1]; break;
            }
        }
    }
    void eng_snapshot(EngSnap& s){ auto&e=engine; int n=e.n; size_t N=(size_t)1<<e.dense.r;
        s.r=e.dense.r; s.dense.assign(e.dense.resident.begin(), e.dense.resident.begin()+N);
        for(auto& z:s.dense) z=cd(z.real()+0.0, z.imag()+0.0);                     // canonicalize signed zero (stable dedup)
        s.M=e.M;
        s.ax.assign(e.inverse_frame.ax.begin(), e.inverse_frame.ax.begin()+n);
        s.az.assign(e.inverse_frame.az.begin(), e.inverse_frame.az.begin()+n);
        s.Xc.assign(e.tableau.Xc.begin(), e.tableau.Xc.begin()+n);
        s.Zc.assign(e.tableau.Zc.begin(), e.tableau.Zc.begin()+n);
        s.pend.clear(); for(auto&pe:e.pending.slots) if(pe.generation==e.pending.gen) s.pend.push_back(pe);
        s.rot_uid=e.rot_uid; }
    int mc_dense_sid=-1;        // Phase-4 lazy dense: which sid the LIVE engine.dense currently holds (-1 stale)
    void mc_materialize_dense(int sid){ const std::vector<cd>& a=bcap_amp[sid]; int rk=(int)__builtin_ctzll(a.size());
        engine.dense.set_state(rk, a.data()); mc_dense_sid=sid; }
    void eng_restore(const EngSnap& s, bool with_dense=true){ auto&e=engine; int n=e.n;
        if(with_dense) e.dense.set_state(s.r, s.dense.data());   // lazy-dense carry skips this on hits (dense carried by sid)
        e.M=s.M;
        for(int i=0;i<n;i++){ e.inverse_frame.ax[i]=s.ax[i]; e.inverse_frame.az[i]=s.az[i]; e.tableau.Xc[i]=s.Xc[i]; e.tableau.Zc[i]=s.Zc[i]; }
        e.pending.gen++;                                                            // invalidate all live; re-create the saved live set
        for(auto&pe:s.pend){ if(pe.uid>=e.pending.slots.size()) e.pending.slots.resize(pe.uid+1, PendingEntry{PackedPauli(e.W),0.0,0,0});
            e.pending.slots[pe.uid]=pe; e.pending.slots[pe.uid].generation=e.pending.gen; }
        e.rot_uid=s.rot_uid; e.inv_dirty=false; e.basis_valid=false; mc_restore++; }
    uint64_t eng_fingerprint(){ auto&e=engine; int n=e.n; size_t N=(size_t)1<<e.dense.r;
        bcap_buf.resize(N); for(size_t j=0;j<N;j++) bcap_buf[j]=cd(e.dense.resident[j].real()+0.0, e.dense.resident[j].imag()+0.0);
        uint64_t h=dfnv(1469598103934665603ULL, bcap_buf.data(), sizeof(cd)*N);
        int rr=e.dense.r; h=dfnv(h,&rr,sizeof(int));
        for(int m:e.M){ int mm=m; h=dfnv(h,&mm,sizeof(int)); }
        for(int i=0;i<n;i++){ h=dfnv(h,&e.inverse_frame.ax[i],sizeof(PackedPauli)); h=dfnv(h,&e.inverse_frame.az[i],sizeof(PackedPauli));
                              h=dfnv(h,&e.tableau.Xc[i],sizeof(PackedPauli)); h=dfnv(h,&e.tableau.Zc[i],sizeof(PackedPauli)); }
        for(auto&pe:e.pending.slots) if(pe.generation==e.pending.gen){ h=dfnv(h,&pe.p,sizeof(PackedPauli)); double th=pe.theta; h=dfnv(h,&th,8); h=dfnv(h,&pe.uid,4); }
        return h; }
    bool eng_equal(const EngSnap& s){ auto&e=engine; int n=e.n; size_t N=(size_t)1<<e.dense.r;
        if(s.r!=e.dense.r || s.M!=e.M) return false;
        for(size_t j=0;j<N;j++){ cd z(e.dense.resident[j].real()+0.0, e.dense.resident[j].imag()+0.0); if(s.dense[j]!=z) return false; }
        for(int i=0;i<n;i++){ if(memcmp(&s.ax[i],&e.inverse_frame.ax[i],sizeof(PackedPauli))) return false;
                              if(memcmp(&s.az[i],&e.inverse_frame.az[i],sizeof(PackedPauli))) return false;
                              if(memcmp(&s.Xc[i],&e.tableau.Xc[i],sizeof(PackedPauli))) return false;
                              if(memcmp(&s.Zc[i],&e.tableau.Zc[i],sizeof(PackedPauli))) return false; }
        size_t np=0; for(auto&pe:e.pending.slots) if(pe.generation==e.pending.gen) np++;
        if(np!=s.pend.size()) return false;
        size_t k=0; for(auto&pe:e.pending.slots) if(pe.generation==e.pending.gen){ const PendingEntry& q=s.pend[k++];
            if(memcmp(&pe.p,&q.p,sizeof(PackedPauli))||pe.theta!=q.theta||pe.uid!=q.uid) return false; }
        return true; }
    int mc_pool_intern(){                                                          // dedup current engine post-state -> pool id
        uint64_t fp=eng_fingerprint(); auto& cand=mc_pool_idx[fp];
        for(int id:cand) if(eng_equal(mc_pool[id])) return id;
        int id=(int)mc_pool.size(); mc_pool.emplace_back(); eng_snapshot(mc_pool.back());
        { const auto& s=mc_pool.back(); mc_pool_bytes_live += sizeof(EngSnap) + s.dense.capacity()*sizeof(cd)
            + s.M.capacity()*sizeof(int)
            + (s.ax.capacity()+s.az.capacity()+s.Xc.capacity()+s.Zc.capacity())*sizeof(PackedPauli)
            + s.pend.capacity()*sizeof(PendingEntry); }
        cand.push_back(id); return id; }
    uint64_t mc_key(int mp,int kind){
        uint64_t h=1469598103934665603ULL; int mm=mp; h=dfnv(h,&mm,4); h=dfnv(h,&kind,4);
        uint32_t a;
        if(mc_mode==3 && mc_cur_sid>=0) a=(uint32_t)mc_cur_sid;             // Phase-4 carry: NO dense re-hash (the 72-86% hit cost)
        else { a=(uint32_t)bcap_sid(engine.dense.resident.data(),engine.dense.r); if(mc_mode==3){ mc_cur_sid=(int)a; mc_dense_sid=(int)a; } }  // live dense just interned -> it holds this sid
        uint32_t b=bcap_inv_sig(), c=bcap_pend_sig(), d=bcap_m_sig();
        h=dfnv(h,&a,4); h=dfnv(h,&b,4); h=dfnv(h,&c,4); h=dfnv(h,&d,4); return h; }
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
        int nrot=0; for (uint8_t k : p.kind) if (k==MO_ARRAY_T||k==MO_ARRAY_T_DAG||k==MO_EXPAND_T||k==MO_EXPAND_T_DAG
                                                  ||k==MO_ARRAY_ROT||k==MO_EXPAND_ROT) nrot++;
        magic_scratch.reserve_for(p.engine_n, nrot + p.engine_n + 8);
        // §5: pre-size the core cache (1 magic measure_z per SWAP_MEAS / MEAS_ACTIVE_* op) so shot 0 does no resize.
        int nmagic=0; for (uint8_t k : p.kind) if (k==MO_SWAP_MEAS_INTERFERE
                                                   ||k==MO_MEAS_ACTIVE_DIAGONAL||k==MO_MEAS_ACTIVE_INTERFERE) nmagic++;
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
        if (bcap_on) bcap_antis = false;          // Phase-0/1/2: per-boundary reset (oracle sets it true on the anti_s branch)
        if (magic_seen == dump_before_magic && !dumped) dump_engine();
        magic_seen++;
        int mp = magic_point++;                       // per-shot magic-point index (cache key)
        if(pb_cap_on()) pb_mp()=mp;   // Step 1: tag boundary for pullback invariance
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
            if (bcap_on) { NativeMagicTrace tr; int oc = magic_execute(engine, pl, rv, &tr); bcap_p0 = tr.p0; return oc; }
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
                int _nr=(int)magic_scratch.rpp_uf.size(); icap.push_back(_nr); for(int _i=0;_i<_nr;_i++) icap.push_back(magic_scratch.rpp_uf[_i]); }
              if(imem_mode && engine.n<=8){ imem_store_verify(_imem_key,_ms,pl.sign); }
              return _oc; }  // hot path: NO trace -> 0 allocation
        }
        // oracle draws its own Born internally -> route udraw through it
        int rin_before = (int)engine.M.size();
        uint64_t _dsh=0; if(dsig_on){ size_t Nin=(size_t)1<<engine.dense.r; _dsh=dfnv(1469598103934665603ULL, engine.dense.resident.data(), sizeof(cd)*Nin); }
        ITIME_BEG(IT_ORACLE); OracleResult R = oracle_measure_magic_counted(q); ITIME_END(IT_ORACLE);
        magic_oracle++; magic_last_p0=R.p0;     // Gate K shadow: stash oracle Born p0 for the edge-cache verify
        if(bcap_on) bcap_p0=R.p0;               // Phase-0/1/2 boundary capture: oracle Born p0
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
        if(ms && ms->valid){                                          // shadow: compare memo vs live (UNFOLDED rpp)
            bool ok=(ms->sign==sign) && (ms->rpp.size()==magic_scratch.rpp_uf.size());
            if(ok) for(size_t j=0;j<ms->rpp.size();j++) if(ms->rpp[j]!=magic_scratch.rpp_uf[j]){ ok=false; break; }
            if(!ok) imem_mismatch++;
            return;
        }
        if(imem_mode==2) imem_misses++;                              // fast miss -> store
        ImemEntry e; e.valid=true; e.sign=sign;
        e.rpp.assign(magic_scratch.rpp_uf.begin(), magic_scratch.rpp_uf.end());
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
        if(bcap_on) bcap_antis = !anti_s.empty();   // Phase-0/1/2: flag the stabilizer ag_measure branch (coin, not Born)
        if (!anti_s.empty()) {
            PackedPauli Pmphys(engine.W); Pmphys.z[PackedPauli::word(q)]=PackedPauli::bit(q);  // physical Z_q
            int out = (int)idraw2();                  // _ag_measure: integers(0,2)
            ORC_T(8, engine.ag_measure(Pmphys, anti_s[0], out));
            bool _rfn; ORC_T(9, _rfn=engine.reduce_full_is_noop());
            if(!_rfn) return {-1,0.0,false,"reduce_full would fire"};
            return {out, 0.0, true, nullptr};
        }
        double sign; int r;
        ORC_T(3, { PackedPauli Pm(engine.W); Pm.z[PackedPauli::word(q)]=PackedPauli::bit(q);
                   pb_kind()=1; PackedPauli pm = engine.pullback(Pm);   // oracle_Pm (counted path)
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
        sg_rot_sign(slot,xb);           // segment-shadow: capture the (noise/feedback-driven) rotation sign bit
        double theta = xb ? -angle : angle;
        if (rot_log_on) rot_log.push_back({(double)slot,(double)xb,angle,theta});
        engine.apply_rotation(q, theta);
    }

    // Adaptive lazy-inverse policy.  lazy is a BIG win for maxM=0 circuits (frame never read -> 0 rebuilds,
    // d5_r1 0.93x->6.6x, d7_r1 ->35000x) but a ~40% LOSS for magic-heavy circuits (full rebuild-on-read vs
    // eager incremental fwd; cult_d5/rx).  Both are BIT-EXACT, so we auto-pick: run_batch probes shot 0
    // (lazy), then uses eager for the rest iff that shot materialized any magic (engine.magic_ever).
    // Env override: MDAM_LAZY=force lazy, MDAM_NOLAZY=force eager.  lazy_env: -2 unread,-1 auto,0 eager,1 lazy.
    int lazy_env_cache=-2; bool batch_lazy_hint=true;
    int lazy_env(){ if(lazy_env_cache==-2) lazy_env_cache = std::getenv("MDAM_LAZY")?1:(std::getenv("MDAM_NOLAZY")?0:-1); return lazy_env_cache; }
    bool frame_log_on=false; std::vector<std::array<uint64_t,2>> frame_log;
    void run(const MdamProgram& p){
        // Lazy inverse frame: defer the AG-projection inverse-frame rebuild to the first frame READ, so a
        // maxM=0 trajectory (the frame is never read) pays ZERO rebuilds.  It is bit-exact on every CORRECT
        // circuit; the prior blocker (it perturbed coherent_rx_d3_r3's R_X drift) is RESOLVED — rx_d3_r3 is
        // now 25/25 under FUSED, and lazy==eager 25/25 on all 8 feasible benches (d3_r1/d3_r3/d5_r1/d5_r5/
        // rx_d3_r1/rx_d3_r3/cult_d3/cult_d5).  So it is now a pure optimization -> DEFAULT ON (opt-out via
        // MDAM_NOLAZY).  Wall effect: d5_r1 0.93x->6.64x, d3_r1 0.11x->0.30x; d5_r5 unchanged (maxM=12
        // materializes the frame on first read = identical to eager).  run_batch auto-picks (batch_lazy_hint).
        { int le=lazy_env(); engine.lazy_inverse = (le==1) ? true : (le==0 ? false : batch_lazy_hint); }
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
                    int _bmp=0; uint32_t _bsi=0,_biv=0,_bpd=0,_bms=0; uint8_t _bxb=0,_bzb=0; int _bmo=0;
                    if(bcap_on){ _bmp=magic_point; _bsi=bcap_sid(engine.dense.resident.data(),engine.dense.r);
                        _biv=bcap_inv_sig(); _bpd=bcap_pend_sig(); _bms=bcap_m_sig();
                        _bxb=frame.xb(a2); _bzb=frame.zb(a2); _bmo=magic_oracle; }
                    int b=measure_z(q);
                    slot2id[a2]=-1;
                    int m_abs = b ^ frame.zb(a2);
                    record.set((uint32_t)i0, m_abs^i1);
                    frame.set_xz(a2,(uint8_t)m_abs,0);
                    if(bcap_on){ uint32_t so=bcap_sid(engine.dense.resident.data(),engine.dense.r);
                        uint8_t orc=(magic_oracle>_bmo)?(bcap_antis?2:1):0;
                        bcap.push_back({_bmp,_bsi,_biv,_bpd,_bms,_bxb,_bzb,(uint8_t)i1,(uint8_t)2,orc,(uint8_t)b,so,(uint8_t)((m_abs^i1)&1),bcap_p0}); }
                } break;
                // ---- Gate L1: coherent opcodes (authoritative path) ----
                case MO_ARRAY_ROT: rot(p,a1,dv); break;                  // arbitrary-theta diagonal rotation
                case MO_EXPAND_ROT: { newq(a1); engine.h(slot2id[a1]); rot(p,a1,dv); } break;  // birth + rotation
                case MO_ARRAY_SWAP: {                                    // pure slot relabel (no engine op)
                    int i_1=slot2id[a1], i_2=slot2id[a2];
                    slot2id[a1]=-1; slot2id[a2]=-1;
                    if(i_1>=0) slot2id[a2]=i_1; if(i_2>=0) slot2id[a1]=i_2;
                    frame.swap(a1,a2);
                } break;
                case MO_MEAS_ACTIVE_DIAGONAL: {                          // Z-basis active-register measurement
                    int q=slot2id[a1]; if(q<0) break;
                    int _bmp=0; uint32_t _bsi=0,_biv=0,_bpd=0,_bms=0; uint8_t _bxb=0,_bzb=0; int _bmo=0;
                    if(bcap_on){ _bmp=magic_point; _bsi=bcap_sid(engine.dense.resident.data(),engine.dense.r);
                        _biv=bcap_inv_sig(); _bpd=bcap_pend_sig(); _bms=bcap_m_sig();
                        _bxb=frame.xb(a1); _bzb=frame.zb(a1); _bmo=magic_oracle; }
                    int b=measure_z(q);
                    slot2id[a1]=-1;
                    int m_abs = b ^ frame.xb(a1);                        // diagonal: XOR X-frame parity
                    record.set((uint32_t)i0, m_abs^i1);                  // i1 = FLAG_SIGN
                    frame.set_xz(a1,(uint8_t)m_abs,0);
                    if(bcap_on){ uint32_t so=bcap_sid(engine.dense.resident.data(),engine.dense.r);
                        uint8_t orc=(magic_oracle>_bmo)?(bcap_antis?2:1):0;
                        bcap.push_back({_bmp,_bsi,_biv,_bpd,_bms,_bxb,_bzb,(uint8_t)i1,(uint8_t)0,orc,(uint8_t)b,so,(uint8_t)((m_abs^i1)&1),bcap_p0}); }
                } break;
                case MO_MEAS_ACTIVE_INTERFERE: {                         // X-basis (H then Z) active measurement
                    int q=slot2id[a1]; if(q<0) break;
                    engine.h(q);
                    int _bmp=0; uint32_t _bsi=0,_biv=0,_bpd=0,_bms=0; uint8_t _bxb=0,_bzb=0; int _bmo=0;
                    if(bcap_on){ _bmp=magic_point; _bsi=bcap_sid(engine.dense.resident.data(),engine.dense.r);
                        _biv=bcap_inv_sig(); _bpd=bcap_pend_sig(); _bms=bcap_m_sig();
                        _bxb=frame.xb(a1); _bzb=frame.zb(a1); _bmo=magic_oracle; }
                    int b=measure_z(q);
                    slot2id[a1]=-1;
                    int m_abs = b ^ frame.zb(a1);                        // interfere: XOR Z-frame parity
                    record.set((uint32_t)i0, m_abs^i1);
                    frame.set_xz(a1,(uint8_t)m_abs,0);
                    if(bcap_on){ uint32_t so=bcap_sid(engine.dense.resident.data(),engine.dense.r);
                        uint8_t orc=(magic_oracle>_bmo)?(bcap_antis?2:1):0;
                        bcap.push_back({_bmp,_bsi,_biv,_bpd,_bms,_bxb,_bzb,(uint8_t)i1,(uint8_t)1,orc,(uint8_t)b,so,(uint8_t)((m_abs^i1)&1),bcap_p0}); }
                } break;
                // ---- Gate L Tier-3 (de-fused dialect): birth |+> + active Hadamard ----
                case MO_EXPAND: { newq(a1); engine.h(slot2id[a1]); } break;   // _birth: new |0> qubit -> H -> |+> (state-prep, frame untouched)
                case MO_ARRAY_H: { int q=slot2id[a1]; if(q>=0) engine.h(q); frame.h(a1); } break;
                // ---- Gate L Tier-3 DIRECT (fused dialect): frame-keyed fused unitaries (preserve maxM) ----
                // == backend.py _apply_u2: select 2x2 by incoming frame (in_state), apply its ZXZ
                // (Rz(d)Rx(c)Rz(b)) on the magic axis WITHOUT frame sign-flip (frame consumed by selection),
                // then reset frame to the node's out.  i0 = cp_idx, i1 unused.
                case MO_ARRAY_U2: {
                    int q=slot2id[a1];
                    int in_state=(frame.zb(a1)<<1)|frame.xb(a1);
                    int idx=i0*4+in_state; const double* bcd=&p.u2_bcd[(size_t)idx*3];
                    double bb=bcd[0], cc=bcd[1], dd=bcd[2];
                    if(q>=0){
                        if(std::abs(dd)>1e-12) engine.apply_rotation_pauli(q,0,1,dd);   // Rz(d)
                        if(std::abs(cc)>1e-12) engine.apply_rotation_pauli(q,1,0,cc);   // Rx(c)
                        if(std::abs(bb)>1e-12) engine.apply_rotation_pauli(q,0,1,bb);   // Rz(b)
                    }
                    uint8_t out=p.u2_out[idx]; frame.set_xz(a1,out&1,(out>>1)&1);
                } break;
                // == backend.py _apply_u4: select 4x4 by 2-axis frame in_state, replay the precomputed
                // cx/cz/rot1 op-list on (lo=a1, hi=a2), then reset both frames to the node's out.
                case MO_ARRAY_U4: {
                    int lo=slot2id[a1], hi=slot2id[a2];
                    int in_state=(frame.zb(a2)<<3)|(frame.xb(a2)<<2)|(frame.zb(a1)<<1)|frame.xb(a1);
                    int idx=i0*16+in_state; int st=p.u4_start[idx], cnt=p.u4_cnt[idx];
                    if(cnt<0){ err="MO_ARRAY_U4: non-structural fused-U4 in_state selected (rot2/general not native-supported)"; break; }
                    if(lo>=0 && hi>=0){
                        for(int kk=0;kk<cnt;kk++){
                            const double* op=&p.u4_ops[(size_t)(st+kk)*5];
                            int ot=(int)op[0], which=(int)op[1], px=(int)op[2], pz=(int)op[3]; double th=op[4];
                            if(ot==0) engine.cx(lo,hi);                      // cx control=lo target=hi
                            else if(ot==1) engine.cz(lo,hi);                 // cz
                            else { int qq=which?hi:lo; engine.apply_rotation_pauli(qq,px,pz,th); } // rot1
                        }
                    }
                    uint8_t out=p.u4_out[idx];
                    frame.set_xz(a1,out&1,(out>>1)&1); frame.set_xz(a2,(out>>2)&1,(out>>3)&1);
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

    // ===== Phase-3 authoritative-edge cache: per-boundary measure with cache (mc_mode) ===============
    // mc_mode 1 SHADOW: always live measure_z; store edge + verify repeats reproduce p0/post-state (no skip).
    // mc_mode 2 FAST: full HIT (key present, drawn branch filled, non-antis) -> draw Born, RESTORE pool snapshot,
    // SKIP measure_z; miss/partial/antis -> live measure_z (+ store).  rng stream stays aligned (full hit draws
    // the same 1 Born; partial injects the pre-draw; antis stays live=coin).  Returns the outcome bit b.
    int mc_measure(int q, int kind){
        uint64_t _t0=mc_time?__rdtsc():0;
        int mp = magic_point;
        uint64_t K = mc_key(mp, kind);
        uint64_t _t1=mc_time?__rdtsc():0; if(mc_time) mc_cyc[0]+=_t1-_t0;           // [0] key-hash
        MEdge* he=nullptr;
        if(mp<(int)mc_edges.size()){ auto it=mc_edges[mp].find(K); if(it!=mc_edges[mp].end()) he=&it->second; }
        if(mc_time) mc_cyc[1]+=__rdtsc()-_t1;                                       // [1] lookup-find
        if(mc_mode>=2 && he){                                     // FAST (snapshot=2, carry=3)
            if(he->antis){ mc_antis++; }                          // coin (idraw2) -> keep live
            else {
                double rv=udraw(); int b=(rv<he->p0)?0:1;
                if(he->has[b]){ uint64_t _tr=mc_time?__rdtsc():0;
                    eng_restore(mc_pool[he->pool[b]], mc_mode!=3);          // mode 3: frame-only (dense carried by sid -> NO full snapshot restore)
                    if(mc_time){ mc_cyc[2]+=__rdtsc()-_tr; mc_cyc[5]+=__rdtsc()-_t0; }   // [2] restore, [5] hit-total
                    magic_point++; magic_seen++; if(he->disp[b]) magic_oracle++; else magic_compiled++;
                    if(mc_mode==3) mc_cur_sid = he->sid_out[b];             // carry id (no re-hash); mc_dense_sid stays stale -> materialize on next live boundary
                    mc_hit++; sg_note_p0(K,he->p0,0); sg_note_boundary(mp,K,b); return b; }
                kfast_inj_rv=rv; kfast_use_inj=true; mc_partial++;  // drawn branch absent -> reuse rv on the live path
            }
        } else if(mc_mode>=2){ mc_miss++; }                        // no entry -> live + store
        if(mc_mode==3 && mc_dense_sid!=mc_cur_sid && mc_cur_sid>=0) mc_materialize_dense(mc_cur_sid);   // lazy dense: bring the carried state live for measure_z
        uint64_t _tm=mc_time?__rdtsc():0;
        int mo=magic_oracle; bcap_on=true; int b=measure_z(q); bcap_on=false;   // LIVE authoritative (miss/partial/antis/shadow)
        if(mc_time) mc_cyc[3]+=__rdtsc()-_tm;                                       // [3] measure_z
        int disp=(magic_oracle>mo)?1:0; bool is_antis=bcap_antis;
        int so=(mc_mode==3)?(int)bcap_sid(engine.dense.resident.data(),engine.dense.r):-1;   // carry: re-id the (changed) post dense (live only = rare)
        if((int)mc_edges.size()<=mp) mc_edges.resize(mp+1);
        auto& ed=mc_edges[mp][K];
        if(ed.p0<=-2.0){ ed.p0=bcap_p0; ed.antis=is_antis?1:0; }
        else if(mc_mode==1){ mc_verify++; if(ed.p0!=bcap_p0||ed.antis!=(is_antis?1:0)) mc_mismatch++; }
        if(!is_antis){ uint64_t _tp=mc_time?__rdtsc():0; int pid=mc_pool_intern();
            if(mc_time) mc_cyc[4]+=__rdtsc()-_tp;                                   // [4] pool_intern
            if(!ed.has[b]){ ed.has[b]=true; ed.pool[b]=pid; ed.disp[b]=(uint8_t)disp; ed.sid_out[b]=so; }
            else if(mc_mode==1){ if(ed.pool[b]!=pid) mc_mismatch++; } }
        if(mc_mode==3){ mc_cur_sid=so; mc_dense_sid=so; }                          // carry after every live boundary (antis too); live dense now holds `so`
        if(mc_time) mc_cyc[6]+=__rdtsc()-_t0;                                       // [6] live-total
        sg_note_p0(K,bcap_p0,is_antis?1:0); sg_note_boundary(mp,K,b);
        return b;
    }
    // run a shot with the authoritative-edge cache active (eager inverse; cache built from authoritative measure_z).
    void run_mcache(const MdamProgram& p){
        engine.lazy_inverse = false;     // eager: inverse frame always materialized (snapshot-able)
        mc_cur_sid = -1; mc_dense_sid = -1;   // Phase-4 carry: per-shot reset (dense resets each shot)
        // segment-shadow: automaton restarts each shot from a fixed VIRTUAL START node, so the first boundary
        // (start --seg0 signs--> node0) is a covered edge too (lean walk needs it to begin without the engine).
        sg_have_prev = true; sg_prev_key = 0xA5A5A5A5A5A5A5A5ULL; sg_prev_mp = -2; sg_prev_out = -2; sg_prev_id = -1;
        sg_seg_signs = 1469598103934665603ULL;
        size_t N=p.kind.size();
        if(mc_fblock && (!fb_blk_built || fb_blk_N!=N)) fb_build_blocks(p);
        uint64_t _lp0=mc_optime?__rdtsc():0;
        for(size_t i=0;i<N && !err;i++){
            if(mc_fblock && fbi_at[i]>=0){ int b=fbi_at[i];      // frame-block superinstruction
                uint64_t _ob=mc_optime?__rdtsc():0;
                if(!(mc_skip&1)) fb_exec(b);
                if(mc_optime) mc_opcyc[0]+=__rdtsc()-_ob;
                i += (size_t)fb_len[b]-1; continue; }
            int a1=p.a1[i], a2=p.a2[i], i0=p.i0[i], i1=p.i1[i]; double dv=p.dval[i];
            uint64_t _ot=mc_optime?__rdtsc():0;
            switch((MdamOp)p.kind[i]){
                case MO_FRAME_H: if(!(mc_skip&1)) frame.h(a1); break;
                case MO_FRAME_CNOT: if(!(mc_skip&1)) frame.cnot(a1,a2); break;
                case MO_FRAME_CZ: if(!(mc_skip&1)) frame.cz(a1,a2); break;
                case MO_FRAME_SWAP: if(!(mc_skip&1)) frame.swap(a1,a2); break;
                case MO_FRAME_S: if(!(mc_skip&1)) frame.s_gate(a1); break;
                case MO_APPLY_PAULI: { int rc=record.get((uint32_t)i0); if(rc==1) apply_mask(p.cp_masks[i1]); } break;
                case MO_NOISE: if(!(mc_skip&8) && sampler.should_fire(i0)) sampler.apply_site(i0, p.noise_sites[i0], frame); break;   // inline-guard: skip the call for non-firing sites
                case MO_NOISE_BLOCK: if(!(mc_skip&8)) for(int s=i0;s<i0+i1;s++){ if(sampler.should_fire(s)) sampler.apply_site(s, p.noise_sites[s], frame); } break;
                case MO_READOUT_NOISE: if(!(mc_skip&8)){ if(udraw()<dv) record.flip((uint32_t)i0); } break;
                case MO_MEAS_DORM_STATIC: if(!(mc_skip&16)) record.set((uint32_t)i0, frame.xb(a1)^i1); break;
                case MO_MEAS_DORM_RANDOM: if(!(mc_skip&16)){ int m=(int)idraw2(); record.set((uint32_t)i0, m^i1); frame.set_xz(a1,(uint8_t)m,0); } break;
                case MO_ARRAY_CNOT: { int u=slot2id[a1], v=slot2id[a2]; if(u>=0&&v>=0&&!(mc_skip&2)) engine.cx(u,v); if(!(mc_skip&1)) frame.cnot(a1,a2); } break;
                case MO_ARRAY_CZ: { int u=slot2id[a1], v=slot2id[a2]; if(u>=0&&v>=0&&!(mc_skip&2)) engine.cz(u,v); if(!(mc_skip&1)) frame.cz(a1,a2); } break;
                case MO_MULTI_CNOT: { int tgt=a1, t=slot2id[tgt]; uint64_t mask=p.mmask[i0];
                    while(mask){ int ctrl=__builtin_ctzll(mask); mask&=mask-1; if(ctrl==tgt) continue;
                        int c=slot2id[ctrl]; if(t>=0&&c>=0&&!(mc_skip&2)) engine.cx(c,t); if(!(mc_skip&1)) frame.cnot(ctrl,tgt); } } break;
                case MO_MULTI_CZ: { uint64_t mask=p.mmask[i0];
                    while(mask){ int tgt=__builtin_ctzll(mask); mask&=mask-1; if(tgt==a1) continue;
                        int u=slot2id[a1], v=slot2id[tgt]; if(u>=0&&v>=0&&!(mc_skip&2)) engine.cz(u,v); if(!(mc_skip&1)) frame.cz(a1,tgt); } } break;
                case MO_ARRAY_T: if(!(mc_skip&4)) rot(p,a1,NV_T_ANGLE); break;
                case MO_ARRAY_T_DAG: if(!(mc_skip&4)) rot(p,a1,-NV_T_ANGLE); break;
                case MO_ARRAY_S: { int q=slot2id[a1]; if(q>=0&&!(mc_skip&2)) engine.s(q,false); if(!(mc_skip&1)) frame.s_gate(a1); } break;
                case MO_EXPAND_T: { newq(a1); engine.h(slot2id[a1]); if(!(mc_skip&4)) rot(p,a1,NV_T_ANGLE); } break;
                case MO_EXPAND_T_DAG: { newq(a1); engine.h(slot2id[a1]); if(!(mc_skip&4)) rot(p,a1,-NV_T_ANGLE); } break;
                case MO_SWAP_MEAS_INTERFERE: {
                    int i_1=slot2id[a1], i_2=slot2id[a2]; slot2id[a1]=-1; slot2id[a2]=-1;
                    if(i_1>=0) slot2id[a2]=i_1; if(i_2>=0) slot2id[a1]=i_2;
                    if(!(mc_skip&1)) frame.swap(a1,a2);
                    int q=slot2id[a2]; if(q<0) break;
                    engine.h(q);
                    int b=(mc_skip&32)?0:mc_measure(q,2);
                    slot2id[a2]=-1;
                    int m_abs = b ^ frame.zb(a2); record.set((uint32_t)i0, m_abs^i1); frame.set_xz(a2,(uint8_t)m_abs,0);
                } break;
                case MO_ARRAY_ROT: if(!(mc_skip&4)) rot(p,a1,dv); break;
                case MO_EXPAND_ROT: { newq(a1); engine.h(slot2id[a1]); if(!(mc_skip&4)) rot(p,a1,dv); } break;
                case MO_ARRAY_SWAP: { int i_1=slot2id[a1], i_2=slot2id[a2]; slot2id[a1]=-1; slot2id[a2]=-1;
                    if(i_1>=0) slot2id[a2]=i_1; if(i_2>=0) slot2id[a1]=i_2; if(!(mc_skip&1)) frame.swap(a1,a2); } break;
                case MO_MEAS_ACTIVE_DIAGONAL: {
                    int q=slot2id[a1]; if(q<0) break;
                    int b=(mc_skip&32)?0:mc_measure(q,0);
                    slot2id[a1]=-1;
                    int m_abs = b ^ frame.xb(a1); record.set((uint32_t)i0, m_abs^i1); frame.set_xz(a1,(uint8_t)m_abs,0);
                } break;
                case MO_MEAS_ACTIVE_INTERFERE: {
                    int q=slot2id[a1]; if(q<0) break;
                    engine.h(q);
                    int b=(mc_skip&32)?0:mc_measure(q,1);
                    slot2id[a1]=-1;
                    int m_abs = b ^ frame.zb(a1); record.set((uint32_t)i0, m_abs^i1); frame.set_xz(a1,(uint8_t)m_abs,0);
                } break;
                case MO_EXPAND: { newq(a1); engine.h(slot2id[a1]); } break;
                case MO_ARRAY_H: { int q=slot2id[a1]; if(q>=0) engine.h(q); frame.h(a1); } break;
                case MO_ARRAY_U2: {
                    int q=slot2id[a1]; int in_state=(frame.zb(a1)<<1)|frame.xb(a1);
                    int idx=i0*4+in_state; const double* bcd=&p.u2_bcd[(size_t)idx*3]; double bb=bcd[0], cc=bcd[1], dd=bcd[2];
                    if(q>=0){ if(std::abs(dd)>1e-12) engine.apply_rotation_pauli(q,0,1,dd);
                              if(std::abs(cc)>1e-12) engine.apply_rotation_pauli(q,1,0,cc);
                              if(std::abs(bb)>1e-12) engine.apply_rotation_pauli(q,0,1,bb); }
                    uint8_t out=p.u2_out[idx]; frame.set_xz(a1,out&1,(out>>1)&1);
                } break;
                case MO_ARRAY_U4: {
                    int lo=slot2id[a1], hi=slot2id[a2];
                    int in_state=(frame.zb(a2)<<3)|(frame.xb(a2)<<2)|(frame.zb(a1)<<1)|frame.xb(a1);
                    int idx=i0*16+in_state; int st=p.u4_start[idx], cnt=p.u4_cnt[idx];
                    if(cnt<0){ err="MO_ARRAY_U4: non-structural fused-U4 in_state selected"; break; }
                    if(lo>=0 && hi>=0){ for(int kk=0;kk<cnt;kk++){ const double* op=&p.u4_ops[(size_t)(st+kk)*5];
                        int ot=(int)op[0], which=(int)op[1], px=(int)op[2], pz=(int)op[3]; double th=op[4];
                        if(ot==0) engine.cx(lo,hi); else if(ot==1) engine.cz(lo,hi);
                        else { int qq=which?hi:lo; engine.apply_rotation_pauli(qq,px,pz,th); } } }
                    uint8_t out=p.u4_out[idx]; frame.set_xz(a1,out&1,(out>>1)&1); frame.set_xz(a2,(out>>2)&1,(out>>3)&1);
                } break;
                case MO_END: default: break;
            }
            if(mc_optime) mc_opcyc[mc_catof(p.kind[i])] += __rdtsc()-_ot;
        }
        if(mc_optime) mc_opcyc[7] += __rdtsc()-_lp0;
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
        engine.lazy_inverse=false;   // F-B path maintains the inverse frame live (snapshot tableau != live)
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
                // ---- Gate L: coherent rotations (arbitrary-theta dv; cap_theta applies xb sign + creates pending) ----
                case MO_ARRAY_ROT: cap_theta(a1,dv); break;
                case MO_EXPAND_ROT: { newq(a1); int q2=slot2id[a1]; if(fast){ ISKIP(ISK_INV_FWD, engine.h_inv(q2)); } else engine.h(q2); cap_theta(a1,dv); } break;
                case MO_ARRAY_SWAP: { int i_1=slot2id[a1], i_2=slot2id[a2]; slot2id[a1]=-1; slot2id[a2]=-1;
                    if(i_1>=0) slot2id[a2]=i_1; if(i_2>=0) slot2id[a1]=i_2; ISKIP(ISK_FRAME, frame.swap(a1,a2)); } break;
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
                case MO_MEAS_ACTIVE_DIAGONAL:                            // Gate L: coherent active measure (no swap; slot a1)
                case MO_MEAS_ACTIVE_INTERFERE: {                         // DIAGONAL: no H, xb; INTERFERE: H, zb
                    bool interfere = ((MdamOp)p.kind[i]==MO_MEAS_ACTIVE_INTERFERE);
                    int q=slot2id[a1]; if(q<0) break;
                    if(interfere){ if(fast){ ISKIP(ISK_INV_FWD, engine.h_inv(q)); } else engine.h(q); }   // boundary H (interfere only)
                    int b=fb_region;
                    if(compiling) fb_record_boundary(b, fb_phase_prev);
                    else if(shadow) fb_shadow_boundary(b, fb_phase_prev);
                    else if(fast) { ITIME_BEG(IT_REGION); fb_load_boundary(b); ITIME_END(IT_REGION); }
                    int bit=measure_z(q);
                    if(!fast) fb_capture_phase(fb_phase_prev);
                    fb_region++;
                    slot2id[a1]=-1;
                    int m_abs = bit ^ (interfere ? frame.zb(a1) : frame.xb(a1));
                    record.set((uint32_t)i0, m_abs^i1);
                    ISKIP(ISK_FRAME, frame.set_xz(a1,(uint8_t)m_abs,0));
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
        batch_lazy_hint = true; engine.magic_ever = false;     // adaptive: probe shot 0 in lazy, then decide
        for (uint64_t sh = 0; sh < num_shots; sh++) {
            uint64_t sd = master.bounded(RNG_EXCL);            // per-shot seed from master stream
            __uint128_t st, inc; SeedExpand::seedseq_pcg64(sd, st, inc);
            reset_shot(p, (uint64_t)(st >> 64), (uint64_t)st, (uint64_t)(inc >> 64), (uint64_t)inc);
            if (fb_mode != FB_OFF) run_fb(p); else run(p);
            // after shot 0, if this circuit materialized magic, eager is faster for the remaining shots.
            if (sh == 0 && lazy_env() == -1) batch_lazy_hint = !engine.magic_ever;
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

    // Phase-3: batch driver for run_mcache (same master-seed expansion as run_batch -> fair vs sample_batch).
    int run_mcache_batch(const MdamProgram& p, uint64_t num_shots,
                         uint64_t mshi, uint64_t mslo, uint64_t mihi, uint64_t milo,
                         uint8_t* out_record, char* out_err, int errlen) {
        NativeRng master; master.seed_from_state(mshi, mslo, mihi, milo);
        const uint64_t RNG_EXCL = ((uint64_t)1 << 63) - 1;
        const size_t nm = (size_t)p.num_measurements;
        for (uint64_t sh = 0; sh < num_shots; sh++) {
            uint64_t sd = master.bounded(RNG_EXCL);
            __uint128_t st, inc; SeedExpand::seedseq_pcg64(sd, st, inc);
            reset_shot(p, (uint64_t)(st >> 64), (uint64_t)st, (uint64_t)(inc >> 64), (uint64_t)inc);
            run_mcache(p);
            std::memcpy(out_record + (size_t)sh * nm, record.bits.data(), nm);
            if (err) { if(out_err){ std::strncpy(out_err, err, errlen-1); out_err[errlen-1]=0; } return 1; }
        }
        if (out_err) out_err[0] = 0;
        return 0;
    }

    // run_lean: one shot via the reduced lean walk (frame layer + automaton; NO engine gate-walk).
    void run_lean(const MdamProgram& p){
        size_t N=p.kind.size();
        if(mc_fblock && (!lfb_built || lfb_N!=N)) lfb_build(p);              // lean frame-block (batches FRAME_* + ARRAY_{CNOT,CZ,S,H})
        ln_cur_id=-1; ln_prev_out=-2;                                        // virtual start (== sg_prev_id/sg_prev_out)
        ln_incomplete=false; sg_seg_signs=1469598103934665603ULL; ln_active=true;
        for(size_t i=0;i<N && !err && !ln_incomplete;i++){
            if(mc_fblock && lfb_at[i]>=0){ if(!(mc_skip&1)) lfb_exec(lfb_at[i]); i += (size_t)lfb_len[lfb_at[i]]-1; continue; }
            switch((MdamOp)p.kind[i]){                                        // operands loaded lazily per-case (fewer SoA loads/op)
                case MO_FRAME_H: if(!(mc_skip&1)) frame.h(p.a1[i]); break;
                case MO_FRAME_CNOT: if(!(mc_skip&1)) frame.cnot(p.a1[i],p.a2[i]); break;
                case MO_FRAME_CZ: if(!(mc_skip&1)) frame.cz(p.a1[i],p.a2[i]); break;
                case MO_FRAME_SWAP: if(!(mc_skip&1)) frame.swap(p.a1[i],p.a2[i]); break;
                case MO_FRAME_S: if(!(mc_skip&1)) frame.s_gate(p.a1[i]); break;
                case MO_APPLY_PAULI: { int rc=record.get((uint32_t)p.i0[i]); if(rc==1) apply_mask(p.cp_masks[p.i1[i]]); } break;
                case MO_NOISE: if(!(mc_skip&8) && sampler.should_fire(p.i0[i])) sampler.apply_site(p.i0[i], p.noise_sites[p.i0[i]], frame); break;
                case MO_NOISE_BLOCK: if(!(mc_skip&8)){ int i0=p.i0[i],e=i0+p.i1[i]; for(int s=i0;s<e;s++){ if(sampler.should_fire(s)) sampler.apply_site(s, p.noise_sites[s], frame); } } break;
                case MO_READOUT_NOISE: if(!(mc_skip&8)){ if(udraw()<p.dval[i]) record.flip((uint32_t)p.i0[i]); } break;
                case MO_MEAS_DORM_STATIC: record.set((uint32_t)p.i0[i], frame.xb(p.a1[i])^p.i1[i]); break;
                case MO_MEAS_DORM_RANDOM: { int m=(int)idraw2(); record.set((uint32_t)p.i0[i], m^p.i1[i]); frame.set_xz(p.a1[i],(uint8_t)m,0); } break;
                case MO_ARRAY_CNOT: if(!(mc_skip&1)) frame.cnot(p.a1[i],p.a2[i]); break;         // engine.cx SKIPPED (dead on hit)
                case MO_ARRAY_CZ: if(!(mc_skip&1)) frame.cz(p.a1[i],p.a2[i]); break;
                case MO_MULTI_CNOT: if(!(mc_skip&1)){ int tgt=p.a1[i]; uint64_t mask=p.mmask[p.i0[i]];
                    while(mask){ int ctrl=__builtin_ctzll(mask); mask&=mask-1; if(ctrl==tgt) continue; frame.cnot(ctrl,tgt); } } break;
                case MO_MULTI_CZ: if(!(mc_skip&1)){ int a1=p.a1[i]; uint64_t mask=p.mmask[p.i0[i]];
                    while(mask){ int tgt=__builtin_ctzll(mask); mask&=mask-1; if(tgt==a1) continue; frame.cz(a1,tgt); } } break;
                case MO_ARRAY_S: if(!(mc_skip&1)) frame.s_gate(p.a1[i]); break;
                case MO_ARRAY_H: if(!(mc_skip&1)) frame.h(p.a1[i]); break;
                case MO_ARRAY_T: case MO_ARRAY_T_DAG: case MO_ARRAY_ROT: if(!(mc_skip&4)) lean_rot(p.a1[i]); break;   // sign only, pending SKIPPED
                case MO_EXPAND_T: case MO_EXPAND_T_DAG: case MO_EXPAND_ROT: { newq(p.a1[i]); if(!(mc_skip&4)) lean_rot(p.a1[i]); } break;  // engine.h SKIPPED
                case MO_EXPAND: { newq(p.a1[i]); } break;                     // engine.h SKIPPED (frame untouched)
                case MO_ARRAY_SWAP: { int a1=p.a1[i],a2=p.a2[i]; int i_1=slot2id[a1], i_2=slot2id[a2]; slot2id[a1]=-1; slot2id[a2]=-1;
                    if(i_1>=0) slot2id[a2]=i_1; if(i_2>=0) slot2id[a1]=i_2; if(!(mc_skip&1)) frame.swap(a1,a2); } break;
                case MO_SWAP_MEAS_INTERFERE: {
                    int a1=p.a1[i],a2=p.a2[i]; int i_1=slot2id[a1], i_2=slot2id[a2]; slot2id[a1]=-1; slot2id[a2]=-1;
                    if(i_1>=0) slot2id[a2]=i_1; if(i_2>=0) slot2id[a1]=i_2;
                    if(!(mc_skip&1)) frame.swap(a1,a2);
                    int q=slot2id[a2]; if(q<0) break;
                    int b=(mc_skip&32)?0:lean_measure(); if(b<0) break;
                    slot2id[a2]=-1;
                    int m_abs=b^frame.zb(a2); record.set((uint32_t)p.i0[i], m_abs^p.i1[i]); frame.set_xz(a2,(uint8_t)m_abs,0);
                } break;
                case MO_MEAS_ACTIVE_DIAGONAL: {
                    int a1=p.a1[i]; int q=slot2id[a1]; if(q<0) break;
                    int b=(mc_skip&32)?0:lean_measure(); if(b<0) break;
                    slot2id[a1]=-1;
                    int m_abs=b^frame.xb(a1); record.set((uint32_t)p.i0[i], m_abs^p.i1[i]); frame.set_xz(a1,(uint8_t)m_abs,0);
                } break;
                case MO_MEAS_ACTIVE_INTERFERE: {
                    int a1=p.a1[i]; int q=slot2id[a1]; if(q<0) break;
                    int b=(mc_skip&32)?0:lean_measure(); if(b<0) break;
                    slot2id[a1]=-1;
                    int m_abs=b^frame.zb(a1); record.set((uint32_t)p.i0[i], m_abs^p.i1[i]); frame.set_xz(a1,(uint8_t)m_abs,0);
                } break;
                case MO_ARRAY_U2: case MO_ARRAY_U4: ln_incomplete=true; break;  // non-structural; unsupported in lean
                case MO_END: default: break;
            }
        }
        ln_active=false;
        if(ln_incomplete) ln_incomplete_shots++;
    }
    // run_lean_batch: same master-seed expansion as run_batch/run_mcache_batch.  out_incomplete[sh]=1 if the
    // shot hit an uncached edge (excluded from the record comparison).  Uses the pre-warmed sg table.
    int run_lean_batch(const MdamProgram& p, uint64_t num_shots,
                       uint64_t mshi, uint64_t mslo, uint64_t mihi, uint64_t milo,
                       uint8_t* out_record, uint8_t* out_incomplete, char* out_err, int errlen){
        NativeRng master; master.seed_from_state(mshi, mslo, mihi, milo);
        const uint64_t RNG_EXCL = ((uint64_t)1 << 63) - 1;
        const size_t nm = (size_t)p.num_measurements;
        for (uint64_t sh = 0; sh < num_shots; sh++) {
            uint64_t sd = master.bounded(RNG_EXCL);
            __uint128_t st, inc; SeedExpand::seedseq_pcg64(sd, st, inc);
            reset_shot(p, (uint64_t)(st >> 64), (uint64_t)st, (uint64_t)(inc >> 64), (uint64_t)inc);
            run_lean(p);
            std::memcpy(out_record + (size_t)sh * nm, record.bits.data(), nm);
            if(out_incomplete) out_incomplete[sh] = ln_incomplete?1:0;
            if (err) { if(out_err){ std::strncpy(out_err, err, errlen-1); out_err[errlen-1]=0; } return 1; }
        }
        if (out_err) out_err[0] = 0;
        return 0;
    }

    // run_lean_fb_batch: FAST-PATH driver — lean walk with miss-fallback to run_mcache (full engine, bit-exact).
    // On an uncached edge run_lean aborts (ln_incomplete); we re-seed the SAME per-shot seed and replay the shot
    // through run_mcache so the record is exact.  For a saturating (warm) automaton the fallback ~never fires.
    // Uses the pre-warmed sg table + mc cache (mc_mode/fblock/rb set by caller).  ln_fb_count = #fallback shots.
    long ln_fb_count=0;
    int run_lean_fb_batch(const MdamProgram& p, uint64_t num_shots,
                          uint64_t mshi, uint64_t mslo, uint64_t mihi, uint64_t milo,
                          uint8_t* out_record, char* out_err, int errlen){
        NativeRng master; master.seed_from_state(mshi, mslo, mihi, milo);
        const uint64_t RNG_EXCL = ((uint64_t)1 << 63) - 1;
        const size_t nm = (size_t)p.num_measurements;
        for (uint64_t sh = 0; sh < num_shots; sh++) {
            uint64_t sd = master.bounded(RNG_EXCL);
            __uint128_t st, inc; SeedExpand::seedseq_pcg64(sd, st, inc);
            uint64_t shi=(uint64_t)(st>>64), slo=(uint64_t)st, ihi=(uint64_t)(inc>>64), ilo=(uint64_t)inc;
            reset_shot(p, shi,slo,ihi,ilo);
            run_lean(p);
            if(ln_incomplete){ ln_fb_count++; reset_shot(p, shi,slo,ihi,ilo); run_mcache(p); }
            std::memcpy(out_record + (size_t)sh * nm, record.bits.data(), nm);
            if (err) { if(out_err){ std::strncpy(out_err, err, errlen-1); out_err[errlen-1]=0; } return 1; }
        }
        if (out_err) out_err[0] = 0;
        return 0;
    }

    // ==== Adaptive bounded-regret executor (opt-in via run_lean_adapt_batch; default path unchanged) ====
    // lean optimistic start; conservative sticky demote to SLOW_ONLY (= run_mcache direct, NOT raw auth).
    // Runtime lazy cache only (no offline prefill) — every shot is a real output, cache builds inline.
    // Demote ONLY on (hard table/memory cap) OR (past horizon AND node_rate still above floor AND tail lean
    // cost > slow cost) sustained over ad_bad_needed windows.  node_rate floor (not absolute fb) is the key
    // signal: a slow-saturating lean winner (cult_d3) has node_rate -> ~0 past horizon so it is NEVER demoted;
    // a genuinely non-saturating circuit (cult_d5) keeps node_rate high.  Config via nvm_adapt_config.
    long   ad_window=4096, ad_node_cap=1000000, ad_edge_cap=4000000, ad_mem_cap=512L*1024*1024, ad_horizon=100000;
    double ad_node_floor=0.02, ad_cost_margin=1.10; int ad_bad_needed=3;
    // Fine-grained OOM-safety demote (checked every 64 shots, separate from the 4096 perf-window): if the cache
    // memory crosses ad_mem_cap AND the recent miss(fallback) rate is ~1.0, LEAN is pure waste on a heavy-core
    // circuit (d5_r5: 3.75MB/shot, fb=1.0) -> demote to AUTH BEFORE OOM.  Two independent gates each protect a
    // different LEAN winner: the mem gate spares small-core 100%-fb circuits (coherent_d3_r3, maxM=4); the fb gate
    // spares heavy-but-hitting circuits (cult_d5, fb~0.5<floor).  Cost(lean vs slow) does NOT separate them.
    double ad_fb_demote=0.95;
    int    ad_final_policy=0; long ad_demote_shot=-1, ad_windows=0, ad_slow_shots=0;   // filled by adapt batch
    double ad_node_rate_init=-1, ad_node_rate_last=-1, ad_lean_ns_last=-1, ad_slow_ns_last=-1, ad_fb_rate_last=-1;
    // EXACT, O(pool): stats only (called ~1x/run).
    size_t ad_mem_est() const { return ln_id.size()*64 + ln_edge.size()*48 + ln_p0v.size()*9
                                     + mc_pool_bytes(); }
    // O(1) memory-budget estimate for the shot loop: lean tables + the running dense-core byte counter
    // (mc_pool_bytes_live, maintained at the single mc_pool_intern site).  mc_pool dense is the dominant term,
    // so this tracks the exact ad_mem_est() closely (measured cult_d5: live vs exact within ~2%).
    size_t ad_mem_est_live() const { return ln_id.size()*64 + ln_edge.size()*48 + ln_p0v.size()*9
                                          + mc_pool_bytes_live; }

    int run_lean_adapt_batch(const MdamProgram& p, uint64_t num_shots,
                             uint64_t mshi, uint64_t mslo, uint64_t mihi, uint64_t milo,
                             uint8_t* out_record, char* out_err, int errlen){
        NativeRng master; master.seed_from_state(mshi, mslo, mihi, milo);
        const uint64_t RNG_EXCL = ((uint64_t)1 << 63) - 1;
        const size_t nm = (size_t)p.num_measurements;
        int policy = 0;                          // 0=LEAN, 1=AUTH (sticky).  run_mcache is NOT a policy:
        bool auth_first=false;                   //   it is only the LEAN-miss recovery fallback.  Demote => AUTH.
        engine.magic_ever=false;                 // high-water reset: track whether THIS batch ever materializes magic
        ad_final_policy=0; ad_demote_shot=-1; ad_windows=0; ad_slow_shots=0;
        ad_node_rate_init=-1; ad_node_rate_last=-1; ad_lean_ns_last=-1; ad_slow_ns_last=-1; ad_fb_rate_last=-1;
        long   w_fb0=ln_fb_count; size_t w_node0=ln_id.size(); long w_shots=0;
        size_t f_node0=ln_id.size();                 // fine-window (64-shot) node baseline for the memory-budget gate
        double w_t0=now_ns(), w_slow_sum=0; long w_slow_n=0; int bad_windows=0;
        // Demote = "cache won't close": stop growing it (sg_shadow off), free lean tables + dense-core mcache,
        // switch to AUTH (run(), constant memory).  run_mcache is NOT the demote target (it kept interning + is
        // slower than auth on localization).  Shared by the fine OOM-safety check and the perf-window decision.
        auto do_demote = [&](long at){
            policy=1; ad_demote_shot=at; sg_shadow=0;
            { std::unordered_map<uint64_t,int> a,b; ln_id.swap(a); ln_edge.swap(b); }
            { std::unordered_map<uint64_t,double> a; sg_p0.swap(a); }
            { std::unordered_map<uint64_t,uint8_t> a; sg_antis.swap(a); }
            { std::unordered_map<uint64_t,uint64_t> a; sg_trans.swap(a); }
            ln_p0v.clear(); ln_p0v.shrink_to_fit(); ln_antisv.clear(); ln_antisv.shrink_to_fit();
            mc_pool_free();
            batch_lazy_hint=true; engine.magic_ever=false; auth_first=true;   // clean AUTH lazy-probe
        };
        for (uint64_t sh=0; sh<num_shots; sh++){
            uint64_t sd=master.bounded(RNG_EXCL);
            __uint128_t st,inc; SeedExpand::seedseq_pcg64(sd,st,inc);
            uint64_t shi=(uint64_t)(st>>64),slo=(uint64_t)st,ihi=(uint64_t)(inc>>64),ilo=(uint64_t)inc;
            if(policy==1){                        // AUTH sticky: authoritative path (== sample_batch), no cache/shadow.
                reset_shot(p,shi,slo,ihi,ilo);
                if (fb_mode != FB_OFF) run_fb(p); else run(p);
                if (auth_first){ if (lazy_env()==-1) batch_lazy_hint = !engine.magic_ever; auth_first=false; }
                ad_slow_shots++;
            } else {                              // LEAN optimistic + miss fallback (builds cache lazily)
                reset_shot(p,shi,slo,ihi,ilo); run_lean(p);
                if(ln_incomplete){ ln_fb_count++; double ts=now_ns();
                    reset_shot(p,shi,slo,ihi,ilo); run_mcache(p);
                    w_slow_sum+=now_ns()-ts; w_slow_n++; }
            }
            std::memcpy(out_record+(size_t)sh*nm, record.bits.data(), nm);
            if(err){ if(out_err){ std::strncpy(out_err,err,errlen-1); out_err[errlen-1]=0; } return 1; }
            w_shots++;
            // MEMORY BUDGET (every 64 shots, O(1) via the live byte counter): a cache that has grown past the
            // budget AND is still non-saturating (new nodes in this 64-window) => AUTH.  No fb gate: this fires for
            // BOTH heavy-core all-miss (d5_r5, ~3.75MB/shot -> @191) AND light-core hitting-but-non-saturating
            // caches (cult_d5, ~0.029MB/shot -> demotes near the budget instead of ballooning to multi-GB).  A
            // saturating cache (node_rate~0) is spared even above budget; a small cache never reaches the budget.
            // Bounding the cache is what makes shot-parallel scaling safe (else memory = budget x workers).
            if(policy==0 && (sh & 63)==63){
                double nr_fine=(double)(ln_id.size()-f_node0)/64.0; f_node0=ln_id.size();
                if((long)ad_mem_est_live()>ad_mem_cap && nr_fine>ad_node_floor) do_demote((long)sh);
            }
            if(policy==0 && w_shots>=ad_window){  // window boundary: conservative demote decision
                double lean_ns=(now_ns()-w_t0)/(double)w_shots;
                size_t node_now=ln_id.size();
                double node_rate=(double)(node_now-w_node0)/(double)w_shots;
                double slow_ns=w_slow_n>0? w_slow_sum/(double)w_slow_n : 0.0;
                ad_windows++;
                if(ad_node_rate_init<0) ad_node_rate_init=node_rate;
                ad_node_rate_last=node_rate; ad_lean_ns_last=lean_ns; ad_slow_ns_last=slow_ns;
                ad_fb_rate_last=(double)(ln_fb_count-w_fb0)/(double)w_shots;
                // Node/edge COUNT cap removed: the memory budget (fine check above) counts lean-table bytes too,
                // so a node-table explosion (maxM=0 localization) trips the byte budget the same way -- and those
                // circuits are already caught earlier by loc_demote.  Memory is now the single OOM/size backstop.
                bool past_horizon=(long)sh>=ad_horizon;
                bool not_decaying=node_rate>ad_node_floor;      // sustained new-node growth => non-saturating
                bool cost_bad=slow_ns>0 && lean_ns>ad_cost_margin*slow_ns;
                if(past_horizon && not_decaying && cost_bad) bad_windows++; else bad_windows=0;
                // Early localization demote: a circuit that NEVER materialized magic (maxM=0, pure Clifford r<<k
                // localization) with ~all misses and a still-growing node table -> AUTH is trivially cheap and
                // the cache is pure waste.  Fires at the FIRST window (no long horizon needed): !magic_ever
                // excludes every magic circuit (cult_*, coherent_d3_r3); the fb gate excludes maxM=0 circuits
                // that DO saturate (surface, coherent_d3_r1: fb=0).  d7_r1/d5_r1 -> AUTH within one window.
                bool loc_demote = !engine.magic_ever && ad_fb_rate_last>ad_fb_demote && not_decaying;
                if(loc_demote || bad_windows>=ad_bad_needed) do_demote((long)sh);
                w_fb0=ln_fb_count; w_node0=ln_id.size(); w_shots=0; w_t0=now_ns(); w_slow_sum=0; w_slow_n=0;
            }
        }
        ad_final_policy=policy; if(out_err) out_err[0]=0; return 0;
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
