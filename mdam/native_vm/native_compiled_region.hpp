// native_compiled_region.hpp — Gate J Phase-2: Region-Compiled MDAM Sampler (compiler + shadow).
//
// Phase-1 proved the Pauli-frame layer is F2-parity-compilable.  This header builds the compiled
// artifact by SYMBOLIC EXECUTION (abstract interpretation) of the opcode stream over the dynamic-bit
// basis dynbits = (noise (site,channel) fired bits, dormant-random bits, readout-flip bits, magic
// outcomes), and a SHADOW evaluator that runs the authoritative VM, populates dynbits from concrete
// events, and compares every theta_sign / final-record bit against the compiled parity query.
//
// Phase 2A milestone: compiled frame queries reproduce the VM EXACTLY (mismatch=0); no live path is
// disabled, no speed claim.  Default native path is untouched (this header is only used by the Gate J
// shadow entry points, default-off).
#pragma once
#include <vector>
#include <cstdint>
#include <cstring>
#include <x86intrin.h>          // __rdtsc for the 2D-3 timing breakdown (default-off)
#include "native_mdam_shot.hpp"

namespace mdam {

// ---- GF(2) symbolic expression over the dynbit basis:  value = cst ^ parity(mask & dynbits) ----
static constexpr int JDYN_WORDS = 40;            // up to 2560 dynbits (cultivation_d3 uses 2466)
// Multiword packed signatures: theta_sig (bit r = rotation r's xb) and rec_sig (bit r = record r) span
// SIG_MAX_WORDS uint64 each, so nrot/record_cap > 64 are supported (coherent_d5_r1: nrot=129, rc=98).
static constexpr int SIG_MAX_WORDS = 16;         // up to 1024 rotations / record bits
#define JSIGBIT(sig, i) (((sig)[(i)>>6] >> ((i)&63)) & 1ULL)
struct SymBit {
    uint64_t m[JDYN_WORDS]; uint8_t cst;
    SymBit(){ clear(); }
    void clear(){ for(int i=0;i<JDYN_WORDS;i++) m[i]=0; cst=0; }
    void set_const(int c){ clear(); cst=(uint8_t)(c&1); }
    void set_dyn(int b){ clear(); m[b>>6]|=1ULL<<(b&63); }
    void xor_dyn(int b){ m[b>>6]^=1ULL<<(b&63); }
    void xor_in(const SymBit& o){ for(int i=0;i<JDYN_WORDS;i++) m[i]^=o.m[i]; cst^=o.cst; }
};

struct ParityQuery {
    uint64_t m[JDYN_WORDS]; uint8_t cst;
    void from(const SymBit& s){ for(int i=0;i<JDYN_WORDS;i++) m[i]=s.m[i]; cst=s.cst; }
};
inline int eval_parity(const ParityQuery& q, const uint64_t* dyn){
    int p=q.cst; for(int i=0;i<JDYN_WORDS;i++) p^=__builtin_popcountll(q.m[i]&dyn[i]); return p&1;
}

// ---- dynbit layout (built once from the program) ----
struct DynLayout {
    std::vector<int> noise_base;     // noise_base[site] = first dynbit for that site's channels
    int dormant_base=0, readout_base=0, outcome_base=0, ndyn=0;
    int n_noise=0;                   // total (site,channel) dynbits
};

// ---- the compiled program (Phase 2A: frame queries; magic/phase plans added in 2A+) ----
struct CompiledMdamProgram {
    DynLayout dyn;
    std::vector<ParityQuery> theta_q;       // per rotation opcode (ARRAY_T/T_DAG/EXPAND_T/T_DAG), in order
    std::vector<ParityQuery> final_rec_q;   // per record index, the final symbolic record bit
    int num_qubits=0, record_cap=0, nrot=0, nmagic=0;
    // ---- Gate J 2B: event-driven accumulation tables (TRANSPOSE of the queries) ----
    // Instead of evaluating each query as popcount(mask & dynbits) over all ndyn bits, the fast path
    // keeps a packed signature theta_sig (bit r = xb at rotation r) + rec_sig (bit r = record r) and,
    // when dynbit e fires, XORs the precomputed columns: theta_sig ^= ev_theta[e]; rec_sig ^= ev_rec[e].
    // Cost is O(fired events), not O(queries × ndyn).  (nrot, record_cap ≤ 64 here → uint64 sigs.)
    int theta_words=1, rec_words=1;         // ceil(nrot/64), ceil(record_cap/64)
    std::vector<uint64_t> theta_init, rec_init;   // constant parts (query .cst), theta_words/rec_words long
    std::vector<uint64_t> ev_theta, ev_rec; // per dynbit e: rotations / record bits it flips (flat e*words+w)
    bool fast_ok=false;                     // theta_words,rec_words <= SIG_MAX_WORDS (else fast path unsupported)
};

// transpose the compiled queries into per-dynbit contribution columns (built once, after compile).
inline void build_event_tables(CompiledMdamProgram& cp){
    int nrot=cp.nrot, rc=cp.record_cap, ndyn=cp.dyn.ndyn;
    int tw=(nrot+63)>>6, rw=(rc+63)>>6; if(tw<1) tw=1; if(rw<1) rw=1;
    cp.theta_words=tw; cp.rec_words=rw;
    cp.fast_ok = (tw<=SIG_MAX_WORDS && rw<=SIG_MAX_WORDS);
    cp.theta_init.assign(tw,0); for(int r=0;r<nrot;r++) if(cp.theta_q[r].cst) cp.theta_init[r>>6]|=1ULL<<(r&63);
    cp.rec_init.assign(rw,0);   for(int r=0;r<rc;r++)   if(cp.final_rec_q[r].cst) cp.rec_init[r>>6]|=1ULL<<(r&63);
    cp.ev_theta.assign((size_t)ndyn*tw,0); cp.ev_rec.assign((size_t)ndyn*rw,0);
    if(!cp.fast_ok) return;
    for(int r=0;r<nrot;r++){ const ParityQuery& q=cp.theta_q[r];
        for(int w=0;w<JDYN_WORDS;w++){ uint64_t m=q.m[w]; while(m){ int b=__builtin_ctzll(m); m&=m-1;
            int e=(w<<6)+b; if(e<ndyn) cp.ev_theta[(size_t)e*tw+(r>>6)]|=1ULL<<(r&63); } } }
    for(int r=0;r<rc;r++){ const ParityQuery& q=cp.final_rec_q[r];
        for(int w=0;w<JDYN_WORDS;w++){ uint64_t m=q.m[w]; while(m){ int b=__builtin_ctzll(m); m&=m-1;
            int e=(w<<6)+b; if(e<ndyn) cp.ev_rec[(size_t)e*rw+(r>>6)]|=1ULL<<(r&63); } } }
}

// ============================ symbolic compiler ============================
// Abstract-interpret the opcode stream: maintain symbolic frame fx[slot]/fz[slot] and record frec[idx]
// over the dynbit basis, mirroring run() exactly (frame ops only; engine/dense ops are not symbolic).
inline CompiledMdamProgram compile_jprogram(const MdamProgram& p) {
    CompiledMdamProgram cp; cp.num_qubits=p.num_qubits; cp.record_cap=p.record_cap;
    int NS=(int)p.noise_sites.size();
    cp.dyn.noise_base.assign(NS,0); int base=0;
    for(int s=0;s<NS;s++){ cp.dyn.noise_base[s]=base; base+=(int)p.noise_sites[s].channels.size(); }
    cp.dyn.n_noise=base;
    int ndorm=0,nread=0,nmag=0;
    for(uint8_t k:p.kind){ if(k==MO_MEAS_DORM_RANDOM)ndorm++; else if(k==MO_READOUT_NOISE)nread++;
        else if(k==MO_SWAP_MEAS_INTERFERE||k==MO_MEAS_ACTIVE_DIAGONAL||k==MO_MEAS_ACTIVE_INTERFERE)nmag++; }
    cp.dyn.dormant_base=base; base+=ndorm;
    cp.dyn.readout_base=base; base+=nread;
    cp.dyn.outcome_base=base; base+=nmag;
    cp.dyn.ndyn=base; cp.nmagic=nmag;

    // SymBit packs the dynbit basis into a FIXED uint64_t m[JDYN_WORDS].  If a program has more dynbits
    // than that holds (e.g. cultivation_d5: ndyn=16056 needs 251 words), the abstract-interp xor_dyn()
    // below would write past m[] and corrupt the heap.  Refuse to compile (fast_ok=false) and bail BEFORE
    // any OOB write; callers gate on fast_ok and fall back to the authoritative path.
    if (base > JDYN_WORDS*64) { cp.fast_ok=false; cp.nrot=0; return cp; }

    int NQ=p.num_qubits, RC=p.record_cap;
    std::vector<SymBit> fx(NQ), fz(NQ), frec(RC);   // all start at const 0
    auto noise_site=[&](int s){ const NoiseSite& st=p.noise_sites[s]; int b=cp.dyn.noise_base[s];
        for(size_t c=0;c<st.channels.size();c++){ const NoiseChannel& ch=st.channels[c];
            for(size_t wi=0;wi<ch.x_words.size();wi++){ uint64_t w=ch.x_words[wi];
                while(w){ int bit=__builtin_ctzll(w); w&=w-1; int q=(int)(wi*64+bit); if(q<NQ) fx[q].xor_dyn(b+(int)c); } }
            for(size_t wi=0;wi<ch.z_words.size();wi++){ uint64_t w=ch.z_words[wi];
                while(w){ int bit=__builtin_ctzll(w); w&=w-1; int q=(int)(wi*64+bit); if(q<NQ) fz[q].xor_dyn(b+(int)c); } } } };
    int dorm_i=0, read_i=0, mag_i=0, rot_i=0;
    size_t N=p.kind.size();
    for(size_t i=0;i<N;i++){
        int a1=p.a1[i], a2=p.a2[i], i0=p.i0[i], i1=p.i1[i];
        switch((MdamOp)p.kind[i]){
            case MO_FRAME_H: std::swap(fx[a1],fz[a1]); break;
            case MO_FRAME_CNOT: fx[a2].xor_in(fx[a1]); fz[a1].xor_in(fz[a2]); break;
            case MO_FRAME_CZ: fz[a1].xor_in(fx[a2]); fz[a2].xor_in(fx[a1]); break;
            case MO_FRAME_SWAP: std::swap(fx[a1],fx[a2]); std::swap(fz[a1],fz[a2]); break;
            case MO_FRAME_S: fz[a1].xor_in(fx[a1]); break;
            case MO_APPLY_PAULI: { const NoiseSite& cm=p.cp_masks[i1]; if(cm.channels.empty()) break;
                const NoiseChannel& ch=cm.channels[0];
                for(size_t wi=0;wi<ch.x_words.size();wi++){ uint64_t w=ch.x_words[wi]; while(w){ int bit=__builtin_ctzll(w); w&=w-1; int q=(int)(wi*64+bit); if(q<NQ) fx[q].xor_in(frec[i0]); } }
                for(size_t wi=0;wi<ch.z_words.size();wi++){ uint64_t w=ch.z_words[wi]; while(w){ int bit=__builtin_ctzll(w); w&=w-1; int q=(int)(wi*64+bit); if(q<NQ) fz[q].xor_in(frec[i0]); } } } break;
            case MO_NOISE: noise_site(i0); break;
            case MO_NOISE_BLOCK: for(int s=i0;s<i0+i1;s++) noise_site(s); break;
            case MO_READOUT_NOISE: { int b=cp.dyn.readout_base+(read_i++); frec[i0].xor_dyn(b); } break;
            case MO_MEAS_DORM_STATIC: { frec[i0]=fx[a1]; frec[i0].cst^=(uint8_t)(i1&1); } break;
            case MO_MEAS_DORM_RANDOM: { int b=cp.dyn.dormant_base+(dorm_i++);
                frec[i0].set_dyn(b); frec[i0].cst^=(uint8_t)(i1&1);
                fx[a1].set_dyn(b); fz[a1].set_const(0); } break;
            case MO_ARRAY_CNOT: fx[a2].xor_in(fx[a1]); fz[a1].xor_in(fz[a2]); break;
            case MO_ARRAY_CZ: fz[a1].xor_in(fx[a2]); fz[a2].xor_in(fx[a1]); break;
            case MO_MULTI_CNOT: { int tgt=a1; uint64_t mask=p.mmask[i0];
                while(mask){ int ctrl=__builtin_ctzll(mask); mask&=mask-1; if(ctrl==tgt) continue; fx[tgt].xor_in(fx[ctrl]); fz[ctrl].xor_in(fz[tgt]); } } break;
            case MO_MULTI_CZ: { uint64_t mask=p.mmask[i0];
                while(mask){ int tgt=__builtin_ctzll(mask); mask&=mask-1; if(tgt==a1) continue; fz[a1].xor_in(fx[tgt]); fz[tgt].xor_in(fx[a1]); } } break;
            case MO_ARRAY_T: case MO_ARRAY_T_DAG:
            case MO_ARRAY_ROT: { cp.theta_q.resize(rot_i+1); cp.theta_q[rot_i].from(fx[a1]); rot_i++; } break;
            case MO_ARRAY_S: fz[a1].xor_in(fx[a1]); break;
            case MO_EXPAND_T: case MO_EXPAND_T_DAG:
            case MO_EXPAND_ROT: { cp.theta_q.resize(rot_i+1); cp.theta_q[rot_i].from(fx[a1]); rot_i++; } break;
            case MO_ARRAY_SWAP: std::swap(fx[a1],fx[a2]); std::swap(fz[a1],fz[a2]); break;  // pure relabel
            case MO_SWAP_MEAS_INTERFERE: { std::swap(fx[a1],fx[a2]); std::swap(fz[a1],fz[a2]);
                int b=cp.dyn.outcome_base+(mag_i++);
                SymBit mabs; mabs.set_dyn(b); mabs.xor_in(fz[a2]);   // m_abs = outcome ^ frame.zb(a2)
                frec[i0]=mabs; frec[i0].cst^=(uint8_t)(i1&1);
                fx[a2]=mabs; fz[a2].set_const(0); } break;
            case MO_MEAS_ACTIVE_DIAGONAL: {                          // m_abs = outcome ^ frame.xb(a1)
                int b=cp.dyn.outcome_base+(mag_i++);
                SymBit mabs; mabs.set_dyn(b); mabs.xor_in(fx[a1]);
                frec[i0]=mabs; frec[i0].cst^=(uint8_t)(i1&1);
                fx[a1]=mabs; fz[a1].set_const(0); } break;
            case MO_MEAS_ACTIVE_INTERFERE: {                         // m_abs = outcome ^ frame.zb(a1)
                int b=cp.dyn.outcome_base+(mag_i++);
                SymBit mabs; mabs.set_dyn(b); mabs.xor_in(fz[a1]);
                frec[i0]=mabs; frec[i0].cst^=(uint8_t)(i1&1);
                fx[a1]=mabs; fz[a1].set_const(0); } break;
            case MO_ARRAY_H: std::swap(fx[a1],fz[a1]); break;   // active Hadamard: frame H is symbolic (fx<->fz); engine H is not symbolic (handled in run_jfast_2e/shadow)
            default: break;
        }
    }
    cp.nrot=rot_i;
    cp.final_rec_q.resize(RC); for(int r=0;r<RC;r++) cp.final_rec_q[r].from(frec[r]);
    build_event_tables(cp);
    return cp;
}

// ============================ shadow evaluator ============================
// Run the authoritative VM (mirror of MdamShot::run) while populating dynbits from concrete events and
// comparing each theta_sign (frame.xb at a rotation) and each final-record bit to the compiled query.
struct JShadowStats { long theta_checks=0, theta_mismatch=0, rec_checks=0, rec_mismatch=0;
    long opcode_dispatch=0, frame_fwd=0; int first_bad_rot=-1, first_bad_rec=-1; };

inline void run_jshadow(MdamShot& s, const MdamProgram& p, const CompiledMdamProgram& cp, JShadowStats& st) {
    std::vector<uint64_t> dyn((size_t)JDYN_WORDS, 0);
    s.sampler.log_on=true; s.sampler.fire_log.clear(); size_t fire_cur=0;
    // map a fired (site,xw,zw) -> dynbit
    auto set_noise=[&](int site,uint64_t ci,uint64_t){ int bdyn=cp.dyn.noise_base[site]+(int)ci; dyn[bdyn>>6]|=1ULL<<(bdyn&63); };   // fire_log stores the channel index (multiword-safe)
    auto drain_fires=[&](){ for(;fire_cur<s.sampler.fire_log.size();++fire_cur){ auto&f=s.sampler.fire_log[fire_cur]; set_noise((int)f[0],f[1],f[2]); } };
    int dorm_i=0, read_i=0, mag_i=0, rot_i=0;
    size_t N=p.kind.size();
    for(size_t i=0;i<N && !s.err;i++){
        st.opcode_dispatch++;
        int a1=p.a1[i], a2=p.a2[i], i0=p.i0[i], i1=p.i1[i]; double dv=p.dval[i];
        switch((MdamOp)p.kind[i]){
            case MO_FRAME_H: s.frame.h(a1); st.frame_fwd++; break;
            case MO_FRAME_CNOT: s.frame.cnot(a1,a2); st.frame_fwd++; break;
            case MO_FRAME_CZ: s.frame.cz(a1,a2); st.frame_fwd++; break;
            case MO_FRAME_SWAP: s.frame.swap(a1,a2); st.frame_fwd++; break;
            case MO_FRAME_S: s.frame.s_gate(a1); st.frame_fwd++; break;
            case MO_APPLY_PAULI: { int rc=s.record.get((uint32_t)i0); if(rc==1) s.apply_mask(p.cp_masks[i1]); } break;
            case MO_NOISE: s.sampler.apply_site(i0, p.noise_sites[i0], s.frame); drain_fires(); break;
            case MO_NOISE_BLOCK: for(int si=i0;si<i0+i1;si++) s.sampler.apply_site(si, p.noise_sites[si], s.frame); drain_fires(); break;
            case MO_READOUT_NOISE: { int b=cp.dyn.readout_base+(read_i++); if(s.udraw()<dv){ s.record.flip((uint32_t)i0); dyn[b>>6]|=1ULL<<(b&63);} } break;
            case MO_MEAS_DORM_STATIC: s.record.set((uint32_t)i0, s.frame.xb(a1)^i1); break;
            case MO_MEAS_DORM_RANDOM: { int m=(int)s.idraw2(); int b=cp.dyn.dormant_base+(dorm_i++); if(m) dyn[b>>6]|=1ULL<<(b&63);
                s.record.set((uint32_t)i0, m^i1); s.frame.set_xz(a1,(uint8_t)m,0); } break;
            case MO_ARRAY_CNOT: { int u=s.slot2id[a1], v=s.slot2id[a2]; if(u>=0&&v>=0) s.engine.cx(u,v); s.frame.cnot(a1,a2); } break;
            case MO_ARRAY_CZ: { int u=s.slot2id[a1], v=s.slot2id[a2]; if(u>=0&&v>=0) s.engine.cz(u,v); s.frame.cz(a1,a2); } break;
            case MO_MULTI_CNOT: { int tgt=a1, t=s.slot2id[tgt]; uint64_t mask=p.mmask[i0];
                while(mask){ int ctrl=__builtin_ctzll(mask); mask&=mask-1; if(ctrl==tgt) continue; int c=s.slot2id[ctrl]; if(t>=0&&c>=0) s.engine.cx(c,t); s.frame.cnot(ctrl,tgt); } } break;
            case MO_MULTI_CZ: { uint64_t mask=p.mmask[i0];
                while(mask){ int tgt=__builtin_ctzll(mask); mask&=mask-1; if(tgt==a1) continue; int u=s.slot2id[a1], v=s.slot2id[tgt]; if(u>=0&&v>=0) s.engine.cz(u,v); s.frame.cz(a1,tgt); } } break;
            case MO_ARRAY_T: { int conc=s.frame.xb(a1); int comp=eval_parity(cp.theta_q[rot_i],dyn.data());
                st.theta_checks++; if(conc!=comp){ st.theta_mismatch++; if(st.first_bad_rot<0)st.first_bad_rot=rot_i; } rot_i++;
                s.rot(p,a1,NV_T_ANGLE); } break;
            case MO_ARRAY_T_DAG: { int conc=s.frame.xb(a1); int comp=eval_parity(cp.theta_q[rot_i],dyn.data());
                st.theta_checks++; if(conc!=comp){ st.theta_mismatch++; if(st.first_bad_rot<0)st.first_bad_rot=rot_i; } rot_i++;
                s.rot(p,a1,-NV_T_ANGLE); } break;
            case MO_ARRAY_S: { int q=s.slot2id[a1]; if(q>=0) s.engine.s(q,false); s.frame.s_gate(a1); } break;
            case MO_EXPAND_T: { s.newq(a1); s.engine.h(s.slot2id[a1]); int conc=s.frame.xb(a1); int comp=eval_parity(cp.theta_q[rot_i],dyn.data());
                st.theta_checks++; if(conc!=comp){ st.theta_mismatch++; if(st.first_bad_rot<0)st.first_bad_rot=rot_i; } rot_i++;
                s.rot(p,a1,NV_T_ANGLE); } break;
            case MO_EXPAND_T_DAG: { s.newq(a1); s.engine.h(s.slot2id[a1]); int conc=s.frame.xb(a1); int comp=eval_parity(cp.theta_q[rot_i],dyn.data());
                st.theta_checks++; if(conc!=comp){ st.theta_mismatch++; if(st.first_bad_rot<0)st.first_bad_rot=rot_i; } rot_i++;
                s.rot(p,a1,-NV_T_ANGLE); } break;
            case MO_SWAP_MEAS_INTERFERE: {
                int i_1=s.slot2id[a1], i_2=s.slot2id[a2]; s.slot2id[a1]=-1; s.slot2id[a2]=-1;
                if(i_1>=0) s.slot2id[a2]=i_1; if(i_2>=0) s.slot2id[a1]=i_2;
                s.frame.swap(a1,a2);
                int q=s.slot2id[a2]; if(q<0) break;
                s.engine.h(q); int b=s.measure_z(q);
                int bdyn=cp.dyn.outcome_base+(mag_i++); if(b) dyn[bdyn>>6]|=1ULL<<(bdyn&63);
                s.slot2id[a2]=-1; int m_abs=b ^ s.frame.zb(a2);
                s.record.set((uint32_t)i0, m_abs^i1); s.frame.set_xz(a2,(uint8_t)m_abs,0);
            } break;
            // ---- distillation / coherent dialect: mirror MdamShot::run authoritative cases ----
            case MO_ARRAY_ROT: { int conc=s.frame.xb(a1); int comp=eval_parity(cp.theta_q[rot_i],dyn.data());
                st.theta_checks++; if(conc!=comp){ st.theta_mismatch++; if(st.first_bad_rot<0)st.first_bad_rot=rot_i; } rot_i++;
                s.rot(p,a1,dv); } break;
            case MO_EXPAND_ROT: { s.newq(a1); s.engine.h(s.slot2id[a1]); int conc=s.frame.xb(a1); int comp=eval_parity(cp.theta_q[rot_i],dyn.data());
                st.theta_checks++; if(conc!=comp){ st.theta_mismatch++; if(st.first_bad_rot<0)st.first_bad_rot=rot_i; } rot_i++;
                s.rot(p,a1,dv); } break;
            case MO_ARRAY_SWAP: { int i_1=s.slot2id[a1], i_2=s.slot2id[a2]; s.slot2id[a1]=-1; s.slot2id[a2]=-1;
                if(i_1>=0) s.slot2id[a2]=i_1; if(i_2>=0) s.slot2id[a1]=i_2; s.frame.swap(a1,a2); } break;
            case MO_MEAS_ACTIVE_DIAGONAL: { int q=s.slot2id[a1]; int bdyn=cp.dyn.outcome_base+(mag_i++); if(q<0) break;
                int b=s.measure_z(q); if(b) dyn[bdyn>>6]|=1ULL<<(bdyn&63);
                s.slot2id[a1]=-1; int m_abs=b ^ s.frame.xb(a1);
                s.record.set((uint32_t)i0, m_abs^i1); s.frame.set_xz(a1,(uint8_t)m_abs,0); } break;
            case MO_MEAS_ACTIVE_INTERFERE: { int q=s.slot2id[a1]; int bdyn=cp.dyn.outcome_base+(mag_i++); if(q<0) break;
                s.engine.h(q); int b=s.measure_z(q); if(b) dyn[bdyn>>6]|=1ULL<<(bdyn&63);
                s.slot2id[a1]=-1; int m_abs=b ^ s.frame.zb(a1);
                s.record.set((uint32_t)i0, m_abs^i1); s.frame.set_xz(a1,(uint8_t)m_abs,0); } break;
            case MO_EXPAND: { s.newq(a1); s.engine.h(s.slot2id[a1]); } break;
            case MO_ARRAY_H: { int q=s.slot2id[a1]; if(q>=0) s.engine.h(q); s.frame.h(a1); } break;
            default: break;
        }
    }
    s.sampler.log_on=false;
    // final record comparison
    for(int r=0;r<p.num_measurements && r<cp.record_cap;r++){
        int conc=s.record.get((uint32_t)r); int comp=eval_parity(cp.final_rec_q[r],dyn.data());
        st.rec_checks++; if(conc!=comp){ st.rec_mismatch++; if(st.first_bad_rec<0)st.first_bad_rec=r; }
    }
}

// DEBUG: run authoritative NativeFrame and the symbolic SymBit frame in PARALLEL for one shot,
// comparing eval_parity(fx/fz[slot],dyn) vs s.frame.xb/zb(slot) after EVERY opcode.  Reports the FIRST
// opcode where they diverge (out[0]=opno, out[1]=slot, out[2]=kind, out[3]=is_z, out[4]=found).  This
// isolates which symbolic frame update in compile_jprogram disagrees with the live NativeFrame.
inline void frame_first_divergence(MdamShot& s, const MdamProgram& p, const CompiledMdamProgram& cp, long* out){
    out[0]=out[1]=out[2]=out[3]=out[4]=-1; out[4]=0;
    std::vector<uint64_t> dyn((size_t)JDYN_WORDS,0);
    int NQ=p.num_qubits, RC=p.record_cap;
    std::vector<SymBit> fx(NQ), fz(NQ), frec(RC);
    s.sampler.log_on=true; s.sampler.fire_log.clear(); size_t fire_cur=0;
    auto set_noise=[&](int site,uint64_t ci,uint64_t){ int bdyn=cp.dyn.noise_base[site]+(int)ci; dyn[bdyn>>6]|=1ULL<<(bdyn&63); };   // fire_log stores the channel index (multiword-safe)
    auto drain_fires=[&](){ for(;fire_cur<s.sampler.fire_log.size();++fire_cur){ auto&f=s.sampler.fire_log[fire_cur]; set_noise((int)f[0],f[1],f[2]); } };
    auto sym_noise=[&](int site){ const NoiseSite& st=p.noise_sites[site]; int b=cp.dyn.noise_base[site];
        for(size_t c=0;c<st.channels.size();c++){ const NoiseChannel& ch=st.channels[c];
            for(size_t wi=0;wi<ch.x_words.size();wi++){ uint64_t w=ch.x_words[wi]; while(w){ int bit=__builtin_ctzll(w); w&=w-1; int q=(int)(wi*64+bit); if(q<NQ) fx[q].xor_dyn(b+(int)c); } }
            for(size_t wi=0;wi<ch.z_words.size();wi++){ uint64_t w=ch.z_words[wi]; while(w){ int bit=__builtin_ctzll(w); w&=w-1; int q=(int)(wi*64+bit); if(q<NQ) fz[q].xor_dyn(b+(int)c); } } } };
    int dorm_i=0, read_i=0, mag_i=0, rot_i=0, dorm_i2=0, read_i2=0, mag_i2=0;
    size_t N=p.kind.size();
    for(size_t i=0;i<N && !s.err;i++){
        int a1=p.a1[i], a2=p.a2[i], i0=p.i0[i], i1=p.i1[i]; double dv=p.dval[i];
        MdamOp k=(MdamOp)p.kind[i];
        // ---- authoritative (mirror run_jshadow) ----
        switch(k){
            case MO_FRAME_H: s.frame.h(a1); break;
            case MO_FRAME_CNOT: s.frame.cnot(a1,a2); break;
            case MO_FRAME_CZ: s.frame.cz(a1,a2); break;
            case MO_FRAME_SWAP: s.frame.swap(a1,a2); break;
            case MO_FRAME_S: s.frame.s_gate(a1); break;
            case MO_APPLY_PAULI: { int rc=s.record.get((uint32_t)i0); if(rc==1) s.apply_mask(p.cp_masks[i1]); } break;
            case MO_NOISE: s.sampler.apply_site(i0, p.noise_sites[i0], s.frame); drain_fires(); break;
            case MO_NOISE_BLOCK: for(int si=i0;si<i0+i1;si++) s.sampler.apply_site(si, p.noise_sites[si], s.frame); drain_fires(); break;
            case MO_READOUT_NOISE: { int b=cp.dyn.readout_base+(read_i++); if(s.udraw()<dv){ s.record.flip((uint32_t)i0); dyn[b>>6]|=1ULL<<(b&63);} } break;
            case MO_MEAS_DORM_STATIC: s.record.set((uint32_t)i0, s.frame.xb(a1)^i1); break;
            case MO_MEAS_DORM_RANDOM: { int m=(int)s.idraw2(); int b=cp.dyn.dormant_base+(dorm_i++); if(m) dyn[b>>6]|=1ULL<<(b&63);
                s.record.set((uint32_t)i0,m^i1); s.frame.set_xz(a1,(uint8_t)m,0); } break;
            case MO_ARRAY_CNOT: { int u=s.slot2id[a1],v=s.slot2id[a2]; if(u>=0&&v>=0) s.engine.cx(u,v); s.frame.cnot(a1,a2); } break;
            case MO_ARRAY_CZ: { int u=s.slot2id[a1],v=s.slot2id[a2]; if(u>=0&&v>=0) s.engine.cz(u,v); s.frame.cz(a1,a2); } break;
            case MO_MULTI_CNOT: { int tgt=a1,t=s.slot2id[tgt]; uint64_t mask=p.mmask[i0];
                while(mask){ int ctrl=__builtin_ctzll(mask); mask&=mask-1; if(ctrl==tgt) continue; int c=s.slot2id[ctrl]; if(t>=0&&c>=0) s.engine.cx(c,t); s.frame.cnot(ctrl,tgt); } } break;
            case MO_MULTI_CZ: { uint64_t mask=p.mmask[i0];
                while(mask){ int tgt=__builtin_ctzll(mask); mask&=mask-1; if(tgt==a1) continue; int u=s.slot2id[a1],v=s.slot2id[tgt]; if(u>=0&&v>=0) s.engine.cz(u,v); s.frame.cz(a1,tgt); } } break;
            case MO_ARRAY_T: rot_i++; s.rot(p,a1,NV_T_ANGLE); break;
            case MO_ARRAY_T_DAG: rot_i++; s.rot(p,a1,-NV_T_ANGLE); break;
            case MO_ARRAY_S: { int q=s.slot2id[a1]; if(q>=0) s.engine.s(q,false); s.frame.s_gate(a1); } break;
            case MO_EXPAND_T: { s.newq(a1); s.engine.h(s.slot2id[a1]); rot_i++; s.rot(p,a1,NV_T_ANGLE); } break;
            case MO_EXPAND_T_DAG: { s.newq(a1); s.engine.h(s.slot2id[a1]); rot_i++; s.rot(p,a1,-NV_T_ANGLE); } break;
            case MO_ARRAY_ROT: rot_i++; s.rot(p,a1,dv); break;
            case MO_EXPAND_ROT: { s.newq(a1); s.engine.h(s.slot2id[a1]); rot_i++; s.rot(p,a1,dv); } break;
            case MO_ARRAY_SWAP: { int i_1=s.slot2id[a1],i_2=s.slot2id[a2]; s.slot2id[a1]=-1; s.slot2id[a2]=-1; if(i_1>=0) s.slot2id[a2]=i_1; if(i_2>=0) s.slot2id[a1]=i_2; s.frame.swap(a1,a2); } break;
            case MO_SWAP_MEAS_INTERFERE: { int i_1=s.slot2id[a1],i_2=s.slot2id[a2]; s.slot2id[a1]=-1; s.slot2id[a2]=-1;
                if(i_1>=0) s.slot2id[a2]=i_1; if(i_2>=0) s.slot2id[a1]=i_2; s.frame.swap(a1,a2);
                int q=s.slot2id[a2]; int bdyn=cp.dyn.outcome_base+(mag_i++); if(q<0) break;
                s.engine.h(q); int b=s.measure_z(q); if(b) dyn[bdyn>>6]|=1ULL<<(bdyn&63);
                s.slot2id[a2]=-1; int m_abs=b ^ s.frame.zb(a2); s.record.set((uint32_t)i0,m_abs^i1); s.frame.set_xz(a2,(uint8_t)m_abs,0); } break;
            case MO_MEAS_ACTIVE_DIAGONAL: { int q=s.slot2id[a1]; int bdyn=cp.dyn.outcome_base+(mag_i++); if(q<0) break;
                int b=s.measure_z(q); if(b) dyn[bdyn>>6]|=1ULL<<(bdyn&63);
                s.slot2id[a1]=-1; int m_abs=b ^ s.frame.xb(a1); s.record.set((uint32_t)i0,m_abs^i1); s.frame.set_xz(a1,(uint8_t)m_abs,0); } break;
            case MO_MEAS_ACTIVE_INTERFERE: { int q=s.slot2id[a1]; int bdyn=cp.dyn.outcome_base+(mag_i++); if(q<0) break;
                s.engine.h(q); int b=s.measure_z(q); if(b) dyn[bdyn>>6]|=1ULL<<(bdyn&63);
                s.slot2id[a1]=-1; int m_abs=b ^ s.frame.zb(a1); s.record.set((uint32_t)i0,m_abs^i1); s.frame.set_xz(a1,(uint8_t)m_abs,0); } break;
            case MO_EXPAND: { s.newq(a1); s.engine.h(s.slot2id[a1]); } break;
            case MO_ARRAY_H: { int q=s.slot2id[a1]; if(q>=0) s.engine.h(q); s.frame.h(a1); } break;
            default: break;
        }
        // ---- symbolic (mirror compile_jprogram) ----
        switch(k){
            case MO_FRAME_H: std::swap(fx[a1],fz[a1]); break;
            case MO_FRAME_CNOT: fx[a2].xor_in(fx[a1]); fz[a1].xor_in(fz[a2]); break;
            case MO_FRAME_CZ: fz[a1].xor_in(fx[a2]); fz[a2].xor_in(fx[a1]); break;
            case MO_FRAME_SWAP: std::swap(fx[a1],fx[a2]); std::swap(fz[a1],fz[a2]); break;
            case MO_FRAME_S: fz[a1].xor_in(fx[a1]); break;
            case MO_APPLY_PAULI: { const NoiseSite& cm=p.cp_masks[i1]; if(cm.channels.empty()) break; const NoiseChannel& ch=cm.channels[0];
                for(size_t wi=0;wi<ch.x_words.size();wi++){ uint64_t w=ch.x_words[wi]; while(w){ int bit=__builtin_ctzll(w); w&=w-1; int q=(int)(wi*64+bit); if(q<NQ) fx[q].xor_in(frec[i0]); } }
                for(size_t wi=0;wi<ch.z_words.size();wi++){ uint64_t w=ch.z_words[wi]; while(w){ int bit=__builtin_ctzll(w); w&=w-1; int q=(int)(wi*64+bit); if(q<NQ) fz[q].xor_in(frec[i0]); } } } break;
            case MO_NOISE: sym_noise(i0); break;
            case MO_NOISE_BLOCK: for(int si=i0;si<i0+i1;si++) sym_noise(si); break;
            case MO_READOUT_NOISE: { int b=cp.dyn.readout_base+(read_i2++); frec[i0].xor_dyn(b); } break;
            case MO_MEAS_DORM_STATIC: { frec[i0]=fx[a1]; frec[i0].cst^=(uint8_t)(i1&1); } break;
            case MO_MEAS_DORM_RANDOM: { int b=cp.dyn.dormant_base+(dorm_i2++); frec[i0].set_dyn(b); frec[i0].cst^=(uint8_t)(i1&1); fx[a1].set_dyn(b); fz[a1].set_const(0); } break;
            case MO_ARRAY_CNOT: fx[a2].xor_in(fx[a1]); fz[a1].xor_in(fz[a2]); break;
            case MO_ARRAY_CZ: fz[a1].xor_in(fx[a2]); fz[a2].xor_in(fx[a1]); break;
            case MO_MULTI_CNOT: { int tgt=a1; uint64_t mask=p.mmask[i0]; while(mask){ int ctrl=__builtin_ctzll(mask); mask&=mask-1; if(ctrl==tgt) continue; fx[tgt].xor_in(fx[ctrl]); fz[ctrl].xor_in(fz[tgt]); } } break;
            case MO_MULTI_CZ: { uint64_t mask=p.mmask[i0]; while(mask){ int tgt=__builtin_ctzll(mask); mask&=mask-1; if(tgt==a1) continue; fz[a1].xor_in(fx[tgt]); fz[tgt].xor_in(fx[a1]); } } break;
            case MO_ARRAY_S: fz[a1].xor_in(fx[a1]); break;
            case MO_ARRAY_SWAP: std::swap(fx[a1],fx[a2]); std::swap(fz[a1],fz[a2]); break;
            case MO_SWAP_MEAS_INTERFERE: { std::swap(fx[a1],fx[a2]); std::swap(fz[a1],fz[a2]); int b=cp.dyn.outcome_base+(mag_i2++);
                SymBit mabs; mabs.set_dyn(b); mabs.xor_in(fz[a2]); frec[i0]=mabs; frec[i0].cst^=(uint8_t)(i1&1); fx[a2]=mabs; fz[a2].set_const(0); } break;
            case MO_MEAS_ACTIVE_DIAGONAL: { int b=cp.dyn.outcome_base+(mag_i2++); SymBit mabs; mabs.set_dyn(b); mabs.xor_in(fx[a1]);
                frec[i0]=mabs; frec[i0].cst^=(uint8_t)(i1&1); fx[a1]=mabs; fz[a1].set_const(0); } break;
            case MO_MEAS_ACTIVE_INTERFERE: { int b=cp.dyn.outcome_base+(mag_i2++); SymBit mabs; mabs.set_dyn(b); mabs.xor_in(fz[a1]);
                frec[i0]=mabs; frec[i0].cst^=(uint8_t)(i1&1); fx[a1]=mabs; fz[a1].set_const(0); } break;
            case MO_ARRAY_H: std::swap(fx[a1],fz[a1]); break;
            default: break;   // MO_ARRAY_T/T_DAG/ROT, EXPAND_*, MO_EXPAND: no symbolic frame effect
        }
        // ---- compare every slot ----
        ParityQuery pq;
        for(int q=0;q<NQ;q++){
            pq.from(fx[q]); if((eval_parity(pq,dyn.data())&1) != (s.frame.xb(q)&1)){ out[0]=(long)i; out[1]=q; out[2]=(long)k; out[3]=0; out[4]=1; return; }
            pq.from(fz[q]); if((eval_parity(pq,dyn.data())&1) != (s.frame.zb(q)&1)){ out[0]=(long)i; out[1]=q; out[2]=(long)k; out[3]=1; out[4]=1; return; }
        }
    }
    s.sampler.log_on=false;
}

// ====================================================================================
// Gate J Phase-2B : NativeFrame-OFF fast path via event-driven accumulation
// ====================================================================================
// The compiled frame queries are consumed WITHOUT the NativeFrame: a packed signature theta_sig
// (bit r = xb at rotation r) + rec_sig (bit r = final record bit r) is carried, and when a dynbit
// fires (noise channel / dormant-random / readout flip / magic outcome) the precomputed contribution
// columns are XORed in: theta_sig ^= ev_theta[e]; rec_sig ^= ev_rec[e].  Cost is O(fired events), NOT
// O(queries × ndyn).  No frame.h/cnot/cz/swap/s_gate (frame_fwd=0) and no frame.xb/zb (frame_read=0);
// the noise sampler runs with noapply=true (RNG draws unchanged, no frame mask write).  Records are
// written ONCE at the end from rec_sig (each record index is set-once; conditional-Pauli reads are
// compiled into the symbolic frame).  The engine (inverse frame / pending / dense / magic) stays LIVE
// in 2B — only the frame is removed.  Bit-exactness vs the authoritative VM is the 2B gate.
struct JFastStats { long opcode_dispatch=0, fires=0, accum_xor=0, rotations=0, frame_fwd=0, frame_read=0; };
inline int& jfast_dbg(){ static int d=0; return d; }   // 0=full; 1=skip fire-handling (timing bisect only)

inline int run_jfast(MdamShot& s, const MdamProgram& p, const CompiledMdamProgram& cp, JFastStats& st){
    if(!cp.fast_ok){ s.err="run_jfast: nrot/record_cap > 64 (fast sig unsupported)"; return 1; }
    s.engine.lazy_inverse=false;
    int j_tw=cp.theta_words, j_rw=cp.rec_words; uint64_t theta_sig[SIG_MAX_WORDS], rec_sig[SIG_MAX_WORDS]; for(int j_w=0;j_w<j_tw;j_w++) theta_sig[j_w]=cp.theta_init[j_w]; for(int j_w=0;j_w<j_rw;j_w++) rec_sig[j_w]=cp.rec_init[j_w];
    s.sampler.log_on=true; s.sampler.noapply=true; s.sampler.fire_log.clear(); size_t fire_cur=0;
    auto fire_dynbit=[&](int site,uint64_t ci,uint64_t)->int{ return cp.dyn.noise_base[site]+(int)ci; };   // fire_log stores the channel index (multiword-safe)
    auto fire=[&](int e){ if(e<0) return; for(int j_w=0;j_w<j_tw;j_w++) theta_sig[j_w]^=cp.ev_theta[(size_t)e*j_tw+j_w]; for(int j_w=0;j_w<j_rw;j_w++) rec_sig[j_w]^=cp.ev_rec[(size_t)e*j_rw+j_w]; st.fires++; st.accum_xor+=2; };
    int dbg=jfast_dbg();
    auto drain=[&](){ if(dbg) { s.sampler.fire_log.clear(); fire_cur=0; return; }
        for(;fire_cur<s.sampler.fire_log.size();++fire_cur){ auto&f=s.sampler.fire_log[fire_cur];
        fire(fire_dynbit((int)f[0],f[1],f[2])); } };
    int dorm_i=0, read_i=0, mag_i=0, rot_i=0;
    size_t N=p.kind.size();
    for(size_t i=0;i<N && !s.err;i++){
        st.opcode_dispatch++;
        int a1=p.a1[i], a2=p.a2[i], i1=p.i1[i]; int i0=p.i0[i]; double dv=p.dval[i];
        switch((MdamOp)p.kind[i]){
            // frame-only ops: fully compiled away (no NativeFrame)
            case MO_FRAME_H: case MO_FRAME_CNOT: case MO_FRAME_CZ: case MO_FRAME_SWAP:
            case MO_FRAME_S: case MO_APPLY_PAULI: case MO_MEAS_DORM_STATIC: break;
            case MO_NOISE: s.sampler.apply_site(p.i0[i], p.noise_sites[p.i0[i]], s.frame); drain(); break;
            case MO_NOISE_BLOCK: for(int si=i0;si<i0+i1;si++) s.sampler.apply_site(si, p.noise_sites[si], s.frame); drain(); break;
            case MO_READOUT_NOISE: { int e=cp.dyn.readout_base+(read_i++); if(s.udraw()<dv) fire(e); } break;
            case MO_MEAS_DORM_RANDOM: { int m=(int)s.idraw2(); int e=cp.dyn.dormant_base+(dorm_i++); if(m) fire(e); } break;
            case MO_ARRAY_CNOT: { int u=s.slot2id[a1],v=s.slot2id[a2]; if(u>=0&&v>=0) s.engine.cx(u,v); } break;
            case MO_ARRAY_CZ: { int u=s.slot2id[a1],v=s.slot2id[a2]; if(u>=0&&v>=0) s.engine.cz(u,v); } break;
            case MO_MULTI_CNOT: { int tgt=a1, t=s.slot2id[tgt]; uint64_t mask=p.mmask[i0];
                while(mask){ int ctrl=__builtin_ctzll(mask); mask&=mask-1; if(ctrl==tgt) continue;
                    int c=s.slot2id[ctrl]; if(t>=0&&c>=0) s.engine.cx(c,t); } } break;
            case MO_MULTI_CZ: { uint64_t mask=p.mmask[i0];
                while(mask){ int tgt=__builtin_ctzll(mask); mask&=mask-1; if(tgt==a1) continue;
                    int u=s.slot2id[a1],v=s.slot2id[tgt]; if(u>=0&&v>=0) s.engine.cz(u,v); } } break;
            case MO_ARRAY_S: { int q=s.slot2id[a1]; if(q>=0) s.engine.s(q,false); } break;
            case MO_ARRAY_T: { int q=s.slot2id[a1]; if(q>=0){ int xb=(int)(JSIGBIT(theta_sig,rot_i));
                s.engine.apply_rotation(q, xb?-NV_T_ANGLE:NV_T_ANGLE); } rot_i++; st.rotations++; } break;
            case MO_ARRAY_T_DAG: { int q=s.slot2id[a1]; if(q>=0){ int xb=(int)(JSIGBIT(theta_sig,rot_i));
                s.engine.apply_rotation(q, xb?NV_T_ANGLE:-NV_T_ANGLE); } rot_i++; st.rotations++; } break;
            case MO_EXPAND_T: { s.newq(a1); int q=s.slot2id[a1]; s.engine.h(q);
                int xb=(int)(JSIGBIT(theta_sig,rot_i)); s.engine.apply_rotation(q, xb?-NV_T_ANGLE:NV_T_ANGLE); rot_i++; st.rotations++; } break;
            case MO_EXPAND_T_DAG: { s.newq(a1); int q=s.slot2id[a1]; s.engine.h(q);
                int xb=(int)(JSIGBIT(theta_sig,rot_i)); s.engine.apply_rotation(q, xb?NV_T_ANGLE:-NV_T_ANGLE); rot_i++; st.rotations++; } break;
            // ---- Gate L: coherent opcodes (arbitrary-theta dv; xb from compiled theta_sig) ----
            case MO_ARRAY_ROT: { int q=s.slot2id[a1]; int xb=(int)(JSIGBIT(theta_sig,rot_i));
                if(q>=0){ if(s.rot_log_on) s.rot_log.push_back({(double)a1,(double)xb,dv,xb?-dv:dv});
                    s.engine.apply_rotation(q, xb?-dv:dv); } rot_i++; st.rotations++; } break;
            case MO_EXPAND_ROT: { s.newq(a1); int q=s.slot2id[a1]; s.engine.h(q);
                int xb=(int)(JSIGBIT(theta_sig,rot_i));
                if(s.rot_log_on) s.rot_log.push_back({(double)a1,(double)xb,dv,xb?-dv:dv});
                s.engine.apply_rotation(q, xb?-dv:dv); rot_i++; st.rotations++; } break;
            case MO_ARRAY_SWAP: {   // frame compiled away -> slot relabel only
                int i_1=s.slot2id[a1], i_2=s.slot2id[a2]; s.slot2id[a1]=-1; s.slot2id[a2]=-1;
                if(i_1>=0) s.slot2id[a2]=i_1; if(i_2>=0) s.slot2id[a1]=i_2; } break;
            case MO_SWAP_MEAS_INTERFERE: {
                int i_1=s.slot2id[a1], i_2=s.slot2id[a2]; s.slot2id[a1]=-1; s.slot2id[a2]=-1;
                if(i_1>=0) s.slot2id[a2]=i_1; if(i_2>=0) s.slot2id[a1]=i_2;
                int q=s.slot2id[a2]; if(q<0){ mag_i++; break; }
                s.engine.h(q); int b=s.measure_z(q);
                int e=cp.dyn.outcome_base+(mag_i++); if(b) fire(e);
                s.slot2id[a2]=-1;   // m_abs / record write deferred to end (rec_sig already has it)
            } break;
            case MO_MEAS_ACTIVE_DIAGONAL: {   // no swap, no H; m_abs symbolic in rec_sig
                int q=s.slot2id[a1]; if(q<0){ mag_i++; break; }
                int b=s.measure_z(q); int e=cp.dyn.outcome_base+(mag_i++); if(b) fire(e);
                s.slot2id[a1]=-1; } break;
            case MO_MEAS_ACTIVE_INTERFERE: {  // H then measure
                int q=s.slot2id[a1]; if(q<0){ mag_i++; break; }
                s.engine.h(q); int b=s.measure_z(q); int e=cp.dyn.outcome_base+(mag_i++); if(b) fire(e);
                s.slot2id[a1]=-1; } break;
            default: break;
        }
    }
    s.sampler.log_on=false; s.sampler.noapply=false;
    int nm=p.num_measurements; for(int r=0;r<nm;r++) s.record.bits[r]=(uint8_t)(JSIGBIT(rec_sig,r));
    return s.err?1:0;
}

// ====================================================================================
// Gate J Phase-2A+ : magic-side phase_pack compiler + shadow (the inverse-frame phases)
// ====================================================================================
// Phase-1 proved the magic side is dense-amplitude-coupled, BUT the inverse-frame MASKS are
// shot-static at every magic boundary (Gate I) — only the 2n PHASES (ax[i].phase, az[i].phase)
// vary.  Those 2n Z4 values are phase_pack.  The forward active-gate evolution of the inverse
// frame between two boundaries is a *fixed Z4-affine map* on phase_pack: each forward gate's
// phase update is (sum of operand phases) + 2*cross, and 2*cross is a STATIC constant (masks
// static).  So phase_pack_out = A_region * phase_pack_in + b_region (mod 4).
//
// This block compiles (A,b) per region by SYMBOLIC Z4 EXECUTION of the inverse-frame forward
// ops (a parallel SymInvFrame carrying concrete masks + a 2n-var Z4-affine phase per row), and
// a shadow that, each shot, re-syncs phase_pack from the live inverse frame at region start,
// applies the compiled (A,b), and compares to the live inverse phases at the boundary.  The
// comparison is GATED by region-start mask equality: matching masks ⇒ the map MUST reproduce
// the live phases exactly (a mismatch there is a compiler bug); differing masks ⇒ a rare
// dense-coupled M-variant (the upstream commit's right-folds differed), counted separately and
// handled by per-variant maps in the next sub-step.  No live path is disabled (shadow only).

// ---- Z4-affine phase value: (cst + sum_k coeff[k]*var_k) mod 4, vars = phase_pack[0..2n-1] ----
struct SymZ4 { std::vector<uint8_t> coeff; uint8_t cst=0;
    void init(int twoN){ coeff.assign(twoN,0); cst=0; } };
struct SymRow { PackedPauli mask{1}; SymZ4 ph; };

// symbolic Pauli multiply: mask via concrete XOR; phase = a.ph + b.ph + 2*cross  (cross concrete)
inline SymRow sym_mul(const SymRow& a, const SymRow& b){
    SymRow r; r.mask=PackedPauli(a.mask.W); int cross=0;
    for(int i=0;i<a.mask.W;i++){ r.mask.x[i]=a.mask.x[i]^b.mask.x[i]; r.mask.z[i]=a.mask.z[i]^b.mask.z[i];
        cross+=__builtin_popcountll(a.mask.z[i]&b.mask.x[i]); }
    int K=(int)a.ph.coeff.size(); r.ph.coeff.resize(K);
    for(int k=0;k<K;k++) r.ph.coeff[k]=(uint8_t)((a.ph.coeff[k]+b.ph.coeff[k])&3);
    r.ph.cst=(uint8_t)((a.ph.cst+b.ph.cst+2*cross)&3);
    return r;
}
// Symbolic forward inverse frame: masks evolve concretely (shot-independent), phases stay Z4-affine
// in the region-start phase_pack.  Mirrors NativeInverseFrame::fwd_h/fwd_s/fwd_cx EXACTLY.
struct SymInvFrame {
    int n=0,W=1,twoN=0; std::vector<SymRow> ax,az;
    void init(int n_,int W_){ n=n_;W=W_;twoN=2*n; ax.resize(n);az.resize(n);
        for(int i=0;i<n;i++){ ax[i].ph.init(twoN); az[i].ph.init(twoN); } }
    void reset_to_live(const NativeInverseFrame& f){
        for(int i=0;i<n;i++){ ax[i].mask=f.ax[i]; ax[i].mask.phase=0; ax[i].ph.init(twoN); ax[i].ph.coeff[i]=1;
                              az[i].mask=f.az[i]; az[i].mask.phase=0; az[i].ph.init(twoN); az[i].ph.coeff[n+i]=1; } }
    void fwd_h(int q){ std::swap(ax[q],az[q]); }
    void fwd_cx(int c,int t){ SymRow a=sym_mul(ax[c],ax[t]); SymRow b=sym_mul(az[c],az[t]); ax[c]=a; az[t]=b; }
    void cz(int a,int b){ fwd_h(b); fwd_cx(a,b); fwd_h(b); }
    void fwd_s(int q,bool dag){
        PackedPauli Xq(W); Xq.x[PackedPauli::word(q)]=PackedPauli::bit(q); pconj_s(Xq,q,!dag);
        SymRow out; out.mask=PackedPauli(W); out.ph.init(twoN); out.ph.cst=(uint8_t)(Xq.phase&3);
        for(int wi=0;wi<W;wi++){ uint64_t xi=Xq.x[wi]; while(xi){ int b=__builtin_ctzll(xi); xi&=xi-1; out=sym_mul(out, ax[(wi<<6)+b]); } }
        for(int wi=0;wi<W;wi++){ uint64_t zi=Xq.z[wi]; while(zi){ int b=__builtin_ctzll(zi); zi&=zi-1; out=sym_mul(out, az[(wi<<6)+b]); } }
        ax[q]=out;
    }
};

// commit right-fold phase delta, keyed by the boundary magic-axis vector M (the same key F4 uses):
// the Wout/localizer gate list — hence its phase delta — is a function of M (≤4 variants/boundary,
// gate_f_audit), even though the post-commit MASKS are shot-static.  One rfd per observed M-variant.
struct JCommitVariant { std::vector<int> M_key; std::vector<uint8_t> rfd; };
// per-region compiled forward map + masks + per-variant commit deltas
struct JRegionMap {
    std::vector<uint8_t> A, b;                 // forward Z4-affine map (phase_pack_out = A·in + b)
    std::vector<PackedPauli> ref_ax, ref_az;   // region-start masks (forward variant gate)
    std::vector<PackedPauli> bnd_ax, bnd_az;   // boundary (measure read-point) masks (2C reconstruct)
    std::vector<PackedPauli> post_ax, post_az; // post-commit masks (foldx_q delta source, static)
    std::vector<JCommitVariant> commits;       // rfd per boundary M-variant (lazily populated)
    bool valid=false;
};
struct JPhaseCompiled { int n=0, twoN=0, nmagic=0; std::vector<JRegionMap> maps; bool built=false; };
struct JPhaseStats {
    long regions_total=0, regions_match=0, phase_checks=0, phase_mismatch=0, regions_variant=0;
    long commit_checks=0, commit_mismatch=0, commit_new_variant=0, commit_maskbad=0, commit_rebuild=0;
    int first_bad_region=-1, first_bad_slot=-1, first_bad_commit_region=-1; };

// One shot of the run() mirror.  compile=true: ride SymInvFrame, emit forward map (A,b)+masks+rfd
// per region.  compile=false: CHAIN phase_pack across the whole shot WITHOUT re-sync — apply the
// forward map, then the commit (rfd + the logged foldx q's via post-mask 2·z_q), comparing to the
// live inverse phases at the boundary AND after the commit.  Live path runs authoritatively (shadow).
inline void jphase_run(MdamShot& s, const MdamProgram& p, JPhaseCompiled& cp, JPhaseStats& st, bool compile){
    s.engine.lazy_inverse=false;   // compile/shadow pass rides the live inverse frame (sf) + reads .ax/.az
    int n=s.engine.n, W=s.engine.W, twoN=2*n;
    SymInvFrame sf; if(compile) sf.init(n,W);
    std::vector<uint8_t> ppin(twoN,0), pred, live, pp_b; bool cur_match=true; std::vector<int> foldxlog;
    auto read_phases=[&](std::vector<uint8_t>& out){ out.resize(twoN);
        for(int i=0;i<n;i++){ out[i]=s.engine.inverse_frame.ax[i].phase&3; out[n+i]=s.engine.inverse_frame.az[i].phase&3; } };
    auto masks_eq=[&](const std::vector<PackedPauli>& ax,const std::vector<PackedPauli>& az)->bool{
        for(int i=0;i<n;i++){ const PackedPauli&la=s.engine.inverse_frame.ax[i];
            for(int w=0;w<W;w++) if(la.x[w]!=ax[i].x[w]||la.z[w]!=ax[i].z[w]) return false;
            const PackedPauli&lz=s.engine.inverse_frame.az[i];
            for(int w=0;w<W;w++) if(lz.x[w]!=az[i].x[w]||lz.z[w]!=az[i].z[w]) return false; }
        return true; };
    auto mask_match_ref=[&](int mag)->bool{ if(mag<0||mag>=(int)cp.maps.size()||!cp.maps[mag].valid) return false;
        return masks_eq(cp.maps[mag].ref_ax, cp.maps[mag].ref_az); };
    auto capture_masks=[&](std::vector<PackedPauli>& ax,std::vector<PackedPauli>& az){ ax.resize(n); az.resize(n);
        for(int i=0;i<n;i++){ ax[i]=s.engine.inverse_frame.ax[i]; ax[i].phase=0; az[i]=s.engine.inverse_frame.az[i]; az[i].phase=0; } };
    // pp += Σ_{q in log} 2·z_q over the (static) post-commit masks  (fold_x adds 2·z_q to every row)
    auto apply_foldx=[&](std::vector<uint8_t>& pp, const JRegionMap& m, const std::vector<int>& log){
        for(int q : log){ for(int i=0;i<n;i++){ pp[i]=(uint8_t)((pp[i]+2*m.post_ax[i].getz(q))&3);
                                                pp[n+i]=(uint8_t)((pp[n+i]+2*m.post_az[i].getz(q))&3); } } };
    auto start_region_compile=[&](int mag){ if(mag>=(int)cp.maps.size()) return;
        sf.reset_to_live(s.engine.inverse_frame); JRegionMap& m=cp.maps[mag];
        capture_masks(m.ref_ax, m.ref_az); };
    auto emit_map=[&](int mag){ if(mag>=(int)cp.maps.size()) return; JRegionMap& m=cp.maps[mag];
        m.A.assign(twoN*twoN,0); m.b.assign(twoN,0); m.bnd_ax.resize(n); m.bnd_az.resize(n);
        for(int i=0;i<n;i++){ for(int k=0;k<twoN;k++) m.A[i*twoN+k]=sf.ax[i].ph.coeff[k]; m.b[i]=sf.ax[i].ph.cst;
                              for(int k=0;k<twoN;k++) m.A[(n+i)*twoN+k]=sf.az[i].ph.coeff[k]; m.b[n+i]=sf.az[i].ph.cst;
                              m.bnd_ax[i]=sf.ax[i].mask; m.bnd_ax[i].phase=0;
                              m.bnd_az[i]=sf.az[i].mask; m.bnd_az[i].phase=0; }
        m.valid=true; };
    auto apply_map=[&](int mag){ JRegionMap& m=cp.maps[mag]; pred.assign(twoN,0);
        for(int o=0;o<twoN;o++){ int v=m.b[o]; for(int k=0;k<twoN;k++) v+=m.A[o*twoN+k]*ppin[k]; pred[o]=(uint8_t)(v&3); } };
    int mag=0;
    if(compile) start_region_compile(0);
    else { read_phases(ppin); cur_match=mask_match_ref(0); }
    size_t N=p.kind.size();
    for(size_t i=0;i<N && !s.err;i++){
        int a1=p.a1[i], a2=p.a2[i], i0=p.i0[i], i1=p.i1[i]; double dv=p.dval[i];
        switch((MdamOp)p.kind[i]){
            case MO_FRAME_H: s.frame.h(a1); break;
            case MO_FRAME_CNOT: s.frame.cnot(a1,a2); break;
            case MO_FRAME_CZ: s.frame.cz(a1,a2); break;
            case MO_FRAME_SWAP: s.frame.swap(a1,a2); break;
            case MO_FRAME_S: s.frame.s_gate(a1); break;
            case MO_APPLY_PAULI: { int rc=s.record.get((uint32_t)i0); if(rc==1) s.apply_mask(p.cp_masks[i1]); } break;
            case MO_NOISE: s.sampler.apply_site(i0, p.noise_sites[i0], s.frame); break;
            case MO_NOISE_BLOCK: for(int si=i0;si<i0+i1;si++) s.sampler.apply_site(si, p.noise_sites[si], s.frame); break;
            case MO_READOUT_NOISE: if(s.udraw()<dv) s.record.flip((uint32_t)i0); break;
            case MO_MEAS_DORM_STATIC: s.record.set((uint32_t)i0, s.frame.xb(a1)^i1); break;
            case MO_MEAS_DORM_RANDOM: { int m=(int)s.idraw2(); s.record.set((uint32_t)i0, m^i1); s.frame.set_xz(a1,(uint8_t)m,0); } break;
            case MO_ARRAY_CNOT: { int u=s.slot2id[a1],v=s.slot2id[a2]; if(u>=0&&v>=0){ s.engine.cx(u,v); if(compile) sf.fwd_cx(u,v);} s.frame.cnot(a1,a2); } break;
            case MO_ARRAY_CZ: { int u=s.slot2id[a1],v=s.slot2id[a2]; if(u>=0&&v>=0){ s.engine.cz(u,v); if(compile) sf.cz(u,v);} s.frame.cz(a1,a2); } break;
            case MO_MULTI_CNOT: { int tgt=a1, t=s.slot2id[tgt]; uint64_t mask=p.mmask[i0];
                while(mask){ int ctrl=__builtin_ctzll(mask); mask&=mask-1; if(ctrl==tgt) continue;
                    int c=s.slot2id[ctrl]; if(t>=0&&c>=0){ s.engine.cx(c,t); if(compile) sf.fwd_cx(c,t);} s.frame.cnot(ctrl,tgt); } } break;
            case MO_MULTI_CZ: { uint64_t mask=p.mmask[i0];
                while(mask){ int tgt=__builtin_ctzll(mask); mask&=mask-1; if(tgt==a1) continue;
                    int u=s.slot2id[a1],v=s.slot2id[tgt]; if(u>=0&&v>=0){ s.engine.cz(u,v); if(compile) sf.cz(u,v);} s.frame.cz(a1,tgt); } } break;
            case MO_ARRAY_T: s.rot(p,a1,NV_T_ANGLE); break;
            case MO_ARRAY_T_DAG: s.rot(p,a1,-NV_T_ANGLE); break;
            case MO_ARRAY_S: { int q=s.slot2id[a1]; if(q>=0){ s.engine.s(q,false); if(compile) sf.fwd_s(q,false);} s.frame.s_gate(a1); } break;
            case MO_EXPAND_T: { s.newq(a1); int q2=s.slot2id[a1]; s.engine.h(q2); if(compile) sf.fwd_h(q2); s.rot(p,a1,NV_T_ANGLE); } break;
            case MO_EXPAND_T_DAG: { s.newq(a1); int q2=s.slot2id[a1]; s.engine.h(q2); if(compile) sf.fwd_h(q2); s.rot(p,a1,-NV_T_ANGLE); } break;
            // ---- Gate L: coherent rotations (diagonal magic on dense; inverse frame untouched) + slot relabel ----
            case MO_ARRAY_ROT: s.rot(p,a1,dv); break;
            case MO_EXPAND_ROT: { s.newq(a1); int q2=s.slot2id[a1]; s.engine.h(q2); if(compile) sf.fwd_h(q2); s.rot(p,a1,dv); } break;
            case MO_ARRAY_SWAP: { int i_1=s.slot2id[a1], i_2=s.slot2id[a2]; s.slot2id[a1]=-1; s.slot2id[a2]=-1;
                if(i_1>=0) s.slot2id[a2]=i_1; if(i_2>=0) s.slot2id[a1]=i_2; s.frame.swap(a1,a2); } break;
            case MO_SWAP_MEAS_INTERFERE: {
                int i_1=s.slot2id[a1], i_2=s.slot2id[a2]; s.slot2id[a1]=-1; s.slot2id[a2]=-1;
                if(i_1>=0) s.slot2id[a2]=i_1; if(i_2>=0) s.slot2id[a1]=i_2;
                s.frame.swap(a1,a2);
                int q=s.slot2id[a2]; if(q<0) break;
                s.engine.h(q); if(compile) sf.fwd_h(q);          // boundary h: last op of region mag
                bool fmatch = (mag<(int)cp.maps.size() && cp.maps[mag].valid && cur_match);
                if(compile){ emit_map(mag); read_phases(pp_b); }   // boundary forward map + boundary phases
                else if(mag<(int)cp.maps.size() && cp.maps[mag].valid){
                    st.regions_total++; apply_map(mag); pp_b=pred;  // pp_b = compiled (chained) boundary
                    if(fmatch){ st.regions_match++; st.phase_checks++; read_phases(live);
                        bool bad=false; for(int o=0;o<twoN;o++) if(pred[o]!=live[o]){ bad=true;
                            if(st.first_bad_slot<0){ st.first_bad_region=mag; st.first_bad_slot=o; } }
                        if(bad) st.phase_mismatch++; }
                    else st.regions_variant++;
                }
                std::vector<int> Mkey = s.engine.M;                // boundary M-vector = commit-variant key
                long ag0 = s.engine.ag_fired;                       // detect stabilizer-branch rebuild
                foldxlog.clear(); s.engine.foldx_log=&foldxlog;   // capture commit foldx (dense byproduct)
                int b=s.measure_z(q);
                s.engine.foldx_log=nullptr;
                bool rebuilt = (s.engine.ag_fired > ag0);           // ag_measure rebuilt the inverse phases
                if(mag<(int)cp.maps.size()){ JRegionMap& m=cp.maps[mag];
                    std::vector<uint8_t> pc; read_phases(pc);       // post-commit live phases
                    if(compile){ capture_masks(m.post_ax,m.post_az);
                        if(!rebuilt){ std::vector<uint8_t> tmp=pp_b; apply_foldx(tmp,m,foldxlog);
                            JCommitVariant cv; cv.M_key=Mkey; cv.rfd.assign(twoN,0);
                            for(int o=0;o<twoN;o++) cv.rfd[o]=(uint8_t)((pc[o]-tmp[o])&3);
                            m.commits.push_back(cv); } }
                    else if(rebuilt){ st.commit_rebuild++; ppin=pc; }  // stabilizer rebuild: re-sync (separate case)
                    else {
                        bool cmatch=masks_eq(m.post_ax,m.post_az); if(!cmatch) st.commit_maskbad++;
                        int vi=-1; for(size_t k=0;k<m.commits.size();k++) if(m.commits[k].M_key==Mkey){ vi=(int)k; break; }
                        if(vi<0){                                   // new M-variant: capture rfd from live, count
                            std::vector<uint8_t> tmp=pp_b; apply_foldx(tmp,m,foldxlog);
                            JCommitVariant cv; cv.M_key=Mkey; cv.rfd.assign(twoN,0);
                            for(int o=0;o<twoN;o++) cv.rfd[o]=(uint8_t)((pc[o]-tmp[o])&3);
                            m.commits.push_back(cv); st.commit_new_variant++; ppin=pc;   // chain exact
                        } else {                                    // known variant: predict + compare + chain
                            std::vector<uint8_t> pa=pp_b;
                            for(int o=0;o<twoN;o++) pa[o]=(uint8_t)((pa[o]+m.commits[vi].rfd[o])&3);
                            apply_foldx(pa,m,foldxlog);
                            st.commit_checks++; bool bad=false;
                            for(int o=0;o<twoN;o++) if(pa[o]!=pc[o]){ bad=true;
                                if(st.first_bad_commit_region<0) st.first_bad_commit_region=mag; }
                            if(bad) st.commit_mismatch++;
                            ppin=pa;                                // CHAIN: carry compiled phase_pack forward
                        }
                    }
                }
                mag++;
                if(compile) start_region_compile(mag);
                else cur_match=mask_match_ref(mag);                // next region forward gate (NO re-sync)
                s.slot2id[a2]=-1;
                int m_abs=b ^ s.frame.zb(a2);
                s.record.set((uint32_t)i0, m_abs^i1);
                s.frame.set_xz(a2,(uint8_t)m_abs,0);
            } break;
            case MO_MEAS_ACTIVE_DIAGONAL:                            // Gate L: coherent active measure boundary (no swap; slot a1)
            case MO_MEAS_ACTIVE_INTERFERE: {                         // DIAGONAL: no boundary H, xb; INTERFERE: boundary H, zb
                bool interfere = ((MdamOp)p.kind[i]==MO_MEAS_ACTIVE_INTERFERE);
                int q=s.slot2id[a1]; if(q<0) break;
                if(interfere){ s.engine.h(q); if(compile) sf.fwd_h(q); }   // boundary h: last op of region mag (interfere only)
                bool fmatch = (mag<(int)cp.maps.size() && cp.maps[mag].valid && cur_match);
                if(compile){ emit_map(mag); read_phases(pp_b); }   // boundary forward map + boundary phases
                else if(mag<(int)cp.maps.size() && cp.maps[mag].valid){
                    st.regions_total++; apply_map(mag); pp_b=pred;  // pp_b = compiled (chained) boundary
                    if(fmatch){ st.regions_match++; st.phase_checks++; read_phases(live);
                        bool bad=false; for(int o=0;o<twoN;o++) if(pred[o]!=live[o]){ bad=true;
                            if(st.first_bad_slot<0){ st.first_bad_region=mag; st.first_bad_slot=o; } }
                        if(bad) st.phase_mismatch++; }
                    else st.regions_variant++;
                }
                std::vector<int> Mkey = s.engine.M;                // boundary M-vector = commit-variant key
                long ag0 = s.engine.ag_fired;                       // detect stabilizer-branch rebuild
                foldxlog.clear(); s.engine.foldx_log=&foldxlog;   // capture commit foldx (dense byproduct)
                int b=s.measure_z(q);
                s.engine.foldx_log=nullptr;
                bool rebuilt = (s.engine.ag_fired > ag0);           // ag_measure rebuilt the inverse phases
                if(mag<(int)cp.maps.size()){ JRegionMap& m=cp.maps[mag];
                    std::vector<uint8_t> pc; read_phases(pc);       // post-commit live phases
                    if(compile){ capture_masks(m.post_ax,m.post_az);
                        if(!rebuilt){ std::vector<uint8_t> tmp=pp_b; apply_foldx(tmp,m,foldxlog);
                            JCommitVariant cv; cv.M_key=Mkey; cv.rfd.assign(twoN,0);
                            for(int o=0;o<twoN;o++) cv.rfd[o]=(uint8_t)((pc[o]-tmp[o])&3);
                            m.commits.push_back(cv); } }
                    else if(rebuilt){ st.commit_rebuild++; ppin=pc; }  // stabilizer rebuild: re-sync (separate case)
                    else {
                        bool cmatch=masks_eq(m.post_ax,m.post_az); if(!cmatch) st.commit_maskbad++;
                        int vi=-1; for(size_t k=0;k<m.commits.size();k++) if(m.commits[k].M_key==Mkey){ vi=(int)k; break; }
                        if(vi<0){                                   // new M-variant: capture rfd from live, count
                            std::vector<uint8_t> tmp=pp_b; apply_foldx(tmp,m,foldxlog);
                            JCommitVariant cv; cv.M_key=Mkey; cv.rfd.assign(twoN,0);
                            for(int o=0;o<twoN;o++) cv.rfd[o]=(uint8_t)((pc[o]-tmp[o])&3);
                            m.commits.push_back(cv); st.commit_new_variant++; ppin=pc;   // chain exact
                        } else {                                    // known variant: predict + compare + chain
                            std::vector<uint8_t> pa=pp_b;
                            for(int o=0;o<twoN;o++) pa[o]=(uint8_t)((pa[o]+m.commits[vi].rfd[o])&3);
                            apply_foldx(pa,m,foldxlog);
                            st.commit_checks++; bool bad=false;
                            for(int o=0;o<twoN;o++) if(pa[o]!=pc[o]){ bad=true;
                                if(st.first_bad_commit_region<0) st.first_bad_commit_region=mag; }
                            if(bad) st.commit_mismatch++;
                            ppin=pa;                                // CHAIN: carry compiled phase_pack forward
                        }
                    }
                }
                mag++;
                if(compile) start_region_compile(mag);
                else cur_match=mask_match_ref(mag);                // next region forward gate (NO re-sync)
                s.slot2id[a1]=-1;
                int m_abs=b ^ (interfere ? s.frame.zb(a1) : s.frame.xb(a1));
                s.record.set((uint32_t)i0, m_abs^i1);
                s.frame.set_xz(a1,(uint8_t)m_abs,0);
            } break;
            default: break;
        }
    }
    s.engine.foldx_log=nullptr;
    if(compile){ cp.n=n; cp.twoN=twoN; cp.nmagic=mag; cp.built=true; }
}

// ====================================================================================
// Gate J Phase-2C-A : NativeInverseFrame-OFF fast path (phase_pack carried, reconstruct at measure)
// ====================================================================================
// Builds on 2B (frame removed via event accumulation).  The per-gate inverse FORWARD is removed
// (inverse_fwd=0): active gates use the _noinv variants (tableau + pending only), and phase_pack is
// carried via the compiled forward Z4 map.  At each measure boundary the live inverse frame is
// RECONSTRUCTED from the static boundary masks + carried phase_pack (proving phase_pack is the state),
// measure_z runs (compiled magic uses Imem inject → 0 pullback; the rare oracle uses the reconstructed
// inverse), and phase_pack is read back from the post-commit inverse.  The 0.16% oracle ag_measure
// rebuild is absorbed by the reconstruct+read-back and reported as oracle_count (transitional; closed
// in 2C+).  drop_keepbit/foldx are dense-kernel byproducts (no recompute).
struct JFast2CStats { long opcode_dispatch=0, reconstructs=0, pullback_calls=0, imem_miss=0,
    oracle_count=0, phase_mismatch=0; int first_bad=-1; };

inline int run_jfast_2c(MdamShot& s, const MdamProgram& p, const CompiledMdamProgram& cp,
                        const JPhaseCompiled& jp, JFast2CStats& st){
    if(!cp.fast_ok){ s.err="run_jfast_2c: fast sig unsupported"; return 1; }
    s.engine.lazy_inverse=false;   // fast path maintains/reconstructs the inverse frame explicitly
    int n=s.engine.n, twoN=2*n;
    int j_tw=cp.theta_words, j_rw=cp.rec_words; uint64_t theta_sig[SIG_MAX_WORDS], rec_sig[SIG_MAX_WORDS]; for(int j_w=0;j_w<j_tw;j_w++) theta_sig[j_w]=cp.theta_init[j_w]; for(int j_w=0;j_w<j_rw;j_w++) rec_sig[j_w]=cp.rec_init[j_w];
    s.sampler.log_on=true; s.sampler.noapply=true; s.sampler.fire_log.clear(); size_t fire_cur=0;
    std::vector<uint8_t> pp(twoN,0), bnd(twoN,0);
    auto fire_dynbit=[&](int site,uint64_t ci,uint64_t)->int{ return cp.dyn.noise_base[site]+(int)ci; };   // fire_log stores the channel index (multiword-safe)
    auto fire=[&](int e){ if(e<0) return; for(int j_w=0;j_w<j_tw;j_w++) theta_sig[j_w]^=cp.ev_theta[(size_t)e*j_tw+j_w]; for(int j_w=0;j_w<j_rw;j_w++) rec_sig[j_w]^=cp.ev_rec[(size_t)e*j_rw+j_w]; };
    auto drain=[&](){ for(;fire_cur<s.sampler.fire_log.size();++fire_cur){ auto&f=s.sampler.fire_log[fire_cur];
        fire(fire_dynbit((int)f[0],f[1],f[2])); } };
    auto fwd_map=[&](int mag){ const JRegionMap& m=jp.maps[mag];   // bnd = A·pp + b
        for(int o=0;o<twoN;o++){ int v=m.b[o]; for(int k=0;k<twoN;k++) v+=m.A[o*twoN+k]*pp[k]; bnd[o]=(uint8_t)(v&3); } };
    int dorm_i=0, read_i=0, mag_i=0, rot_i=0; size_t N=p.kind.size();
    for(size_t i=0;i<N && !s.err;i++){
        st.opcode_dispatch++;
        int a1=p.a1[i], a2=p.a2[i], i1=p.i1[i]; int i0=p.i0[i]; double dv=p.dval[i];
        switch((MdamOp)p.kind[i]){
            case MO_FRAME_H: case MO_FRAME_CNOT: case MO_FRAME_CZ: case MO_FRAME_SWAP:
            case MO_FRAME_S: case MO_APPLY_PAULI: case MO_MEAS_DORM_STATIC: break;
            case MO_NOISE: s.sampler.apply_site(i0, p.noise_sites[i0], s.frame); drain(); break;
            case MO_NOISE_BLOCK: for(int si=i0;si<i0+i1;si++) s.sampler.apply_site(si, p.noise_sites[si], s.frame); drain(); break;
            case MO_READOUT_NOISE: { int e=cp.dyn.readout_base+(read_i++); if(s.udraw()<dv) fire(e); } break;
            case MO_MEAS_DORM_RANDOM: { int m=(int)s.idraw2(); int e=cp.dyn.dormant_base+(dorm_i++); if(m) fire(e); } break;
            case MO_ARRAY_CNOT: { int u=s.slot2id[a1],v=s.slot2id[a2]; if(u>=0&&v>=0) s.engine.cx_noinv(u,v); } break;
            case MO_ARRAY_CZ: { int u=s.slot2id[a1],v=s.slot2id[a2]; if(u>=0&&v>=0) s.engine.cz_noinv(u,v); } break;
            case MO_MULTI_CNOT: { int tgt=a1, t=s.slot2id[tgt]; uint64_t mask=p.mmask[i0];
                while(mask){ int ctrl=__builtin_ctzll(mask); mask&=mask-1; if(ctrl==tgt) continue;
                    int c=s.slot2id[ctrl]; if(t>=0&&c>=0) s.engine.cx_noinv(c,t); } } break;
            case MO_MULTI_CZ: { uint64_t mask=p.mmask[i0];
                while(mask){ int tgt=__builtin_ctzll(mask); mask&=mask-1; if(tgt==a1) continue;
                    int u=s.slot2id[a1],v=s.slot2id[tgt]; if(u>=0&&v>=0) s.engine.cz_noinv(u,v); } } break;
            case MO_ARRAY_S: { int q=s.slot2id[a1]; if(q>=0) s.engine.s_noinv(q,false); } break;
            case MO_ARRAY_T: { int q=s.slot2id[a1]; if(q>=0){ int xb=(int)(JSIGBIT(theta_sig,rot_i));
                s.engine.apply_rotation(q, xb?-NV_T_ANGLE:NV_T_ANGLE); } rot_i++; } break;
            case MO_ARRAY_T_DAG: { int q=s.slot2id[a1]; if(q>=0){ int xb=(int)(JSIGBIT(theta_sig,rot_i));
                s.engine.apply_rotation(q, xb?NV_T_ANGLE:-NV_T_ANGLE); } rot_i++; } break;
            case MO_EXPAND_T: { s.newq(a1); int q=s.slot2id[a1]; s.engine.h_noinv(q);
                int xb=(int)(JSIGBIT(theta_sig,rot_i)); s.engine.apply_rotation(q, xb?-NV_T_ANGLE:NV_T_ANGLE); rot_i++; } break;
            case MO_EXPAND_T_DAG: { s.newq(a1); int q=s.slot2id[a1]; s.engine.h_noinv(q);
                int xb=(int)(JSIGBIT(theta_sig,rot_i)); s.engine.apply_rotation(q, xb?NV_T_ANGLE:-NV_T_ANGLE); rot_i++; } break;
            case MO_SWAP_MEAS_INTERFERE: {
                int i_1=s.slot2id[a1], i_2=s.slot2id[a2]; s.slot2id[a1]=-1; s.slot2id[a2]=-1;
                if(i_1>=0) s.slot2id[a2]=i_1; if(i_2>=0) s.slot2id[a1]=i_2;
                int q=s.slot2id[a2]; if(q<0){ mag_i++; break; }
                int mag=mag_i;
                s.engine.h_noinv(q);                          // boundary h: tableau+pending (inverse via map)
                fwd_map(mag);                                  // phase_pack at boundary (= A·pp + b)
                const JRegionMap& m=jp.maps[mag];              // reconstruct inverse from masks + phase_pack
                s.engine.reconstruct_inverse(m.bnd_ax, m.bnd_az, bnd.data()); st.reconstructs++;
                long pb0=s.engine.pullback_calls; int mo0=s.magic_oracle;
                int b=s.measure_z(q);
                st.pullback_calls += (s.engine.pullback_calls-pb0);
                if(s.magic_oracle>mo0) st.oracle_count++;
                int e=cp.dyn.outcome_base+(mag_i++); if(b) fire(e);
                s.engine.read_phase_pack(pp.data());           // post-commit phase_pack (region mag+1 start)
                s.slot2id[a2]=-1;
            } break;
            default: break;
        }
    }
    s.sampler.log_on=false; s.sampler.noapply=false;
    st.imem_miss = s.imem_misses;
    int nm=p.num_measurements; for(int r=0;r<nm;r++) s.record.bits[r]=(uint8_t)(JSIGBIT(rec_sig,r));
    return s.err?1:0;
}

// ====================================================================================
// Gate J Phase-2D-1 : compiled magic WITHOUT reconstruct (phase-only + Imem inject + rfd commit)
// ====================================================================================
// Removes the 2C-A transitional cost: the 4 compiled magics no longer materialize the inverse frame.
// A compiled magic that is fully warm (plan_cache feasible + Imem key present + rfd variant present)
// runs with: set_inverse_phases (12 bytes, for the Imem key) + inverse_off=true (commit folds tableau
// only) + Imem-injected rpp/sign (0 pullback) + phase_pack commit = bnd + rfd[M] + Σ foldx_log (the
// 2A+ formula; foldx_log is the dense-kernel byproduct).  compiled_reconstruct=0, compiled_pullback=0.
// A cold variant or the oracle falls back to full reconstruct + read-back (SELF-WARMING: the fallback
// captures rfd + populates Imem), counted as reconstructs/oracle/cold.  inverse_fwd stays 0.
struct JFast2DStats { long opcode_dispatch=0, compiled_fast=0, reconstructs=0, pullback_calls=0,
    imem_miss=0, oracle_count=0, cold_fallback=0; };
// 2D-3 timing instrumentation (runtime, default OFF = release path unchanged).  j2d_dbg = A/B skip
// bitmask (correctness-breaking, timing-only, RNG-preserving); j2d_time = rdtsc accumulate on/off.
// j2d_cyc: [0]=whole-shot cycles, [1]=measure-block cycles (the rest is derived by A/B deltas).
enum { J2D_SKIP_FIRE=1, J2D_SKIP_ENGINE=2, J2D_SKIP_ROT=4, J2D_SKIP_FWDMAP=8, J2D_SKIP_RECORD=16 };
inline int& j2d_dbg(){ static int d=0; return d; }
inline int& j2d_time(){ static int t=0; return t; }
inline uint64_t* j2d_cyc(){ static uint64_t c[4]={0,0,0,0}; return c; }

inline int run_jfast_2d(MdamShot& s, const MdamProgram& p, const CompiledMdamProgram& cp,
                        JPhaseCompiled& jp, JFast2DStats& st){
    if(!cp.fast_ok){ s.err="run_jfast_2d: fast sig unsupported"; return 1; }
    s.engine.lazy_inverse=false;   // fast path maintains/reconstructs the inverse frame explicitly
    int n=s.engine.n, twoN=2*n;
    int j_tw=cp.theta_words, j_rw=cp.rec_words; uint64_t theta_sig[SIG_MAX_WORDS], rec_sig[SIG_MAX_WORDS]; for(int j_w=0;j_w<j_tw;j_w++) theta_sig[j_w]=cp.theta_init[j_w]; for(int j_w=0;j_w<j_rw;j_w++) rec_sig[j_w]=cp.rec_init[j_w];
    s.sampler.log_on=true; s.sampler.noapply=true; s.sampler.fire_log.clear(); size_t fire_cur=0;
    std::vector<uint8_t> pp(twoN,0), bnd(twoN,0); std::vector<int> foldxlog;
    int dbg=j2d_dbg(), tm=j2d_time(); uint64_t* C=j2d_cyc(); uint64_t _tsh=tm?__rdtsc():0;
    auto fire_dynbit=[&](int site,uint64_t ci,uint64_t)->int{ return cp.dyn.noise_base[site]+(int)ci; };   // fire_log stores the channel index (multiword-safe)
    auto fire=[&](int e){ if(e<0) return; for(int j_w=0;j_w<j_tw;j_w++) theta_sig[j_w]^=cp.ev_theta[(size_t)e*j_tw+j_w]; for(int j_w=0;j_w<j_rw;j_w++) rec_sig[j_w]^=cp.ev_rec[(size_t)e*j_rw+j_w]; };
    auto drain=[&](){ for(;fire_cur<s.sampler.fire_log.size();++fire_cur){ auto&f=s.sampler.fire_log[fire_cur];
        fire(fire_dynbit((int)f[0],f[1],f[2])); } };
    auto fwd_map=[&](int mag){ const JRegionMap& m=jp.maps[mag];
        for(int o=0;o<twoN;o++){ int v=m.b[o]; for(int k=0;k<twoN;k++) v+=m.A[o*twoN+k]*pp[k]; bnd[o]=(uint8_t)(v&3); } };
    // Imem key from phase_pack + M (replicates measure_z's _imem_key, n<=8 perfect pack)
    auto imem_key=[&](int mag, const std::vector<uint8_t>& ph, const std::vector<int>& M)->uint64_t{
        long mpack=(long)M.size(); for(size_t k=0;k<M.size();k++) mpack|=((long)(M[k]&15))<<(4*(k+1));
        uint64_t ip=0; for(int i=0;i<n;i++){ ip|=((uint64_t)(ph[i]&3))<<(4*i); ip|=((uint64_t)(ph[n+i]&3))<<(4*i+2); }
        return (uint64_t)mag | (ip<<4) | ((uint64_t)mpack<<(4+4*n)); };
    auto plan_compiled=[&](int mag, const std::vector<int>& M)->bool{
        if(mag>=(int)s.plan_cache.size()) return false;
        for(auto& sp : s.plan_cache[mag]) if(sp.M_key==M) return sp.state==1; return false; };
    auto commit_find=[&](int mag, const std::vector<int>& M)->int{
        if(mag>=(int)jp.maps.size()) return -1; auto& cv=jp.maps[mag].commits;
        for(size_t k=0;k<cv.size();k++) if(cv[k].M_key==M) return (int)k; return -1; };
    auto apply_foldx=[&](std::vector<uint8_t>& a, const JRegionMap& m, const std::vector<int>& log){
        for(int q : log){ for(int i=0;i<n;i++){ a[i]=(uint8_t)((a[i]+2*m.post_ax[i].getz(q))&3);
                                                a[n+i]=(uint8_t)((a[n+i]+2*m.post_az[i].getz(q))&3); } } };
    int dorm_i=0, read_i=0, mag_i=0, rot_i=0; size_t N=p.kind.size();
    for(size_t i=0;i<N && !s.err;i++){
        st.opcode_dispatch++;
        int a1=p.a1[i], a2=p.a2[i], i1=p.i1[i]; int i0=p.i0[i]; double dv=p.dval[i];
        switch((MdamOp)p.kind[i]){
            case MO_FRAME_H: case MO_FRAME_CNOT: case MO_FRAME_CZ: case MO_FRAME_SWAP:
            case MO_FRAME_S: case MO_APPLY_PAULI: case MO_MEAS_DORM_STATIC: break;
            case MO_NOISE: s.sampler.apply_site(i0, p.noise_sites[i0], s.frame); if(!(dbg&J2D_SKIP_FIRE)) drain(); break;
            case MO_NOISE_BLOCK: for(int si=i0;si<i0+i1;si++) s.sampler.apply_site(si, p.noise_sites[si], s.frame); if(!(dbg&J2D_SKIP_FIRE)) drain(); break;
            case MO_READOUT_NOISE: { int e=cp.dyn.readout_base+(read_i++); if(s.udraw()<dv) fire(e); } break;
            case MO_MEAS_DORM_RANDOM: { int m=(int)s.idraw2(); int e=cp.dyn.dormant_base+(dorm_i++); if(m) fire(e); } break;
            case MO_ARRAY_CNOT: { int u=s.slot2id[a1],v=s.slot2id[a2]; if(u>=0&&v>=0&&!(dbg&J2D_SKIP_ENGINE)) s.engine.cx_noinv(u,v); } break;
            case MO_ARRAY_CZ: { int u=s.slot2id[a1],v=s.slot2id[a2]; if(u>=0&&v>=0&&!(dbg&J2D_SKIP_ENGINE)) s.engine.cz_noinv(u,v); } break;
            case MO_MULTI_CNOT: { int tgt=a1, t=s.slot2id[tgt]; uint64_t mask=p.mmask[i0];
                while(mask){ int ctrl=__builtin_ctzll(mask); mask&=mask-1; if(ctrl==tgt) continue;
                    int c=s.slot2id[ctrl]; if(t>=0&&c>=0&&!(dbg&J2D_SKIP_ENGINE)) s.engine.cx_noinv(c,t); } } break;
            case MO_MULTI_CZ: { uint64_t mask=p.mmask[i0];
                while(mask){ int tgt=__builtin_ctzll(mask); mask&=mask-1; if(tgt==a1) continue;
                    int u=s.slot2id[a1],v=s.slot2id[tgt]; if(u>=0&&v>=0&&!(dbg&J2D_SKIP_ENGINE)) s.engine.cz_noinv(u,v); } } break;
            case MO_ARRAY_S: { int q=s.slot2id[a1]; if(q>=0&&!(dbg&J2D_SKIP_ENGINE)) s.engine.s_noinv(q,false); } break;
            case MO_ARRAY_T: { int q=s.slot2id[a1]; if(q>=0&&!(dbg&J2D_SKIP_ROT)){ int xb=(int)(JSIGBIT(theta_sig,rot_i));
                s.engine.apply_rotation(q, xb?-NV_T_ANGLE:NV_T_ANGLE); } rot_i++; } break;
            case MO_ARRAY_T_DAG: { int q=s.slot2id[a1]; if(q>=0&&!(dbg&J2D_SKIP_ROT)){ int xb=(int)(JSIGBIT(theta_sig,rot_i));
                s.engine.apply_rotation(q, xb?NV_T_ANGLE:-NV_T_ANGLE); } rot_i++; } break;
            case MO_EXPAND_T: { s.newq(a1); int q=s.slot2id[a1]; if(!(dbg&J2D_SKIP_ENGINE)) s.engine.h_noinv(q);
                int xb=(int)(JSIGBIT(theta_sig,rot_i)); if(!(dbg&J2D_SKIP_ROT)) s.engine.apply_rotation(q, xb?-NV_T_ANGLE:NV_T_ANGLE); rot_i++; } break;
            case MO_EXPAND_T_DAG: { s.newq(a1); int q=s.slot2id[a1]; if(!(dbg&J2D_SKIP_ENGINE)) s.engine.h_noinv(q);
                int xb=(int)(JSIGBIT(theta_sig,rot_i)); if(!(dbg&J2D_SKIP_ROT)) s.engine.apply_rotation(q, xb?NV_T_ANGLE:-NV_T_ANGLE); rot_i++; } break;
            case MO_SWAP_MEAS_INTERFERE: {
                int i_1=s.slot2id[a1], i_2=s.slot2id[a2]; s.slot2id[a1]=-1; s.slot2id[a2]=-1;
                if(i_1>=0) s.slot2id[a2]=i_1; if(i_2>=0) s.slot2id[a1]=i_2;
                int q=s.slot2id[a2]; if(q<0){ mag_i++; break; }
                int mag=mag_i; mag_i++;
                if(!(dbg&J2D_SKIP_ENGINE)) s.engine.h_noinv(q); if(!(dbg&J2D_SKIP_FWDMAP)) fwd_map(mag);
                uint64_t _tm=tm?__rdtsc():0;
                std::vector<int> Mkey = s.engine.M;
                int cvi = commit_find(mag, Mkey);
                bool compiled_fast = plan_compiled(mag, Mkey) && cvi>=0 && s.imem.count(imem_key(mag,bnd,Mkey));
                const JRegionMap& m=jp.maps[mag];
                foldxlog.clear(); s.engine.foldx_log=&foldxlog;
                int b;
                if(compiled_fast){
                    s.engine.set_inverse_phases(bnd.data());   // phases only (Imem key); no mask reconstruct
                    s.engine.inverse_off=true; b=s.measure_z(q); s.engine.inverse_off=false;
                    // phase_pack commit = bnd + rfd[M] + Σ foldx_log   (the 2A+ formula, no inverse)
                    for(int o=0;o<twoN;o++) pp[o]=(uint8_t)((bnd[o]+jp.maps[mag].commits[cvi].rfd[o])&3);
                    apply_foldx(pp, m, foldxlog); st.compiled_fast++;
                } else {                                       // oracle or cold variant: reconstruct + readback
                    s.engine.reconstruct_inverse(m.bnd_ax, m.bnd_az, bnd.data()); st.reconstructs++;
                    long pb0=s.engine.pullback_calls; int mo0=s.magic_oracle;
                    b=s.measure_z(q);
                    st.pullback_calls += (s.engine.pullback_calls-pb0);
                    bool is_oracle=(s.magic_oracle>mo0); if(is_oracle) st.oracle_count++; else st.cold_fallback++;
                    s.engine.read_phase_pack(pp.data());
                    if(!is_oracle && cvi<0){                   // SELF-WARM: capture rfd for this variant
                        std::vector<uint8_t> tmp=bnd; apply_foldx(tmp, m, foldxlog);
                        JCommitVariant ncv; ncv.M_key=Mkey; ncv.rfd.assign(twoN,0);
                        for(int o=0;o<twoN;o++) ncv.rfd[o]=(uint8_t)((pp[o]-tmp[o])&3);
                        if(jp.maps[mag].post_ax.empty()){ jp.maps[mag].post_ax.resize(n); jp.maps[mag].post_az.resize(n);
                            for(int ii=0;ii<n;ii++){ jp.maps[mag].post_ax[ii]=s.engine.inverse_frame.ax[ii]; jp.maps[mag].post_ax[ii].phase=0;
                                                     jp.maps[mag].post_az[ii]=s.engine.inverse_frame.az[ii]; jp.maps[mag].post_az[ii].phase=0; } }
                        jp.maps[mag].commits.push_back(ncv);
                    }
                }
                s.engine.foldx_log=nullptr;
                if(tm) C[1]+=__rdtsc()-_tm;
                int e=cp.dyn.outcome_base+mag; if(b) fire(e);
                s.slot2id[a2]=-1;
            } break;
            default: break;
        }
    }
    s.sampler.log_on=false; s.sampler.noapply=false;
    st.imem_miss = s.imem_misses;
    int nm=p.num_measurements; if(!(dbg&J2D_SKIP_RECORD)) for(int r=0;r<nm;r++) s.record.bits[r]=(uint8_t)(JSIGBIT(rec_sig,r));
    if(tm) C[0]+=__rdtsc()-_tsh;
    return s.err?1:0;
}

// ===== Gate J Phase-2E: Gate-J compiled-control + Gate-F-B region snapshot, merged ====================
// The 2D-3 breakdown proved the opcode-dispatch loop is ~0; the big eliminable cost is the engine
// FORWARD (tableau + pending conjugation ~2.9us + rotation pending.create ~1.3us).  Gate F-B already
// solved that with a per-measurement-boundary region snapshot.  2E splices F-B's tableau/pending
// snapshot onto 2D's compiled inverse/magic block:
//   * region active gates (cx/cz/s/h)  -> SKIPPED (FAST): no tableau conjugation, no pending conjugation
//   * rotations                        -> cap_theta: capture theta (from theta_sig), NO pending.create
//   * at each boundary                 -> fb_load_boundary: load static tableau masks + region phase
//                                         delta + rebuild the (static) snapshot pending set
//   * inverse-frame / magic            -> VERBATIM from run_jfast_2d (phase_pack fwd_map + compiled
//                                         magic + Imem inject; oracle/cold transitional + counted)
// shadow=true keeps the live _noinv forward AND verifies the snapshot at every boundary
// (fb_shadow_boundary); measure uses the live tableau (== run_jfast_2d + boundary checks).  Requires
// the F-B snapshot already built (run via nvm_mdam_sample_batch fb_mode=COMPILE) + jp/cp/imem warmed.
struct JFast2EStats { long opcode_dispatch=0, compiled_fast=0, reconstructs=0, pullback_calls=0,
    imem_miss=0, oracle_count=0, cold_fallback=0,
    tableau_conj=0,          // _noinv gate calls = tableau+pending conjugation (FAST: 0)
    pending_create_rot=0,    // apply_rotation in region = rotation pending.create (FAST: 0)
    cap_theta_count=0,       // theta captures without pending (FAST: = nrot)
    boundary_loads=0,        // fb_load_boundary calls (FAST)
    boundary_pending=0,      // pending rebuilt from snapshot at boundaries (the only pending creation)
    fb_mismatch=0,           // SHADOW: snapshot-vs-live boundary mismatch (must be 0)
    // Gate J Phase-2F-M counters (cmode==2: compiled magic bypasses generic measure_z plan/commit):
    dense_only_calls=0,      // 2F compiled magics executed via magic_compiled_fast (target 4/shot)
    phasepack_updates=0,     // direct phase_pack updates on the compiled path (target 4/shot)
    generic_measure_calls=0, // measure_z calls (generic plan/commit) — 2F: ONLY the oracle (1/shot)
    // Gate J Phase-2G (BoundaryPlan) counters (cmode==3):
    bplan_resolve=0,         // boundary-variant resolves (1/boundary = 5/shot)
    bplan_build=0,           // cold builds (= variant first-seen; linear scan only here, 0 after warm)
    imem_keybuild=0,         // imem keys built (target = compiled boundaries = 4/shot, was 8 in 2F)
    imem_probe=0;            // imem hash probes  (target = compiled boundaries = 4/shot, was 8 in 2F)
};
// 2E component-breakdown instrumentation (runtime, default OFF = release path unchanged).  j2e_dbg = A/B
// skip bitmask (correctness-breaking, timing-only, RNG-draw-count preserved); j2e_time = rdtsc accumulate.
// j2e_cyc: [0]=whole-shot, [1]=measure block (set_inverse_phases..commit), [2]=boundary prep (fb_load).
enum { J2E_SKIP_FIRE=1, J2E_SKIP_CAP=2, J2E_SKIP_BNDLOAD=4, J2E_SKIP_FWDMAP=8, J2E_SKIP_RECORD=16 };
inline int& j2e_dbg(){ static int d=0; return d; }
inline int& j2e_time(){ static int t=0; return t; }
// Noise-parity coarse wall-delta knob (TIMING-ONLY; trajectory-divergent for modes 1/2 -> NOT for correctness).
// 0 = full (sample+apply), 1 = draw-only (noise site loop + RNG draw + fault sample, NO frame/record apply),
// 2 = off (no site loop, no RNG, no apply).  Measurement Born RNG is untouched in all modes.
inline int& j2e_noise_mode(){ static int m=0; return m; }
// skip-to-next-fire: the gap-sampler already knows next_idx (the next firing site), so non-firing sites are
// pure no-ops -> visit ONLY the block(s) containing next_idx instead of scanning every site.  EXACT (correctness-
// preserving), default ON (shadow-verified bit-exact: 25/25 + 128k 0, draws/fires unchanged).  1 = skip-to-next-
// fire, 0 = per-site loop.  Affects only the j-fast modes (run_jfast_2e); authoritative run_shot is untouched.
inline int& j2e_noise_skip(){ static int s=1; return s; }
// [0]=whole-shot [1]=measure block [2]=boundary prep [4]=noise apply_site(sample) [5]=noise drain(apply)
// counters (tm-gated, EXACT regardless of rdtsc perturbation): [8]=noise_site_calls [9]=noise_draws [10]=noise_fires
//   [11]=noise_block_checks (skip-to-next-fire range checks) [12]=noise_blocks_skipped
inline uint64_t* j2e_cyc(){ static uint64_t c[16]={0}; return c; }
// cmode: 1 = SHADOW (live forward + fb_shadow_boundary verify), 0 = FAST-2E (snapshot load + compiled
// magic via measure_z), 2 = FAST-2F (snapshot load + compiled magic via magic_compiled_fast = NO generic
// measure_z plan/commit; dense kernel fed directly from StaticPlan + Imem + fb_theta), 3 = FAST-2G
// (2F + BoundaryPlan: per-(mag,M-variant) memoized dispatch — no Mkey heap copy, no plan/commit linear
// scan, imem key built ONCE & probed ONCE per compiled boundary).  The oracle/cold boundary always
// routes through measure_z (transitional, counted).  (1/0 preserves the old shadow-int API.)
inline int run_jfast_2e(MdamShot& s, const MdamProgram& p, const CompiledMdamProgram& cp,
                        JPhaseCompiled& jp, JFast2EStats& st, int cmode){
    if(!cp.fast_ok){ s.err="run_jfast_2e: fast sig unsupported"; return 1; }
    s.engine.lazy_inverse=false;   // fast path reconstructs the inverse frame at boundaries (reads .ax/.az directly)
    bool shadow=(cmode==1), dense_only=(cmode==2||cmode==3||cmode==4||cmode==5), use_bplan=(cmode==3||cmode==4||cmode==5);
    bool kshadow=(cmode==4);   // Gate K Step-2: maintain+verify the boundary-edge cache (NO live skip)
    bool kfast=(cmode==5);     // Gate K Step-4A: FAST — full edge HIT skips boundary_load/imem/dense/commit
    int n=s.engine.n, twoN=2*n;
    if(kshadow||kfast){ s.cur_sid=s.intern_state(s.engine.dense.resident.data(), s.engine.dense.r); s.dense_sid=s.cur_sid; }   // Step-4B-1: id of the initial resident state; Step-4B-4: engine.dense holds it (live)
    int j_tw=cp.theta_words, j_rw=cp.rec_words; uint64_t theta_sig[SIG_MAX_WORDS], rec_sig[SIG_MAX_WORDS]; for(int j_w=0;j_w<j_tw;j_w++) theta_sig[j_w]=cp.theta_init[j_w]; for(int j_w=0;j_w<j_rw;j_w++) rec_sig[j_w]=cp.rec_init[j_w];
    s.sampler.log_on=true; s.sampler.noapply=true; s.sampler.fire_log.clear(); size_t fire_cur=0;
    std::vector<uint8_t> pp(twoN,0), bnd(twoN,0); std::vector<int> foldxlog;
    std::vector<uint8_t> fb_prev; if(shadow) fb_prev.assign(4*n,0);   // post-commit phase of prev boundary
    if((int)s.fb_theta.size()<cp.nrot) s.fb_theta.resize(cp.nrot,0.0);
    int dbg=j2e_dbg(), tm=j2e_time(); uint64_t* C=j2e_cyc(); uint64_t _tsh=tm?__rdtsc():0;
    int nmode=j2e_noise_mode();   // 0 full / 1 draw-only / 2 off (timing-only)
    int nskip=j2e_noise_skip();   // 1 = skip-to-next-fire (visit only blocks containing next_idx), 0 = per-site loop
    auto fire_dynbit=[&](int site,uint64_t ci,uint64_t)->int{ return cp.dyn.noise_base[site]+(int)ci; };   // fire_log stores the channel index (multiword-safe)
    auto fire=[&](int e){ if(e<0) return; for(int j_w=0;j_w<j_tw;j_w++) theta_sig[j_w]^=cp.ev_theta[(size_t)e*j_tw+j_w]; for(int j_w=0;j_w<j_rw;j_w++) rec_sig[j_w]^=cp.ev_rec[(size_t)e*j_rw+j_w]; };
    auto drain=[&](){ for(;fire_cur<s.sampler.fire_log.size();++fire_cur){ auto&f=s.sampler.fire_log[fire_cur];
        fire(fire_dynbit((int)f[0],f[1],f[2])); } };
    auto fwd_map=[&](int mag){ const JRegionMap& m=jp.maps[mag];
        for(int o=0;o<twoN;o++){ int v=m.b[o]; for(int k=0;k<twoN;k++) v+=m.A[o*twoN+k]*pp[k]; bnd[o]=(uint8_t)(v&3); } };
    auto imem_key=[&](int mag, const std::vector<uint8_t>& ph, const std::vector<int>& M)->uint64_t{
        long mpack=(long)M.size(); for(size_t k=0;k<M.size();k++) mpack|=((long)(M[k]&15))<<(4*(k+1));
        uint64_t ip=0; for(int i=0;i<n;i++){ ip|=((uint64_t)(ph[i]&3))<<(4*i); ip|=((uint64_t)(ph[n+i]&3))<<(4*i+2); }
        return (uint64_t)mag | (ip<<4) | ((uint64_t)mpack<<(4+4*n)); };
    auto plan_compiled=[&](int mag, const std::vector<int>& M)->bool{
        if(mag>=(int)s.plan_cache.size()) return false;
        for(auto& sp : s.plan_cache[mag]) if(sp.M_key==M) return sp.state==1; return false; };
    auto plan_find=[&](int mag, const std::vector<int>& M)->StaticPlan*{   // 2F: get the StaticPlan skeleton
        if(mag>=(int)s.plan_cache.size()) return nullptr;
        for(auto& sp : s.plan_cache[mag]) if(sp.M_key==M) return &sp; return nullptr; };
    auto imem_find=[&](uint64_t key)->MdamShot::ImemEntry*{ auto it=s.imem.find(key); return it!=s.imem.end()?&it->second:nullptr; };
    auto commit_find=[&](int mag, const std::vector<int>& M)->int{
        if(mag>=(int)jp.maps.size()) return -1; auto& cv=jp.maps[mag].commits;
        for(size_t k=0;k<cv.size();k++) if(cv[k].M_key==M) return (int)k; return -1; };
    auto apply_foldx=[&](std::vector<uint8_t>& a, const JRegionMap& m, const std::vector<int>& log){
        for(int q : log){ for(int i=0;i<n;i++){ a[i]=(uint8_t)((a[i]+2*m.post_ax[i].getz(q))&3);
                                                a[n+i]=(uint8_t)((a[n+i]+2*m.post_az[i].getz(q))&3); } } };
    std::vector<double> cthetas;   // 2F: per-shot core rotation thetas (= fb_theta[core_uid]) — persistent (no per-boundary realloc)
    std::vector<int> Mpre;         // 2G: pre-measure M for the oracle/cold self-warm only (persistent buffer, no per-boundary malloc)
    std::vector<int> kM_in; std::vector<double> koracle_th; std::vector<uint8_t> ktp; int k_sid_in=0;   // Gate K: pre-boundary M + (Step-3) oracle thetas + (Step-4) tableau phase + (Step-4B-1) pre-boundary state id
    std::vector<uint8_t> k_pp_in;   // Step-4B-2: pre-boundary carried phase_pack snapshot (key+collision ingredient; captured before fwd_map/pp-overwrite)
    // 2G BoundaryPlan: pack M exactly as the Imem key's mpack field (so it doubles as the variant key
    // AND the cached mpack for the per-shot Imem key build).  O(|M|), no heap.
    auto imem_mpack=[&](const std::vector<int>& M)->uint64_t{
        uint64_t mpack=(uint64_t)M.size(); for(size_t k=0;k<M.size();k++) mpack|=((uint64_t)(M[k]&15))<<(4*(k+1)); return mpack; };
    auto ip_of=[&](const std::vector<uint8_t>& ph)->uint64_t{
        uint64_t ip=0; for(int i=0;i<n;i++){ ip|=((uint64_t)(ph[i]&3))<<(4*i); ip|=((uint64_t)(ph[n+i]&3))<<(4*i+2); } return ip; };
    // resolve (or build, once) the BoundaryVariant for (mag, mpack).  Scan is over <=4 uint64 mpacks
    // (NOT std::vector<int> M_key); the build (plan/commit index resolution) runs only on first-seen.
    auto bplan_resolve=[&](int mag, uint64_t mpack, const std::vector<int>& M)->MdamShot::BoundaryVariant*{
        if((int)s.boundary_cache.size()<=mag) s.boundary_cache.resize(mag+1);
        auto& vs=s.boundary_cache[mag];
        for(auto& v : vs) if(v.mpack==mpack) return &v;
        vs.push_back({mpack,false,false,-1,-1}); MdamShot::BoundaryVariant* bv=&vs.back(); st.bplan_build++;
        // cold build: resolve plan_idx (state==1) + commit_idx by M (linear, ONCE per variant)
        if(mag<(int)s.plan_cache.size()) for(size_t k=0;k<s.plan_cache[mag].size();k++) if(s.plan_cache[mag][k].M_key==M){ bv->plan_idx=(int)k; break; }
        if(mag<(int)jp.maps.size()){ auto& cv=jp.maps[mag].commits; for(size_t k=0;k<cv.size();k++) if(cv[k].M_key==M){ bv->commit_idx=(int)k; break; } }
        bv->compiled = (bv->plan_idx>=0 && s.plan_cache[mag][bv->plan_idx].state==1 && bv->commit_idx>=0);
        bv->built=true; return bv; };
    // cap_theta: store rotation theta (from the event-accumulated theta_sig bit) WITHOUT creating a
    // pending entry (FAST).  bit rot is stable once the rotation executes (its noise contributors all
    // fired before it), so fb_load_boundary rebuilds the snapshot pending with the correct theta.
    auto cap=[&](int rot,double theta){ s.fb_theta[rot]=theta; st.cap_theta_count++; };
    // Gate K key: FNV(state_id, M, pp, thetas).  Step-4B-2: keyed on the carried PRE-fwd_map phase_pack `pp`
    // (was `bnd`=fwd_map(pp)).  fwd_map is a deterministic per-mag affine fn, so the pp-key has the SAME edge
    // determinism as the bnd-key (it can only SPLIT bnd-merged entries, never merge distinct ones) — but the
    // hit path no longer needs fwd_map to build the key.  Used IDENTICALLY by kverify (shadow) and the FAST
    // early-out so the lookup key == the shadow key.  rpp/sign stay OUT of the key (fns of bnd+M via Imem).
    auto kkey=[&](int sid, const std::vector<int>& Mv, const std::vector<uint8_t>& ppv, const std::vector<double>* th)->uint64_t{
        uint64_t k=MdamShot::dfnv(1469598103934665603ULL, &sid, sizeof(int));   // Step-4B-1: 4-byte state id, NOT the 256-byte resident
        k=MdamShot::dfnv(k, Mv.data(), sizeof(int)*Mv.size());
        k=MdamShot::dfnv(k, ppv.data(), ppv.size());                           // Step-4B-2: carried pre-fwd_map pp (== pp_in); A/B-proven == bnd-key distinct count (fwd_map injective on reachable set)
        if(th&&!th->empty()) k=MdamShot::dfnv(k, th->data(), sizeof(double)*th->size());
        return k; };
    // Gate K Step-2: verify/store the boundary EDGE.  kphi_in/kM_in = pre-boundary snapshot; outputs read
    // LIVE from the engine post-boundary (resident=survivor, M, pp).  NO live skip — pure shadow.  Raw
    // inputs stored + compared on a key match (FNV-collision defense); p0 + per-outcome post-state verified.
    auto kverify=[&](int mag, bool is_oracle, const std::vector<int>* rpp, double sgn,
                     const std::vector<double>* thetas, int outcome, double p0){
        if((int)s.kcache.size()<=mag) s.kcache.resize(mag+1);
        uint64_t key=kkey(k_sid_in, kM_in, k_pp_in, thetas); (void)rpp; (void)sgn;
        int nsid=s.intern_state(s.engine.dense.resident.data(), s.engine.dense.r);   // survivor id; carry it for the next boundary's key
        s.cur_sid=nsid;
        auto& mm=s.kcache[mag]; s.k_lookup++; if(is_oracle) s.k_lookup_o++;
        bool antis=(is_oracle && !s.magic_scratch.anti_s.empty());   // Step-4B-3: stabilizer ag_measure branch (idraw2+out) — mark NOT fast-eligible
        size_t Nout=(size_t)1<<s.engine.dense.r;     // survivor size (resident AFTER the boundary)
        ktp.clear(); for(int i=0;i<n;i++){ ktp.push_back((uint8_t)(s.engine.tableau.Xc[i].phase&3)); ktp.push_back((uint8_t)(s.engine.tableau.Zc[i].phase&3)); }  // carried tableau phase
        auto it=mm.find(key);
        if(it==mm.end()){                            // MISS -> store this outcome's branch
            s.k_miss++; if(is_oracle) s.k_miss_o++;
            MdamShot::KEdge e; e.oracle=is_oracle; e.antis=antis; e.p0=p0; e.sid_in=k_sid_in; e.M_in=kM_in;
            if(rpp) e.rpp_in=*rpp; e.sign_in=sgn; if(thetas) e.th_in=*thetas; e.pp_in=k_pp_in; e.has[outcome]=true;
            e.surv[outcome].assign(s.engine.dense.resident.begin(), s.engine.dense.resident.begin()+Nout);
            e.Mout[outcome]=s.engine.M; e.ppout[outcome]=pp; e.tphase[outcome]=ktp; e.rout[outcome]=s.engine.dense.r; e.next_sid[outcome]=nsid;
            mm.emplace(key,std::move(e)); return;
        }
        s.k_hit++; if(is_oracle) s.k_hit_o++; MdamShot::KEdge& e=it->second;
        bool coll=(e.sid_in!=k_sid_in || e.M_in!=kM_in || e.pp_in!=k_pp_in);   // collision defense: small KEY inputs (state id, M, pp, thetas)
        if(thetas && e.th_in!=*thetas) coll=true;
        if(coll){ s.k_collision++; return; }
        bool mis=(e.p0!=p0);                         // p0 + per-outcome post-state determinism (bit-exact)
        if(e.has[outcome]){
            if(e.Mout[outcome]!=s.engine.M || e.ppout[outcome]!=pp || e.surv[outcome].size()!=Nout
               || e.tphase[outcome]!=ktp || e.rout[outcome]!=s.engine.dense.r || e.next_sid[outcome]!=nsid) mis=true;
            else for(size_t j=0;j<Nout;j++) if(e.surv[outcome][j]!=s.engine.dense.resident[j]){ mis=true; break; }
        } else { e.has[outcome]=true; e.surv[outcome].assign(s.engine.dense.resident.begin(), s.engine.dense.resident.begin()+Nout);
                 e.Mout[outcome]=s.engine.M; e.ppout[outcome]=pp; e.tphase[outcome]=ktp; e.rout[outcome]=s.engine.dense.r; e.next_sid[outcome]=nsid; }
        if(mis){ s.k_mismatch++; if(is_oracle) s.k_mismatch_o++; }
    };
    int dorm_i=0, read_i=0, mag_i=0, rot_i=0; size_t N=p.kind.size();
    for(size_t i=0;i<N && !s.err;i++){
        st.opcode_dispatch++;
        int a1=p.a1[i], a2=p.a2[i], i1=p.i1[i]; int i0=p.i0[i]; double dv=p.dval[i];
        switch((MdamOp)p.kind[i]){
            case MO_FRAME_H: case MO_FRAME_CNOT: case MO_FRAME_CZ: case MO_FRAME_SWAP:
            case MO_FRAME_S: case MO_APPLY_PAULI: case MO_MEAS_DORM_STATIC: break;
            case MO_NOISE: if(nmode<2){ uint64_t _tn=tm?__rdtsc():0; bool fired=false;
                              if(nskip){ if(s.sampler.next_idx==i0){ s.sampler.apply_site(i0, p.noise_sites[i0], s.frame); fired=true; if(tm)C[8]++; } if(tm){C[11]++; if(!fired)C[12]++;} }   // skip-to-next-fire: 1 site = 1 range check
                              else { s.sampler.apply_site(i0, p.noise_sites[i0], s.frame); fired=true; if(tm)C[8]++; }
                              if(tm)C[4]+=__rdtsc()-_tn;
                              if(nmode==0 && !(dbg&J2E_SKIP_FIRE) && (!nskip || fired)){ uint64_t _ta=tm?__rdtsc():0; drain(); if(tm)C[5]+=__rdtsc()-_ta; } } break;   // nmode 1=sample(no drain/apply), 2=skip
            case MO_NOISE_BLOCK: if(nmode<2){ uint64_t _tn=tm?__rdtsc():0; int lo=i0, hi=i0+i1; bool fired=false;
                              if(nskip){ if(s.sampler.next_idx>=lo && s.sampler.next_idx<hi){      // a fire in this block -> process firing sites only
                                             while(s.sampler.next_idx>=lo && s.sampler.next_idx<hi){ int fs=s.sampler.next_idx; s.sampler.apply_site(fs, p.noise_sites[fs], s.frame); if(tm)C[8]++; } fired=true; }
                                         if(tm){C[11]++; if(!fired)C[12]++;} }                      // else: whole block skipped (no fire in range)
                              else { for(int si=lo;si<hi;si++) s.sampler.apply_site(si, p.noise_sites[si], s.frame); fired=true; if(tm)C[8]+=i1; }
                              if(tm)C[4]+=__rdtsc()-_tn;
                              if(nmode==0 && !(dbg&J2E_SKIP_FIRE) && (!nskip || fired)){ uint64_t _ta=tm?__rdtsc():0; drain(); if(tm)C[5]+=__rdtsc()-_ta; } } break;
            case MO_READOUT_NOISE: { int e=cp.dyn.readout_base+(read_i++); double r=(nmode<2)?s.udraw():0.0; if(nmode==0 && r<dv) fire(e); } break;   // nmode 1=draw(no fire), 2=no draw
            case MO_MEAS_DORM_RANDOM: { int m=(nmode<2)?(int)s.idraw2():0; int e=cp.dyn.dormant_base+(dorm_i++); if(nmode==0 && m) fire(e); } break;
            case MO_ARRAY_CNOT: { int u=s.slot2id[a1],v=s.slot2id[a2]; if(u>=0&&v>=0&&shadow){ s.engine.cx_noinv(u,v); st.tableau_conj++; } } break;
            case MO_ARRAY_CZ: { int u=s.slot2id[a1],v=s.slot2id[a2]; if(u>=0&&v>=0&&shadow){ s.engine.cz_noinv(u,v); st.tableau_conj++; } } break;
            case MO_MULTI_CNOT: { int tgt=a1, t=s.slot2id[tgt]; uint64_t mask=p.mmask[i0];
                while(mask){ int ctrl=__builtin_ctzll(mask); mask&=mask-1; if(ctrl==tgt) continue;
                    int c=s.slot2id[ctrl]; if(t>=0&&c>=0&&shadow){ s.engine.cx_noinv(c,t); st.tableau_conj++; } } } break;
            case MO_MULTI_CZ: { uint64_t mask=p.mmask[i0];
                while(mask){ int tgt=__builtin_ctzll(mask); mask&=mask-1; if(tgt==a1) continue;
                    int u=s.slot2id[a1],v=s.slot2id[tgt]; if(u>=0&&v>=0&&shadow){ s.engine.cz_noinv(u,v); st.tableau_conj++; } } } break;
            case MO_ARRAY_S: { int q=s.slot2id[a1]; if(q>=0&&shadow){ s.engine.s_noinv(q,false); st.tableau_conj++; } } break;
            case MO_ARRAY_T: { int q=s.slot2id[a1]; if(q>=0){ int xb=(int)(JSIGBIT(theta_sig,rot_i)); double th=xb?-NV_T_ANGLE:NV_T_ANGLE;
                if(!(dbg&J2E_SKIP_CAP)) cap(rot_i,th); if(shadow){ s.engine.apply_rotation(q,th); st.pending_create_rot++; } } rot_i++; } break;
            case MO_ARRAY_T_DAG: { int q=s.slot2id[a1]; if(q>=0){ int xb=(int)(JSIGBIT(theta_sig,rot_i)); double th=xb?NV_T_ANGLE:-NV_T_ANGLE;
                if(!(dbg&J2E_SKIP_CAP)) cap(rot_i,th); if(shadow){ s.engine.apply_rotation(q,th); st.pending_create_rot++; } } rot_i++; } break;
            case MO_EXPAND_T: { s.newq(a1); int q=s.slot2id[a1]; if(shadow){ s.engine.h_noinv(q); st.tableau_conj++; }
                int xb=(int)(JSIGBIT(theta_sig,rot_i)); double th=xb?-NV_T_ANGLE:NV_T_ANGLE;
                if(!(dbg&J2E_SKIP_CAP)) cap(rot_i,th); if(shadow){ s.engine.apply_rotation(q,th); st.pending_create_rot++; } rot_i++; } break;
            case MO_EXPAND_T_DAG: { s.newq(a1); int q=s.slot2id[a1]; if(shadow){ s.engine.h_noinv(q); st.tableau_conj++; }
                int xb=(int)(JSIGBIT(theta_sig,rot_i)); double th=xb?NV_T_ANGLE:-NV_T_ANGLE;
                if(!(dbg&J2E_SKIP_CAP)) cap(rot_i,th); if(shadow){ s.engine.apply_rotation(q,th); st.pending_create_rot++; } rot_i++; } break;
            // ---- Gate L: coherent rotations (arbitrary-theta dv; xb from compiled theta_sig) + slot relabel ----
            case MO_ARRAY_ROT: { int q=s.slot2id[a1]; if(q>=0){ int xb=(int)(JSIGBIT(theta_sig,rot_i)); double th=xb?-dv:dv;
                if(!(dbg&J2E_SKIP_CAP)) cap(rot_i,th); if(shadow){ s.engine.apply_rotation(q,th); st.pending_create_rot++; } } rot_i++; } break;
            case MO_EXPAND_ROT: { s.newq(a1); int q=s.slot2id[a1]; if(shadow){ s.engine.h_noinv(q); st.tableau_conj++; }
                int xb=(int)(JSIGBIT(theta_sig,rot_i)); double th=xb?-dv:dv;
                if(!(dbg&J2E_SKIP_CAP)) cap(rot_i,th); if(shadow){ s.engine.apply_rotation(q,th); st.pending_create_rot++; } rot_i++; } break;
            case MO_ARRAY_SWAP: { int i_1=s.slot2id[a1], i_2=s.slot2id[a2]; s.slot2id[a1]=-1; s.slot2id[a2]=-1;
                if(i_1>=0) s.slot2id[a2]=i_1; if(i_2>=0) s.slot2id[a1]=i_2; } break;   // frame compiled away -> slot relabel only
            case MO_SWAP_MEAS_INTERFERE: {
                int i_1=s.slot2id[a1], i_2=s.slot2id[a2]; s.slot2id[a1]=-1; s.slot2id[a2]=-1;
                if(i_1>=0) s.slot2id[a2]=i_1; if(i_2>=0) s.slot2id[a1]=i_2;
                int q=s.slot2id[a2]; if(q<0){ mag_i++; break; }
                int mag=mag_i; mag_i++;
                if(kfast){
                    // === Gate K FAST: edge hit on the DRAWN branch skips boundary_load/imem/dense/commit ===
                    // hit-eligibility = has[drawn b] (NOT has[0]&&has[1]: near-deterministic measurements never fill the
                    // minority branch).  Double-draw avoided by injecting the pre-drawn rv into the live path (udraw()).
                    // Step-4B-2b: NO fwd_map — key is on the carried pp (not bnd), so a HIT needs no bnd; fwd_map runs
                    // only on the live miss/oracle fall-through (line ~940).
                    // Step-4B-3: ORACLE edges are FAST too — the oracle's core thetas == core_cache+fb_theta (proven,
                    // no flush_core) so the SAME key (line below) works for compiled AND oracle; its single Born udraw
                    // is reused via udraw()-injection on a partial.  he->oracle just selects the right counter.
                    kM_in=s.engine.M;
                    cthetas.clear(); for(uint32_t uid : s.core_cache[mag]) cthetas.push_back(uid<s.fb_theta.size()?s.fb_theta[uid]:0.0);
                    uint64_t key=kkey(s.cur_sid, kM_in, pp, &cthetas);        // Step-4B-2: key on the carried pre-fwd_map pp (== pp_in), NOT bnd
                    MdamShot::KEdge* he=nullptr;
                    if(mag<(int)s.kcache.size()){ auto it=s.kcache[mag].find(key); if(it!=s.kcache[mag].end()) he=&it->second; }
                    bool handled=false;
                    if(he && !he->antis){                                    // compiled OR Born-oracle edge (stabilizer ag_measure oracle stays live: idraw2+out, not Born)
                        bool okraw=(he->sid_in==s.cur_sid && he->M_in==kM_in && he->pp_in==pp && he->th_in==cthetas);  // collision defense (small KEY inputs)
                        if(okraw){
                            double rv=s.udraw();                             // ONE Born draw (same position as the live path's udraw)
                            int bb=(rv<he->p0)?0:1;
                            if(he->has[bb]){                                 // FULL HIT on the drawn branch
                                s.k_full_hit++;
                                // Step-4B-4: lazy carry — NO survivor byte copy; carry only cur_sid (engine.dense
                                // goes stale, re-materialized from state_amp[cur_sid] on the next live boundary).
                                s.engine.M=he->Mout[bb]; pp=he->ppout[bb]; s.cur_sid=he->next_sid[bb];   // carry the cached survivor's id (no re-hash, no byte copy)
                                for(int i=0;i<n;i++){ s.engine.tableau.Xc[i].phase=he->tphase[bb][2*i]; s.engine.tableau.Zc[i].phase=he->tphase[bb][2*i+1]; }
                                s.magic_point++; s.magic_seen++; if(he->oracle) s.magic_oracle++; else s.magic_compiled++;  // counters in sync (oracle measure_z indexes plan_cache[magic_point])
                                int e=cp.dyn.outcome_base+mag; if(bb) fire(e);
                                s.slot2id[a2]=-1; break;                     // boundary_load/imem/dense/commit (+ oracle reconstruct/measure_z) all SKIPPED
                            }
                            s.kfast_inj_rv=rv; s.kfast_use_inj=true; s.k_partial++; handled=true;  // drawn branch absent -> live reuses rv via udraw()
                        }
                    }
                    if(!handled){ s.k_miss5++; if(he && he->antis) s.k_antis_live++; }   // no entry / collision / anti_s -> fall through, live draws its own Born
                    // Step-4B-4: this is a live boundary -> materialize the (possibly stale) dense from cur_sid
                    if(s.dense_sid!=s.cur_sid){ s.materialize_dense(s.cur_sid); s.dense_sid=s.cur_sid; s.k_materialize++; }
                }
                uint64_t _tb=tm?__rdtsc():0;
                if(shadow){ s.engine.h_noinv(q); st.tableau_conj++; s.fb_shadow_boundary(mag, fb_prev); }   // live tableau + verify snapshot
                else if(!(dbg&J2E_SKIP_BNDLOAD)){ s.fb_load_boundary(mag); st.boundary_loads++;             // load snapshot tableau+pending (incl. boundary h)
                       if(mag<(int)s.fb_snap.size()) st.boundary_pending += (long)s.fb_snap[mag].puid.size(); }
                if(tm) C[2]+=__rdtsc()-_tb;
                if(!(dbg&J2E_SKIP_FWDMAP)){ fwd_map(mag); if(kfast) s.k_fwdmap++; }   // Step-4B-2b: fwd_map runs ONLY on the live (miss/oracle) path — HITS broke out before here, so fwd_map_on_hit=0
                uint64_t _tm=tm?__rdtsc():0;
                const JRegionMap& m=jp.maps[mag];
                // --- resolve compiled-ness + plan/commit/imem ---
                StaticPlan* pcp=nullptr; MdamShot::ImemEntry* ie=nullptr; int cvi=-1; bool compiled_fast=false;
                if(use_bplan){                                              // 2G: BoundaryPlan O(1) dispatch (no Mkey heap copy, key built+probed ONCE)
                    uint64_t mpack=imem_mpack(s.engine.M);
                    MdamShot::BoundaryVariant* bv=bplan_resolve(mag, mpack, s.engine.M); st.bplan_resolve++;
                    if(bv->compiled){
                        cvi=bv->commit_idx; pcp=&s.plan_cache[mag][bv->plan_idx];
                        uint64_t ikey=(uint64_t)mag | (ip_of(bnd)<<4) | (mpack<<(4+4*n)); st.imem_keybuild++;
                        ie=imem_find(ikey); st.imem_probe++; compiled_fast=(ie!=nullptr);
                    }
                } else {                                                    // 2E/2F: original dispatch (Mkey copy + double key/probe)
                    std::vector<int> Mkey = s.engine.M;
                    cvi = commit_find(mag, Mkey);
                    if(plan_compiled(mag, Mkey) && cvi>=0 && s.imem.count(imem_key(mag,bnd,Mkey))){
                        compiled_fast=true; pcp=plan_find(mag,Mkey); ie=imem_find(imem_key(mag,bnd,Mkey)); }
                }
                foldxlog.clear(); s.engine.foldx_log=&foldxlog;
                if(kshadow||kfast){ k_sid_in=s.cur_sid; kM_in=s.engine.M; k_pp_in=pp; }   // Step-4B-1/2: pre-boundary state id + M + carried pp (captured BEFORE pp is overwritten to pp_out)
                double kp0=-2.0;
                int b;
                if(compiled_fast && dense_only){                            // === 2F/2G: dense-only, NO generic measure_z ===
                    st.compiled_fast++;
                    cthetas.clear(); for(uint32_t uid : s.core_cache[mag]) cthetas.push_back(uid<s.fb_theta.size()?s.fb_theta[uid]:0.0);
                    double rv=s.udraw();  // Step-4B-3: udraw() now reuses the early-out's injected rv (compiled+oracle unified) — no double-draw
                    s.engine.inverse_off=true;
                    b=magic_compiled_fast(s.engine,*pcp,ie->rpp,ie->sign,cthetas,rv,s.magic_scratch, (kshadow||kfast)?&kp0:nullptr);
                    s.engine.inverse_off=false; st.dense_only_calls++;
                    s.magic_point++; s.magic_seen++; s.magic_compiled++;     // keep counters in sync (oracle's measure_z indexes plan_cache[magic_point])
                    for(int o=0;o<twoN;o++) pp[o]=(uint8_t)((bnd[o]+jp.maps[mag].commits[cvi].rfd[o])&3);
                    apply_foldx(pp, m, foldxlog); st.phasepack_updates++;
                    if(kshadow||kfast) kverify(mag,false,&ie->rpp,ie->sign,&cthetas,b,kp0);   // store/verify the edge (miss-path on kfast)
                } else if(compiled_fast){                                    // === 2E: compiled magic via measure_z (generic plan/commit) ===
                    s.engine.set_inverse_phases(bnd.data());                 // phases only (Imem key); no reconstruct
                    s.engine.inverse_off=true; b=s.measure_z(q); s.engine.inverse_off=false; st.generic_measure_calls++;
                    for(int o=0;o<twoN;o++) pp[o]=(uint8_t)((bnd[o]+jp.maps[mag].commits[cvi].rfd[o])&3);
                    apply_foldx(pp, m, foldxlog); st.compiled_fast++;
                } else {                                                     // oracle/cold variant: reconstruct + readback (transitional, counted)
                    Mpre.assign(s.engine.M.begin(), s.engine.M.end());       // pre-measure M (persistent buffer) for the cold self-warm
                    ORC_T(0, s.engine.reconstruct_inverse(m.bnd_ax, m.bnd_az, bnd.data())); st.reconstructs++;
                    long pb0=s.engine.pullback_calls; int mo0=s.magic_oracle;
                    ORC_T(7, b=s.measure_z(q)); st.generic_measure_calls++;
                    st.pullback_calls += (s.engine.pullback_calls-pb0);
                    bool is_oracle=(s.magic_oracle>mo0); if(is_oracle) st.oracle_count++; else st.cold_fallback++;
                    ORC_T(6, s.engine.read_phase_pack(pp.data()));
                    if(!is_oracle && cvi<0){                                 // SELF-WARM: capture rfd for this variant
                        std::vector<uint8_t> tmp=bnd; apply_foldx(tmp, m, foldxlog);
                        JCommitVariant ncv; ncv.M_key=Mpre; ncv.rfd.assign(twoN,0);
                        for(int o=0;o<twoN;o++) ncv.rfd[o]=(uint8_t)((pp[o]-tmp[o])&3);
                        if(jp.maps[mag].post_ax.empty()){ jp.maps[mag].post_ax.resize(n); jp.maps[mag].post_az.resize(n);
                            for(int ii=0;ii<n;ii++){ jp.maps[mag].post_ax[ii]=s.engine.inverse_frame.ax[ii]; jp.maps[mag].post_ax[ii].phase=0;
                                                     jp.maps[mag].post_az[ii]=s.engine.inverse_frame.az[ii]; jp.maps[mag].post_az[ii].phase=0; } }
                        jp.maps[mag].commits.push_back(ncv);
                    }
                    if((kshadow||kfast) && is_oracle){
                        if(s.core_cache[mag].empty()) for(auto* e : s.magic_scratch.core) s.core_cache[mag].push_back(e->uid);   // Step-4: cache oracle core uids (shot-static) so the FAST early-out builds cthetas WITHOUT flush_core
                        koracle_th.clear(); for(auto* e : s.magic_scratch.core) koracle_th.push_back(e->theta);   // Step-3: oracle core thetas (flush_core payload, deterministic scr.core order)
                        kverify(mag,true,nullptr,0.0,&koracle_th,b,s.magic_last_p0); }   // store/verify the edge (miss-path on kfast)
                }
                s.engine.foldx_log=nullptr;
                if(tm) C[1]+=__rdtsc()-_tm;
                if(shadow) s.fb_capture_phase(fb_prev);                      // post-commit phase for next region
                // Step-4B-4: this live boundary produced a survivor in engine.dense -> re-sync the carried identity.
                // Covers ALL miss-path branches uniformly (compiled, oracle, AND the cold self-warm path which has
                // no kverify); re-interns the same survivor kverify stored as next_sid (idempotent, ~miss-rate/shot).
                if(kfast){ s.cur_sid=s.intern_state(s.engine.dense.resident.data(), s.engine.dense.r); s.dense_sid=s.cur_sid; }
                int e=cp.dyn.outcome_base+mag; if(b) fire(e);
                s.slot2id[a2]=-1;
            } break;
            case MO_MEAS_ACTIVE_DIAGONAL:                            // Gate L: coherent active measure boundary (no swap; slot a1)
            case MO_MEAS_ACTIVE_INTERFERE: {                         // DIAGONAL: no boundary H; INTERFERE: boundary H. rec parity baked in rec_sig.
                bool interfere = ((MdamOp)p.kind[i]==MO_MEAS_ACTIVE_INTERFERE);
                int q=s.slot2id[a1]; if(q<0){ mag_i++; break; }
                int mag=mag_i; mag_i++;
                if(kfast){
                    kM_in=s.engine.M;
                    cthetas.clear(); for(uint32_t uid : s.core_cache[mag]) cthetas.push_back(uid<s.fb_theta.size()?s.fb_theta[uid]:0.0);
                    uint64_t key=kkey(s.cur_sid, kM_in, pp, &cthetas);        // Step-4B-2: key on the carried pre-fwd_map pp
                    MdamShot::KEdge* he=nullptr;
                    if(mag<(int)s.kcache.size()){ auto it=s.kcache[mag].find(key); if(it!=s.kcache[mag].end()) he=&it->second; }
                    bool handled=false;
                    if(he && !he->antis){
                        bool okraw=(he->sid_in==s.cur_sid && he->M_in==kM_in && he->pp_in==pp && he->th_in==cthetas);
                        if(okraw){
                            double rv=s.udraw();
                            int bb=(rv<he->p0)?0:1;
                            if(he->has[bb]){                                 // FULL HIT on the drawn branch
                                s.k_full_hit++;
                                s.engine.M=he->Mout[bb]; pp=he->ppout[bb]; s.cur_sid=he->next_sid[bb];
                                for(int i=0;i<n;i++){ s.engine.tableau.Xc[i].phase=he->tphase[bb][2*i]; s.engine.tableau.Zc[i].phase=he->tphase[bb][2*i+1]; }
                                s.magic_point++; s.magic_seen++; if(he->oracle) s.magic_oracle++; else s.magic_compiled++;
                                int e=cp.dyn.outcome_base+mag; if(bb) fire(e);
                                s.slot2id[a1]=-1; break;                     // boundary_load/imem/dense/commit all SKIPPED
                            }
                            s.kfast_inj_rv=rv; s.kfast_use_inj=true; s.k_partial++; handled=true;
                        }
                    }
                    if(!handled){ s.k_miss5++; if(he && he->antis) s.k_antis_live++; }
                    if(s.dense_sid!=s.cur_sid){ s.materialize_dense(s.cur_sid); s.dense_sid=s.cur_sid; s.k_materialize++; }
                }
                uint64_t _tb=tm?__rdtsc():0;
                if(shadow){ if(interfere){ s.engine.h_noinv(q); st.tableau_conj++; } s.fb_shadow_boundary(mag, fb_prev); }   // INTERFERE-only boundary H + verify
                else if(!(dbg&J2E_SKIP_BNDLOAD)){ s.fb_load_boundary(mag); st.boundary_loads++;             // load snapshot (incl. boundary h for INTERFERE; none for DIAGONAL)
                       if(mag<(int)s.fb_snap.size()) st.boundary_pending += (long)s.fb_snap[mag].puid.size(); }
                if(tm) C[2]+=__rdtsc()-_tb;
                if(!(dbg&J2E_SKIP_FWDMAP)){ fwd_map(mag); if(kfast) s.k_fwdmap++; }
                uint64_t _tm=tm?__rdtsc():0;
                const JRegionMap& m=jp.maps[mag];
                StaticPlan* pcp=nullptr; MdamShot::ImemEntry* ie=nullptr; int cvi=-1; bool compiled_fast=false;
                if(use_bplan){
                    uint64_t mpack=imem_mpack(s.engine.M);
                    MdamShot::BoundaryVariant* bv=bplan_resolve(mag, mpack, s.engine.M); st.bplan_resolve++;
                    if(bv->compiled){
                        cvi=bv->commit_idx; pcp=&s.plan_cache[mag][bv->plan_idx];
                        uint64_t ikey=(uint64_t)mag | (ip_of(bnd)<<4) | (mpack<<(4+4*n)); st.imem_keybuild++;
                        ie=imem_find(ikey); st.imem_probe++; compiled_fast=(ie!=nullptr);
                    }
                } else {
                    std::vector<int> Mkey = s.engine.M;
                    cvi = commit_find(mag, Mkey);
                    if(plan_compiled(mag, Mkey) && cvi>=0 && s.imem.count(imem_key(mag,bnd,Mkey))){
                        compiled_fast=true; pcp=plan_find(mag,Mkey); ie=imem_find(imem_key(mag,bnd,Mkey)); }
                }
                foldxlog.clear(); s.engine.foldx_log=&foldxlog;
                if(kshadow||kfast){ k_sid_in=s.cur_sid; kM_in=s.engine.M; k_pp_in=pp; }
                double kp0=-2.0;
                int b;
                if(compiled_fast && dense_only){
                    st.compiled_fast++;
                    cthetas.clear(); for(uint32_t uid : s.core_cache[mag]) cthetas.push_back(uid<s.fb_theta.size()?s.fb_theta[uid]:0.0);
                    double rv=s.udraw();
                    s.engine.inverse_off=true;
                    b=magic_compiled_fast(s.engine,*pcp,ie->rpp,ie->sign,cthetas,rv,s.magic_scratch, (kshadow||kfast)?&kp0:nullptr);
                    s.engine.inverse_off=false; st.dense_only_calls++;
                    s.magic_point++; s.magic_seen++; s.magic_compiled++;
                    for(int o=0;o<twoN;o++) pp[o]=(uint8_t)((bnd[o]+jp.maps[mag].commits[cvi].rfd[o])&3);
                    apply_foldx(pp, m, foldxlog); st.phasepack_updates++;
                    if(kshadow||kfast) kverify(mag,false,&ie->rpp,ie->sign,&cthetas,b,kp0);
                } else if(compiled_fast){
                    s.engine.set_inverse_phases(bnd.data());
                    s.engine.inverse_off=true; b=s.measure_z(q); s.engine.inverse_off=false; st.generic_measure_calls++;
                    for(int o=0;o<twoN;o++) pp[o]=(uint8_t)((bnd[o]+jp.maps[mag].commits[cvi].rfd[o])&3);
                    apply_foldx(pp, m, foldxlog); st.compiled_fast++;
                } else {
                    Mpre.assign(s.engine.M.begin(), s.engine.M.end());
                    ORC_T(0, s.engine.reconstruct_inverse(m.bnd_ax, m.bnd_az, bnd.data())); st.reconstructs++;
                    long pb0=s.engine.pullback_calls; int mo0=s.magic_oracle;
                    ORC_T(7, b=s.measure_z(q)); st.generic_measure_calls++;
                    st.pullback_calls += (s.engine.pullback_calls-pb0);
                    bool is_oracle=(s.magic_oracle>mo0); if(is_oracle) st.oracle_count++; else st.cold_fallback++;
                    ORC_T(6, s.engine.read_phase_pack(pp.data()));
                    if(!is_oracle && cvi<0){
                        std::vector<uint8_t> tmp=bnd; apply_foldx(tmp, m, foldxlog);
                        JCommitVariant ncv; ncv.M_key=Mpre; ncv.rfd.assign(twoN,0);
                        for(int o=0;o<twoN;o++) ncv.rfd[o]=(uint8_t)((pp[o]-tmp[o])&3);
                        if(jp.maps[mag].post_ax.empty()){ jp.maps[mag].post_ax.resize(n); jp.maps[mag].post_az.resize(n);
                            for(int ii=0;ii<n;ii++){ jp.maps[mag].post_ax[ii]=s.engine.inverse_frame.ax[ii]; jp.maps[mag].post_ax[ii].phase=0;
                                                     jp.maps[mag].post_az[ii]=s.engine.inverse_frame.az[ii]; jp.maps[mag].post_az[ii].phase=0; } }
                        jp.maps[mag].commits.push_back(ncv);
                    }
                    if((kshadow||kfast) && is_oracle){
                        if(s.core_cache[mag].empty()) for(auto* e : s.magic_scratch.core) s.core_cache[mag].push_back(e->uid);
                        koracle_th.clear(); for(auto* e : s.magic_scratch.core) koracle_th.push_back(e->theta);
                        kverify(mag,true,nullptr,0.0,&koracle_th,b,s.magic_last_p0); }
                }
                s.engine.foldx_log=nullptr;
                if(tm) C[1]+=__rdtsc()-_tm;
                if(shadow) s.fb_capture_phase(fb_prev);
                if(kfast){ s.cur_sid=s.intern_state(s.engine.dense.resident.data(), s.engine.dense.r); s.dense_sid=s.cur_sid; }
                int e=cp.dyn.outcome_base+mag; if(b) fire(e);
                s.slot2id[a1]=-1;
            } break;
            case MO_ARRAY_H: break;   // active Hadamard: frame H compiled symbolically (compile_jprogram fx<->fz); no engine op (mirrors ARRAY_CNOT/S, which touch the engine in shadow only)
            default: break;
        }
    }
    s.sampler.log_on=false; s.sampler.noapply=false;
    st.imem_miss = s.imem_misses; st.fb_mismatch = s.fb_mismatch;
    int nm=p.num_measurements; if(!(dbg&J2E_SKIP_RECORD)) for(int r=0;r<nm;r++) s.record.bits[r]=(uint8_t)(JSIGBIT(rec_sig,r));
    if(tm){ C[0]+=__rdtsc()-_tsh; C[9]+=s.sampler.draws; C[10]+=(uint64_t)s.sampler.fire_log.size(); }   // per-shot noise RNG draws + fires (gap-sampling: draws ~ fires+1)
    return s.err?1:0;
}

} // namespace mdam
