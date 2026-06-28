// native_magic_measure.hpp — C2-B: native port of compiled_core.try_compiled_measure.
// Split into magic_plan() (guards + kernel-input build; NO rng, NO state mutation) and
// magic_execute() (kernel + survivor commit + pending consume + frame folds + drop), so the
// integrated shot loop draws the single Born rng EXACTLY where Python does (after guards, before
// kernel) and falls back to the oracle path with NO wasted draw.
//
// Gate E-B (§4): ALL per-measurement temporaries + plan outputs live in a persistent MagicScratch
// (owned by the VM, sized once, clear()-only).  magic_plan writes into the scratch and returns a
// lightweight MagicPlan view (scalars + the scratch ptr); 0 heap allocation in the shot hot loop.
// Identical math / operation order / RNG semantics to the pre-scratch version (storage change only).
#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <x86intrin.h>          // __rdtsc for the 2F-M dense-vs-commit split (default-off)
#include "native_magic_state.hpp"
#include "native_inverse_frame.hpp"   // pconj_h/s/cx

namespace mdam {

struct WGate { int type; int a; int b; bool sdag; };   // type 0=h 1=s 2=cx

// Persistent per-measurement scratch — allocated once, reused via clear() (no realloc, no per-shot heap).
struct MagicScratch {
    std::vector<PendingEntry*> live;          // dynamic_core: live entries
    std::vector<char>          in_core;        // dynamic_core: membership
    std::vector<int>           stack;          // dynamic_core: DFS stack
    std::vector<PendingEntry*> core;           // selected core entries
    std::vector<int>           M_mat;          // magic-axis layout
    std::vector<PackedPauli>   pulled_p;       // pulled-back core Paulis
    std::vector<uint8_t>       pulled_pp;      // their phases
    std::vector<double>        pulled_theta;
    std::vector<int>           supp;
    std::vector<WGate>         Wg;             // localizer gate list (pre-conjugation)
    std::vector<WGate>         Wout;           // frame-fold localizer
    std::vector<std::array<int,3>> lm;
    // kernel inputs
    std::vector<uint64_t> rx, rz; std::vector<int> rpp; std::vector<double> rc, rs, rtheta;
    std::vector<int> lt, la, lb;
    std::vector<int> M_A;                       // post-measurement M
    std::vector<int> anti_s;                    // oracle stabilizer-branch candidates

    // Reserve to safe per-measurement upper bounds ONCE (no realloc in the hot loop afterwards).
    // n = engine qubit count; caps are generous (every qubit could be a magic axis / localizer op).
    void reserve_for(int n, int max_pending) {
        int P = max_pending > 0 ? max_pending : (n + 8);
        live.reserve(P); in_core.reserve(P); stack.reserve(P); core.reserve(P);
        M_mat.reserve(n + 4); pulled_p.reserve(P); pulled_pp.reserve(P); pulled_theta.reserve(P);
        supp.reserve(n + 4); Wg.reserve(2 * (n + 4)); Wout.reserve(2 * (n + 4)); lm.reserve(2 * (n + 4));
        rx.reserve(P); rz.reserve(P); rpp.reserve(P); rc.reserve(P); rs.reserve(P); rtheta.reserve(P);
        lt.reserve(2 * (n + 4)); la.reserve(2 * (n + 4)); lb.reserve(2 * (n + 4)); M_A.reserve(n + 4);
        anti_s.reserve(n + 4);
    }
};

// Gate F (F4): per-magic-point STATIC plan skeleton.  The audit (gate_f_audit.py, 50 seeds) proves
// that for a fixed magic-point the measurement's structural plan — M_mat layout, feasibility, the
// localizer gate list (Wout) + its kernel encoding (lt/la/lb), the rotation Pauli MASKS over the
// M_mat layout (rx/rz), and all the integer ranks (m_idx/r_qubit/rin/rmat/rout) — is SHOT-STATIC.
// Only the pullback PHASES (rpp), the rotation THETA (sign, frame-dependent), and the measurement
// SIGN are dynamic.  So we build the skeleton once (state=1 feasible / 2 infeasible) and on later
// shots load it and recompute ONLY the dynamic phases — eliminating the localizer/M_mat/encode cost.
// gate_f_audit (2000 seeds) proves the ONLY structural-variant source is st.M (the magic-axis vector):
// tableau masks and pending masks are shot-static; M varies (<=4 variants/boundary) because
// drop_residual_products drops different axes on rare seeds.  Since the skeleton is a deterministic
// function of (M, static masks), st.M ALONE is a complete & sufficient cache key.  We therefore keep a
// SMALL list of StaticPlan per magic-point, one per observed M; an unseen M takes the exact full path.
struct StaticPlan {
    int state = 0;                 // 0=unknown(not yet built), 1=feasible, 2=infeasible (cached)
    std::vector<int> M_key;        // the st.M this skeleton was built for (complete structural key)
    int m_idx=-1, r_qubit=-1, rin=0, rmat=0, rout=0;
    std::vector<int> M_mat;        // static magic-axis layout
    std::vector<WGate> Wout;       // static localizer gate list (also used by commit right_*)
    std::vector<int> lt, la, lb;   // static kernel localizer encoding
    std::vector<uint64_t> rx, rz;  // static rotation Pauli masks over the M_mat layout
};

struct MagicPlan {
    bool feasible = false;
    MagicScratch* s = nullptr;              // backing scratch (core/M_mat/rx../lt../Wout live here)
    int m_idx = -1, rin = 0, rmat = 0, rout = 0, r_qubit = -1;
    double sign = 1.0;
    // convenience accessors mirroring the old field names (so call-sites read unchanged)
    std::vector<int>&     M_mat() const { return s->M_mat; }
    std::vector<uint64_t>& rx()   const { return s->rx; }
    std::vector<uint64_t>& rz()   const { return s->rz; }
    std::vector<int>&     rpp()   const { return s->rpp; }
    std::vector<double>&  rc()    const { return s->rc; }
    std::vector<double>&  rs()    const { return s->rs; }
    std::vector<double>&  rtheta()const { return s->rtheta; }
    std::vector<int>&     lt()    const { return s->lt; }
    std::vector<int>&     la()    const { return s->la; }
    std::vector<int>&     lb()    const { return s->lb; }
    std::vector<WGate>&   Wout()  const { return s->Wout; }
    std::vector<PendingEntry*>& core() const { return s->core; }
};

struct NativeMagicTrace {
    bool fell_back = false; PackedPauli pulled_meas{1};
    std::vector<int> M_mat;
    std::vector<std::array<long long,3>> rots_xzpp; std::vector<double> rots_theta;
    std::vector<std::array<int,3>> lm;
    int m_bit=-1, rin=0, rmat=0, rout=0; double sign=1.0, p0=0.0; int outcome=-1;
    std::vector<int> M_after;
};

// Allocating variants (used by the oracle fallback path, 1/shot).  The compiled hot path uses the
// scratch versions below (0 allocation).
inline std::vector<PendingEntry*> live_entries(PendingLedger& L) {
    std::vector<PendingEntry*> v; for (auto& e : L.slots) if (e.generation == L.gen) v.push_back(&e); return v;
}
inline std::vector<PendingEntry*> dynamic_core(PendingLedger& L, int q, int W) {
    auto entries = live_entries(L); int N = (int)entries.size();
    std::vector<char> in_core(N, 0); std::vector<int> stack;
    PackedPauli Pm(W); Pm.z[PackedPauli::word(q)] = PackedPauli::bit(q);
    for (int j = 0; j < N; j++) if (!commute(Pm, entries[j]->p)) { in_core[j]=1; stack.push_back(j); }
    while (!stack.empty()) { int j = stack.back(); stack.pop_back();
        for (int k = 0; k < N; k++) if (!in_core[k] && !commute(entries[j]->p, entries[k]->p)) { in_core[k]=1; stack.push_back(k); } }
    std::vector<PendingEntry*> core; for (int j = 0; j < N; j++) if (in_core[j]) core.push_back(entries[j]); return core;
}

// dynamic_core into persistent scratch (live/in_core/stack/core); no allocation after warmup.
// §5 (E-C) cache: the core membership depends ONLY on the pending Paulis' X/Z bits, which evolve
// ONLY via the static forward Clifford gates (ag_measure touches Xc/Zc, NOT pending) and on the
// structural live-set at this measurement point.  Hence the core uid list is UNCONDITIONALLY
// shot-static (proven by code dependency).  cache_slot (optional): if non-empty, resolve uids ->
// live entries (skip the O(N^2) commute closure); if empty + provided, run closure AND fill it.
inline void dynamic_core_scr(PendingLedger& L, int q, int W, MagicScratch& s,
                             std::vector<uint32_t>* cache_slot=nullptr) {
    if (cache_slot && !cache_slot->empty()) {
        s.core.clear();
        for (uint32_t uid : *cache_slot)
            if (uid < L.slots.size() && L.slots[uid].generation == L.gen) s.core.push_back(&L.slots[uid]);
        return;
    }
    s.live.clear();
    for (auto& e : L.slots) if (e.generation == L.gen) s.live.push_back(&e);
    int N = (int)s.live.size();
    s.in_core.assign(N, 0); s.stack.clear();
    PackedPauli Pm(W); Pm.z[PackedPauli::word(q)] = PackedPauli::bit(q);
    for (int j = 0; j < N; j++) if (!commute(Pm, s.live[j]->p)) { s.in_core[j]=1; s.stack.push_back(j); }
    while (!s.stack.empty()) { int j = s.stack.back(); s.stack.pop_back();
        for (int k = 0; k < N; k++) if (!s.in_core[k] && !commute(s.live[j]->p, s.live[k]->p)) { s.in_core[k]=1; s.stack.push_back(k); } }
    s.core.clear();
    for (int j = 0; j < N; j++) if (s.in_core[j]) s.core.push_back(s.live[j]);
    if (cache_slot) { cache_slot->clear(); for (auto* e : s.core) cache_slot->push_back(e->uid); }
}

// Build the compiled-measurement plan for Z_q into `scr`.  Returns plan.feasible=false (== None) on
// stabilizer/X-on-nonmagic/deterministic/localizer-fail.  No rng, no state mutation, no heap alloc.
//
// Gate F (F4): optional `pc` = per-magic-point StaticPlan cache.  state==2 -> return infeasible (the
// branch is shot-static, proven by gate_f_audit).  state==1 -> FAST PATH: load the static skeleton
// (M_mat/Wout/lt/la/lb/rx/rz + ranks) and recompute ONLY the dynamic phases (rpp via pullback, theta
// sign, measurement sign) -> the localizer/M_mat/encode construction is eliminated.  state==0 ->
// full path, which also POPULATES pc.  pc==nullptr -> original behaviour (no cache).
inline MagicPlan magic_plan(NativeDenseEngineState& st, int q, MagicScratch& scr, double* psub=nullptr,
                            std::vector<uint32_t>* core_cache_slot=nullptr, StaticPlan* pc=nullptr,
                            const std::vector<int>* inj_rpp=nullptr, const double* inj_sign=nullptr) {
    MagicPlan P; P.s = &scr; int n = st.n, W = st.W;
#ifdef MDAM_PROFILE
    auto _nn=[](){ return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    double _ps0 = psub? _nn():0.0;
#endif
    dynamic_core_scr(st.pending, q, W, scr, core_cache_slot);
#ifdef MDAM_PROFILE
    double _ps1 = psub? _nn():0.0; if(psub) psub[0]+=_ps1-_ps0;
#endif
    // ---- F4 FAST PATH: cached static skeleton ----
    if (pc && pc->state == 2) { P.feasible = false; return P; }
    if (pc && pc->state == 1) {
        scr.M_mat = pc->M_mat; scr.Wout = pc->Wout;
        scr.lt = pc->lt; scr.la = pc->la; scr.lb = pc->lb;
        scr.rx = pc->rx; scr.rz = pc->rz;
        int nr = (int)scr.core.size();
        scr.rpp.resize(nr); scr.rc.resize(nr); scr.rs.resize(nr); scr.rtheta.resize(nr);
        for (int i=0;i<nr;i++) { PendingEntry* e = scr.core[i];
            double th = e->theta; scr.rtheta[i]=th; scr.rc[i]=std::cos(th/2.0); scr.rs[i]=std::sin(th/2.0);
            // Gate I (Imem): inject the cached pullback phase (hit) or compute it live (miss/shadow).
            if (inj_rpp) { scr.rpp[i] = (*inj_rpp)[i]; }
            else ISKIP(ISK_PULLBACK, {
            PackedPauli pb = st.pullback(e->p);
            scr.rpp[i] = (int)((pb.phase + e->p.phase) & 3); }); }
        if (inj_sign) { P.sign = *inj_sign; }
        else { P.sign = 1.0;
        ISKIP(ISK_SIGN, {
        PackedPauli Pm_log(W); Pm_log.z[PackedPauli::word(q)] = PackedPauli::bit(q);
        PackedPauli Pp = st.pullback(Pm_log);
        for (auto& g : pc->Wout) { if (g.type==0) pconj_h(Pp,g.a); else if (g.type==1) pconj_s(Pp,g.a,g.sdag); else pconj_cx(Pp,g.a,g.b); }
        P.sign = ((Pp.phase & 3)==0) ? 1.0 : -1.0; }); }
        P.m_idx=pc->m_idx; P.r_qubit=pc->r_qubit; P.rin=pc->rin; P.rmat=pc->rmat; P.rout=pc->rout;
#ifdef MDAM_PROFILE
        if(psub) psub[1]+= _nn()-_ps1;
#endif
#ifdef F4_DEBUG
        {   // recompute full plan into a temp scratch and compare every field
            static thread_local MagicScratch tmp; static thread_local bool init=false;
            if(!init){ tmp.reserve_for(st.n, st.n+64); init=true; }
            MagicPlan F = magic_plan(st, q, tmp, nullptr, nullptr, nullptr);
            auto bad=[&](const char* w){ fprintf(stderr,"[F4 DIFF] q=%d field=%s\n", q, w); };
            if(F.feasible!=P.feasible) bad("feasible");
            if(F.sign!=P.sign) bad("sign");
            if(F.m_idx!=P.m_idx) bad("m_idx"); if(F.r_qubit!=P.r_qubit) bad("r_qubit");
            if(F.rin!=P.rin) bad("rin"); if(F.rmat!=P.rmat) bad("rmat"); if(F.rout!=P.rout) bad("rout");
            if(scr.M_mat!=tmp.M_mat) bad("M_mat");
            if(scr.rx!=tmp.rx) bad("rx"); if(scr.rz!=tmp.rz) bad("rz");
            if(scr.rpp!=tmp.rpp) bad("rpp"); if(scr.rtheta!=tmp.rtheta) bad("rtheta");
            if(scr.rc!=tmp.rc) bad("rc"); if(scr.rs!=tmp.rs) bad("rs");
            if(scr.lt!=tmp.lt) bad("lt"); if(scr.la!=tmp.la) bad("la"); if(scr.lb!=tmp.lb) bad("lb");
            if(scr.core.size()!=tmp.core.size()) bad("core.size");
        }
#endif
        P.feasible = true; return P;
    }
    std::vector<int>& M_mat = scr.M_mat; M_mat.assign(st.M.begin(), st.M.end());
    auto in_Mmat = [&](int qq){ for (int m : M_mat) if (m==qq) return true; return false; };
    scr.pulled_p.clear(); scr.pulled_pp.clear(); scr.pulled_theta.clear();
    for (auto* e : scr.core) {
        PackedPauli pb = st.pullback(e->p); uint8_t pp = (uint8_t)((pb.phase + e->p.phase) & 3);
        scr.pulled_p.push_back(pb); scr.pulled_pp.push_back(pp); scr.pulled_theta.push_back(e->theta);
        for (int qq = 0; qq < n; qq++) if (pb.getx(qq) && !in_Mmat(qq)) M_mat.push_back(qq);
    }
#ifdef MDAM_PROFILE
    double _ps2 = psub? _nn():0.0; if(psub) psub[1]+=_ps2-_ps1;
#endif
    for (int i = 0; i < n; i++) if (!in_Mmat(i) && !st.tableau.Zc_commutes_with_Zq(i, q)) { if(pc) pc->state=2; P.feasible=false; return P; }
    PackedPauli Pm_log(W); Pm_log.z[PackedPauli::word(q)] = PackedPauli::bit(q);
    PackedPauli pm = st.pullback(Pm_log);
    auto tb = [&](const PackedPauli& X, uint64_t& xb, uint64_t& zb){ xb=zb=0;
        for (size_t l=0;l<M_mat.size();l++){ int qq=M_mat[l]; if (X.getx(qq)) xb|=1ULL<<l; if (X.getz(qq)) zb|=1ULL<<l; } };
    uint64_t Mx, Mz; tb(pm, Mx, Mz);
    auto x_on_nonmagic = [&](){ for (int qq=0;qq<n;qq++) if (pm.getx(qq) && !in_Mmat(qq)) return true; return false; };
    if (x_on_nonmagic() || (Mx==0 && Mz==0)) { if(pc) pc->state=2; P.feasible=false; return P; }
    std::vector<int>& supp = scr.supp; supp.clear();
    for (int qq : M_mat) if (pm.getx(qq) || pm.getz(qq)) supp.push_back(qq);
    std::sort(supp.begin(), supp.end());
    int r = q; { bool qin=false; for (int s2:supp) if (s2==q) qin=true; if (!qin) r=supp[0]; }
    int m_idx=-1; for (size_t i=0;i<M_mat.size();i++) if (M_mat[i]==r) m_idx=(int)i;
    PackedPauli Pp = pm;
    auto mmidx = [&](int qq){ for (size_t i=0;i<M_mat.size();i++) if (M_mat[i]==qq) return (int)i; return -1; };
    scr.Wout.clear(); scr.lm.clear(); scr.Wg.clear();
    // build localizer gate list (the old Wg pass folded inline into Wout/lm)
    for (int s2 : supp) { int xs=Pp.getx(s2), zs=Pp.getz(s2);
        if (xs && zs) { scr.Wg.push_back({1,s2,0,true}); scr.Wg.push_back({0,s2,0,false}); } else if (xs) scr.Wg.push_back({0,s2,0,false}); }
    for (int s2 : supp) if (s2 != r) scr.Wg.push_back({2,s2,r,false});
    for (auto& g : scr.Wg) {
        if (g.type==0){ pconj_h(Pp,g.a); scr.Wout.push_back(g); scr.lm.push_back({0,mmidx(g.a),0}); }
        else if (g.type==1){ pconj_s(Pp,g.a,g.sdag); scr.Wout.push_back(g); scr.lm.push_back({g.sdag?2:1,mmidx(g.a),0}); }
        else { pconj_cx(Pp,g.a,g.b); scr.Wout.push_back(g); scr.lm.push_back({3,mmidx(g.a),mmidx(g.b)}); }
    }
    bool reached = true; for (int wi=0;wi<W;wi++) if (Pp.x[wi]!=0) reached=false;
    { PackedPauli Zr(W); Zr.z[PackedPauli::word(r)]=PackedPauli::bit(r);
      for (int wi=0;wi<W;wi++) if (Pp.z[wi]!=Zr.z[wi]) reached=false; }
    if (!reached) { if(pc) pc->state=2; P.feasible=false; return P; }
    P.sign = ((Pp.phase & 3)==0) ? 1.0 : -1.0;
    P.m_idx = m_idx; P.r_qubit = r;
    P.rin = (int)st.M.size(); P.rmat = (int)M_mat.size(); P.rout = P.rmat-1;
    int nr = (int)scr.pulled_p.size();
    scr.rx.resize(nr); scr.rz.resize(nr); scr.rpp.resize(nr); scr.rc.resize(nr); scr.rs.resize(nr); scr.rtheta.resize(nr);
    for (int i=0;i<nr;i++){ uint64_t xb,zb; tb(scr.pulled_p[i],xb,zb);
        scr.rx[i]=xb; scr.rz[i]=zb; scr.rpp[i]=scr.pulled_pp[i]; scr.rtheta[i]=scr.pulled_theta[i];
        scr.rc[i]=std::cos(scr.pulled_theta[i]/2.0); scr.rs[i]=std::sin(scr.pulled_theta[i]/2.0); }
    scr.lt.resize(scr.lm.size()); scr.la.resize(scr.lm.size()); scr.lb.resize(scr.lm.size());
    for (size_t i=0;i<scr.lm.size();i++){ scr.lt[i]=scr.lm[i][0]; scr.la[i]=scr.lm[i][1]; scr.lb[i]=scr.lm[i][2]; }
#ifdef MDAM_PROFILE
    if(psub) psub[2]+= _nn()-_ps2;
#endif
    // F4: populate the static skeleton for subsequent shots (masks/encoding/ranks are shot-static).
    if (pc) {
        pc->M_mat=scr.M_mat; pc->Wout=scr.Wout; pc->lt=scr.lt; pc->la=scr.la; pc->lb=scr.lb;
        pc->rx=scr.rx; pc->rz=scr.rz;
        pc->m_idx=P.m_idx; pc->r_qubit=P.r_qubit; pc->rin=P.rin; pc->rmat=P.rmat; pc->rout=P.rout;
        pc->state=1;
    }
    P.feasible = true; return P;
}

// Execute the planned compiled measurement with a predetermined Born uniform; commit survivor + frame.
// prof_kernel/prof_commit (PROFILE only): split the dense kernel (arithmetic) from the symbolic commit.
inline int magic_execute(NativeDenseEngineState& st, MagicPlan& P, double rand_val, NativeMagicTrace* tr=nullptr,
                         double* prof_kernel=nullptr, double* prof_commit=nullptr) {
    MagicScratch& s = *P.s;
    int nrot=(int)s.rx.size(), nlm=(int)s.lt.size();
#ifdef MDAM_PROFILE
    double _tk0 = prof_kernel ? (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() : 0.0;
#endif
    CoreResult R = st.dense.execute_core(P.rmat,
        nrot?s.rx.data():nullptr, nrot?s.rz.data():nullptr, nrot?s.rpp.data():nullptr,
        nrot?s.rc.data():nullptr, nrot?s.rs.data():nullptr, nrot,
        nlm?s.lt.data():nullptr, nlm?s.la.data():nullptr, nlm?s.lb.data():nullptr, nlm,
        P.m_idx, P.sign, USE_RANDOM, rand_val);
#ifdef MDAM_PROFILE
    if (prof_kernel) { double _tk1=(double)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count(); *prof_kernel += _tk1-_tk0; }
#endif
    int outcome = R.outcome;
    // Gate F5: in F-B FAST the pending ledger is rebuilt from the snapshot at the next boundary, so the
    // pending consume is DISCARDED and fb_commit_mode==1 skips it.  The tableau right folds + fold_x are
    // NOT skipped: shadow/record verification proved the carried tableau PHASE is read by the oracle
    // measurement's ag_measure / inverse rebuild (inverse-only commit DIVERGES).  The expensive commit
    // op is the inverse-frame right fold (~12 O(n)/shot) which is needed to keep the live inverse.
    if (st.fb_commit_mode!=1) for (auto* e : s.core) { st.pending.consume(e->uid);
#ifdef FB_COUNT
        fbc().consume++;
#endif
    }
    s.M_A.clear(); for (int i=0;i<P.rmat;i++) if (i!=P.m_idx) s.M_A.push_back(s.M_mat[i]);
    st.M = s.M_A;
#ifdef FB_COUNT
    fbc().mupd++;
#endif
    ISKIP(ISK_RIGHTFOLD, { for (auto& g : s.Wout) { if (g.type==0) st.right_h(g.a); else if (g.type==1) st.right_s(g.a, !g.sdag); else st.right_cx(g.a,g.b); } });
    int plus_bit = (P.sign>0)?0:1; int keepbit = (outcome==0)?plus_bit:(1-plus_bit);
    ISKIP(ISK_FOLDX, { if (keepbit==1) st.fold_x(P.r_qubit); });
    ITIME_BEG(IT_DROP); st.drop_residual_products(); ITIME_END(IT_DROP);
    (void)prof_commit;   // commit = (execute total - kernel) computed by the caller (measure_z)
    if (tr) { tr->M_mat=s.M_mat; tr->m_bit=P.m_idx; tr->rin=P.rin; tr->rmat=P.rmat; tr->rout=P.rout;
              tr->sign=P.sign; tr->p0=R.p0; tr->outcome=outcome;
              for (int i=0;i<nrot;i++){ tr->rots_xzpp.push_back({(long long)s.rx[i],(long long)s.rz[i],(long long)s.rpp[i]}); tr->rots_theta.push_back(s.rtheta[i]); }
              for (int i=0;i<nlm;i++) tr->lm.push_back({s.lt[i],s.la[i],s.lb[i]});
              tr->M_after=st.M; }
    return outcome;
}

// Gate J Phase-2F-M: dedicated compiled-magic execution.  BYPASSES magic_plan entirely (no skeleton
// vector copies, no dynamic_core_scr, no Imem-key build, no pullback/sign recompute) AND magic_execute's
// generic dispatch.  Feeds the dense kernel DIRECTLY from the precompiled StaticPlan (rx/rz/lt/la/lb/
// m_idx/rmat/Wout/M_mat) + Imem-injected rpp/sign + per-shot rotation thetas (the caller reads
// fb_theta[core_uid]).  Commit = exactly the magic_execute ops that matter on the compiled path: M
// update + right-folds(Wout) + fold_x + drop_residual_products.  Pending consume is SKIPPED (the F-B
// snapshot rebuilds/invalidates pending at the next boundary).  inverse_off must be set by the caller
// (commit folds the tableau only; the phase_pack carries the inverse — set by the caller via rfd+foldx).
// Bit-identical to measure_z's compiled path by construction (same kernel inputs, same commit ops).
// 2F-M dense-vs-commit timing split (default OFF; rdtsc, no correctness change).  cyc[0]=dense kernel,
// cyc[1]=commit (M-update + right-folds + fold_x + drop_residual).
inline int& mcf_time(){ static int t=0; return t; }
inline uint64_t* mcf_cyc(){ static uint64_t c[2]={0,0}; return c; }
// 2C+ oracle-path dissection (default OFF; rdtsc).  cyc slots:
//   [0]=reconstruct_inverse  [1]=flush_core  [2]=anti_s scan  [3]=pullback+oracle_localize
//   [4]=branch_sqnorm+project+normalize  [5]=drop+reduce  [6]=read_phase_pack  [7]=measure_z(oracle) total
inline int& orc_time(){ static int t=0; return t; }
inline uint64_t* orc_cyc(){ static uint64_t c[8]={0,0,0,0,0,0,0,0}; return c; }
#define ORC_T(slot, code) do{ if(mdam::orc_time()){ uint64_t _o=__rdtsc(); code; mdam::orc_cyc()[slot]+=__rdtsc()-_o; } else { code; } }while(0)
inline int magic_compiled_fast(NativeDenseEngineState& st, const StaticPlan& pc,
                               const std::vector<int>& rpp, double sign,
                               const std::vector<double>& thetas, double rand_val, MagicScratch& scr,
                               double* p0_out=nullptr){
    int nr=(int)pc.rx.size(), nlm=(int)pc.lt.size();
    scr.rc.resize(nr); scr.rs.resize(nr);
    for(int i=0;i<nr;i++){ double th=thetas[i]; scr.rc[i]=std::cos(th*0.5); scr.rs[i]=std::sin(th*0.5); }
    int _tm=mcf_time(); uint64_t* _C=mcf_cyc(); uint64_t _t0=_tm?__rdtsc():0;
    CoreResult R = st.dense.execute_core(pc.rmat,
        nr?pc.rx.data():nullptr, nr?pc.rz.data():nullptr, nr?rpp.data():nullptr,
        nr?scr.rc.data():nullptr, nr?scr.rs.data():nullptr, nr,
        nlm?pc.lt.data():nullptr, nlm?pc.la.data():nullptr, nlm?pc.lb.data():nullptr, nlm,
        pc.m_idx, sign, USE_RANDOM, rand_val);
    if(_tm){ _C[0]+=__rdtsc()-_t0; _t0=__rdtsc(); }
    if(p0_out) *p0_out=R.p0;       // Gate K shadow: expose Born p0 for the edge-cache verify
    int outcome=R.outcome;
    scr.M_A.clear(); for(int i=0;i<pc.rmat;i++) if(i!=pc.m_idx) scr.M_A.push_back(pc.M_mat[i]); st.M=scr.M_A;
    for(const auto& g:pc.Wout){ if(g.type==0) st.right_h(g.a); else if(g.type==1) st.right_s(g.a,!g.sdag); else st.right_cx(g.a,g.b); }
    int plus_bit=(sign>0)?0:1; int keepbit=(outcome==0)?plus_bit:(1-plus_bit);
    if(keepbit==1) st.fold_x(pc.r_qubit);
    st.drop_residual_products();
    if(_tm) _C[1]+=__rdtsc()-_t0;
    return outcome;
}

// Compatibility wrapper for the C2-B snapshot test (predetermined rand_val): owns a local scratch.
inline int native_measure_magic_z(NativeDenseEngineState& st, int q, double rand_val, NativeMagicTrace* tr) {
    static thread_local MagicScratch scr;
    MagicPlan P = magic_plan(st, q, scr);
    if (!P.feasible) { if (tr) tr->fell_back = true; return -1; }
    return magic_execute(st, P, rand_val, tr);
}

} // namespace mdam
