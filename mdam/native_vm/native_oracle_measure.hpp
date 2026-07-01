// native_oracle_measure.hpp — C2-C3: native port of the ORACLE measure_z magic branch
// (bounded.measure_z when try_compiled_measure returns None).  cultivation_d3 has NO stabilizer-
// branch measurements (verified: anti_s always empty) and _reduce_full/_drop_residual never fire,
// so this ports the magic branch: _flush_core (apply core rotations to phi via lincomb + promote)
// -> _localize_to_Z (h/s/cnot axis ops + frame folds) -> Born (branch_sqnorm) -> outcome -> project
// -> normalize -> drop_localized_core -> guarded no-op reduce.  Draws exactly ONE rng.next_double().
#pragma once
#include <vector>
#include <cmath>
#include "native_magic_state.hpp"
#include "native_magic_measure.hpp"     // dynamic_core, live_entries
#include "native_inverse_frame.hpp"     // pconj_*

namespace mdam {

struct OracleResult { int outcome; double p0; bool ok; const char* err; };

// _flush_core(0,1<<q): flush the anticommuting core into phi (== _do_flush -> _flush_one each).
// Gate E-B: core built into scratch (0 alloc); the compiled & oracle paths never overlap per measure.
inline void oracle_flush_core(NativeDenseEngineState& st, int q, MagicScratch& scr) {
    dynamic_core_scr(st.pending, q, st.W, scr);
    for (auto* e : scr.core) {
        // _flush_one(x,z,theta,phase): pullback -> pp+phase -> _masks(promote) -> lincomb(c,-i s)
        pb_kind()=2; PackedPauli pb = st.pullback(e->p);   // flush_pullback
        int pp = (pb.phase + e->p.phase) & 3;
        // _masks(promote=True): promote X-support qubits, then mx/mz over M layout
        for (int qq = 0; qq < st.n; qq++) if (pb.getx(qq)) { bool inM=false; for(int m:st.M) if(m==qq) inM=true; if(!inM) st.promote(qq); }
        uint64_t mx=0, mz=0;
        for (size_t j=0;j<st.M.size();j++){ int qq=st.M[j]; if (pb.getx(qq)) mx|=1ULL<<j; if (pb.getz(qq)) mz|=1ULL<<j; }
        double c = std::cos(e->theta/2.0), s = std::sin(e->theta/2.0);
        st.lincomb(mx, mz, pp, cd(c,0), cd(0,-s));
    }
    for (auto* e : scr.core) st.pending.consume(e->uid);
}

// _localize_to_Z(xp,zp,pp, prefer=q): returns (r, sign) or r=-1 deterministic(ev in sign).
// Applies W=H/S/CX to phi (axis ops) and folds W^dag into the frame (right_*).
inline int oracle_localize(NativeDenseEngineState& st, const PackedPauli& pm, int prefer, double& sign_out, MagicScratch& scr) {
    std::vector<int>& supp = scr.supp; supp.clear();
    for (int s : st.M) if (pm.getx(s) || pm.getz(s)) supp.push_back(s);
    std::sort(supp.begin(), supp.end());
    if (supp.empty()) {                          // no magic support -> deterministic +-1
        double ev = std::real(NativeDenseEngineState::iphase(pm.phase & 3));
        sign_out = (ev >= 0) ? 1.0 : -1.0; return -1;
    }
    int r = prefer; { bool in=false; for (int s:supp) if (s==prefer) in=true; if (!in) r=supp[0]; }
    PackedPauli P = pm;
    auto idx = [&](int qq){ for (size_t i=0;i<st.M.size();i++) if (st.M[i]==qq) return (int)i; return -1; };
    std::vector<WGate>& W = scr.Wg; W.clear();
    for (int s : supp) { int xb=P.getx(s), zb=P.getz(s);
        if (xb && zb) { W.push_back({1,s,0,true}); W.push_back({0,s,0,false}); } else if (xb) W.push_back({0,s,0,false}); }
    for (int s : supp) if (s != r) W.push_back({2,s,r,false});
    for (auto& g : W) {
        if (g.type==0){ st.h_axis(idx(g.a)); pconj_h(P,g.a); st.right_h(g.a); }
        else if (g.type==1){ st.s_axis(idx(g.a), g.sdag); pconj_s(P,g.a,g.sdag); st.right_s(g.a, !g.sdag); }
        else { st.cnot_axes(idx(g.a), idx(g.b)); pconj_cx(P,g.a,g.b); st.right_cx(g.a,g.b); }
    }
    sign_out = ((P.phase & 3)==0) ? 1.0 : -1.0;
    return r;
}

// Full oracle magic-branch measure_z.  Draws ONE rng.next_double() at the Born point.
inline OracleResult oracle_measure_magic(NativeDenseEngineState& st, int q, NativeRng& rng, MagicScratch& scr) {
    oracle_flush_core(st, q, scr);
    // anti_s stabilizer check: cultivation_d3 -> always empty; guard otherwise
    for (int i = 0; i < st.n; i++) { bool inM=false; for(int m:st.M) if(m==i) inM=true;
        if (!inM && !st.tableau.Zc_commutes_with_Zq(i, q)) return {-1,0.0,false,"stabilizer branch (unsupported for cultivation_d3)"}; }
    PackedPauli Pm_log(st.W); Pm_log.z[PackedPauli::word(q)] = PackedPauli::bit(q);
    pb_kind()=1; PackedPauli pm = st.pullback(Pm_log);   // oracle Pm_pullback
    double sign;
    int r = oracle_localize(st, pm, q, sign, scr);
    double p0; int outcome;
    if (r < 0) {                                  // deterministic
        p0 = std::max(0.0, std::min(1.0, (1.0 + sign) / 2.0));
        outcome = (rng.next_double() < p0) ? 0 : 1;
        if (!st.reduce_full_is_noop()) return {-1,0.0,false,"reduce_full would fire"};
        return {outcome, p0, true, nullptr};
    }
    int jr = -1; for (size_t i=0;i<st.M.size();i++) if (st.M[i]==r) jr=(int)i;
    dense_flop_collapse() += (uint64_t)12 << st.dense.r;   // collapse (Born+project+norm), == Clifft meas convention
    double s0 = st.branch_sqnorm(jr, 0), s1 = st.branch_sqnorm(jr, 1), tot = s0 + s1;
    p0 = (tot > 1e-300) ? ((sign > 0 ? s0 : s1) / tot) : 0.5;
    p0 = std::max(0.0, std::min(1.0, p0));
    outcome = (rng.next_double() < p0) ? 0 : 1;
    int plus_bit = (sign > 0) ? 0 : 1;
    int keepbit = (outcome == 0) ? plus_bit : (1 - plus_bit);
    // project: zero the (1-keepbit) branch at bit jr
    size_t N = (size_t)1 << st.dense.r;
    for (size_t s = 0; s < N; s++) if ((int)((s >> jr) & 1) == (1 - keepbit)) st.dense.resident[s] = cd(0,0);
    double nrm2 = (keepbit == 0) ? s0 : s1;
    if (nrm2 > 1e-24) { double inv = 1.0 / std::sqrt(nrm2); for (size_t s=0;s<N;s++) st.dense.resident[s] *= inv; }
    st.drop_localized_core(jr, keepbit);          // drop the localized measured axis
    st.drop_residual_products();                  // no-op for cultivation_d3 (verified)
    if (!st.reduce_full_is_noop()) return {-1,0.0,false,"reduce_full would fire"};
    return {outcome, p0, true, nullptr};
}

} // namespace mdam
