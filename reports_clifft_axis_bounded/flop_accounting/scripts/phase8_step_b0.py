"""Step B0 -- event-level differential shadow (NO engine change, NO dispatch activation).

Proves, per T, that a candidate Policy-3 DIAGONAL path produces the IDENTICAL physical state as the
existing exact path (the off-diagonal butterfly that cultivation actually runs, since its peak rank
10 < the _loc_min_size=2^14 localizer gate).  The candidate is run ONLY on CLONES / a throwaway
engine; the authoritative Phase-2 path is never touched.  a05843e / tag / fallbacks preserved.

Design (why this is faithful AND tractable):
  The butterfly leaves the Clifford frame (tableau Xc/Zc) UNTOUCHED.  The candidate is built
  frame-preserving (apply V, diagonal T/T^dag, UNDO V), so its tableau is BIT-IDENTICAL to the
  butterfly's.  For two bounded states with the SAME U_C, |Psi1>=U_C(phi1 (x) |0>) equals
  gamma|Psi2>=gamma U_C(phi2 (x) |0>) IFF phi1 = gamma phi2 (U_C unitary => injective).  So comparing
  the 2^r magic register `phi` up-to-global-phase is EXACTLY comparing the full 2^n physical states
  -- no 2^n materialization needed (cultivation_d5 n=16).  For d3 (n=6) we ALSO materialize the full
  statevector via the independent U_C-matrix path as a cross-check of this reduction.

Candidate construction (explicit, in the MAGIC-BIT (mx,mz,pp) space the butterfly uses):
  pullback -> (mx,mz,pp) ; free CNOT-collapse the X-string onto pivot a ; (S^dag if local Y) ; the
  ONE born-basis H (X_a->Z_a) ; free CNOT-collapse the Z-string -> generator is sign*Z_a.  Then the
  Policy-3 DIAGONAL step: apply gate T (sign>0) or T^dag (sign<0) on bit a, gamma <- gamma * e^{-i s theta/2}
  (the rotation<->gate convention phase), and UNDO V.  sign==-1 <=> an effective X-residue on the
  axis <=> T->T^dag, the rule  T X^x Z^z = omega^x X^x Z^z T^{(-1)^x}.
"""
import sys; sys.path.insert(0, "/home/jung/clifft-paper"); sys.setrecursionlimit(400000)
import copy
import numpy as np
import clifft  # noqa
import nearclifford_backend.backend as bk
from nearclifford_backend.lazy import _conj_h, _conj_s, _conj_cx
from nearclifford_backend.clifft_axis.bounded import compile_bounded, CliftAxisBoundedNearClifford as C

OMEGA = np.exp(1j * np.pi / 4)


def _conj(P, g):
    if g[0] == "h":
        return _conj_h(P, g[1])
    if g[0] == "s":
        return _conj_s(P, g[1], g[2])
    return _conj_cx(P, g[1], g[2])


def candidate_decompose(eng, x, z, theta, phase):
    """Build the Policy-3 diagonal decomposition for ONE rotation on `eng` (mutating eng.phi only).
    Returns metadata. Frame (Xc/Zc) is left untouched (V applied then undone)."""
    xp, zp, pp = eng._pullback(x, z)
    pp = (pp + phase) & 3
    mx, mz = eng._masks(xp, zp, promote=True, where="rot")     # magic-bit masks; promotes like real
    k = len(eng.M)
    weight = int(xp | zp).bit_count()
    if mx == 0 and mz == 0:                                     # generator is a global phase i^pp
        s_sign = 1 if (pp & 3) == 0 else -1
        return dict(ok=True, a=None, born="I", s_sign=s_sign, gamma=np.exp(-1j * s_sign * theta / 2.0),
                    pp=pp, weight=weight, px=(1 if s_sign < 0 else 0), pz=0, W=[], mx=mx, mz=mz)
    P = (mx, mz, pp); W = []
    xsupp = [j for j in range(k) if (mx >> j) & 1]
    if xsupp:
        a = xsupp[0]
        for b in xsupp:
            if b != a:
                g = ("cx", a, b); W.append(g); P = _conj(P, g)          # collapse X-string onto a (free)
        if (P[0] >> a) & 1 and (P[1] >> a) & 1:
            g = ("s", a, True); W.append(g); P = _conj(P, g); born = "Y" # local Y -> S^dag -> X
        else:
            born = "X"
        g = ("h", a); W.append(g); P = _conj(P, g)                       # the ONE born-basis H: X_a->Z_a
        for b in [j for j in range(k) if j != a and (P[1] >> j) & 1]:
            g = ("cx", b, a); W.append(g); P = _conj(P, g)               # collapse Z-string onto a (free)
        pz_local = 0
    else:                                                                 # pure-Z (diagonal already)
        zsupp = [j for j in range(k) if (mz >> j) & 1]
        a = zsupp[0]; born = "Z"
        for b in zsupp:
            if b != a:
                g = ("cx", b, a); W.append(g); P = _conj(P, g)           # collapse Z-string onto a (free)
        pz_local = 1
    ok = (P[0] == 0 and P[1] == (1 << a) and (P[2] & 1) == 0)
    if not ok:
        return dict(ok=False, why=f"collapse->{P}", weight=weight, mx=mx, mz=mz)
    s_sign = 1 if (P[2] & 3) == 0 else -1
    return dict(ok=True, a=a, born=born, s_sign=s_sign, gamma=np.exp(-1j * s_sign * theta / 2.0),
                pp=pp, weight=weight, px=(1 if s_sign < 0 else 0), pz=pz_local, W=W, mx=mx, mz=mz,
                theta=theta)


def candidate_apply(eng, meta, theta):
    """Apply the decomposed candidate path to eng.phi (V, diagonal T/T^dag, undo V). Returns gamma."""
    if meta["a"] is None:                       # global phase only
        return meta["gamma"]
    W = meta["W"]; a = meta["a"]; s = meta["s_sign"]
    for g in W:
        if g[0] == "h":
            eng._h_axis(g[1])
        elif g[0] == "s":
            eng._s_axis(g[1], g[2])
        else:
            eng._cnot_axes(g[1], g[2])
    v = eng.phi.reshape(-1, 2, 1 << a)
    v[:, 1, :] *= np.exp(1j * s * theta)        # gate T (s>0) / T^dag (s<0) on bit a
    for g in reversed(W):
        if g[0] == "h":
            eng._h_axis(g[1])
        elif g[0] == "s":
            eng._s_axis(g[1], not g[2])
        else:
            eng._cnot_axes(g[1], g[2])
    return meta["gamma"]


def candidate_flush(eng, x, z, theta, phase=0):
    """Authoritative candidate _flush_one (drops the unobservable global phase). Used by the
    throwaway whole-run candidate engine (Part 2)."""
    meta = candidate_decompose(eng, x, z, theta, phase)
    if not meta["ok"]:
        raise RuntimeError(f"candidate collapse failed: {meta}")
    candidate_apply(eng, meta, theta)


# ----------------------------- independent full-state materialization (small n) ---------------- #
def materialize(eng):
    """Full 2^n statevector |Psi> = U_C (phi (x) |0>) via the independent U_C-matrix path
    (verification only; small n).  Clears pending on a deepcopy so only the CURRENT phi+frame
    materialize (no future rotations)."""
    e = copy.deepcopy(eng)
    e.pending = {}
    from nearclifford_backend.simulator import NearClifford
    return NearClifford.statevector(e)


def upto_phase(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-300 or nb < 1e-300:
        return float(abs(na - nb))
    return float(abs(abs(np.vdot(a, b)) - na * nb))      # 0 iff a ∝ b


# ============================== PART 1: per-T differential (fork) ============================== #
def part1(circ, seed, full_state_check=False):
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    of1 = C._flush_one
    rows = []
    rng_state = {"last_clone": None}

    def f1(self, x, z, theta, phase=0):
        pre = copy.deepcopy(self)
        rank = self.phi.size.bit_length() - 1
        r = of1(self, x, z, theta, phase)               # SOURCE OF TRUTH (butterfly)
        phiS = self.phi.copy()
        cand = copy.deepcopy(pre); cand.budget.enforce = False
        meta = candidate_decompose(cand, x, z, theta, phase)
        if not meta["ok"]:
            rows.append(dict(ok=False, rank=rank, **meta))
            return r
        candidate_apply(cand, meta, theta)
        g = meta["gamma"]; phiC = cand.phi
        # comparisons
        same_shape = (phiS.shape == phiC.shape)
        incl = float(np.max(np.abs(phiS - g * phiC))) if same_shape else 9.9     # item 2 (incl gamma)
        upg = upto_phase(phiS, phiC) if same_shape else 9.9                       # item 1 (up to phase)
        nrm = float(abs(np.linalg.norm(phiS) - np.linalg.norm(phiC)))            # item 5 (norm)
        fr_ok = (self.Xc == cand.Xc and self.Zc == cand.Zc and self.M == cand.M) # item 3/7 (frame/map)
        rank_ok = (self.phi.size == cand.phi.size and self.M == cand.M)
        # item 4: random magic-Pauli expectation equality (frame identical -> same Pauli on phi)
        kk = len(self.M); rxp = 0; rzp = 0
        if kk:
            h = (hash((circ, seed, len(rows))) & ((1 << (2 * kk)) - 1))
            rxp = h & ((1 << kk) - 1); rzp = (h >> kk) & ((1 << kk) - 1)
        eS = self._pauli_expectation(rxp, rzp, 0, where="exp") if kk else 0.0
        eC = cand._pauli_expectation(rxp, rzp, 0, where="exp") if kk else 0.0
        pauli_exp_diff = float(abs(eS - eC))
        # full-state independent cross-check (small n only)
        fs = None
        if full_state_check:
            psiS = materialize(self); psiC = materialize(cand)
            fs = dict(upg=upto_phase(psiS, psiC), incl=float(np.max(np.abs(psiS - g * psiC))))
        rows.append(dict(ok=True, rank=rank, born=meta["born"], a=meta["a"], px=meta["px"],
                         pz=meta["pz"], pp=meta["pp"], s_sign=meta["s_sign"], weight=meta["weight"],
                         incl=incl, upg=upg, nrm=nrm, fr_ok=fr_ok, rank_ok=rank_ok,
                         pexp=pauli_exp_diff, fs=fs, theta=theta))
        return r

    C._flush_one = f1
    try:
        be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                    structure_once=False, clifft_axis_enforce=True)
        rec = tuple(be.run_shot(prog, seed))
        pk = be.nc.budget.peak_resident.bit_length() - 1
    finally:
        C._flush_one = of1
    return rows, rec, pk


# ===================== PART 2: whole-run authoritative candidate engine ======================= #
def part2(circ, seed):
    """Run the circuit with the real (butterfly) engine and with the authoritative candidate path
    (the Policy-3 DIAGONAL dispatch made authoritative for _flush_one on a THROWAWAY run; the
    unobservable global phase is dropped). Compares records, peak rank, per-measurement Born p0,
    and the |M| before/after schedule -- i.e. validates the candidate THROUGH real measurements/drops."""
    prog = compile_bounded(open(f"qec_bench/circuits/{circ}.stim").read())
    be = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                structure_once=False, clifft_axis_enforce=True)
    recR = tuple(be.run_shot(prog, seed)); pkR = be.nc.budget.peak_resident.bit_length() - 1
    p0R = [c.get("p0") for c in be.nc.core_log]; mR = [(c["M_before"], c["M_after"]) for c in be.nc.core_log]

    of1 = C._flush_one
    try:
        C._flush_one = candidate_flush                # authoritative diagonal dispatch (throwaway)
        be2 = bk.NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False,
                                     structure_once=False, clifft_axis_enforce=True)
        recC = tuple(be2.run_shot(prog, seed)); pkC = be2.nc.budget.peak_resident.bit_length() - 1
        p0C = [c.get("p0") for c in be2.nc.core_log]; mC = [(c["M_before"], c["M_after"]) for c in be2.nc.core_log]
    finally:
        C._flush_one = of1
    rec_ok = (recR == recC)
    pk_ok = (pkR == pkC)
    p0_ok = (len(p0R) == len(p0C) and all((a is None and b is None) or
             (a is not None and b is not None and abs(a - b) < 1e-9) for a, b in zip(p0R, p0C)))
    map_ok = (mR == mC)
    p0maxdiff = max((abs(a - b) for a, b in zip(p0R, p0C) if a is not None and b is not None), default=0.0)
    return dict(rec_ok=rec_ok, pk_ok=pk_ok, p0_ok=p0_ok, map_ok=map_ok, p0maxdiff=p0maxdiff,
                pkR=pkR, pkC=pkC, nmeas=len(p0R))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    args = ap.parse_args()

    print("=" * 78)
    print("STEP B0 -- event-level differential shadow (candidate Policy-3 diagonal vs butterfly)")
    print("=" * 78)

    # ---- Part 1: cultivation_d5 seed 1 full table summary + aggregate over seeds ----
    print("\n--- PART 1: per-T differential (fork; phi up-to-phase == full physical state) ---")
    for circ, nseed in [("cultivation_d3", 8), ("cultivation_d5", 4)]:
        agg = dict(nT=0, ok=0, max_incl=0.0, max_upg=0.0, max_nrm=0.0, max_pexp=0.0,
                   fr_bad=0, rank_bad=0, fails=[], max_fs_upg=0.0, max_fs_incl=0.0, borns={})
        fsflag = (circ == "cultivation_d3")
        for s in range(1, nseed + 1):
            rows, rec, pk = part1(circ, s, full_state_check=fsflag)
            for d in rows:
                agg["nT"] += 1
                if not d["ok"]:
                    agg["fails"].append((s, d.get("why"), d.get("weight")))
                    continue
                agg["ok"] += 1
                agg["max_incl"] = max(agg["max_incl"], d["incl"])
                agg["max_upg"] = max(agg["max_upg"], d["upg"])
                agg["max_nrm"] = max(agg["max_nrm"], d["nrm"])
                agg["max_pexp"] = max(agg["max_pexp"], d["pexp"])
                agg["fr_bad"] += (0 if d["fr_ok"] else 1)
                agg["rank_bad"] += (0 if d["rank_ok"] else 1)
                agg["borns"][d["born"]] = agg["borns"].get(d["born"], 0) + 1
                if d["fs"]:
                    agg["max_fs_upg"] = max(agg["max_fs_upg"], d["fs"]["upg"])
                    agg["max_fs_incl"] = max(agg["max_fs_incl"], d["fs"]["incl"])
        print(f"\n{circ}: T={agg['nT']} (over {nseed} seeds)  candidate_ok={agg['ok']}/{agg['nT']}")
        print(f"  max|phiS - gamma*phiC| (incl gamma)   = {agg['max_incl']:.2e}")
        print(f"  max up-to-global-phase residual       = {agg['max_upg']:.2e}")
        print(f"  max |norm_S - norm_C|                 = {agg['max_nrm']:.2e}")
        print(f"  max random-Pauli <P> mismatch         = {agg['max_pexp']:.2e}")
        print(f"  frame/tableau mismatches              = {agg['fr_bad']}    rank/map mismatches = {agg['rank_bad']}")
        print(f"  born-basis distribution               = {agg['borns']}")
        if fsflag:
            print(f"  INDEP full-statevector(2^6) up-to-phase= {agg['max_fs_upg']:.2e}   incl-gamma = {agg['max_fs_incl']:.2e}")
        if agg["fails"]:
            print(f"  *** COLLAPSE FAILS: {agg['fails'][:6]}")

    # ---- Part 2: whole-run authoritative candidate engine vs real (records/p0/rank) ----
    print("\n--- PART 2: whole-run authoritative candidate engine (covers next-p0 + rank/map through measurements) ---")
    for circ, nseed in [("cultivation_d3", 8), ("cultivation_d5", 4)]:
        allok = True; mp0 = 0.0
        for s in range(1, nseed + 1):
            d = part2(circ, s)
            ok = d["rec_ok"] and d["pk_ok"] and d["p0_ok"] and d["map_ok"]
            allok &= ok; mp0 = max(mp0, d["p0maxdiff"])
            if not ok:
                print(f"  {circ} seed {s}: FAIL rec={d['rec_ok']} pk={d['pk_ok']}({d['pkR']}/{d['pkC']}) "
                      f"p0={d['p0_ok']} map={d['map_ok']} p0maxdiff={d['p0maxdiff']:.2e}")
        print(f"{circ}: {'ALL PASS' if allok else 'FAIL'}  records/rank/p0/map identical over {nseed} seeds, "
              f"max p0 diff={mp0:.2e}")
