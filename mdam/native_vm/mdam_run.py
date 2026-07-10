"""MDAM production batch entry point: run_batch(stim_text, N) = full algorithm
(probe -> LEAN/AUTH routing -> walk-compile gate -> race -> cruise), bit-exact record stream.
Separate entry point; run()/run_mcache/run_lean_*/run_lean_adapt_batch are untouched.
Tiered plan for a batch of N shots (all segments emit EXACT records; every shot is a valid sample):

  1. PROBE   : nvm_run_lean_adapt_batch on the first P shots (the UNCHANGED adaptive executor).  It
               auto-selects LEAN vs sticky-AUTH and reports lean ns/shot + fallback rate.
  2. ROUTE   : policy==AUTH  -> rest via the authoritative sample_batch (what sticky-AUTH does anyway).
               fb_rate>FB_MAX -> automaton not saturating; the codegen fast path would miss constantly
               -> rest via adaptive (warm re-entry).  No codegen.
  3. GATE    : engage codegen iff the SELF-RELATIVE amortization holds:
                    N_rem * lean_ns * S_MIN  >  fixed_cost
               fixed_cost = COMPILE_EST if the .so cache misses, ~0 on a hit.  S_MIN is a conservative
               expected saving fraction of codegen vs the interpreted lean walk.  This gate compares MDAM
               paths against EACH OTHER only (no external baseline enters the algorithm).
  4. RACE    : compile/load the circuit .so (persistent cache keyed on cpp+hpp+flags), then time one lean
               chunk and one codegen chunk (both are real output shots).  Winner is sticky for the cruise.
               Bounded regret if codegen disappoints: compile_wall + R*(gen_ns-lean_ns).
  5. CRUISE  : remaining shots via the winner (codegen path = gen_run_lean_fb_batch: unrolled fast walk,
               miss -> SAME per-shot seed -> run_mcache, bit-exact to run_lean_fb_batch by construction).

Seeds: per-segment master seeds derived as seed+shots_done (same convention as the existing chunked
drivers).  The full-record stream therefore equals the reference composition of run_lean_fb_batch /
sample_batch calls with the same segment seeds (verified), not a single monolithic call.
taskset -c 2, single-thread env assumed by the caller."""
import os, sys, ctypes, time, hashlib
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
ROOT="/home/jung/clifft-paper"; HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(ROOT,"mdam")); sys.path.insert(0,ROOT)
import walk_compile as cg
from verify_mdam_oneshot import translate, make_prog, pcg
P_=ctypes.c_void_p; U=ctypes.c_uint64; C=ctypes.c_int; D=ctypes.c_double
lib=cg.lib
lib.nvm_run_lean_adapt_batch.restype=C; lib.nvm_run_lean_adapt_batch.argtypes=[P_,P_,U]+[U]*4+[P_,P_,C]
lib.nvm_mdam_sample_batch.restype=C; lib.nvm_mdam_sample_batch.argtypes=[P_,P_,U]+[U]*4+[P_,P_,P_,C]
lib.nvm_adapt_stats.argtypes=[P_,ctypes.POINTER(D)]; lib.nvm_adapt_stats.restype=None
lib.nvm_mc_pool_off.argtypes=[P_,C]; lib.nvm_mc_pool_off.restype=None
lib.nvm_lean_stats.argtypes=[P_,ctypes.POINTER(ctypes.c_long)]; lib.nvm_lean_stats.restype=None
lib.nvm_mc_canon.argtypes=[P_,C]; lib.nvm_mc_canon.restype=None
lib.nvm_mdam_run_bcap.restype=C; lib.nvm_mdam_run_bcap.argtypes=[P_,P_]+[U]*4+[P_,P_,C]
lib.nvm_bcap_n.restype=ctypes.c_long; lib.nvm_bcap_n.argtypes=[P_]
lib.nvm_bcap_get.argtypes=[P_,ctypes.POINTER(ctypes.c_longlong),ctypes.POINTER(D)]; lib.nvm_bcap_get.restype=None
lib.nvm_bcap_distinct_states.restype=ctypes.c_long; lib.nvm_bcap_distinct_states.argtypes=[P_]
lib.nvm_diag_canon_map.restype=ctypes.c_long; lib.nvm_diag_canon_map.argtypes=[P_,C,ctypes.POINTER(ctypes.c_long),ctypes.c_long]

CFG=dict(
    PROBE=2048,           # adaptive probe shots (policy + lean_ns + fb_rate come from here).  Sized to the
                          # MEASURED v2 decision latency: every demote fires inside calibration/first windows
                          # (<=639 shots, all benches) and calibration itself is <=512 shots, so 2048 settles
                          # the policy with >3x margin; a longer adaptive probe only pays the executor's
                          # per-window bookkeeping on shots the lean walk could run cheaper (cult_d5: 20k
                          # probe cost +3.8us/shot amortized at N=1M).  fb measured at 4096 overshoots the
                          # steady rate, so near-FB_MAX circuits may spend one re-evaluate chunk on leanfb
                          # before codegen engages (bounded: <=1 chunk).
    RACE=20000,           # race chunk per contender
    CHUNK=100000,         # cruise chunk
    S_MIN=0.25,           # conservative saving fraction for the engage gate
    FB_MAX=0.02,          # probe fallback rate above which codegen is pointless (misses dominate)
    COMPILE_EST_S=7.0,    # a-priori g++ wall estimate used by the gate on a cache miss (measured 5.3-6.4s
                          # with the fb-batch instantiation; conservative so marginal N never net-loses)
    CANON="auto",         # canonical (phase + 1e-9 grid) BoundaryKey sid interning: "auto" = decide by the
                          # canon probe below, 0/1 = force.  ON merges representation variants of one
                          # physical boundary state -> lower fb + ~half the cache memory on diagonal-magic
                          # circuits (cult_d5 −5% wall, d3_r3 −5%); pure transform overhead on off-axis
                          # circuits (rx_d3_r3 +6%) -> the probe keeps those OFF.  Records exact modulo
                          # Born draws inside the intra-class p0 window (~1e-8/run; measured 0 mismatches).
    CANON_PROBE=512,     # throwaway capture shots for the auto decision (records NOT emitted, so the
                          # production record stream is unchanged either way)
    CANON_MIN=0.25,       # SID-merge fraction above which canon turns ON.  sid merge = representation-
                          # duplication density of the boundary-state population; it separates early and
                          # wide (1024 shots: d3_r3 83%, cult_d5 57%, distillation 48% vs rx 11%) where
                          # edge merge is still forming (d3_r3 0.5%@1k -> 27%@1M).  Off-axis states are
                          # genuinely distinct, so low sid merge <=> transform-cost-only regime.
)

def _canon_probe(ph, nm, nprobe, seed):
    """Throwaway pre-pass: capture boundary-key components over nprobe authoritative shots on a
    SCRATCH vm, rewrite sid -> phase-canonical class (nvm_diag_canon_map) and measure the edge-merge
    fraction.  Internal-structure measurement only (no external baseline).  Returns (merge, n_edges)."""
    vm=lib.nvm_mdam_vm_create(ph); eb=ctypes.create_string_buffer(256)
    rec=np.zeros(nm,np.uint8); rows=[]
    t0=time.perf_counter()
    for i in range(nprobe):
        if i>=256 and (i&255)==0 and time.perf_counter()-t0>1.0: break   # wall budget: slow-auth circuits
        if lib.nvm_mdam_run_bcap(ph,vm,*pcg(seed+i),rec.ctypes.data,eb,256): continue
        n=lib.nvm_bcap_n(vm)
        if n:
            bi=np.zeros((n,13),np.int64)
            lib.nvm_bcap_get(vm,bi.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
                             np.zeros(n).ctypes.data_as(ctypes.POINTER(D)))
            rows.append(bi[:,[0,8,1,2,3,4,10]].copy())    # (mp,kind,sid,inv,pend,m,outcome)
    if not rows: return 0.0, 0.0                          # no magic boundaries -> canon is a no-op
    E=np.concatenate(rows)
    nsid=int(lib.nvm_bcap_distinct_states(vm))
    cmap=np.zeros(nsid,np.int64)
    lib.nvm_diag_canon_map(vm,1,cmap.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),nsid)
    sid_merge=1.0-len(np.unique(cmap))/max(1,nsid)
    raw=len(np.unique(E,axis=0))
    Ec=E.copy(); Ec[:,2]=cmap[E[:,2]]
    can=len(np.unique(Ec,axis=0))
    return sid_merge, 1.0-can/max(1,raw)

def _setup_lean(vm): lib.nvm_rb_static_reset(); lib.nvm_rb_static(1); lib.nvm_mcache_set_mode(vm,3); lib.nvm_mcache_set_fblock(vm,1)
def _fresh(vm): lib.nvm_mcache_reset(vm); lib.nvm_sg_reset(vm); lib.nvm_sg_signs(vm,1); lib.nvm_sg_shadow(vm,1); lib.nvm_lean_reset_counts(vm)

def prewarm_so(stim_text, cache_dir=None):
    """explicit OFFLINE .so preparation (opt-in; run_batch never compiles unless its own gate passes —
    without this, a small-N batch declines to compile and the cache stays empty forever).  returns
    (so_path, cache_hit, compile_wall_s)."""
    import clifft
    t=translate(clifft.compile(stim_text))
    return cg.get_so_cached(cg.gen_cpp(t), cache_dir=cache_dir)

def run_batch(stim_text, N, seed=40000, cfg=None, cache_dir=None, log=lambda s: None):
    """returns (records uint8[N,nm], info dict).  info['plan'] = list of (segment, path, shots, ns/shot)."""
    import clifft
    c=dict(CFG); c.update(cfg or {})
    prog=clifft.compile(stim_text); t=translate(prog); nm=t["num_meas"]
    ph=make_prog(lib,t); vm=lib.nvm_mdam_vm_create(ph); eb=ctypes.create_string_buffer(256)
    if c.get("POOL_OFF"): lib.nvm_mc_pool_off(vm,1)   # lean-build pool-snapshot interning off (walk never
                                                      # reads the pool; use when LEAN sustains and fb->0)
    rec=np.zeros((N,nm),np.uint8); done=0; plan=[]; info=dict(plan=plan, engaged=False, cache_hit=None, compile_s=0.0)
    # --- 0 CANON GATE: probe-measured internal criterion (see CFG.CANON); set before ANY shot runs ---
    if c["CANON"]=="auto":
        t0=time.perf_counter()
        sid_merge,edge_merge=_canon_probe(ph,nm,min(int(c["CANON_PROBE"]),N),seed)
        canon_on = sid_merge>c["CANON_MIN"]
        info.update(canon=canon_on, canon_merge=sid_merge, canon_edge_merge=edge_merge,
                    canon_probe_s=time.perf_counter()-t0)
    else:
        canon_on=bool(c["CANON"]); info.update(canon=canon_on, canon_merge=None, canon_probe_s=0.0)
    if canon_on: lib.nvm_mc_canon(vm,1)
    bump=0                                   # per-segment seed bump for reduce_full retries (documented
                                             # ~1/2M-shot event; retry-with-new-seed is the convention and
                                             # the retried segment's records stay valid samples)
    waste=0.0                                # wall spent on failed (retried) attempts -> counted in total_ns
    def seg(path, n, fn):
        nonlocal done, bump, waste
        for k in range(6):
            bump=k
            t0=time.perf_counter(); r=fn(n, rec[done:done+n]); wall=time.perf_counter()-t0
            if r==0: break
            waste+=wall
            if k==5 or b"reduce_full" not in eb.value: raise RuntimeError(f"{path} failed: {eb.value.decode()}")
        bump=0
        plan.append((path+(f"(retry{k})" if k else ""), n, wall/n*1e9)); done+=n; return wall/n*1e9
    run_adapt =lambda n,buf: lib.nvm_run_lean_adapt_batch(ph,vm,n,*pcg(seed+done+bump*7919),buf.ctypes.data,eb,256)
    run_leanfb=lambda n,buf: lib.nvm_run_lean_fb_batch(ph,vm,n,*pcg(seed+done+bump*7919),buf.ctypes.data,eb,256)
    run_auth  =lambda n,buf: lib.nvm_mdam_sample_batch(ph,vm,n,*pcg(seed+done+bump*7919),buf.ctypes.data,None,eb,256)
    # --- 1 PROBE (adaptive, unchanged).  FORCE_LEAN=1 = LEAN-forced ablation: pure LEAN probe, the AUTH
    # route is disabled, everything else (gate/race/cruise criterion) identical to production. ---
    force_lean=bool(c.get("FORCE_LEAN"))
    _setup_lean(vm); _fresh(vm)
    p=min(c["PROBE"],N)
    o=(D*16)(); ls=(ctypes.c_long*3)(); last_fb=0
    if force_lean:
        probe_ns=seg("probe/leanfb",p,run_leanfb)
        lib.nvm_lean_stats(vm,ls); policy=0; lean_ns=probe_ns; fb=ls[2]/max(1,done); last_fb=ls[2]
    else:
        probe_ns=seg("probe/adaptive",p,run_adapt)
        lib.nvm_adapt_stats(vm,o); policy=int(o[0]); lean_ns=o[6] if o[6]>0 else probe_ns
        # fb from the direct fallback counter (window-independent; the executor's own windowed fb stat
        # reads 0 until its first internal evaluation window completes, which a 4096-shot probe may not)
        lib.nvm_lean_stats(vm,ls); fb=ls[2]/max(1,p); last_fb=ls[2]
    info.update(policy=("AUTH" if policy==1 else "LEAN"), probe_lean_ns=lean_ns, probe_fb=fb)
    # --- 3-5 GATE + RACE + CRUISE (self-relative gate; no external baseline enters the decision) ---
    def codegen_stage(cur_lean_ns):
        cpp=cg.gen_cpp(t); key=cg.cache_key(cpp)
        d=cache_dir or os.environ.get("MDAM_CGCACHE") or os.path.join(HERE,".cgcache")
        hit=os.path.exists(os.path.join(d,f"gen_{key}.so"))
        fixed=0.0 if hit else c["COMPILE_EST_S"]
        n_rem=N-done
        if n_rem*cur_lean_ns*1e-9*c["S_MIN"] <= fixed:
            log(f"gate: n_rem={n_rem} * lean_ns={cur_lean_ns:.0f} * S_MIN not > {fixed:.1f}s -> no codegen")
            seg("cruise/leanfb",n_rem,run_leanfb); return
        so,hit2,cw=cg.get_so_cached(cpp,cache_dir=d); info.update(cache_hit=hit2, compile_s=cw)
        g=ctypes.CDLL(so); g.gen_run_lean_fb_batch.restype=C; g.gen_run_lean_fb_batch.argtypes=[P_,P_,U]+[U]*4+[P_,P_,C]
        run_genfb=lambda n,buf: g.gen_run_lean_fb_batch(ph,vm,n,*pcg(seed+done),buf.ctypes.data,eb,256)
        # RACE (both chunks are real output shots)
        r=min(c["RACE"],(N-done)//2)
        if r>0:
            inc_ns=seg("race/leanfb",r,run_leanfb)
            gen_ns=seg("race/codegen",r,run_genfb)
        else: inc_ns,gen_ns=cur_lean_ns,float("inf")   # race skipped -> conservative: keep the interpreted path
        winner = run_genfb if gen_ns<inc_ns else run_leanfb
        info["engaged"]=gen_ns<inc_ns
        while done<N:                                  # CRUISE (winner sticky, chunked)
            n=min(c["CHUNK"],N-done)
            seg("cruise/%s"%("codegen" if winner is run_genfb else "leanfb"),n,winner)
    # --- 2 ROUTE ---
    if done<N and policy==1:                       # adaptive demoted -> stay authoritative (its own sticky choice)
        seg("cruise/auth",N-done,run_auth)
    elif done<N and fb>c["FB_MAX"]:
        # non-saturating YET: the automaton keeps learning on every fallback, so RE-EVALUATE per chunk and
        # promote to the codegen stage once the miss rate closes below FB_MAX.  Chunks run on the interpreted
        # lean walk (leanfb), NOT the adaptive wrapper: the probe already gave v2 20k shots to demote and
        # every measured demote fires within the calibration/first windows (<=639 shots on all benches), so
        # policy==LEAN here is v2's own sticky decision — re-wrapping each chunk in the adaptive executor
        # only re-buys that settled decision at ~6% per-window bookkeeping cost (measured d3_r3/cult_d5).
        lib.nvm_lean_stats(vm,ls); last_fb=ls[2]
        while done<N:
            # 25k sub-chunks: a reduce_full abort (~1/2M shots) throws away only the failed sub-chunk's
            # partial wall, quartering the expected retry waste vs full CHUNK segments
            n=min(c["CHUNK"]//4,N-done)
            lean2=seg("cruise/leanfb",n,run_leanfb)
            lib.nvm_lean_stats(vm,ls); fb2=(ls[2]-last_fb)/max(1,n); last_fb=ls[2]
            info.update(probe_fb=max(fb2,0.0))
            if 0<=fb2<=c["FB_MAX"] and done<N:     # saturated late -> engage the gate now
                codegen_stage(lean2); break
    elif done<N:
        codegen_stage(lean_ns)
    if force_lean:
        lib.nvm_lean_stats(vm,ls); info["fb_overall"]=ls[2]/max(1,done)
    lib.nvm_sg_shadow(vm,0); lib.nvm_rb_static(0)
    info["total_ns"]=(sum(n*ns for _,n,ns in plan)+waste*1e9+info["canon_probe_s"]*1e9)/N
    info["retry_waste_s"]=waste
    return rec, info

if __name__=="__main__":
    bench=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 1_000_000
    text=open(f"{ROOT}/qec_bench/circuits/{bench}.stim").read()
    rec,info=run_batch(text,N,log=print)
    cm=info.get('canon_merge'); cm="-" if cm is None else f"{cm*100:.1f}%"
    print(f"\n{bench} N={N}  policy={info['policy']} probe_fb={info['probe_fb']*100:.2f}% engaged={info['engaged']} "
          f"canon={info['canon']}(merge={cm}, {info['canon_probe_s']:.2f}s) "
          f"cache_hit={info['cache_hit']} compile={info['compile_s']:.2f}s  amortized={info['total_ns']:.1f} ns/shot")
    for pth,n,ns in info["plan"]: print(f"  {pth:.<20s} {n:>9d} shots @ {ns:>9.1f} ns")
