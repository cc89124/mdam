"""control-plane CODEGEN (step 2): for a given circuit, emit a straight-line C++ shot-runner that unrolls
run_lean's op loop with all operands as immediates (no switch dispatch, no SoA operand loads), reusing the
exact inline methods (frame.cnot/apply_mask/lean_measure/...) so it is BIT-EXACT by construction.  Compile to
a .so, dlopen, run on the SAME warm vm (struct MdamShot -> shared layout across .so).  Verify record ==
run_lean_batch and measure ns/shot vs serial lean (fblock on/off).  default-OFF, hot path untouched.  taskset -c 2."""
import os, sys, ctypes, time, subprocess, hashlib
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
ROOT="/home/jung/clifft-paper"; HERE=os.path.dirname(os.path.abspath(__file__)); SCRATCH="/tmp/claude-1000/-home-jung/14b1f2d8-129b-4a3b-b8a2-ed2f1f3cccf6/scratchpad"
sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(ROOT,"mdam")); sys.path.insert(0,ROOT)
import clifft
from verify_mdam_oneshot import translate, make_prog, pcg, load_lib
P=ctypes.c_void_p; U=ctypes.c_uint64; C=ctypes.c_int
lib=load_lib()
lib.nvm_run_lean_fb_batch.restype=C; lib.nvm_run_lean_fb_batch.argtypes=[P,P,U]+[U]*4+[P,P,C]
lib.nvm_run_lean_batch.restype=C; lib.nvm_run_lean_batch.argtypes=[P,P,U]+[U]*4+[P,P,P,C]
for f in ("nvm_mcache_set_mode","nvm_mcache_set_fblock","nvm_sg_shadow","nvm_sg_signs"):
    getattr(lib,f).argtypes=[P,C]; getattr(lib,f).restype=None
lib.nvm_mcache_reset.argtypes=[P]; lib.nvm_sg_reset.argtypes=[P]; lib.nvm_lean_reset_counts.argtypes=[P]; lib.nvm_rb_static.argtypes=[C]
eb=ctypes.create_string_buffer(256); N=12800; SEED=99
FNV="1469598103934665603ULL"

# ---- emit straight-line C++ for one op (mirrors run_lean's switch EXACTLY) ----
def op_cpp(k,a1,a2,i0,i1,dv):
    if k==0:  return f"s.frame.h({a1});"
    if k==1:  return f"s.frame.cnot({a1},{a2});"
    if k==2:  return f"s.frame.cz({a1},{a2});"
    if k==3:  return f"s.frame.swap({a1},{a2});"
    if k==4:  return f"s.frame.s_gate({a1});"
    if k==5:  return f"if(s.record.get({i0})==1) s.apply_mask(p.cp_masks[{i1}]);"
    if k==6:  return f"if(s.sampler.should_fire({i0})) s.sampler.apply_site({i0}, p.noise_sites[{i0}], s.frame);"
    if k==7:  return f"for(int _s={i0};_s<{i0+i1};_s++){{ if(s.sampler.should_fire(_s)) s.sampler.apply_site(_s, p.noise_sites[_s], s.frame); }}"
    if k==8:  return f"if(s.udraw()<{dv!r}) s.record.flip({i0});"
    if k==9:  return f"s.record.set({i0}, s.frame.xb({a1})^{i1});"
    if k==10: return f"{{ int m=(int)s.idraw2(); s.record.set({i0}, m^{i1}); s.frame.set_xz({a1},(uint8_t)m,0); }}"
    if k==11: return f"s.frame.cnot({a1},{a2});"
    if k==12: return f"s.frame.cz({a1},{a2});"
    if k==13: return f"{{ int tgt={a1}; uint64_t mask=p.mmask[{i0}]; while(mask){{ int ctrl=__builtin_ctzll(mask); mask&=mask-1; if(ctrl==tgt) continue; s.frame.cnot(ctrl,tgt); }} }}"
    if k==14: return f"{{ int aa={a1}; uint64_t mask=p.mmask[{i0}]; while(mask){{ int tgt=__builtin_ctzll(mask); mask&=mask-1; if(tgt==aa) continue; s.frame.cz(aa,tgt); }} }}"
    if k in (15,16,21): return f"s.lean_rot({a1});"
    if k in (18,19,22): return f"s.newq({a1}); s.lean_rot({a1});"
    if k==17: return f"s.frame.s_gate({a1});"
    if k==27: return f"s.frame.h({a1});"
    if k==26: return f"s.newq({a1});"
    if k==23: return (f"{{ int i_1=s.slot2id[{a1}],i_2=s.slot2id[{a2}]; s.slot2id[{a1}]=-1; s.slot2id[{a2}]=-1;"
                      f" if(i_1>=0)s.slot2id[{a2}]=i_1; if(i_2>=0)s.slot2id[{a1}]=i_2; s.frame.swap({a1},{a2}); }}")
    if k==20: return (f"{{ int i_1=s.slot2id[{a1}],i_2=s.slot2id[{a2}]; s.slot2id[{a1}]=-1; s.slot2id[{a2}]=-1;"
                      f" if(i_1>=0)s.slot2id[{a2}]=i_1; if(i_2>=0)s.slot2id[{a1}]=i_2; s.frame.swap({a1},{a2});"
                      f" int q=s.slot2id[{a2}]; if(q>=0){{ int b=s.lean_measure(); if(b<0) goto done; s.slot2id[{a2}]=-1;"
                      f" int m_abs=b^s.frame.zb({a2}); s.record.set({i0}, m_abs^{i1}); s.frame.set_xz({a2},(uint8_t)m_abs,0); }} }}")
    if k==24: return (f"{{ int q=s.slot2id[{a1}]; if(q>=0){{ int b=s.lean_measure(); if(b<0) goto done; s.slot2id[{a1}]=-1;"
                      f" int m_abs=b^s.frame.xb({a1}); s.record.set({i0}, m_abs^{i1}); s.frame.set_xz({a1},(uint8_t)m_abs,0); }} }}")
    if k==25: return (f"{{ int q=s.slot2id[{a1}]; if(q>=0){{ int b=s.lean_measure(); if(b<0) goto done; s.slot2id[{a1}]=-1;"
                      f" int m_abs=b^s.frame.zb({a1}); s.record.set({i0}, m_abs^{i1}); s.frame.set_xz({a1},(uint8_t)m_abs,0); }} }}")
    if k==28: return (f"{{ int in_state=(s.frame.zb({a1})<<1)|s.frame.xb({a1}); s.sg_u2_sign({a1},in_state); "
                      f"uint8_t out=p.u2_out[{i0}*4+in_state]; s.frame.set_xz({a1},out&1,(out>>1)&1); }}")   # frame-only U2 (mirrors run_lean)
    if k==29: return "s.ln_incomplete=true; goto done;"          # U4 unsupported in lean
    if k==30: return ""                                          # MO_END
    return ""

def gen_cpp(t):
    K=list(map(int,t["kind"])); A1=list(map(int,t["a1"])); A2=list(map(int,t["a2"]))
    I0=list(map(int,t["i0"])); I1=list(map(int,t["i1"])); DV=list(map(float,t["dval"]))
    body=[]
    for i in range(len(K)):
        c=op_cpp(K[i],A1[i],A2[i],I0[i],I1[i],DV[i])
        if c: body.append("    "+c)
    src = ['#include "native_mdam_shot.hpp"','#include <cstring>','using namespace mdam;','',
        'extern "C" void gen_run_lean(void* vmp, const void* progp){',
        '  MdamShot& s=*(MdamShot*)vmp; const MdamProgram& p=*(const MdamProgram*)progp; (void)p;',
        f'  s.ln_cur_id=-1; s.ln_prev_out=-2; s.ln_incomplete=false; s.sg_seg_signs={FNV}; s.ln_active=true;']
    src += body
    src += ['  done:','  s.ln_active=false; if(s.ln_incomplete) s.ln_incomplete_shots++;','}','',
        'extern "C" int gen_run_lean_batch(const void* progp, void* vmp, uint64_t num_shots,',
        '   uint64_t mshi,uint64_t mslo,uint64_t mihi,uint64_t milo, uint8_t* out_record, uint8_t* out_incomplete, char* out_err, int errlen){',
        '  MdamShot& s=*(MdamShot*)vmp; const MdamProgram& p=*(const MdamProgram*)progp;',
        '  NativeRng master; master.seed_from_state(mshi,mslo,mihi,milo);',
        '  const uint64_t RNG_EXCL=((uint64_t)1<<63)-1; const size_t nm=(size_t)p.num_measurements;',
        '  for(uint64_t sh=0; sh<num_shots; sh++){',
        '    uint64_t sd=master.bounded(RNG_EXCL); __uint128_t st,inc; SeedExpand::seedseq_pcg64(sd,st,inc);',
        '    s.reset_shot(p,(uint64_t)(st>>64),(uint64_t)st,(uint64_t)(inc>>64),(uint64_t)inc);',
        '    gen_run_lean(vmp,progp);',
        '    std::memcpy(out_record+(size_t)sh*nm, s.record.bits.data(), nm);',
        '    if(out_incomplete) out_incomplete[sh]=s.ln_incomplete?1:0;',
        '    if(s.err){ if(out_err){ std::strncpy(out_err,s.err,errlen-1); out_err[errlen-1]=0; } return 1; }',
        '  }',
        '  if(out_err) out_err[0]=0; return 0;','}','',
        '// fb variant: mirrors MdamShot::run_lean_fb_batch EXACTLY (miss -> SAME per-shot seed -> run_mcache),',
        '// with run_lean replaced by the unrolled gen_run_lean.  gen_run_lean is bit-exact to run_lean, the',
        '// fallback is byte-identical code => whole batch bit-exact to run_lean_fb_batch for the same master seed.',
        'extern "C" int gen_run_lean_fb_batch(const void* progp, void* vmp, uint64_t num_shots,',
        '   uint64_t mshi,uint64_t mslo,uint64_t mihi,uint64_t milo, uint8_t* out_record, char* out_err, int errlen){',
        '  MdamShot& s=*(MdamShot*)vmp; const MdamProgram& p=*(const MdamProgram*)progp;',
        '  NativeRng master; master.seed_from_state(mshi,mslo,mihi,milo);',
        '  const uint64_t RNG_EXCL=((uint64_t)1<<63)-1; const size_t nm=(size_t)p.num_measurements;',
        '  for(uint64_t sh=0; sh<num_shots; sh++){',
        '    uint64_t sd=master.bounded(RNG_EXCL); __uint128_t st,inc; SeedExpand::seedseq_pcg64(sd,st,inc);',
        '    uint64_t shi=(uint64_t)(st>>64),slo=(uint64_t)st,ihi=(uint64_t)(inc>>64),ilo=(uint64_t)inc;',
        '    s.reset_shot(p,shi,slo,ihi,ilo);',
        '    gen_run_lean(vmp,progp);',
        '    if(s.ln_incomplete){ s.ln_fb_count++; s.reset_shot(p,shi,slo,ihi,ilo); s.run_mcache(p); }',
        '    std::memcpy(out_record+(size_t)sh*nm, s.record.bits.data(), nm);',
        '    if(s.err){ if(out_err){ std::strncpy(out_err,s.err,errlen-1); out_err[errlen-1]=0; } return 1; }',
        '  }',
        '  if(out_err) out_err[0]=0; return 0;','}','']
    return "\n".join(src)

GCC_FLAGS=["-O3","-march=native","-std=c++17","-DNDEBUG","-shared","-fPIC"]
def compile_so(cpp, tag, outdir=None):
    d=outdir or SCRATCH
    cpath=os.path.join(d,f"gen_{tag}.cpp"); sopath=os.path.join(d,f"gen_{tag}.so")
    open(cpath,"w").write(cpp)
    # link against the main VM .so: the fb path (run_mcache) needs mdm_execute_core; the dynamic loader
    # dedups by dev+inode so the generated .so binds to the SAME already-loaded native_mdam_vm.so instance.
    r=subprocess.run(["g++"]+GCC_FLAGS+[f"-I{HERE}", cpath,
                      os.path.join(HERE,"native_mdam_vm.so"), f"-Wl,-rpath,{HERE}",
                      "-o", sopath], capture_output=True, text=True)
    if r.returncode!=0: raise RuntimeError("compile failed:\n"+r.stderr[-2000:])
    return sopath

# ---- persistent .so cache (the "same circuit, many runs" amortization).  Key covers EVERYTHING that ----
# ---- affects the binary: generated cpp + native_mdam_shot.hpp bytes (struct layout!) + gcc flags.    ----
def cache_key(cpp):
    h=hashlib.md5()
    for f in sorted(os.listdir(HERE)):          # ALL local headers (native_mdam_shot.hpp includes others)
        if f.endswith(".hpp"): h.update(open(os.path.join(HERE,f),"rb").read())
    h.update((" ".join(GCC_FLAGS)).encode()); h.update(cpp.encode())
    return h.hexdigest()[:16]
def get_so_cached(cpp, cache_dir=None):
    """returns (so_path, hit, compile_wall_s). cache_dir default: native_vm/.cgcache (env MDAM_CGCACHE)."""
    d=cache_dir or os.environ.get("MDAM_CGCACHE") or os.path.join(HERE,".cgcache")
    os.makedirs(d,exist_ok=True)
    tag=cache_key(cpp); sopath=os.path.join(d,f"gen_{tag}.so")
    if os.path.exists(sopath): return sopath, True, 0.0
    t0=time.perf_counter(); compile_so(cpp,tag,outdir=d); return sopath, False, time.perf_counter()-t0

def load(b):
    prog=clifft.compile(open(f"{ROOT}/qec_bench/circuits/{b}.stim").read()); t=translate(prog); return t, make_prog(lib,t), t["num_meas"]
def warm(ph,vm,seed,nm,fblock):
    lib.nvm_rb_static(1); lib.nvm_mcache_set_mode(vm,3); lib.nvm_mcache_set_fblock(vm,1 if fblock else 0)
    lib.nvm_mcache_reset(vm); lib.nvm_sg_reset(vm); lib.nvm_sg_signs(vm,1); lib.nvm_sg_shadow(vm,1); lib.nvm_lean_reset_counts(vm)
    w=np.zeros((N,nm),np.uint8); lib.nvm_run_lean_fb_batch(ph,vm,N,*pcg(seed),w.ctypes.data,eb,256); lib.nvm_sg_shadow(vm,0); lib.nvm_rb_static(0)
def time_serial(ph,vm,nm,fblock,reps=5):
    lib.nvm_mcache_set_fblock(vm,1 if fblock else 0)
    rec=np.zeros((N,nm),np.uint8); inc=np.zeros(N,np.uint8)
    lib.nvm_run_lean_batch(ph,vm,N,*pcg(SEED),rec.ctypes.data,inc.ctypes.data,eb,256)
    best=1e30
    for _ in range(reps):
        t0=time.perf_counter(); lib.nvm_run_lean_batch(ph,vm,N,*pcg(SEED),rec.ctypes.data,inc.ctypes.data,eb,256); best=min(best,(time.perf_counter()-t0)/N*1e9)
    return rec, inc, best

def _driver():
    out=[]
    def emit(s): print(s); out.append(s)
    emit("# control-plane codegen: unrolled straight-line C++ lean shot-runner vs serial run_lean_batch\n")
    hdr=f"{'bench':17s} {'shots':>6s} {'serial_fb_off':>13s} {'serial_fb_on':>12s} {'codegen_ns':>10s} {'sp_vs_fbON':>10s} {'sp_vs_fbOFF':>11s} {'rec_mism':>8s}"
    emit("```"); emit(hdr); emit("-"*len(hdr))
    for b in ["cultivation_d3","distillation","coherent_d3_r1","coherent_rx_d3_r1"]:
        t,ph,nm=load(b)
        cpp=gen_cpp(t); tag=hashlib.md5((b+cpp).encode()).hexdigest()[:10]
        so=compile_so(cpp,tag); g=ctypes.CDLL(so)
        g.gen_run_lean_batch.restype=C; g.gen_run_lean_batch.argtypes=[P,P,U]+[U]*4+[P,P,P,C]
        vm=lib.nvm_mdam_vm_create(ph); warm(ph,vm,SEED,nm,fblock=True)   # warm automaton (fblock irrelevant to automaton content)
        srec_off,sinc_off,ns_off=time_serial(ph,vm,nm,fblock=False)
        srec_on,sinc_on,ns_on=time_serial(ph,vm,nm,fblock=True)
        # codegen run (same warm vm)
        drec=np.zeros((N,nm),np.uint8); dinc=np.zeros(N,np.uint8)
        g.gen_run_lean_batch(ph,vm,N,*pcg(SEED),drec.ctypes.data,dinc.ctypes.data,eb,256)  # warm-up
        best=1e30
        for _ in range(5):
            t0=time.perf_counter(); g.gen_run_lean_batch(ph,vm,N,*pcg(SEED),drec.ctypes.data,dinc.ctypes.data,eb,256); best=min(best,(time.perf_counter()-t0)/N*1e9)
        both=(sinc_on==0)&(dinc==0); rm=int((srec_on[both]!=drec[both]).sum())
        emit(f"{b:17s} {N:6d} {ns_off:13.1f} {ns_on:12.1f} {best:10.1f} {ns_on/best:9.2f}x {ns_off/best:10.2f}x {rm:8d}")
    emit("```\n")
    emit("serial_fb_off = run_lean_batch raw op loop.  serial_fb_on = with frame-block superinstruction (the fast")
    emit("serial baseline).  codegen = fully-unrolled straight-line C++ (no switch/dispatch, immediates, inlined")
    emit("frame ops).  rec_mism=0 = bit-exact vs serial run_lean_batch.  Removes DISPATCH; keeps RNG/hash/measure.")
    with open(f"{ROOT}/results/benchmark_comparison/codegen_lean.md","w") as f: f.write("\n".join(out)+"\n")
    print(f"\n[written] {ROOT}/results/benchmark_comparison/codegen_lean.md")

if __name__=="__main__":
    _driver()
