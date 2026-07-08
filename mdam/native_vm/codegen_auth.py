"""AUTH-path codegen (the REAL target: circuits that lose to the external baseline BECAUSE of control —
coherent_d3_r3 / cultivation_d5 demote to AUTH where 84-92% of the wall is control+machinery, dense kernel
only 8-16%; see results/benchmark_comparison/auth_prof_losers.md).

Emits a straight-line C++ mirror of MdamShot::run() (native_mdam_shot.hpp:801) with all operands as
IMMEDIATES: no switch dispatch, no SoA operand loads.  Every case body is transcribed VERBATIM (including
the bcap capture blocks and the engine/measure_z/rot calls), so it is bit-exact by construction.  Debug-only
branches (frame_log_on / verbose fprintf) are omitted — the generated runner does not support those tools.
gen_run_auth_batch mirrors run_batch (same master-seed expansion + the shot-0 lazy probe; stats omitted).
default-OFF: separate entry point, run()/run_batch untouched.  Reuses walk_compile's compile_so/.so-cache."""
import os, sys, ctypes, time, hashlib
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
ROOT="/home/jung/clifft-paper"; HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(ROOT,"mdam")); sys.path.insert(0,ROOT)
import walk_compile as cg
from verify_mdam_oneshot import translate, make_prog, pcg
P_=ctypes.c_void_p; U=ctypes.c_uint64; C=ctypes.c_int
lib=cg.lib
lib.nvm_mdam_sample_batch.restype=C; lib.nvm_mdam_sample_batch.argtypes=[P_,P_,U]+[U]*4+[P_,P_,P_,C]

# ---- verbatim bcap capture snippets (identical to run()'s bodies, s.-qualified; @A@/@I1@ substituted) ----
def _bcap_pre(a):
    return ("int _bmp=0; uint32_t _bsi=0,_biv=0,_bpd=0,_bms=0; uint8_t _bxb=0,_bzb=0; int _bmo=0; "
            "if(s.bcap_on){ _bmp=s.magic_point; _bsi=s.bcap_sid(s.engine.dense.resident.data(),s.engine.dense.r); "
            "_biv=s.bcap_inv_sig(); _bpd=s.bcap_pend_sig(); _bms=s.bcap_m_sig(); "
            "_bxb=s.frame.xb(@A@), _bzb=s.frame.zb(@A@); _bmo=s.magic_oracle; }").replace("@A@",str(a))
def _bcap_post(kind,i1):
    return ("if(s.bcap_on){ uint32_t so=s.bcap_sid(s.engine.dense.resident.data(),s.engine.dense.r); "
            "uint8_t orc=(s.magic_oracle>_bmo)?(s.bcap_antis?2:1):0; "
            "s.bcap.push_back({_bmp,_bsi,_biv,_bpd,_bms,_bxb,_bzb,(uint8_t)@I1@,(uint8_t)"+str(kind)+
            ",orc,(uint8_t)b,so,(uint8_t)((m_abs^@I1@)&1),s.bcap_p0}); }").replace("@I1@",str(i1))

def auth_op_cpp(k,a1,a2,i0,i1,dv):
    """one straight-line C++ statement mirroring run()'s case for opcode k (operands as immediates)."""
    E="if(s.err) goto done;"   # run() checks !err before each op; engine/measure ops can set it
    if k==0:  return f"s.frame.h({a1});"
    if k==1:  return f"s.frame.cnot({a1},{a2});"
    if k==2:  return f"s.frame.cz({a1},{a2});"
    if k==3:  return f"s.frame.swap({a1},{a2});"
    if k==4:  return f"s.frame.s_gate({a1});"
    if k==5:  return f"if(s.record.get({i0})==1) s.apply_mask(p.cp_masks[{i1}]);"
    if k==6:  return f"s.sampler.apply_site({i0}, p.noise_sites[{i0}], s.frame);"
    if k==7:  return f"for(int _s={i0};_s<{i0+i1};_s++) s.sampler.apply_site(_s, p.noise_sites[_s], s.frame);"
    if k==8:  return f"if(s.udraw()<{dv!r}) s.record.flip({i0});"
    if k==9:  return f"s.record.set({i0}, s.frame.xb({a1})^{i1});"
    if k==10: return f"{{ int m=(int)s.idraw2(); s.record.set({i0}, m^{i1}); s.frame.set_xz({a1},(uint8_t)m,0); }}"
    if k==11: return f"{{ int u=s.slot2id[{a1}], v=s.slot2id[{a2}]; if(u>=0&&v>=0) s.engine.cx(u,v); s.frame.cnot({a1},{a2}); }} "+E
    if k==12: return f"{{ int u=s.slot2id[{a1}], v=s.slot2id[{a2}]; if(u>=0&&v>=0) s.engine.cz(u,v); s.frame.cz({a1},{a2}); }} "+E
    if k==13: return (f"{{ int t=s.slot2id[{a1}]; uint64_t mask=p.mmask[{i0}]; "
                      f"while(mask){{ int ctrl=__builtin_ctzll(mask); mask&=mask-1; if(ctrl=={a1}) continue; "
                      f"int c=s.slot2id[ctrl]; if(t>=0&&c>=0) s.engine.cx(c,t); s.frame.cnot(ctrl,{a1}); }} }} "+E)
    if k==14: return (f"{{ uint64_t mask=p.mmask[{i0}]; "
                      f"while(mask){{ int tgt=__builtin_ctzll(mask); mask&=mask-1; if(tgt=={a1}) continue; "
                      f"int u=s.slot2id[{a1}], v=s.slot2id[tgt]; if(u>=0&&v>=0) s.engine.cz(u,v); s.frame.cz({a1},tgt); }} }} "+E)
    if k==15: return f"s.rot(p,{a1},NV_T_ANGLE); "+E
    if k==16: return f"s.rot(p,{a1},-NV_T_ANGLE); "+E
    if k==17: return f"{{ int q=s.slot2id[{a1}]; if(q>=0) s.engine.s(q,false); s.frame.s_gate({a1}); }} "+E
    if k==18: return f"{{ s.newq({a1}); s.engine.h(s.slot2id[{a1}]); s.rot(p,{a1},NV_T_ANGLE); }} "+E
    if k==19: return f"{{ s.newq({a1}); s.engine.h(s.slot2id[{a1}]); s.rot(p,{a1},-NV_T_ANGLE); }} "+E
    if k==20: return (f"{{ int i_1=s.slot2id[{a1}], i_2=s.slot2id[{a2}]; s.slot2id[{a1}]=-1; s.slot2id[{a2}]=-1; "
                      f"if(i_1>=0) s.slot2id[{a2}]=i_1; if(i_2>=0) s.slot2id[{a1}]=i_2; s.frame.swap({a1},{a2}); "
                      f"int q=s.slot2id[{a2}]; if(q>=0){{ s.engine.h(q); "
                      + _bcap_pre(a2) +
                      f" int b=s.measure_z(q); s.slot2id[{a2}]=-1; int m_abs=b^s.frame.zb({a2}); "
                      f"s.record.set({i0}, m_abs^{i1}); s.frame.set_xz({a2},(uint8_t)m_abs,0); "
                      + _bcap_post(2,i1) + f" }} }} "+E)
    if k==21: return f"s.rot(p,{a1},{dv!r}); "+E
    if k==22: return f"{{ s.newq({a1}); s.engine.h(s.slot2id[{a1}]); s.rot(p,{a1},{dv!r}); }} "+E
    if k==23: return (f"{{ int i_1=s.slot2id[{a1}], i_2=s.slot2id[{a2}]; s.slot2id[{a1}]=-1; s.slot2id[{a2}]=-1; "
                      f"if(i_1>=0) s.slot2id[{a2}]=i_1; if(i_2>=0) s.slot2id[{a1}]=i_2; s.frame.swap({a1},{a2}); }}")
    if k==24: return (f"{{ int q=s.slot2id[{a1}]; if(q>=0){{ "
                      + _bcap_pre(a1) +
                      f" int b=s.measure_z(q); s.slot2id[{a1}]=-1; int m_abs=b^s.frame.xb({a1}); "
                      f"s.record.set({i0}, m_abs^{i1}); s.frame.set_xz({a1},(uint8_t)m_abs,0); "
                      + _bcap_post(0,i1) + f" }} }} "+E)
    if k==25: return (f"{{ int q=s.slot2id[{a1}]; if(q>=0){{ s.engine.h(q); "
                      + _bcap_pre(a1) +
                      f" int b=s.measure_z(q); s.slot2id[{a1}]=-1; int m_abs=b^s.frame.zb({a1}); "
                      f"s.record.set({i0}, m_abs^{i1}); s.frame.set_xz({a1},(uint8_t)m_abs,0); "
                      + _bcap_post(1,i1) + f" }} }} "+E)
    if k==26: return f"{{ s.newq({a1}); s.engine.h(s.slot2id[{a1}]); }} "+E
    if k==27: return f"{{ int q=s.slot2id[{a1}]; if(q>=0) s.engine.h(q); s.frame.h({a1}); }} "+E
    if k==28: return (f"{{ int q=s.slot2id[{a1}]; int in_state=(s.frame.zb({a1})<<1)|s.frame.xb({a1}); "
                      f"int idx={i0}*4+in_state; const double* bcd=&p.u2_bcd[(size_t)idx*3]; "
                      f"double bb=bcd[0], cc=bcd[1], dd=bcd[2]; "
                      f"if(q>=0){{ if(std::abs(dd)>1e-12) s.engine.apply_rotation_pauli(q,0,1,dd); "
                      f"if(std::abs(cc)>1e-12) s.engine.apply_rotation_pauli(q,1,0,cc); "
                      f"if(std::abs(bb)>1e-12) s.engine.apply_rotation_pauli(q,0,1,bb); }} "
                      f"uint8_t out=p.u2_out[idx]; s.frame.set_xz({a1},out&1,(out>>1)&1); }} "+E)
    if k==29: return (f"{{ int lo=s.slot2id[{a1}], hi=s.slot2id[{a2}]; "
                      f"int in_state=(s.frame.zb({a2})<<3)|(s.frame.xb({a2})<<2)|(s.frame.zb({a1})<<1)|s.frame.xb({a1}); "
                      f"int idx={i0}*16+in_state; int st=p.u4_start[idx], cnt=p.u4_cnt[idx]; "
                      f"if(cnt<0){{ s.err=\"MO_ARRAY_U4: non-structural fused-U4 in_state selected (rot2/general not native-supported)\"; goto done; }} "
                      f"if(lo>=0 && hi>=0){{ for(int kk=0;kk<cnt;kk++){{ const double* op=&p.u4_ops[(size_t)(st+kk)*5]; "
                      f"int ot=(int)op[0], which=(int)op[1], px=(int)op[2], pz=(int)op[3]; double th=op[4]; "
                      f"if(ot==0) s.engine.cx(lo,hi); else if(ot==1) s.engine.cz(lo,hi); "
                      f"else {{ int qq=which?hi:lo; s.engine.apply_rotation_pauli(qq,px,pz,th); }} }} }} "
                      f"uint8_t out=p.u4_out[idx]; s.frame.set_xz({a1},out&1,(out>>1)&1); s.frame.set_xz({a2},(out>>2)&1,(out>>3)&1); }} "+E)
    if k==30: return ""
    raise ValueError(f"unhandled opcode {k}")

def gen_auth_cpp(t):
    K=list(map(int,t["kind"])); A1=list(map(int,t["a1"])); A2=list(map(int,t["a2"]))
    I0=list(map(int,t["i0"])); I1=list(map(int,t["i1"])); DV=list(map(float,t["dval"]))
    body=[]
    for i in range(len(K)):
        c=auth_op_cpp(K[i],A1[i],A2[i],I0[i],I1[i],DV[i])
        if c: body.append("    "+c)
    src=['#include "native_mdam_shot.hpp"','#include <cstring>','using namespace mdam;','',
        '// straight-line mirror of MdamShot::run() — operands as immediates, no dispatch.',
        'extern "C" void gen_run_auth(void* vmp, const void* progp){',
        '  MdamShot& s=*(MdamShot*)vmp; const MdamProgram& p=*(const MdamProgram*)progp; (void)p;',
        '  { int le=s.lazy_env(); s.engine.lazy_inverse = (le==1) ? true : (le==0 ? false : s.batch_lazy_hint); }']
    src+=body
    src+=['  done:;','}','',
        '// mirrors run_batch: same master-seed expansion + shot-0 lazy probe (stats omitted).',
        'extern "C" int gen_run_auth_batch(const void* progp, void* vmp, uint64_t num_shots,',
        '   uint64_t mshi,uint64_t mslo,uint64_t mihi,uint64_t milo, uint8_t* out_record, char* out_err, int errlen){',
        '  MdamShot& s=*(MdamShot*)vmp; const MdamProgram& p=*(const MdamProgram*)progp;',
        '  NativeRng master; master.seed_from_state(mshi,mslo,mihi,milo);',
        '  const uint64_t RNG_EXCL=((uint64_t)1<<63)-1; const size_t nm=(size_t)p.num_measurements;',
        '  s.batch_lazy_hint=true; s.engine.magic_ever=false;',
        '  for(uint64_t sh=0; sh<num_shots; sh++){',
        '    uint64_t sd=master.bounded(RNG_EXCL); __uint128_t st,inc; SeedExpand::seedseq_pcg64(sd,st,inc);',
        '    s.reset_shot(p,(uint64_t)(st>>64),(uint64_t)st,(uint64_t)(inc>>64),(uint64_t)inc);',
        '    gen_run_auth(vmp,progp);',
        '    if(sh==0 && s.lazy_env()==-1) s.batch_lazy_hint = !s.engine.magic_ever;',
        '    std::memcpy(out_record+(size_t)sh*nm, s.record.bits.data(), nm);',
        '    if(s.err){ if(out_err){ std::strncpy(out_err,s.err,errlen-1); out_err[errlen-1]=0; } return 1; }',
        '  }',
        '  if(out_err) out_err[0]=0; return 0;','}','']
    return "\n".join(src)

def get_auth_so(t, cache_dir=None):
    return cg.get_so_cached(gen_auth_cpp(t), cache_dir=cache_dir)

if __name__=="__main__":
    import clifft
    try: clifft.set_num_threads(1)
    except Exception: pass
    lib.nvm_rb_static.argtypes=[C]
    N=4000; eb=ctypes.create_string_buffer(256)
    print(f"{'bench':17s} {'rb':>2s} {'auth_ns':>9s} {'genauth_ns':>10s} {'speedup':>7s} {'clifft':>9s} {'cl/gen':>6s} {'mism':>5s}")
    for b in (sys.argv[1:] or ["cultivation_d3","coherent_d3_r1","coherent_d3_r3","cultivation_d5"]):
        text=open(f"{ROOT}/qec_bench/circuits/{b}.stim").read()
        prog=clifft.compile(text); t=translate(prog); nm=t["num_meas"]
        so,hit,cw=get_auth_so(t)
        g=ctypes.CDLL(so); g.gen_run_auth_batch.restype=C; g.gen_run_auth_batch.argtypes=[P_,P_,U]+[U]*4+[P_,P_,C]
        # clifft anchor
        best_cl=1e30
        for _ in range(3):
            t0=time.perf_counter(); clifft.sample(prog,N); best_cl=min(best_cl,(time.perf_counter()-t0)/N*1e9)
        for rb in (1,):                                   # adaptive context runs rb_static ON
            ph=make_prog(lib,t); va=lib.nvm_mdam_vm_create(ph); vg=lib.nvm_mdam_vm_create(ph)
            lib.nvm_rb_static_reset(); lib.nvm_rb_static(rb)
            A=np.zeros((N,nm),np.uint8); B=np.zeros((N,nm),np.uint8)
            # bit-exactness: same master seed
            r1=lib.nvm_mdam_sample_batch(ph,va,N,*pcg(31),A.ctypes.data,None,eb,256); assert r1==0, eb.value
            r2=g.gen_run_auth_batch(ph,vg,N,*pcg(31),B.ctypes.data,eb,256); assert r2==0, eb.value
            mism=int((A!=B).sum())
            # timing (min-of-5, separate vms already warm)
            ta=1e30; tg=1e30
            for _ in range(5):
                t0=time.perf_counter(); lib.nvm_mdam_sample_batch(ph,va,N,*pcg(77),A.ctypes.data,None,eb,256); ta=min(ta,(time.perf_counter()-t0)/N*1e9)
            for _ in range(5):
                t0=time.perf_counter(); g.gen_run_auth_batch(ph,vg,N,*pcg(77),B.ctypes.data,eb,256); tg=min(tg,(time.perf_counter()-t0)/N*1e9)
            lib.nvm_rb_static(0)
            print(f"{b:17s} {rb:>2d} {ta:9.1f} {tg:10.1f} {ta/tg:6.2f}x {best_cl:9.1f} {best_cl/tg:5.2f}x {mism:5d}")
