#!/usr/bin/env python
"""Gate C2-C4: full cultivation_d3 noisy magic-core one-shot in native C++ vs authoritative Python
run_shot, across 25 seeds.  The whole shot runs inside nvm_mdam_run (1 Python->C++ call, 0 callbacks)."""
import os, sys, ctypes, math, cmath
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import clifft
from mdam.backend.backend import NearCliffordBackend, _opname, count_idents
from mdam.backend.backend import _zxz_angles, _u4_decompose   # Gate L Tier-3 DIRECT: fused-unitary decompose
from mdam.frame import frame_layer as fl

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_HERE = os.path.dirname(__file__)
BENCH = os.environ.get("MDAM_BENCH", "cultivation_d3")

(MO_FRAME_H, MO_FRAME_CNOT, MO_FRAME_CZ, MO_FRAME_SWAP, MO_FRAME_S, MO_APPLY_PAULI, MO_NOISE,
 MO_NOISE_BLOCK, MO_READOUT_NOISE, MO_MEAS_DORM_STATIC, MO_MEAS_DORM_RANDOM, MO_ARRAY_CNOT,
 MO_ARRAY_CZ, MO_MULTI_CNOT, MO_MULTI_CZ, MO_ARRAY_T, MO_ARRAY_T_DAG, MO_ARRAY_S, MO_EXPAND_T,
 MO_EXPAND_T_DAG, MO_SWAP_MEAS_INTERFERE,
 MO_ARRAY_ROT, MO_EXPAND_ROT, MO_ARRAY_SWAP, MO_MEAS_ACTIVE_DIAGONAL, MO_MEAS_ACTIVE_INTERFERE,
 MO_EXPAND, MO_ARRAY_H,
 MO_ARRAY_U2, MO_ARRAY_U4,
 MO_END) = range(31)

def translate(prog):
    FS = fl.FLAG_SIGN
    kind, a1l, a2l, i0l, i1l, dvl = [], [], [], [], [], []
    mmask_pool = []; cp_pool = []; max_cidx = [int(prog.num_measurements)-1]
    # Gate L Tier-3 DIRECT: precompute the frame-keyed fused-unitary decompositions (compile-time static).
    u2_nodes = getattr(prog, "fused_u2_nodes", None); u4_nodes = getattr(prog, "fused_u4_nodes", None)
    u2_tab = {}; u4_tab = {}   # cp_idx -> per-in_state decomposition rows
    def note(c):
        if c is not None and int(c) > max_cidx[0]: max_cidx[0] = int(c)
    def emit(k, a1=0, a2=0, i0=0, i1=0, dv=0.0):
        kind.append(k); a1l.append(a1); a2l.append(a2); i0l.append(i0); i1l.append(i1); dvl.append(dv)
    for s in range(len(prog)):
        inst = prog[s]; nm = _opname(inst.opcode); d = fl._d(inst)
        a1 = int(inst.axis_1); a2 = int(inst.axis_2)
        sign = 1 if (int(getattr(inst, "flags", 0)) & FS) else 0
        if nm in fl.IGNORE_OPS: continue
        if nm == "OP_FRAME_H": emit(MO_FRAME_H, a1)
        elif nm == "OP_FRAME_CNOT": emit(MO_FRAME_CNOT, a1, a2)
        elif nm == "OP_FRAME_CZ": emit(MO_FRAME_CZ, a1, a2)
        elif nm == "OP_FRAME_SWAP": emit(MO_FRAME_SWAP, a1, a2)
        elif nm in ("OP_FRAME_S", "OP_FRAME_S_DAG"): emit(MO_FRAME_S, a1)
        elif nm == "OP_APPLY_PAULI":
            cond = d.get("condition_idx"); mi = d.get("cp_mask_idx")
            if cond is None or mi is None: continue
            m = fl._cp_get(prog, fl.CP_MASK_ATTRS, int(mi))
            xw = [int(w) for w in m["x_words"]] if (m and m.get("x_words")) else [0]  # ALL words (n>64 spans >1)
            zw = [int(w) for w in m["z_words"]] if (m and m.get("z_words")) else [0]
            note(cond); cp_pool.append((xw, zw)); emit(MO_APPLY_PAULI, 0, 0, int(cond), len(cp_pool)-1)
        elif nm == "OP_NOISE":
            site = d.get("noise_site_idx")
            if site is not None: emit(MO_NOISE, 0, 0, int(site))
        elif nm == "OP_NOISE_BLOCK":
            st = d.get("start_site", d.get("noise_site_idx", d.get("block_idx"))); cnt = int(d.get("count", 1))
            if st is not None: emit(MO_NOISE_BLOCK, 0, 0, int(st), cnt)
        elif nm == "OP_READOUT_NOISE":
            ei = d.get("readout_noise_idx"); entries = getattr(prog, "readout_noise", None)
            if ei is not None and entries is not None:
                e = entries[int(ei)]; note(e["meas_idx"]); emit(MO_READOUT_NOISE, 0, 0, int(e["meas_idx"]), 0, float(e["prob"]))
        elif nm in ("OP_MEAS_DORMANT_STATIC", "OP_MEAS_DORMANT_STATIC_FORCED"):
            note(d.get("classical_idx", 0)); emit(MO_MEAS_DORM_STATIC, a1, 0, int(d.get("classical_idx", 0)), sign)
        elif nm in ("OP_MEAS_DORMANT_RANDOM", "OP_MEAS_DORMANT_RANDOM_FORCED"):
            note(d.get("classical_idx", 0)); emit(MO_MEAS_DORM_RANDOM, a1, 0, int(d.get("classical_idx", 0)), sign)
        elif nm == "OP_ARRAY_CNOT": emit(MO_ARRAY_CNOT, a1, a2)
        elif nm == "OP_ARRAY_CZ": emit(MO_ARRAY_CZ, a1, a2)
        elif nm == "OP_ARRAY_MULTI_CNOT": mmask_pool.append(int(d["mask"])); emit(MO_MULTI_CNOT, a1, 0, len(mmask_pool)-1)
        elif nm == "OP_ARRAY_MULTI_CZ": mmask_pool.append(int(d["mask"])); emit(MO_MULTI_CZ, a1, 0, len(mmask_pool)-1)
        elif nm == "OP_ARRAY_T": emit(MO_ARRAY_T, a1)
        elif nm == "OP_ARRAY_T_DAG": emit(MO_ARRAY_T_DAG, a1)
        elif nm == "OP_ARRAY_S": emit(MO_ARRAY_S, a1)
        elif nm == "OP_EXPAND_T": emit(MO_EXPAND_T, a1)
        elif nm == "OP_EXPAND_T_DAG": emit(MO_EXPAND_T_DAG, a1)
        elif nm in ("OP_SWAP_MEAS_INTERFERE", "OP_SWAP_MEAS_INTERFERE_FORCED"):
            note(d.get("classical_idx", 0)); emit(MO_SWAP_MEAS_INTERFERE, a1, a2, int(d.get("classical_idx", 0)), sign)
        # ---- Gate L1: coherent opcodes ----
        elif nm == "OP_ARRAY_ROT":                      # arbitrary-theta diagonal rotation; theta in dval
            emit(MO_ARRAY_ROT, a1, 0, 0, 0, cmath.phase(complex(d["weight_re"], d["weight_im"])))
        elif nm == "OP_EXPAND_ROT":                     # birth + arbitrary-theta rotation
            emit(MO_EXPAND_ROT, a1, 0, 0, 0, cmath.phase(complex(d["weight_re"], d["weight_im"])))
        elif nm == "OP_ARRAY_SWAP":
            emit(MO_ARRAY_SWAP, a1, a2)
        elif nm in ("OP_MEAS_ACTIVE_DIAGONAL", "OP_MEAS_ACTIVE_DIAGONAL_FORCED"):
            note(d.get("classical_idx", 0)); emit(MO_MEAS_ACTIVE_DIAGONAL, a1, 0, int(d.get("classical_idx", 0)), sign)
        elif nm in ("OP_MEAS_ACTIVE_INTERFERE", "OP_MEAS_ACTIVE_INTERFERE_FORCED"):
            note(d.get("classical_idx", 0)); emit(MO_MEAS_ACTIVE_INTERFERE, a1, 0, int(d.get("classical_idx", 0)), sign)
        # ---- Gate L Tier-3 (de-fused bytecode_passes=None dialect) ----
        elif nm == "OP_EXPAND": emit(MO_EXPAND, a1)             # birth |+> (newq + H), no rotation
        elif nm == "OP_ARRAY_H": emit(MO_ARRAY_H, a1)          # active-register Hadamard
        # ---- Gate L Tier-3 DIRECT (fused dialect): frame-keyed fused unitaries (preserve maxM) ----
        elif nm == "OP_ARRAY_U2":
            cp = int(d["cp_idx"])
            if cp not in u2_tab:                               # precompute ZXZ for all 4 frame in_states
                node = u2_nodes[cp]; rows = []
                for ins_st in range(4):
                    U = np.asarray(node["matrices"][ins_st], dtype=complex).reshape(2, 2)
                    b, c, dd = _zxz_angles(U)
                    rows.append((b, c, dd, int(node["out_states"][ins_st])))
                u2_tab[cp] = rows
            emit(MO_ARRAY_U2, a1, 0, cp)
        elif nm == "OP_ARRAY_U4":
            cp = int(d["cp_idx"])
            if cp not in u4_tab:                               # precompute op-list for all 16 frame in_states
                node = u4_nodes[cp]; rows = []
                for ins_st in range(16):
                    entry = node["entries"][ins_st]; out = int(entry["out_state"])
                    try:                                       # a never-reached in_state may be non-structural:
                        ops = _u4_decompose(np.asarray(entry["matrix"], dtype=complex).reshape(4, 4))
                        enc = []
                        for (knd, args) in ops:
                            if knd == "cx": enc.append((0, 0, 0, 0, 0.0))
                            elif knd == "cz": enc.append((1, 0, 0, 0, 0.0))
                            elif knd == "rot1":
                                which, x, z, theta = args; enc.append((2, which, x, z, theta))
                            else:
                                raise RuntimeError(f"U4 op {knd} (rot2/general) not native-supported")
                        rows.append((enc, out, True))
                    except Exception:
                        rows.append(([], out, False))          # sentinel: native loud-fails iff selected
                u4_tab[cp] = rows
            emit(MO_ARRAY_U4, a1, a2, cp)
        else:
            raise RuntimeError(f"unsupported opcode {nm}")
    # noise hazards + sites
    probs = fl._noise_probabilities(prog); probs = probs[np.isfinite(probs)]
    probs = np.clip(probs, 0.0, 1.0 - 2.0**-53); hazards = np.cumsum(-np.log1p(-probs))
    sites = getattr(prog, "noise_sites", []) or []
    site_nchan = np.zeros(len(sites), np.int32); ch_prob=[]; ch_x=[]; ch_z=[]
    for i, st in enumerate(sites):
        chans = st if isinstance(st, list) else []
        site_nchan[i] = len(chans)
        for ch in chans:
            ch_prob.append(float(ch.get("prob", 0.0)))
            ch_x.append([int(w) for w in ch.get("x_words", [])] or [0])   # ALL words (n>64)
            ch_z.append([int(w) for w in ch.get("z_words", [])] or [0])
    ch_prob=np.array(ch_prob, np.float64)
    # MULTIWORD flatten: cp/noise masks -> flat (count x MW) uint64, MW=ceil(num_qubits/64) (== native).
    MW = (int(prog.num_qubits) + 63) // 64
    def _flatw(lol):
        out = []
        for wl in lol:
            wl = list(wl)[:MW] + [0] * max(0, MW - len(wl))
            out.extend(wl)
        return np.array(out, np.uint64) if out else np.zeros(0, np.uint64)
    ch_x = _flatw(ch_x); ch_z = _flatw(ch_z)
    # Gate L Tier-3 DIRECT: flatten the U2/U4 decomposition tables (indexed by cp_idx)
    n_u2 = (max(u2_tab) + 1) if u2_tab else 0
    u2_bcd = np.zeros((n_u2, 4, 3), np.float64); u2_out = np.zeros((n_u2, 4), np.uint8)
    for cp, rows in u2_tab.items():
        for ins_st, (b, c, dd, out) in enumerate(rows):
            u2_bcd[cp, ins_st] = (b, c, dd); u2_out[cp, ins_st] = out
    n_u4 = (max(u4_tab) + 1) if u4_tab else 0
    u4_start = np.zeros((n_u4, 16), np.int32); u4_cnt = np.zeros((n_u4, 16), np.int32)
    u4_out = np.zeros((n_u4, 16), np.uint8); u4_ops_list = []
    for cp, rows in u4_tab.items():
        for ins_st, (enc, out, ok) in enumerate(rows):
            u4_out[cp, ins_st] = out
            if ok:
                u4_start[cp, ins_st] = len(u4_ops_list); u4_cnt[cp, ins_st] = len(enc)
                u4_ops_list.extend(enc)
            else:
                u4_cnt[cp, ins_st] = -1            # sentinel: non-structural (never selected at runtime)
    u4_ops = (np.array(u4_ops_list, np.float64).reshape(-1, 5) if u4_ops_list else np.zeros((0, 5), np.float64))
    return dict(kind=np.array(kind, np.uint8), a1=np.array(a1l, np.int32), a2=np.array(a2l, np.int32),
                i0=np.array(i0l, np.int32), i1=np.array(i1l, np.int32), dval=np.array(dvl, np.float64),
                mmask=np.array(mmask_pool, np.uint64), hazards=hazards.astype(np.float64),
                site_nchan=site_nchan, ch_prob=ch_prob, ch_x=ch_x, ch_z=ch_z,
                cp_x=_flatw([c[0] for c in cp_pool]), cp_z=_flatw([c[1] for c in cp_pool]), ncp=len(cp_pool),
                num_qubits=int(prog.num_qubits), num_meas=int(prog.num_measurements),
                engine_n=int(count_idents(prog)),
                max_work=int(getattr(prog, "peak_rank", count_idents(prog))) + 2,
                record_cap=max_cidx[0] + 1,
                n_u2=n_u2, u2_bcd=u2_bcd, u2_out=u2_out,
                n_u4=n_u4, u4_start=u4_start, u4_cnt=u4_cnt, u4_ops=u4_ops, u4_out=u4_out)

def load_lib():
    lib = ctypes.CDLL(os.path.join(_HERE, "native_mdam_vm.so"))
    P = ctypes.c_void_p
    lib.nvm_mdam_create.restype = P
    lib.nvm_mdam_create.argtypes = [ctypes.c_int, P,P,P,P,P,P, P, ctypes.c_int, P, ctypes.c_int,
                                    P,P,P,P, ctypes.c_int, P,P, ctypes.c_int] + [ctypes.c_int]*5
    lib.nvm_mdam_vm_create.restype = P; lib.nvm_mdam_vm_create.argtypes=[P]
    lib.nvm_mdam_run.restype = ctypes.c_int
    lib.nvm_mdam_run.argtypes = [P,P]+[ctypes.c_uint64]*4+[P,P,P,P,P,ctypes.c_int]
    lib.nvm_mdam_set_u2.argtypes = [P, ctypes.c_int, P, P]
    lib.nvm_mdam_set_u4.argtypes = [P, ctypes.c_int, P, P, P, ctypes.c_int, P]
    return lib

def vp(a): return a.ctypes.data if a.size else 0

def make_prog(lib, t):
    ph = lib.nvm_mdam_create(len(t["kind"]), vp(t["kind"]), vp(t["a1"]), vp(t["a2"]), vp(t["i0"]), vp(t["i1"]),
        vp(t["dval"]), vp(t["mmask"]), len(t["mmask"]), vp(t["hazards"]), len(t["hazards"]),
        vp(t["site_nchan"]), vp(t["ch_prob"]), vp(t["ch_x"]), vp(t["ch_z"]), len(t["site_nchan"]),
        vp(t["cp_x"]), vp(t["cp_z"]), t["ncp"],
        t["num_qubits"], t["num_meas"], t["engine_n"], t["max_work"], t["record_cap"])
    # Gate L Tier-3 DIRECT: attach the precomputed fused-unitary decomposition tables (if any).
    # (C side copies immediately; locals keep the buffers alive across the call.)
    if t.get("n_u2", 0) > 0:
        bcd = np.ascontiguousarray(t["u2_bcd"], np.float64); u2o = np.ascontiguousarray(t["u2_out"], np.uint8)
        lib.nvm_mdam_set_u2(ph, t["n_u2"], vp(bcd), vp(u2o))
    if t.get("n_u4", 0) > 0:
        st = np.ascontiguousarray(t["u4_start"], np.int32); cn = np.ascontiguousarray(t["u4_cnt"], np.int32)
        ops = np.ascontiguousarray(t["u4_ops"], np.float64); u4o = np.ascontiguousarray(t["u4_out"], np.uint8)
        lib.nvm_mdam_set_u4(ph, t["n_u4"], vp(st), vp(cn), vp(ops), len(ops), vp(u4o))
    return ph

def pcg(seed):
    rng = np.random.default_rng(seed); s = rng.bit_generator.state["state"]
    st, inc = int(s["state"]), int(s["inc"]); M = (1<<64)-1
    return (st>>64)&M, st&M, (inc>>64)&M, inc&M

if __name__ == "__main__":
    prog = clifft.compile(open(os.path.join(_ROOT, f"qec_bench/circuits/{BENCH}.stim")).read())
    t = translate(prog); nm = t["num_meas"]
    lib = load_lib(); ph = make_prog(lib, t); vm = lib.nvm_mdam_vm_create(ph)

    fixed = [1, 7, 42, 123, 999]
    rs = np.random.RandomState(2026); rnd = [int(x) for x in rs.randint(0, 2**31-1, size=20)]
    seeds = fixed + rnd
    n_pass = 0; first_div = None
    for sd in seeds:
        be = NearCliffordBackend(clifft_axis_bounded=True, drop_dead=False, structure_once=False)
        o = be._reset
        be._reset = lambda prog, _o=o, _b=be: (_o(prog), setattr(_b.nc, "_compiled_core", True))[0]
        rec = be.run_shot(prog, sd)
        pyvec = np.zeros(nm, np.uint8)
        for c, b in rec.items():
            if 0 <= c < nm: pyvec[c] = b
        out = np.zeros(nm, np.uint8); draws = ctypes.c_ulonglong(); comp = ctypes.c_int(); orac = ctypes.c_int()
        errbuf = ctypes.create_string_buffer(256)
        shi, slo, ihi, ilo = pcg(sd)
        rc = lib.nvm_mdam_run(ph, vm, shi, slo, ihi, ilo, out.ctypes.data,
                              ctypes.byref(draws), ctypes.byref(comp), ctypes.byref(orac), errbuf, 256)
        if rc != 0:
            print(f"  [seed {sd}] native ERROR: {errbuf.value.decode()}"); first_div = first_div or (sd, "error", errbuf.value.decode()); continue
        ok = np.array_equal(pyvec, out)
        if ok: n_pass += 1
        else:
            diff = np.where(pyvec != out)[0]
            if first_div is None: first_div = (sd, int(diff[0]), f"py={pyvec[diff[0]]} native={out[diff[0]]}")
            print(f"  [seed {sd}] MISMATCH at meas idx {diff[:6]} (compiled={comp.value} oracle={orac.value} draws={draws.value})")
    print(f"\ncultivation_d3 full native one-shot: {n_pass}/{len(seeds)} seeds record-exact "
          f"(magic compiled={comp.value}/oracle={orac.value} per shot, draws={draws.value})")
    if n_pass == len(seeds):
        print("ALL 25 SEEDS RECORD-EXACT")
    else:
        print(f"first divergence: {first_div}")
    sys.exit(0 if n_pass == len(seeds) else 1)
