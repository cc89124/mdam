"""Save both clifft.sample and clifft.execute results."""
import sys, os, numpy as np
sys.path.insert(0, '.')
import clifft

CIRCUITS = ['distillation', 'cultivation_d3', 'coherent_d3_r1']
SHOTS = 500
SEED = 42

def unwrap(obj, shots):
    try:
        a = np.asarray(obj, dtype=np.uint8)
        if a.ndim >= 1 and a.size > 0: return a.reshape(shots, -1)
    except: pass
    for attr in ('data','bits','results','outcomes','measurements','samples'):
        if hasattr(obj, attr):
            x = getattr(obj, attr)
            if callable(x):
                try: x = x()
                except: continue
            try:
                a = np.asarray(x, dtype=np.uint8)
                if a.ndim >= 1 and a.size > 0: return a.reshape(shots, -1)
            except: pass
    raise TypeError("cannot unwrap")

def sample_via_execute(prog, shots, seed):
    rng = np.random.default_rng(seed)
    n_meas = prog.num_measurements
    out = np.zeros((shots, n_meas), dtype=np.uint8)
    for sh in range(shots):
        ss = int(rng.integers(0, 2**63 - 1))
        state = clifft.State(
            peak_rank=prog.peak_rank,
            num_measurements=prog.num_measurements,
            num_detectors=prog.num_detectors,
            num_observables=prog.num_observables,
            num_exp_vals=prog.num_exp_vals,
            seed=ss,
        )
        clifft.execute(prog, state)
        for i, bit in enumerate(state.meas_record):
            if i < n_meas: out[sh, i] = bit
    return out

os.makedirs('verify_data', exist_ok=True)
for name in CIRCUITS:
    path = f'qec_bench/circuits/{name}.stim'
    if not os.path.exists(path): continue
    with open(path) as h: src = h.read()
    prog = clifft.compile(src)
    A = unwrap(clifft.sample(prog, shots=SHOTS, seed=SEED), SHOTS)
    np.save(f'verify_data/{name}_clifft_sample.npy', A)
    print(f'{name}: clifft.sample saved {A.shape}')

    E = sample_via_execute(prog, SHOTS, SEED)
    np.save(f'verify_data/{name}_clifft_execute.npy', E)
    print(f'{name}: clifft.execute saved {E.shape}')