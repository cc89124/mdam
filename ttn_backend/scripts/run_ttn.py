"""Run TTN backend for each circuit and save to .npy."""
import sys, os, numpy as np, time
sys.path.insert(0, '.')
import clifft
from ttn_backend.backend_spec import export_backend_spec, assign_homes_and_classify
from ttn_backend import TTNBackend
 
CIRCUITS = ['distillation', 'cultivation_d3', 'coherent_d3_r1']
SHOTS = 500
SEED = 42
 
os.makedirs('verify_data', exist_ok=True)
for name in CIRCUITS:
    path = f'qec_bench/circuits/{name}.stim'
    if not os.path.exists(path): continue
    with open(path) as h: src = h.read()
    prog = clifft.compile(src)
    spec = export_backend_spec(prog, strict=False)
    homing = assign_homes_and_classify(spec)
    t0 = time.time()
    backend = TTNBackend(spec, homing)
    T = backend.sample(prog, shots=SHOTS, seed=SEED, num_measurements=prog.num_measurements)
    print(f'{name}: TTN done in {time.time()-t0:.2f}s, shape={T.shape}')
    np.save(f'verify_data/{name}_ttn.npy', T)
 
