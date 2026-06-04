"""Try the three big circuits that fit numpy's 64-dim limit."""
import sys, os, time, resource
import numpy as np
sys.path.insert(0, '.')
import clifft
from ttn_backend.backend_spec import export_backend_spec, assign_homes_and_classify
from ttn_backend import TTNBackend

CIRCUITS = ['coherent_d5_r1', 'coherent_d5_r5', 'coherent_d7_r1']
SEED = 42
CIRC_DIR = 'qec_bench/circuits'

def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

for name in CIRCUITS:
    print(f'\n=== {name} ===')
    path = os.path.join(CIRC_DIR, name + '.stim')
    with open(path) as h: src = h.read()
    prog = clifft.compile(src)
    print(f'  ops={prog.num_instructions}, peak_rank={prog.peak_rank}, n_meas={prog.num_measurements}')
    
    t0 = time.time()
    spec = export_backend_spec(prog, strict=False)
    homing = assign_homes_and_classify(spec)
    print(f'  spec built: {time.time()-t0:.2f}s, union sum2 = {spec["union"]["sum2"]*16/1e6:.2f}MB')
    print(f'  classes: A={homing["stats"]["n_A"]}, B={homing["stats"]["n_B"]}, C={homing["stats"]["n_C"]}')

    backend = TTNBackend(spec, homing)
    t0 = time.time()
    try:
        rec = backend.run_shot(prog, SEED)
        elapsed = time.time() - t0
        m = backend.last_metrics or {}
        print(f'  1 shot: {elapsed:.2f}s, record entries = {len(rec)}')
        print(f'  metrics: stored={m.get("peak_stored_bytes", 0)/1e6:.3f}MB, '
              f'pair_workspace={m.get("peak_pair_workspace_bytes", 0)/1e6:.3f}MB, '
              f'max_bond={m.get("max_bond_dim", 0)}, '
              f'transports={m.get("n_transports", 0)}, qr={m.get("n_qr", 0)}')
        # Estimate cost for 100 shots
        est_100 = elapsed * 100
        if est_100 < 60:
            print(f'  100 shots estimate: {est_100:.0f}s -- FEASIBLE')
        elif est_100 < 600:
            print(f'  100 shots estimate: {est_100/60:.1f}min -- ACCEPTABLE')
        else:
            print(f'  100 shots estimate: {est_100/60:.1f}min -- SLOW')
    except Exception as e:
        elapsed = time.time() - t0
        print(f'  FAILED after {elapsed:.2f}s: {e!r}')
        if getattr(backend, "last_metrics", None):
            print(f'  metrics at fail: {backend.last_metrics}')
        import traceback
        traceback.print_exc()
        continue
    
    print(f'  process RSS: {peak_rss_mb():.0f}MB')
