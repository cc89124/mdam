"""Compare TTN vs both clifft references."""
import numpy as np, os
from collections import Counter

CIRCUITS = ['distillation', 'cultivation_d3', 'coherent_d3_r1']

def tvd_marg(A, B):
    return float(np.abs(A.mean(axis=0) - B.mean(axis=0)).max())

def tvd_joint(A, B):
    cA = Counter(tuple(r) for r in A.tolist())
    cB = Counter(tuple(r) for r in B.tolist())
    keys = set(cA) | set(cB); nA = len(A); nB = len(B)
    return 0.5 * sum(abs(cA[k]/nA - cB[k]/nB) for k in keys)

print(f'{"circuit":22s} | {"sample vs execute":>18s} | {"sample vs TTN":>18s} | {"execute vs TTN":>18s}')
print('-' * 100)
for name in CIRCUITS:
    paths = {
        'sample': f'verify_data/{name}_clifft_sample.npy',
        'execute': f'verify_data/{name}_clifft_execute.npy',
        'ttn': f'verify_data/{name}_ttn.npy',
    }
    if not all(os.path.exists(p) for p in paths.values()):
        print(f'{name}: missing'); continue
    A = np.load(paths['sample'])
    E = np.load(paths['execute'])
    T = np.load(paths['ttn'])
    print(f'{name:22s} | m={tvd_marg(A,E):.3f} j={tvd_joint(A,E):.3f}    '
          f'| m={tvd_marg(A,T):.3f} j={tvd_joint(A,T):.3f}    '
          f'| m={tvd_marg(E,T):.3f} j={tvd_joint(E,T):.3f}')