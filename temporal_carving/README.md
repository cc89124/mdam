# Temporal-Live Carving TTN Layout Optimizer

This package implements a compile-time carving-tree optimizer over time-resolved
live-axis traces. It does not inspect tensors and does not patch the runtime TTN
backend.

## Fixed Cost Contract

Notation: axis set `U`. For a time index `t`, `L(t) ⊆ U` is the live set.
`E_t` is the set of interaction edges (unordered axis pairs) active at `t`.
`d_u` is the local dimension of axis `u` (default 2 for qubits); `ell(X) = Σ_{u∈X} log2 d_u`.

For a fixed `S`, sweep `t` in increasing order maintaining an accumulator `acc`:

```text
acc = 0
for t in timeline:
    Slive  = S      ∩ L(t)
    Sclive = (U\S)  ∩ L(t)
    if Slive == ∅ or Sclive == ∅:
        acc = 0
        C_S[t] = 0
    else:
        cross = #{ (i,j) in E_t : i in S, j in U\S, and i,j both in L(t) }
        acc  += cross
        C_S[t] = acc
```

```text
rhat(S, t) = min( C_S[t], ell(S ∩ L(t)), ell((U\S) ∩ L(t)) )
```

For tree node `v`:

```text
p_v(t) = Σ_{u : leaf(u)=v and u in L(t)} log2 d_u
Ehat_v(t) = p_v(t) + Σ_{e incident to v} rhat(cut_e, t)
```

Objective:

```text
T* = argmin_T max_t max_v Ehat_v(t)
```

At an internal node joining `A` and `B`, with `S=A∪B`:

```text
nodecost(A, B) = max_t [ rhat(A,t) + rhat(B,t) + rhat(S,t) ]
```

The max is over the pointwise sum. Do not take per-cut maxima before summing.

## Modules

```text
temporal_carving/
  io.py
  tree.py
  cost.py
  surrogate.py
  seed.py
  refine.py
  exact.py
  pipeline.py
  synth.py
```

## CLI

```bash
python -m temporal_carving.pipeline \
  --trace path/to/trace_dir \
  --seeder recursive_balanced_mincut \
  --refine nni,spr \
  --exact \
  --out-tree tree.json
```

## Algorithmic Status

The true cut functional is symmetric but non-submodular due to reset, min-cap,
and max-over-time. The package therefore uses established algorithms:

- KL recursive balanced mincut on the submodular surrogate for seeding.
- Louvain only as an ablation seeder.
- NNI/SPR local search on the true `cost.py` objective.
- subset DP exact oracle for small `n`.

The objective is the novelty; the search primitives are standard.
