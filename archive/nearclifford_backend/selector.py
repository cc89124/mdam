"""Offline backend selector: decide *per circuit* whether to simulate with the
dense clifft model or the block near-Clifford backend, BEFORE sampling any shot.

Why this is well-posed (fixed Pauli noise)
------------------------------------------
A stabilizer + non-Clifford-rotation circuit under a FIXED Pauli-noise model has a
*shot-invariant structural schedule*. A sampled Pauli fault is itself a Clifford
(Pauli) gate, so from shot to shot it only relabels the Clifford/Pauli frame --
it does not change the anticommutation pattern between the (fixed) rotation
generators and the (fixed) measured Paulis. That pattern is exactly what sets

    k_t  = clifft's active rank at step t      (its dense register is 2^{k_t}),
    b_t  = near-Clifford's peak block at t      (its largest dense factor is 2^{b_t}).

So `k_max = max_t k_t` and `b_max = max_t b_t` are properties of the CIRCUIT, not
of the shot: one offline structural pass fixes the backend for every subsequent
shot.  (Empirically verified shot-invariant across seeds; see
`tests/test_selector_invariance.py`.)

Decision
--------
Peak memory is `16 * 2^{k_max}` (clifft) vs `16 * 2^{b_max}` (near-Clifford).

    b_max <= k_max - margin   -> 'nc'      (near-Clifford strictly saves memory)
    otherwise                 -> 'clifft'  (NC cannot shrink the state; running it
                                            only adds factor/merge/bookkeeping cost)

Choosing this way makes the *deployed* peak `min(2^{k_max}, 2^{b_max})`, which is
**never worse than clifft** -- the guarantee, obtained by selection rather than by
forcing NC's representation to dominate clifft on every circuit.

`margin` (default 1 qubit = a 2x memory factor) is the cushion that must be earned
before paying NC's constant-factor scan overhead; raise it to be more conservative
about runtime, lower it to 0 to select purely on peak memory.
"""
from __future__ import annotations

import clifft
from nearclifford_backend.backend import NearCliffordBackend
from nearclifford_backend.block_magic import MagicRegister


def _compile(circuit):
    """Accept a compiled program, raw .stim source, or a qec_bench circuit name."""
    if not isinstance(circuit, str):
        return circuit                      # already a compiled program
    if "\n" in circuit or circuit.strip().upper().startswith(("QUBIT", "R ", "H ", "CX")):
        return clifft.compile(circuit)      # raw .stim text
    return clifft.compile(open(f"qec_bench/circuits/{circuit}.stim").read())  # name


def _schedule_once(prog, seed):
    """Run ONE near-Clifford shot, recording the per-step (k_t, b_t) schedule.

    b_t is sampled at the TRANSIENT high-water: inside the rotation merge, before
    factoring peels the block back down -- the honest in-kron peak (the settled
    step-boundary value under-reports it; e.g. cultivation_d5 settles to 10/11 but
    transiently materialises a 14-qubit vector mid-merge). Done with a scoped,
    restored monkeypatch so the library hot path is untouched in normal use."""
    k_sched, b_sched = [], []
    trans = [0]
    orig_merge = MagicRegister._merge

    def merge_w(self, support):
        b = orig_merge(self, support)
        sz = len(self.blocks[b][0])
        if sz > trans[0]:
            trans[0] = sz
        return b

    def rec(step, backend):
        k_sched.append(len(backend.slot2id))
        b_sched.append(max(backend.nc.mag.max_block(), trans[0]))
        trans[0] = backend.nc.mag.max_block()   # rearm to settled for next interval

    MagicRegister._merge = merge_w
    try:
        be = NearCliffordBackend(block=True)
        be.run_shot(prog, seed, step_recorder=rec)
    finally:
        MagicRegister._merge = orig_merge
    return k_sched, b_sched


def analyze(circuit, seeds=(42,)):
    """Structural analysis for the offline decision. Returns a dict with the
    worst-case (over `seeds`) k_max/b_max plus the per-seed schedules so callers
    can confirm shot-invariance and inspect where the peaks occur.

    A single seed suffices when the schedule is shot-invariant (the fixed-Pauli-
    noise case); pass several to take a defensive worst case / audit invariance."""
    prog = _compile(circuit)
    runs = []
    for sd in seeds:
        k_sched, b_sched = _schedule_once(prog, sd)
        runs.append(dict(seed=sd, k_max=max(k_sched, default=0),
                         b_max=max(b_sched, default=0),
                         k_sched=k_sched, b_sched=b_sched))
    k_max = max(r["k_max"] for r in runs)
    b_max = max(r["b_max"] for r in runs)
    invariant = (len({r["k_max"] for r in runs}) == 1 and
                 len({r["b_max"] for r in runs}) == 1)
    return dict(k_max=k_max, b_max=b_max, invariant=invariant, runs=runs)


def select(circuit, margin=1, seeds=(42,)):
    """Return ('nc'|'clifft', info). 'nc' only when it strictly saves >= `margin`
    qubits of peak dimension; otherwise 'clifft' (NC adds overhead with no gain)."""
    info = analyze(circuit, seeds=seeds)
    info["margin"] = margin
    info["backend"] = "nc" if info["b_max"] <= info["k_max"] - margin else "clifft"
    info["peak_qubits"] = min(info["k_max"], info["b_max"]) if info["backend"] == "nc" \
        else info["k_max"]
    return info["backend"], info
