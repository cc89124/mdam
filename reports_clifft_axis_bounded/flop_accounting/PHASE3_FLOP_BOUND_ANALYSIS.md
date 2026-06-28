# Phase 3 — Can bounded FLOP be bounded by Clifft FLOP? (analysis only, no implementation)

Hypothesis under test: *if bounded's dense runtime events are a subset of Clifft's, and bounded's
active rank r ≤ Clifft's active rank k, then with identical kernels F_bnd ≤ F_clf.*

Verdict (jump to §6): **the strong unconditional bound is FALSE for the current representation.**
There is exactly ONE structural bounded-only dense event — the **localization Hadamard** that
re-diagonalizes a generator the *symbolic* Clifford frame rotated off-axis — which Clifft never
pays because it keeps its array in the *physical* basis. It is dominated by the rank gap when
bounded has headroom (r ≤ k−2) but becomes a genuine excess when bounded is all-magic at the cap
(r = k). PLUS one *implementation* gap (the off-diagonal butterfly, c=12) that is fixable and is
the actual cause of the cultivation/RY excess. A precise **conditional bound** holds (§6, Case 2).

All numbers are real event traces (scripts/phase3_event_trace.py, data/phase3_event_trace.txt;
clifft C++ CostMeter unfused, bounded budget.charge).

---

## §1 Common cost model

For both backends, a *dense event* `e` reads or writes the amplitude array. Define

    F = Σ_e c_e · 2^{r_e}

- `r_e` = active rank at event e (clifft: `active_k`; bounded: log2(phi.size)).
- `c_e` = FLOP per amplitude = (6·cmul + 2·rcmul + 2·cadd + 4·sqmag + 8·vdot) / 2^{r_e}.
- Both use the SAME convention (clifft cost_meter.h:94 == bounded BCOEF).

Five DISTINCT quantities — never conflated:
| quantity | meaning | clifft source | bounded source |
|---|---|---|---|
| resident-state size | max 2^{r} live at once | peak active_k | budget.peak_resident |
| **state-size sum** Σ2^{r_e} | Σ over events of 2^{r_e} | CostMeter sum_pow2k | Σ budget.charge resident |
| touched-word traffic | Σ words actually read/written | CostMeter processed | Σ resident touched |
| **FLOP** | Σ c_e·2^{r_e} | CostMeter cmul… | Σ BCOEF·resident |
| wall | seconds (cross-language, not in the bound) | — | — |

---

## §2 Complete enumeration of bounded dense events (every charge site)

| where (event) | c_e | kernel (engine.py / bounded.py) | touches dense state? |
|---|---:|---|---|
| rot:diaghalf | 3 | strided half-array Z-rot `v[:,1,:]*=ρ` | YES (½ array) |
| rot:diag | 6 | general multi-Z parity scale | YES |
| rot:diag0 | 6 | global scalar `phi*=ρ` | YES |
| rot:diag-scalar | 6 | slack-0 scalar Z-rot | YES |
| **rot:offdiag** | **12** | off-diagonal 2×2 butterfly `_pauli_lincomb_inplace` | YES (full, **bounded-only**) |
| **rot:offdiag-scalar** | **12** | slack-0 scalar butterfly | YES (**bounded-only**) |
| purge:h | 4 | `_h_axis` strided Hadamard butterfly | YES (full) |
| purge:s | 2 | `_s_axis` strided ±i on ½ | YES (½) |
| purge:cnot | 0 | `_cnot_axes` in-place sub-block swap | YES (permutation) |
| collapse:offdiag/diag | 12 / 6 | measured-axis rotation collapse | YES |
| sqnorm | 2 | `_branch_sqnorm` einsum ½-view | YES (read ½) |
| normalize | 2 | `phi/=‖·‖` | YES |
| meas / exp / reduce:verify | 10 | streamed ⟨P⟩ / Born | YES (read) |
| reduce:cnot | 0 | `_cnot_inplace` permutation | YES (permutation) |
| reduce:gf2scan | 0 | nonzero-support GF(2) scan | YES (read) |
| drop | 0 | `_drop_axis_inplace` compaction copy | YES (permutation/copy) |
| promote | 0 | resize / zero-fill tail | grows buffer |
| post-reduce / init | 0 | note only | no |

---

## §3 Event-by-event correspondence (REAL traces)

### ry_d3_r1 (k=16, all at/near cap; rank advantage only on sub-gate rotations)
| logical op | Clifft dense event (c, FLOP) | bounded dense event (c, FLOP) | class |
|---|---|---|---|
| Hadamard (localize) | array_h (4, 4.93M) | purge:h (4, 4.92M) | **A matched** |
| diagonal rotation | array_rot (3, 4.12M) | rot:diaghalf+diag (3/6, 3.20M) | **A**, bnd ≤ |
| S (localize) | array_s (3, 1.35M) | purge:s (2, 1.28M) | **A**, bnd ≤ |
| X-meas | meas_interfere (8, 1.11M) | sqnorm+normalize (2+2, 0.92M) | **A**, bnd ≤ |
| CZ | array_cz (0.5, 0.78M) | — (routed to frame) | **C removed** |
| CNOT/SWAP | array_cnot/swap (0) | purge:cnot (0) | A matched (0) |
| **off-diag rotation** | **— (none)** | **rot:offdiag(+scalar) (12, 2.53M)** | **B bounded-only** |
| TOTAL | **12.29M** | **12.85M (+0.56M)** | |

### cultivation_d5 (k=10, ALL-MAGIC at the cap — bounded has NO rank advantage)
| logical op | Clifft dense event (c, FLOP) | bounded dense event (c, FLOP) | class |
|---|---|---|---|
| **T / T† (91×)** | **array_t/t_dag (3, 0.17M, DIAGONAL)** | **rot:offdiag(+scalar) (12, 0.68M)** | **B/perturbed** |
| Pauli-string collapse | array_cnot (0, 277 calls, **free perm**) | — (no pre-collapse before butterfly) | **C removed by clifft** |
| X-meas | meas_interfere (8, 0.033M) | sqnorm+normalize (2+2, 0.029M) | A, bnd ≤ |
| H (localize) | — (0 array_h!) | purge:h (4, 0.012M) | **B bounded-only** |
| CZ/S | array_cz/s (0.5/3, 0.0055M) | — frame | C removed |
| TOTAL | **0.21M** | **0.73M (×3.42)** | |

### d5_r5 (k=24 vs r=13 — rank gap 2^11 dominates everything)
| logical op | Clifft (c, FLOP, Σ2^r) | bounded (c, FLOP, Σ2^r) | class |
|---|---|---|---|
| rotation | array_rot (3, **16.91G**, Σ2^r=5.64G @k≤24) | rot:offdiag (12, **20.74M**, Σ2^r=1.73M @r≤13) | A (bnd c worse) |
| Pauli collapse | array_cnot (0, Σ2^r=**12.01G**, free) | purge:cnot (0, Σ2^r=0.33M) | A (0) |
| meas | meas_diagonal (4, 0.54G) | sqnorm+normalize (2, 2.35M) | A, bnd ≤ |
| TOTAL | **17.99G** | **25.10M (÷716)** | |

**E_bnd ⊆ E_clf is FALSE.** `rot:offdiag` (c=12) and the localization `purge:h` (in cultivation,
where clifft does NO array_h) are bounded-only events with no injective Clifft counterpart at equal
coefficient. The hypothesis's premise does not hold as implemented.

---

## §4 Why clifft is cheap, and the offline-ability of each bounded-only event

**Clifft's mechanism (the key the traces reveal):** Clifft keeps the amplitude array in the
**physical basis** (it applies the Clifford frame TO the array). Therefore (a) a T/rotation is
**diagonal on its native axis** → `array_t`/`array_rot` at c=3, NO Hadamard; (b) a multi-qubit
Pauli string is collapsed to a single axis by **`array_cnot` PERMUTATIONS (c=0)** — cultivation:
277 free CNOTs, d5_r5: 779 — before the diagonal rotation. Clifft's rotation cost ≈ 3·2^k + free
permutations.

Bounded keeps the frame **symbolic** (tableau). A rotation's generator pulled back through the
symbolic frame becomes a general Pauli on the magic register; the current code applies it as the
**off-diagonal butterfly (c=12)** — it does NOT first collapse with free CNOT permutations.

Offline-ability of each bounded-only / extra event:
| extra event | classification | can be removed? |
|---|---|---|
| `rot:offdiag` c=12 → should be CNOT-collapse(0) + [H] + diagonal(3) | **(3) permutation + (4) fusion** | YES at runtime — the Phase-2B collapse-first localizer already does it; it is *gated off* (`_loc_min_size`) and bypassed at slack-0. This is the FIXABLE implementation gap. |
| localization `purge:h` (the H that re-diagonalizes an off-axis-rotated generator) | **(5) essential dense sweep** when r=k; equals clifft's `array_h` when clifft also needs it | NO in general — it is the dual of clifft's array-Clifford cost. Removable ONLY when the generator is pure-Z after collapse (then c=3=clifft) or when r<k (rank-discounted). |
| pullback recompute | **(1) offline / (2) frame-only** | ALREADY removed (incremental inverse-frame, Phase 2B). |
| structure search, dead-rotation drop | **(1) offline** | ALREADY removed (structure-once, drop_dead). |
| sqnorm / normalize / Born | **(5) essential dense**, but c=2 ≤ clifft c=8 | bnd already cheaper. |
| drop / promote / purge:cnot | **(3) permutation/index**, c=0 | free. |

**"Knowing the core offline" removes SEARCH and BOOKKEEPING (pullback, structure, dead drops) — it
does NOT remove the data-dependent dense sweeps** (rotation, Born, normalization, and the
localization H). Those touch live amplitudes and must run at runtime. Offline knowledge *can*
downgrade the off-diagonal butterfly (c=12) to collapse+diagonal (c≤7) because the collapsing
Cliffords are frame-determined, but it cannot erase the localization H when the symbolic frame has
genuinely rotated a generator off-axis.

---

## §5 Residual-0 attribution

**ry_d3_r1: gap +0.56M, attributed to 0.** Bounded EXTRA = rot:offdiag 2.53M. Bounded SAVED vs
clifft = array_cz 0.78M + cheaper diagonal 0.92M + cheaper meas 0.19M + cheaper S 0.07M = 1.96M.
Net = 2.53M − 1.96M = **+0.57M ≈ +0.56M measured (residual ≈ 0).** The entire 5% is the
un-localized off-diagonal rotations: 24 at sub-cap rank (~2^11.7, would localize cheaply) + 2 at
the cap rank 16 (would still cost ~7·2^16 with the H). Removing the gate → ry would drop below
clifft for the sub-cap ones.

**cultivation_d5: ×3.42, attributed to 0.** 91 T/T† done as butterfly (c=12) = 0.68M where clifft
does them diagonal (c=3) = 0.17M. The c=12→c=3 difference (0.51M) decomposes as: (a) FIXABLE
butterfly→collapse part (c=12→7, saves ~0.28M) + (b) STRUCTURAL H part (c=7→3, the 0.23M extra H)
that clifft never pays because its T's are natively pure-Z while bounded's frame (+ AG-measurement
updates) rotated them off-axis. r=k=10 → the H gets NO rank discount. This is the irreducible part.

**d5_r5: ÷716, attributed to rank.** Bounded's rot:offdiag has c=12 (4× clifft's c=3) BUT runs at
r=13 vs clifft's k=24. Per rotation: 12·2^13 = 98k vs clifft 3·2^24 = 50.3M → bounded event is
**512× cheaper despite the 4× coefficient penalty**: the rank gap 2^{k−r}=2^11 = 2048 dominates the
coefficient ratio 4. State-size sum Σ2^r: bnd 4.25M vs clf 18.12G = **4264× smaller** — this, not
the per-amplitude coefficient, is the whole story.

---

## §6 Mathematical verdict

Decompose F_bnd = F_matched + F_extra, where F_matched are events with an injective Clifft
counterpart π(e) and F_extra are bounded-only events (the off-diagonal-butterfly excess + the
localization H with no clifft H).

For matched events, c_e^{bnd} ≤ c_{π(e)}^{clf} (purge:h=array_h=4; diaghalf=array_rot=3; sqnorm
2≤meas 8; purge:s 2≤array_s 3; perms 0=0) AND r_e^{bnd} ≤ k. So **F_matched ≤ F_clf always.**

**This is CASE 2 (conditional), not Case 1.** The strong unconditional bound fails because F_extra
> 0. The bound F_bnd ≤ F_clf holds **iff**

    Σ_{e∈extra} c_e^{bnd} 2^{r_e}  ≤  Σ_{g∈saved} c_g^{clf} 2^{k_g}

i.e. bounded's extra sweeps (off-diagonal-butterfly surplus + localization-H) must be covered by
the Clifft array-Clifford work that bounded routed to the symbolic frame (array_cz/array_s/the H's
Clifft applies, all at rank k).

Equivalent **sufficient conditions** (any one suffices):
1. **Rank headroom:** every flushed rotation has r ≤ k − 2 ⟹ even the c=12 butterfly ≤ 3·2^k.
   (d5_r5: r=13 ≤ 22 ⟹ massively satisfied.)
2. **No off-axis frame rotation:** the pulled-back generator is pure-Z after free CNOT-collapse
   ⟹ no H, c=3 = clifft ⟹ F_bnd ≤ F_clf at any rank.
3. **Clifft pays enough array-Clifford arithmetic** (array_h/cz/s) that bounded routes to the
   frame, exceeding bounded's localization-H bill.

**When ALL three fail simultaneously — bounded all-magic at the cap (r=k, no headroom), the frame
off-diagonalizes the generators (X-character ⟹ H needed), and Clifft keeps its rotations diagonal
essentially for free (cheap/0-FLOP Cliffords) — then F_bnd > F_clf is unavoidable for that
representation.** cultivation_d5 is exactly this corner: r=k=10, off-axis T generators, clifft
keeps T diagonal with only free array_cnot permutations.

### Counterexample (Case-2 proof that no unconditional bound exists)
cultivation_d5: clifft applies T as diagonal array_t = 3·2^k with a FREE array_cnot permutation
collapse; bounded, at the same rank k (all-magic), must apply the off-axis-rotated generator with a
localization H = 4·2^k before the diagonal 3·2^k. Even with a *perfect* collapse+localize kernel
(no butterfly), bounded = 7·2^k > clifft = 3·2^k. No kernel choice fixes it because the H is forced
by the symbolic frame and there is no rank to discount it. ∎

---

## §7 What this means for an achievable bound (no implementation here)

- **Achievable conditional bound:** make bounded **always collapse+localize** (CNOT-permutation
  collapse → at most one H → diagonal), eliminating `rot:offdiag` (c=12). Then every bounded event
  matches a Clifft event at c_bnd ≤ c_clf, and **F_bnd ≤ F_clf whenever any of the three §6
  sufficient conditions holds** — in particular on every circuit with rank headroom (rx, d5_r5,
  coherent_d3_r3, distillation) and on the sub-cap rotations of RY. This is NOT a new kernel and
  NOT a T-specific path — it is the existing Phase-2B localizer applied universally (drop the size
  gate + give the slack-0 path a collapse route). [Design only — not implemented per instruction.]

- **Irreducible residual:** the all-magic-at-cap corner (cultivation) keeps F_bnd ≈ (7/3)·F_clf
  because the localization H has no rank to hide behind. To remove it one would have to abandon the
  symbolic frame for those qubits (apply the frame to the array = become Clifft, losing the rank
  win) OR choose the magic-register basis offline so the T generators stay pure-Z (possible only if
  the frame's H/measurement content on the magic support is trivial — circuit-dependent, not
  general). So **no representation-preserving change makes bounded ≤ Clifft on cultivation.**

---

## §8/§9 Summary

- **§8 (if possible):** the *conditional* bound is reachable by universal collapse+localize (no new
  T kernel). Sufficient condition: §6 inequality / rank headroom / pure-Z-after-collapse.
- **§9 (the counterexample):** cultivation_d5 is a hard counterexample to any UNCONDITIONAL bound —
  r=k, off-axis generators, clifft-diagonal-for-free ⟹ F_bnd = (7/3)F_clf irreducibly.

**Bottom line for the original question** ("bounded uses ≤ state, why can FLOP be larger?"): because
the *only* thing guaranteed ≤ is the **peak rank** (a snapshot); FLOP integrates c_e·2^{r_e} over
the run. Bounded trades Clifft's eager array-Clifford sweeps for a symbolic frame, which is a win
(fewer/cheaper events) *except* it makes rotation generators off-axis, and re-diagonalizing them
costs a localization H that only the **rank gap** can pay for. With a rank gap (the regime bounded
is built for) F_bnd ≪ F_clf; with no rank gap (all-magic at the cap) the H surfaces and F_bnd > F_clf.
