# MDAM current measurement-core & pending-ledger semantics (analysis only — no code change)

목적: 현재 MDAM이 measurement 시점에 pending rotation을 **어떻게 선택·실행·제거·유지**하는지 규명하여,
향후 "마지막 branch-mixing(X/Y-on-measured-axis) rotation 뒤의 measurement-diagonal(I/Z) suffix를
measurement 후로 deferral"하는 아이디어의 패치 위치·exactness 조건을 결정할 근거를 만든다. 이번 단계는
**어떤 코드도 수정하지 않았고**, 모든 결론은 코드 라인 인용(CODE-PROVEN) 또는 read-only runtime
trace(TRACE-PROVEN)로 표시한다. trace는 monkeypatch가 ORIGINAL을 그대로 호출하고 RNG를 소비하지 않아
**record bit-identical(traced≡untraced 검증 완료)** 이다.

산출물: `artifacts/mdam_current_measurement_core_semantics/` (measurement_trace.jsonl,
core_tail_summary.csv, synthetic_trace.json, pending_entry_schema.json).

---

## 1. Executive summary (먼저 한 줄씩)

- **I/Z suffix가 실제 executor에서 실행되는가?** **그렇다 — 단, 회로에 따라 다르다.**
  `coherent_d5_r5`·`coherent_d3_r3`는 **모든** nonempty magic core가 정확히 **1개의 trailing
  diagonal(Z-on-measured-axis) rotation**을 포함하고, 그것이 2^{r_out+1} joint에서 실제 실행된다
  → **Case C**. `cultivation_d5`·`coherent_ry_d3_r3`·`distillation`은 trailing suffix가 **없다**
  (diagonal rotation이 있어도 core의 마지막은 branch-mixing) → **Case A/B**. (TRACE-PROVEN)
- **몇 개 발견됐나?** d5_r5: 144/144 core, tail 144개(max 1); d3_r3: 24/24 core, tail 24개(max 1);
  cult_d5/ry/distill: 0. (TRACE-PROVEN, core_tail_summary.csv)
- **Tail 판정에 쓸 정확한 representation은?** core entry를 `_pullback`한 P' = (xp,zp)를, `_pullback`한
  measured Pauli P_meas = pullback(Z_q) = (xpq,zpq)와의 **symplectic 반교환**으로 분류한다.
  이는 physical `(x>>q)&1`(X-support on q) 분류와 **모든 core rotation에서 일치**(basis_agree=True,
  TRACE-PROVEN — Clifford pullback이 반교환을 보존). 즉 branch-mixing ⟺ "P'가 measured axis에 X/Y" ⟺
  "physical X on q" ⟺ "anti-commute with Z_q". (CODE-PROVEN + TRACE-PROVEN)
- **Measurement internal outcome 변수는?** 내부 Born branch bit = `outcome`(compiled:
  `mdm_execute_core` 반환 / fused·oracle: `out`); survivor가 유지하는 measured-axis eigenvalue =
  `keepbit = plus_bit if outcome==0 else 1-plus_bit`, `plus_bit = 0 if sign>0 else 1`
  (compiled_core.py:164-165). tail-deferral의 (-1)^b의 **b = keepbit**. (CODE-PROVEN)
- **Measured axis drop 시 pending mask는 어떻게 변하나?** core entry는 측정 전에 pending에서 **삭제**
  (`del pending[uid]`)되어 dense state로 들어가므로, drop되는 measured/pivot axis를 **포함한 채 남는
  pending entry는 없다**. drop은 survivor(2^{r_out}) 재작성 + frame fold이고 **남은 pending의 mask를
  reindex하지 않는다**(pending은 physical-qubit basis라 active-axis drop과 무관). (CODE-PROVEN)
- **State sharing 분리가 존재하나?** **부분적.** dense state(`phi`/`_storage`)와 Clifford
  tableau(`Xc/Zc`,`Ax/Az`)와 dormant Pauli frame(`backend.frame`)과 pending ledger(`pending`)는
  서로 다른 필드지만 **모두 한 `nc` 객체 안에 있고 `measure_z`가 in-place로 변형**한다. shot마다 `nc`가
  새로 생성된다(`_reset`). 같은 dense state + 다른 frame/ledger를 표현하는 구조는 **현재 없음**. (CODE-PROVEN)

---

## 2. Actual call graph — 하나의 active Z measurement

opcode → record 까지. 파일:라인은 실제 코드 기준.

```
OP_MEAS_ACTIVE_DIAGONAL (backend.py run_shot:611-622 / _run_shot_compiled H_MEAS_DIAG)
  q = self.slot2id.get(a1)                         # active slot -> nc qubit index
  b = self.nc.measure_z(q)            ───────────────┐  internal Born outcome (0/1)
  del self.slot2id[a1]; self._reduce_dead()          │
  m_abs = b ^ self.frame.xb(a1)                       │  dormant Pauli-frame X-correction
  self.record[cidx] = m_abs ^ sign                    │  classical record bit (+ static sign)
  self.frame.set_xz(a1, m_abs, 0)                     │
                                                      │
  nc.measure_z (bounded.py:874)  ─────────────────────┘
    [S2] if _compiled_frame: _flush_tableau()           # flush deferred tableau gates (no-op default)
    [compiled] if _compiled_core: try_compiled_measure(self,q)  (compiled_core.py:106)
    [fused]    elif _fused_measure: info=_fused_setup(q); _fused_commit(q,info)  (bounded.py)
    [oracle]   else: _flush_core(0,1<<q); <stabilizer|magic localize/Born/drop>
```

세 executor 경로(compiled/fused/oracle)의 **공통 전단**:

```
core = _fused_core_entries(q)            (bounded.py:503)  # READ-ONLY: _fast_cores[_meas_ctr] 또는 _dynamic_core
P_meas = _pullback(0, 1<<q)              (simulator.py:276) # measured Pauli, executor basis
for each core entry (x,z,p,theta,uid):
    (xp,zp,pp0) = _pullback(x,z)         # rotation generator, executor basis
    pp = (pp0 + p) & 3
M_mat = M_in + {qubits with X-support in any xp}    # promote work axes
build per-rotation mask (xb,zb) over M_mat layout; build L-M localizer plan W; pivot r, m=M_mat.index(r)
```

표 A(핵심 함수)는 §15에.

### 2.1 실제-코드 pseudocode (compiled path; fused/oracle 동치)

```python
# compiled_core.try_compiled_measure(eng, q)   (compiled_core.py:106-189)
core   = eng._fused_core_entries(q)                 # ordered core (increasing-uid)  [READ-ONLY]
M_mat  = list(eng.M); pulled = []
for (x,z,p,theta,uid) in core:
    xp,zp,pp0 = eng._pullback(x,z); pp=(pp0+p)&3
    pulled.append((xp,zp,pp,theta))
    promote qubits with X-support(xp) into M_mat
if any non-magic stabilizer anticommutes with Z_q:  return None     # -> oracle (stabilizer)
xpq,zpq,ppq = eng._pullback(0,1<<q)                                 # P_meas
if X on non-magic / deterministic:                  return None     # -> oracle
build localizer W (H/S then CNOTs -> ±Z_r), m = M_mat.index(r)      # EXACTLY _localize_to_Z
rots = [(tb(xp,zp), pp, theta) for ...]            # masks over M_mat layout (executor input)
rand = eng.rng.random()                            # ONE rng draw (after all None-guards)
outcome,p0,surv = executor.execute(eng.phi, r_in, r_mat, rots, lm, m, sign, rand)  # C++ ONE call
eng._meas_ctr += 1
for ce in core: del eng.pending[ce[4]]             # CONSUME core uids
keepbit = (0 if sign>0 else 1) if outcome==0 else (1 if sign>0 else 0)
eng.phi = surv (2^{r_out}); eng.M = M_mat without m
fold localizer W into frame (right_h/s/cx); if keepbit==1: fold X_r into frame
_drop_residual_products()                          # drop any 2nd product axis
return outcome
```

executor 내부(`mdm_execute_core`, mdm_core_executor.cpp:160-222, CODE-PROVEN):
```
joint = phi_in (low) ⊗ |0>_new (high, zero)          # 2^{r_mat} = 2^{r_out+1}
for i in 0..nrot-1: direct_rot(joint, rot[i])        # ORDERED, ONE dense apply per uid, ON THE FULL JOINT
apply L-M localizer ops to joint                     # localize P_meas -> sign*Z_{m_bit}
Born from the two m_bit branches; outcome = (rand < p0)?0:1
survivor = keepbit m-block (m dropped), normalized    # 2^{r_out}
```
**핵심: 모든 rotation(diagonal 포함)이 measured axis를 포함한 2^{r_out+1} joint에 적용된다.** 따라서
trailing diagonal rotation은 현재 2^{r_out+1}에서 실행된다.

---

## 3. State representation (measurement 직전)

모든 필드는 단일 near-Clifford 객체 `nc`(= `CliftAxisBoundedNearClifford`)에 있다. shot마다 `_reset`이
새 `nc`를 만든다(backend.py:212).

### Dense state — CODE-PROVEN
- 필드: `self.phi` = `self._storage[:self._sz]` (capacity buffer의 contiguous prefix view; bounded.py:221-271).
- shape: 1-D complex128, 길이 2^{|M|}.
- axis 의미: bit j ↔ `self.M[j]` (M[0]=LSB; simulator.py 주석 + engine.py kernels). `M` = ordered magic
  qubit 리스트.
- resident vs work: resident = `phi`(2^{r_in}); branch-pair **work** state = executor의 `joint`
  (2^{r_mat}=2^{r_out+1}), measurement 시점에만 C++ caller buffer로 생성(mdm_core_executor.cpp:167-171).
- measured axis가 dense에 포함되는 시점: oracle은 `_flush_core`가 measured qubit을 promote한 뒤 dense에
  포함; compiled/fused는 measured axis가 `joint`에만 등장하고 survivor에서 drop(2^{r_out}).
- rank 결정: |M| (promote/drop). work rank = r_mat = |M_mat| (promote된 work axes 포함).

### Clifford/tableau state — CODE-PROVEN
- `self.Xc[i]`,`self.Zc[i]` = (x,z,p) = **forward** images U_C X_i U_C^† / U_C Z_i U_C^† (simulator.py:84-85).
- `self._inv_ax[i]`,`self._inv_az[i]` = **inverse** images U_C^† X_i U_C / U_C^† Z_i U_C (simulator.py:102-103);
  `_inv_enabled=True`(engine.py:71)라 hot `_pullback`은 inverse-frame lookup(simulator.py:280-291).
- pending Pauli는 **physical** basis(아래 §4). `_pullback`이 physical→executor(pre-frame) basis로 매핑.
- measurement observable pullback: `_pullback(0,1<<q)` = (xpq,zpq,ppq) (executor/active basis).

### Pauli frame (dormant) — CODE-PROVEN
- `backend.frame = ds_mod.PauliFrame()` (backend.py:183) — **nc와 별개 객체**. dormant/모든 slot의
  X/Z bit, readout, dormant 측정 담당.
- shot별 Pauli noise는 `frame`에 기록(`_apply_noise_site(.., frame, rng, ..)`, backend.py:486).
- non-Clifford rotation의 angle/sign 반영: `_rot`에서 `theta = -angle if frame.xb(slot) else angle`
  (backend.py:345) — frame X-bit가 rotation 각도 부호를 뒤집음.
- measurement bit 관계: internal `b` → `m_abs = b ^ frame.xb(a1)` → `record = m_abs ^ sign`
  (backend.py:618-619). readout noise는 record를 별도로 flip(backend.py:502-503).

---

## 4. Pending ledger schema & ordering — CODE-PROVEN

entry 1개 = `list [x, z, phase, theta, uid]` (lazy.py apply_rotation:142). 상세는
`artifacts/.../pending_entry_schema.json`.

| 필드 | idx | 타입 | 의미 | 변경 함수 |
|---|---|---|---|---|
| x | 0 | int mask | physical Pauli X-support | apply_rotation 설정; `_conj_*`/S2 in-place |
| z | 1 | int mask | physical Pauli Z-support | 동상 |
| phase | 2 | int mod4 | 누적 i^phase(Y=i^1 XZ 등) | apply_rotation=0; `_conj_*`; `_flush_one`에 forward |
| theta | 3 | float | 회전각 | 불변 |
| uid | 4 | int | 단조 생성 id | 불변(안정 identity) |

10개 질문 답:
1. **container**: `dict {uid: list}`, insertion-ordered (lazy.py:74). `list(pending.values())`는 항상
   증가-uid 순서(lazy.py:70-73). **CODE-PROVEN**
2. **실행 순서**: container(=uid) 순서. `_dynamic_core`/`_do_flush`가 그 순서 보존(lazy.py:188,241).
   별도 dependency-topo-sort 없음. **CODE-PROVEN**
3. **Clifford 만날 때 즉시 conjugate?**: 예 — lazy.h/s/cx가 매 gate마다 pending 전체를 `_conj_*`로
   재작성(lazy.py:111-134). **CODE-PROVEN**
4. **S2 compiled frame의 conjugation 시점**: 동일하게 매 gate(즉시), 단 dict-재생성 대신 **in-place
   변형**(bounded.py h/s/cx override). tableau만 deferral; pending은 deferral 아님. **CODE-PROVEN**
5. **측정 후 non-selected pending**: 그대로 dict에 남음(객체 동일, mask 변형 없음). **CODE-PROVEN**
6. **selected core uid 제거 시점**: `_do_flush` 진입 직후 `del pending[uid]`(lazy.py:225-226);
   compiled은 `for ce in core: del eng.pending[ce[4]]`(compiled_core.py:163). **CODE-PROVEN**
7. **uid 단조 증가?**: 예(lazy.py:138-139). **CODE-PROVEN**
8. **동일 uid가 다른 representation으로 이동?**: 아니오. uid는 안정; conjugation은 같은 entry의 x/z/phase만
   변형. **CODE-PROVEN**
9. **measured axis 제거 시 residual pending mask reindex?**: 아니오. pending은 **physical-qubit** mask라
   active-axis(M) drop과 독립. drop은 `M`/`phi`/frame만 변형. **CODE-PROVEN**
10. **적용 여부 추적 별도 상태?**: 없음. pending에 있으면 미적용, `del`되면 consumed. **CODE-PROVEN**

---

## 5. Core-selection 알고리즘 — 수학 + 코드 (CODE-PROVEN)

측정 physical Pauli = Z_q = (qx,qz)=(0,1<<q). pending = R_1..R_m (uid 순서).

`_core_indices(qx,qz)` (lazy.py:162-182):
- **seed**: `not _commute_xz(qx,qz, x_i,z_i)` 인 모든 i. `_commute_xz(ax,az,bx,bz) =
  ((popc(ax&bz)+popc(az&bx))&1)==0` (lazy.py:63-64). 여기서 anti-commute(Z_q, R_i) ⟺
  `popc((1<<q)&x_i)` 홀수 ⟺ **R_i의 physical x가 q에 X-support**.
- **closure**: stack에서 꺼낸 core member R_j와 anti-commute하는 모든 비-core R_k 추가(transitive).
  즉 **반교환 connected component(transitive closure)**.
- **ordering**: `list(pending.values())` 순서 = 증가-uid. executor에 그 순서로 전달.

수학적 의미: core = {Z_q를 포함한 반교환 그래프에서 Z_q와 연결된 rotation}. **seed = measured axis에
X/Y인 rotation(branch-mixing)**; **I/Z-on-q(diagonal) rotation은 seed에 없고, 다른 축에서 seed member와
반교환할 때만 closure로 편입**된다. → diagonal rotation 중 **uncoupled한 것은 자동으로 제외(pending에
남음)** 되고, **coupled trailing diagonal만 실행**된다(이것이 Case C의 tail).

§5 예제(요청) 결과(synthetic_trace.json, TRACE-PROVEN):
```
R1=X_q(seed), R2=Z_q(diagonal,R1과 반교환→closure), measure Z_q
  → core = [R1, R2],  seq = X I,  trailing diagonal = [R2], tail_len=1
R1=X_q, R2=Z_q, R3=Z_q  → core=[R1,R2,R3], seq=X I I, tail_len=2 (구조적으로 가능)
```
(주: 예제의 "I/Z on measured axis" rotation이 **uncoupled** 라면 core에 안 들어가 tail=0.)

### offline `_fast_cores` 와의 관계
`_structure_for(prog)`(backend.py:253-301)가 3개 seed로 discovery shot을 돌려 per-measurement core uid
집합(`_record_cores`)이 seed-invariant인지 확인하고, 같으면 `_fast_cores = {meas_idx: [uid,...]}`로 캐시.
runtime은 `_flush_core`/`_fused_core_entries`가 `_fast_cores[_meas_ctr]`를 uid 순서로 gather
(lazy.py:200-217, bounded.py:508-515). **TRACE-PROVEN: 5개 벤치 모두 seed_invariant_cores=True** →
core membership은 shot-static. **단 `_fast_cores`는 core uid 집합만 static이고, Pauli mask/localizer/
sign은 runtime에 frame으로부터 계산**(§6).

---

## 6. Offline plan / `_fast_cores` (CODE-PROVEN + TRACE-PROVEN)

1. `_structure_for`가 계산: dead uids(never-flushed) + `_fast_cores`{meas_idx→core uids} + (block일 때)
   peels. (backend.py:253-301)
2. key=meas_idx(int), value=core uid 리스트(증가-uid). (lazy.py:197)
3. core uid 집합은 physical circuit(active-gate stream)만으로 결정 — record/outcome 독립(backend.py
   주석 81-92: feedback은 전부 Pauli-frame으로만 라우팅). **seed-invariant 확인됨**.
4. **Pauli frame sign은 plan에 없음** — runtime에 `_pullback`로 계산. **static = core uid 집합 + 실행
   순서; dynamic = pulled-back mask, localizer, sign, outcome.**
5. core **support**(어떤 uid)는 static; **actual pulled-back mask**는 frame 의존(dynamic). 단 frame
   support(x,z bits)는 shot-static, frame phase/sign만 dynamic([[mdam-fullshot-cpp-runtime-phase1]] 참조).
6. localizer는 plan에 **미포함** — `try_compiled_measure`/`_localize_to_Z`가 runtime에 재구성.
7. `_fast_cores` 없을 때 `_dynamic_core` 결과와 동일(설계상; `_debug_compare`가 cross-check, lazy.py:204-208).
8. `_fast_cores`는 **minimal**(반교환 closure의 정확한 집합) — conservative prefix 아님. (lazy.py:162-182)
9. dead uid = 모든 seed에서 한 번도 flush되지 않은 uid(backend.py:283).
10. **core plan은 I/Z suffix를 포함할 수 있다** — closure가 coupled diagonal을 넣으므로(§5,§11). TRACE로 확인.

---

## 7. Effective executor representation & measured-axis index (CODE-PROVEN)

표 C — 시점별 mask basis와 measured-axis index:

| 시점 | mask representation | measured-axis index | 변환 함수 |
|---|---|---|---|
| 1 source physical gate | physical (x,z) over qubits | 물리 qubit q | clifft compile → apply_rotation |
| 2 pending ledger | physical (x,z,phase) | 물리 qubit q | apply_rotation(lazy.py:142) |
| 3 frame pullback | pre-frame/active (xp,zp,pp) | 물리 qubit q (pullback은 qubit index 보존) | `_pullback`(simulator.py:276) |
| 4 active-slot basis | = 3 (M에 한정) | q∈M | `_masks`(engine.py:105) |
| 5 work-axis basis (M_mat) | bit (xb,zb) over M_mat layout | bit `m`=M_mat.index(r), r=pivot | `tb`(compiled_core.py:126) |
| 6 localizer 적용 전 | (xb,zb) over M_mat | pivot bit m, P_meas not yet Z | — |
| 7 localizer 적용 후 | W·P·W^† | bit m (pure Z_m) | L-M ops(mdm_core_executor.cpp:179-186) |
| 8 compiled executor 입력 | `rot_x/rot_z` = (xb,zb), `m_bit` | `m_bit` | executor.execute(compiled_core.py:160) |
| 9 branch-pair measured bit | joint bit `m_bit` | `m_bit` | mdm_execute_core |

measured axis q의 index: 시점 1–4는 **물리 qubit q**(pullback은 qubit index 보존), 시점 5–9는 **M_mat
layout의 bit m = M_mat.index(pivot r)**, r = (q∈support면 q, 아니면 support[0]) (compiled_core.py:137-138).

> **Tail eligibility 검사 representation (단일 답):** core entry의 **pulled-back P'=(xp,zp)**
> (`_pullback(x,z)`)를 measured Pauli **P_meas=(xpq,zpq)**(`_pullback(0,1<<q)`)와 **symplectic
> 반교환**으로 분류한다. 이는 executor가 받는 `rots[i]`(M_mat basis)에서 `m_bit` 기준 X/Y와 동치이며
> (localizer는 반교환을 보존), physical `(x>>q)&1`와도 **전 case 일치(basis_agree=True, TRACE-PROVEN)**.

---

## 8. Numerical executor paths (CODE-PROVEN)

| 경로 | eligibility | 입력 rotation list | mask basis | measured axis | probability | survivor | pending 소비 |
|---|---|---|---|---|---|---|---|
| compiled C++ | `_compiled_core` & magic & diagonal-localizable | `rots` (core 순서) | M_mat bit | m_bit | Born s0/s1 on joint | gather keepbit block | `del pending[uid]` after execute |
| fused symbolic | `_fused_measure` & diagonal-magic | `pulled`(core 순서)→symbolic U | M_mat bit | pivot m | `_fused_born` ⟨M'(t)⟩ | `_fused_survivor` (Kb on r_out) | `del` in `_fused_commit` |
| oracle dense | else (or fused/compiled None) | core entries via `_flush_core` | M(magic) bit | localized Z_r | branch sqnorm | strided-zero project + drop | `del` in `_do_flush` |
| stabilizer/GK | non-magic stabilizer anticommutes Z_q | — (core 무관) | — | — | uniform `rng.integers` | frame update | (core still flushed first) |

답:
1. core rotation list 최종 확정: `_fused_core_entries`(bounded.py:503) — 세 경로 공통(같은 uid 집합/순서).
2. executor별 재확장/변환: compiled은 `rots`(M_mat mask)로 변환; fused는 symbolic U로 곱; oracle은
   `_flush_one`로 dense 적용. **rotation 집합/순서는 동일, representation만 다름.**
3. compiled vs Python UID sequence: **동일**(둘 다 `_fused_core_entries` 순서). TRACE 확인.
4. localizer가 I/Z/X/Y 분류 바꾸나?: 아니오 — localizer(Clifford)는 measured Pauli와의 반교환을 보존하므로
   branch-mixing/diagonal 분류 불변(§7, basis_agree=True).
5. executor가 0/1 양쪽 branch 모두 만드나?: joint에 양쪽 다 존재(Born s0,s1 모두 계산,
   mdm_core_executor.cpp:190-200) → **p0,p1 모두 산출**, 그러나 survivor는 **sampled keepbit 1개만**
   gather(line 212-218). **TRACE/CODE-PROVEN: 양쪽 norm은 얻지만 survivor는 단일 branch.**
6. tail을 executor에 전달하지 않으려는 공통 split 지점: **`rots`/`pulled`/core-entry 리스트가 executor에
   넘어가기 직전, 마지막 branch-mixing 이후의 trailing diagonal entry**. 세 경로 공통의 source는
   `_fused_core_entries` 반환 리스트(또는 그 뒤 `pulled`/`rots` 구성 루프). (구조적 위치만 — 패치는 다음 단계)

---

## 9. Outcome-bit semantics (CODE-PROVEN)

표 D:

| 변수 | 의미 | 생성 시점 | frame/readout 관계 |
|---|---|---|---|
| `outcome`/`out`/`b` | 내부 Born branch bit | executor 반환(compiled mdm_execute_core:203-204 / fused `_fused_commit` / oracle measure_z) | frame/readout 이전 |
| `keepbit` | 살아남는 m-eigenvalue bit | compiled_core.py:165 / mdm_core_executor.cpp:205 | `plus_bit=0 if sign>0 else 1`; `keepbit = plus_bit if outcome==0 else 1-plus_bit` |
| `sign` | M' → sign·Z_r 부호 | localizer 결과 P[2] (compiled_core.py:154) | static-ish (frame phase 의존) |
| `m_abs` | frame-corrected 측정값 | backend.py:618 (`b ^ frame.xb(a1)`) | dormant frame X-bit 반영 |
| `record[cidx]` | classical record bit | backend.py:619 (`m_abs ^ sign`) | + static op-sign; readout noise가 backend.py:502-503에서 별도 flip |

> **`Z_measured ⊗ Q` rotation을 survivor rotation으로 바꿀 때 (-1)^b 의 b** = **`keepbit`**
> (compiled_core.py:165 / mdm_core_executor.cpp:205). measured axis가 keepbit으로 collapse되면 Z_m →
> (-1)^keepbit 이므로 R = exp(-i θ Z_m Q /2) → exp(-i θ (-1)^keepbit Q /2) (CODE-PROVEN: 변수 존재).
> 단 deferral 시 angle 부호·frame fold와의 상호작용 exactness는 **다음 설계 단계에서 검증**(현재 미구현).

---

## 10. Axis drop & pending transformation (CODE-PROVEN)

측정 후 measured/pivot axis 제거:
1. collapse 직후 shape: compiled은 survivor 2^{r_out}(mdm_core_executor.cpp:212-218); oracle은 strided
   project 후 `_drop_localized_core`(bounded.py:345).
2. removal helper: compiled은 executor 내부 gather(m 제외); oracle/fused는 `_drop_localized_core` /
   `_swap_axes`+size 축소(bounded.py:345-368, 288-315).
3. dense reindex: in-place 또는 survivor 재작성.
4. slot↔axis: `eng.M = [M_mat[i] for i≠m]`(compiled_core.py:166).
5. internal tableau: localizer W를 `right_h/s/cx`로 frame fold(compiled_core.py:174-177); keepbit==1이면
   `Zc[r]` 부호 flip + `_inv_fold_x`(compiled_core.py:178-180).
6. dormant frame: backend.py:620 `frame.set_xz`.
7. **pending mask reindex: 없음**(pending은 physical basis; active-axis drop과 독립). **CODE-PROVEN**
8. localizer/frame fold-back: `right_*`(compiled_core.py:174-177).
9. resident rank: r_in → r_out(=r_mat-1) → `_drop_residual_products`로 더 줄 수 있음(bounded.py:393).
10. work state 해제: executor의 `joint`는 caller buffer(재사용); 별도 free 없음.

질문 답:
- **measured axis 포함한 채 남는 pending entry?**: **없음** — core(측정에 쓰이는 rotation)는 측정 전
  pending에서 삭제되고, 남는 pending은 measured axis와 무관한(반교환 안 하는) physical rotation뿐.
  **CODE-PROVEN**
- 따라서 "outcome에 따라 줄어드는 pending"은 현재 없음.
- **현재 `Z_q→±I` / branch-conditioned Pauli reduction helper 존재?**: **부분.** keepbit==1일 때
  `Zc[r]` 부호 flip(X_q frame fold, compiled_core.py:178-180)이 outcome-conditioned Pauli 변형. 그러나
  **pending rotation을 outcome으로 변환해 다시 ledger로 넘기는 helper는 없음**(=tail-deferral 미존재).
  **CODE-PROVEN**
- dormant demotion vs active measurement demotion: 다른 경로. active = `measure_z`+`_drop_localized`;
  dormant = `backend.frame` bit. 같은 helper 아님.
- pending reindex가 uid/order 보존?: pending은 reindex되지 않음 → uid/order 자명 보존.

향후 재사용 가능 helper 후보(이름만): `_drop_localized_core`(measured axis drop), `right_h/s/cx`(frame
fold), `_inv_fold_x`(outcome X-fold), `apply_rotation`(ledger 재삽입). **수정안은 작성 안 함.**

---

## 11. Runtime tail trace (TRACE-PROVEN)

설정: compiled_core=True, fused OFF; warmed(structure cached); 5 benches × seeds {7,42,123}, 1 shot.
분류는 **executor basis 반교환**(physical과 전 case 일치). measure_z wrapper는 read-only,
record bit-identical 검증 완료.

표 E:

| benchmark | magic cores | nonempty core | cores with tail | exec rots | tail rots | max tail | mean tail/hit | basis_agree | seed_inv | **Case** |
|---|--:|--:|--:|--:|--:|--:|--:|:--:|:--:|---|
| cultivation_d5 | 45 | 42 | 0 | 273 | 0 | 0 | 0.00 | ✓ | ✓ | A/B |
| coherent_ry_d3_r3 | 99 | 51 | 0 | 459 | 0 | 0 | 0.00 | ✓ | ✓ | A/B |
| **coherent_d5_r5** | 180 | 144 | **144** | 1209 | **144** | **1** | 1.00 | ✓ | ✓ | **C** |
| **coherent_d3_r3** | 36 | 24 | **24** | 195 | **24** | **1** | 1.00 | ✓ | ✓ | **C** |
| distillation | 15 | 6 | 0 | 30 | 0 | 0 | 0.00 | ✓ | ✓ | A/B |

해석:
- **Case C (d5_r5, d3_r3)**: nonempty magic core **전부**가 **마지막에 정확히 1개의 trailing diagonal**
  rotation을 가짐. 그 trailing rotation은 **전부 Z-on-measured-axis(x=0,z=1 on q)** (48/48, 8/8;
  synthetic_tail part a). 예: d5_r5 meas12 `seq=XXXXXXXI`(uid 162가 tail). 이 rotation은 measured
  qubit에 마지막으로 적용된 coherent Z-rotation으로, Z_q와 commute하지만 X/Y seed와 반교환해 closure로
  편입 → **현재 2^{r_out+1} joint에서 실행됨**.
- **Case A/B (cult_d5, ry, distillation)**: trailing diagonal suffix **없음**. diagonal rotation이 core
  안에 **interspersed**로 존재(예 ry meas1 `seq=IXXIII…X`)하지만 **마지막 rotation은 branch-mixing** →
  단순 suffix-deferral 대상 0. (closure가 이미 uncoupled diagonal은 제외; coupled diagonal은 중간에
  있어 trailing 아님.)

> 즉 현재 구현은 **uncoupled diagonal은 이미 제외(Case B 성격, pending에 잔류)**, **coupled diagonal은
> 실행**한다. tail-deferral 아이디어의 실효 대상 = **coupled trailing diagonal(Case C)** 뿐이며, 현재
> d5_r5/d3_r3에서 **측정당 1개**(2^{r_out+1}에서 실행 중)이다.

---

## 12. Synthetic trace (TRACE-PROVEN)

stim 회로 합성은 clifft compile 매핑이 불투명하여, **engine-level**로 pending을 직접 구성(identity frame
→ physical==pulled-back)하고 `_fused_core_entries`+`_pullback`로 executor가 받을 core를 read-only로 관찰.
(synthetic_trace.json)

| 패턴 | core seq | trailing diagonal uids | tail_len |
|---|---|---|---|
| X_q, Z_q, MZ | X I | [1] | 1 |
| X_q, Z_q, Z_q, MZ | X I I | [1,2] | **2** |
| X_q, (Z_q·Z₁), (X₁), MZ | X I I | [1,2] | 2 |
| Y_q, Z_q, MZ | X(=Y) I | [1] | 1 |

결론: trailing diagonal suffix는 **구조적으로 길이 ≥1 가능**하며 합성 시 2도 관찰. Z-on-q와 (coupled)
I-on-q 모두 suffix에 등장 가능. 벤치마크에서 max=1인 것은 회로 구조(측정 직전 단일 Z-rotation) 때문이지
구현 제약이 아님.

---

## 13. State-sharing readiness (CODE-PROVEN / INFERRED)

1. shot마다 dense state 독립 생성?: 예 — `_reset`이 `nc = sim_cls(n)`(backend.py:212), shot별 새 객체.
   **CODE-PROVEN**
2. immutable/shared reference 구조?: 현재 `phi`/`_storage`는 in-place 변형 전제(capacity buffer 재사용,
   bounded.py:213-271). shared 구조 아님. **INFERRED(장애요소)**
3. `measure_z` in-place?: 예 — phi project/normalize/drop 모두 in-place(bounded.py:846-856 등). **CODE-PROVEN**
4. 0/1 survivor 양쪽 계산?: **아니오** — joint에서 p0,p1 norm은 둘 다 얻지만 survivor는 sampled
   keepbit 1개만(mdm_core_executor.cpp:205-218). 양쪽 survivor를 만들려면 추가 gather 필요. **CODE-PROVEN**
5. p0,p1 모두 얻나?: 예 — `*p0_out`,`*p1_out`(mdm_core_executor.cpp:220). **CODE-PROVEN**
6. frame과 dense 분리?: 필드는 분리(`Xc/Zc`,`_inv_*` vs `phi`) but 같은 `nc` 객체. **CODE-PROVEN**
7. pending이 state 내부?: `nc.pending`은 `nc` 멤버(별도 객체 아님). **CODE-PROVEN**
8. 같은 dense + 다른 frame/ledger?: **현재 불가** — frame/pending/phi가 한 객체에 결합, 측정이 셋을
   동시에 in-place 변형. **INFERRED(가장 큰 장애)**
9. core sign signature helper?: localizer가 sign 계산(compiled_core.py:154); 전용 "sign signature"
   helper는 없음. **CODE-PROVEN(부재)**
10. cohort/group 확장의 최대 장애: (a) survivor가 단일 branch만(양쪽 필요), (b) phi/frame/pending이 한
    객체에 in-place 결합되어 "shared dense + per-shot frame/ledger" 분리가 없음, (c) measure_z가 frame
    fold와 drop을 한 번에 수행. **INFERRED**

요청 구조(shared dense + per-cohort frame/ledger/record) 대비: **dense·frame·ledger·record는 필드로는
분리되나 객체로는 결합**되어 있어, cohort 공유를 위해선 `nc`를 (immutable dense) + (per-shot frame/pending/
record) 로 재구성하는 리팩터가 필요(현 코드 미지원). 설계안은 다음 단계.

---

## 14. Unknowns & unresolved risks

- **UNKNOWN**: stim-level synthetic(§12 요청의 정확한 stim 회로)은 clifft compile→pending 매핑 불투명으로
  미수행. engine-level로 메커니즘은 TRACE-PROVEN 했으나, compile 경로에서 동일 패턴이 어떤 stim gate로
  나오는지는 미확인.
- **INFERRED(미검증)**: tail-deferral의 exactness — deferred Z-rotation의 (-1)^keepbit 부호가 frame
  fold(`right_*`, `_inv_fold_x`)와 `_drop_residual_products` 이후에도 정확히 ledger에 재삽입되는지는
  **현재 미구현이라 미검증**. b=keepbit 변수는 존재(CODE-PROVEN)하나 상호작용은 설계 단계에서 증명 필요.
- **범위 외**: cult_d5/ry의 interspersed diagonal을 trailing으로 reorder 후 deferral하는 일반화는 executor
  reorder가 필요(반교환 순서 변경 → exactness 별도 증명). 이번 분석은 **trailing suffix만** 판정.
- d7/더 큰 거리, feedback 회로(structure-once 비활성)에서의 tail 분포는 미측정.

---

## 15. 핵심 표 모음

### 표 A — 핵심 함수
| 단계 | 파일:라인 | 함수 | 입력 | 출력 | mutation |
|---|---|---|---|---|---|
| 측정 dispatch | backend.py:611 | run_shot OP_MEAS_ACTIVE_DIAGONAL | q, frame | record bit | slot2id del, record, frame |
| nc 측정 | bounded.py:874 | measure_z | q | outcome | phi/M/frame/pending |
| core 선택(읽기) | bounded.py:503 | _fused_core_entries | q | core entries | none(read-only) |
| core 선택(실행) | lazy.py:190 | _flush_core | (0,1<<q) | — | _meas_ctr++, pending del, dense |
| 반교환 closure | lazy.py:162 | _core_indices | qx,qz | in_core mask | counters만 |
| pullback | simulator.py:276 | _pullback | x,z | xp,zp,pp | none |
| compiled 실행 | compiled_core.py:106 | try_compiled_measure | eng,q | outcome | phi/M/frame/pending |
| C++ core | mdm_core_executor.cpp:160 | mdm_execute_core | phi,rots,lm,m | outcome,p0,surv | joint/survivor buffers |
| measured drop | bounded.py:345 | _drop_localized_core | q,keep | — | M,phi,frame |

### 표 B — Pending entry schema → §4.
### 표 C — Axis/basis 변환 → §7.
### 표 D — Outcome bits → §9.
### 표 E — Benchmark tail 결과 → §11.

### 표 F — 향후 패치 후보 위치 (구현안 아님, 구조적 interception 지점만)
| 후보 위치 | 장점 | 위험 | 모든 executor 공통 |
|---|---|---|---|
| `_fused_core_entries` 반환 직후 core 리스트 split (trailing diagonal 분리) | 한 곳; compiled/fused/oracle 공통 source | core 순서/closure 의미 변경 위험; deferred entry의 frame/sign 재계산 | **예**(세 경로 공통) |
| compiled `rots` 구성 루프(compiled_core.py:155) | executor 입력 직전, mask 확정됨 | compiled 전용; fused/oracle 별도 필요 | 아니오 |
| executor 내부(mdm_execute_core) tail 분리 | numerical 단일 지점 | C++/Python 경로 모두 수정; survivor에 deferred 적용 로직 | 아니오 |
| measurement 후 `apply_rotation` 재삽입(ledger로 deferral) | 기존 ledger 재사용 | (-1)^keepbit 부호·frame fold·drop 순서 exactness 증명 필요 | 공통(후처리) |

> 본 분석 결론: 현재 구현은 uncoupled diagonal을 이미 제외하고 coupled trailing diagonal(Case C: d5_r5/
> d3_r3 측정당 1개, Z-on-measured-axis)만 2^{r_out+1}에서 실행한다. tail-deferral의 정확한 패치 지점과
> exactness 조건(특히 b=keepbit 부호와 frame fold/drop 상호작용)은 이 보고서를 근거로 **다음 단계에서**
> 결정한다. 이번 단계에서는 어떤 구현·단정도 하지 않았다.
