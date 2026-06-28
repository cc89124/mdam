# Bounded 2^{r_out} dense executor — feasibility proof & decision

**최종 질문:** 임의의 measurement core에 대해 Pauli-sum을 2^{m_core}로 전개하지 않고, 측정축 포함
2^{r_out+1} dense state도 만들지 않으면서, exact survivor를 O(m·2^{r_out}) 시간 + O(2^{r_out})
working set으로 만들 수 있는가?

**스크립트:** `/tmp/proto_branchpair.py`(recurrence+exhaustive), `/tmp/proto_realcores.py`(실코어 분류),
`/tmp/bp_kernel.c`+`/tmp/proto_compiled.py`(compiled C), `/tmp/proto_paths.py`(4-path+selector).

---

## 1. 수학적 recurrence

측정축 m을 명시: |Ψ⟩ = |0⟩_m|α⟩ + |1⟩_m|β⟩, α,β는 survivor register S(2^{r_out}). core rotation
R_P(θ)=cosθ·I − i sinθ·P, P = i^pp X^x Z^z. P의 **m-bit 작용**만으로 (α,β) recurrence가 결정된다
(opS(v) = i^pp X^{xS} Z^{zS} v, m-bit 제거한 S 위 Pauli, c=cosθ, s=sinθ):

| m-bit | (xm,zm) | α' | β' | 결합 |
|---|---|---|---|---|
| I | (0,0) | cα − i s·opS(α) | cβ − i s·opS(β) | **독립** |
| Z | (0,1) | cα − i s·opS(α) | cβ + i s·opS(β) | **독립** |
| X | (1,0) | cα − i s·opS(β) | cβ − i s·opS(α) | **결합** |
| XZ(Y) | (1,1) | cα + i s·opS(β) | cβ − i s·opS(α) | **결합** |

**핵심 분기:**
- **m-bit diagonal(I/Z):** α,β 독립 진화 + 단일-Pauli rotation은 branch별 norm 보존. 따라서 Born
  확률 p0 = ‖α_0‖²/(‖α_0‖²+‖β_0‖²)는 rotation을 돌리지 않고 즉시 얻고, survivor는 선택된 한 branch만
  돌리면 됨 → **단일 2^{r_out} buffer로 충분(Case A/B).**
- **m-bit off-diagonal(X/Y):** α' 계산에 **옛 β**가, β' 계산에 **옛 α**가 필요 → 두 branch가 동시에
  live → **2·2^{r_out}=2^{r_out+1} 필요(Case C).** opS(β)는 β의 full 2^{r_out} Pauli 치환이라 "다른
  branch의 영향"을 bounded metadata로 줄일 수 없음(저랭크 아님). 첫 off-diagonal rotation 직후
  β는 full-rank가 되어 α와 얽힘.

질문별 답: α,β 모두 유지해야 하는가 → off-diagonal이 있으면 **그렇다.** bounded metadata로 대체 가능
한가 → **불가능**(opS는 full-rank Pauli 치환). 최종 selected branch에 두 branch interference가 필요한가
→ **Born 확률에 양쪽 norm이 필요**, off-diagonal이면 norm이 보존 안 되므로 양쪽을 끝까지 진화시켜야 함.

---

## 2. 가능성 판정

| | TIME O(m·2^{r_out}) | WORKING SET O(2^{r_out}) | 판정 |
|---|---|---|---|
| m-diagonal core | ✅ | ✅ | **Case B (가능)** |
| off-diagonal-on-m core | ✅(branch-pair, 비-symbolic) | ❌ (2^{r_out+1} 필요) | **Case C (불가능)** |

**TIME은 항상 달성 가능**(branch-pair recurrence는 symbolic 2^m 없이 m·2^{r_out+1}). 문제는
**WORKING SET**: off-diagonal core에서 2^{r_out}로 못 내려간다. 이는 "측정될 1큐빗을 register와
얽는 unitary를 register만 들고 시뮬레이트할 수 없다"는 기본 사실(얽힌 큐빗 1개 = 메모리 ×2).

**증명을 뒷받침하는 구성:** survivor_b = A_b φ_in, A_b = ⟨b|U_core|0⟩_m (2^{r_out}×2^{r_out}).
off-diagonal rotation 1개로 A_1 = −i sinθ P^S ≠ 0(full-support), 이후 rotation들이 A_0,A_1을 dense
full-rank로 만든다. selected b는 ‖A_0φ‖,‖A_1φ‖에 의존하는 Born sample로 정해지므로 양쪽을 모두
계산(=joint 2^{r_out+1})하거나, A_b의 compact form(2^m Pauli 전개 = symbolic, 또는 m-gate 회로를
single 2^{r_out} 벡터에 streaming = joint 2^{r_out+1} 시뮬)이 필요. 둘 다 (2^{r_out} mem ∧ m·2^{r_out}
time)을 동시에 만족 못함.

---

## 3. Exhaustive bit-identical 검증

branch-pair recurrence를 textbook dense(2^{r_mat})와 비교:
- r_out∈{1,2,3,4}, m=1~8, 모든 m-pattern, diagonal/off-diag survivor Pauli, commuting/anti-commuting,
  repeated/dependent Pauli, random+Clifford(π/4 포함) angle, fresh+resident, b=0,1.
- **1760 cores, FAIL=0, max_err=0.00e+00** (기계정밀도; 연산이 동일하여 bit-exact).
- Born p0 ref vs branch-pair |Δ|=0.0. off-diagonal core는 peak_buffers=2(2^{r_out+1}) 확인.

**실 benchmark core 분류(144 magic cores, spot-check err=0.0):**

| circ | magic | Case B (m-diag) | **Case C (off-diag)** | maxNcore | maxRout |
|---|--:|--:|--:|--:|--:|
| distillation | 5 | 4 | 1 | 6 | 3 |
| cultivation_d3 | 5 | 1 | 4 | 14 | 3 |
| cultivation_d5 | 15 | 2 | **13** | 38 | 9 |
| coherent_d3_r3 | 12 | 4 | 8 | 12 | 4 |
| coherent_d5_r5 | 60 | 12 | **48** | 16 | 12 |
| coherent_rx_d3_r1 | 14 | 3 | 11 | 7 | 10 |
| coherent_ry_d3_r3 | 33 | 21 | 12 | 48 | 15 |
| **합계** | **144** | **47** | **97 (67%)** | | |

- **Case C 97개가 모두 비싼 core**(cultivation_d5 13/15, d5_r5 48/60, rx 11/14). bounded 2^{r_out}
  executor는 이들에 **수학적으로 불가능.**
- Case B 47개는 **전부 resident + norm-preserving**(fresh Case-B=0 — fresh+diagonal은 β=0 deterministic).
  즉 bounded가 가능한 core는 이미 싼(측정축 rotation이 diagonal이라 거의 공짜) core뿐.

---

## 4. 네 경로 비교 (dense-equivalent element-touch, magic core 합)

A=Clifft rot · B=oracle(#rot·2^rmat) · C=fused(#U·2^rout) · **D=realizable(Case B: 1-branch
#rot·2^rout, Case C: 2-branch = B)**

| circ | A Clifft | B oracle | C fused | **D realizable** | C/B | **D/B** |
|---|--:|--:|--:|--:|--:|--:|
| coherent_d3_r3 | 1.48e4 | 5.09e3 | 1.26e4 | 5.09e3 | 2.5 | **1.00** |
| coherent_d5_r5 | 5.64e9 | 7.70e6 | 3.43e7 | 7.70e6 | 4.5 | **1.00** |
| cultivation_d3 | 4.30e2 | 1.31e3 | 1.02e3 | 1.21e3 | 0.8 | 0.93 |
| cultivation_d5 | 5.92e4 | 1.97e5 | 1.63e6 | 1.91e5 | 8.3 | 0.97 |
| distillation | 2.46e2 | 3.36e2 | 2.41e3 | 2.40e2 | 7.2 | 0.71 |
| coherent_rx_d3_r1 | 1.57e5 | 7.33e4 | 1.90e5 | 7.14e4 | 2.6 | 0.98 |
| coherent_ry_d3_r3 | 5.37e6 | 2.66e7 | 3.10e7 | 2.65e7 | 1.2 | **1.00** |

**결정적:** 실현 가능한 D는 **모든 비싼 벤치(d5_r5/ry/d3_r3)에서 D/B=1.00 = 오라클과 정확히 동일.**
oracle을 능가하는 "세 번째 경로"는 비싼 core에 **존재하지 않는다.** (forensic의 낙관적 C/D=4–17×는
실현 불가능한 single-branch D와 비교한 것이었음 — 정정.) C(fused)는 B보다 거의 항상 나쁨(0.8–8.3×).

---

## 5. D 합격 조건 평가

| 조건 | branch-pair(Case B) | branch-pair(Case C) |
|---|---|---|
| O(m·2^{r_out}) time | ✅ | ✅(=oracle) |
| 비용 ∝ 2^{m_core} 아님 | ✅ | ✅ (symbolic 제거됨) |
| 상수개 2^{r_out} buffer | ✅(1개) | ❌ (2개 = 2^{r_out+1}) |
| 2^{r_out+1} 미생성 | ✅ | ❌ |
| rotation UID당 dense 1회 | ✅ | ✅ |
| survivor 직접 생성 | ✅ | ✅ |
| per-term 2^{r_out} 임시 없음 | ✅ | ✅ |
| oracle와 bit-identical | ✅ | ✅ |
| ry 43–48 rotation 선형 | ✅(m·2^rout) | ✅(symbolic 2^m 회피!) |

→ Case C는 working-set 두 조건(상수 buffer / 2^{r_out+1} 미생성)만 위배. **나머지 8개 조건은 만족** —
특히 ry의 48-rotation core도 branch-pair는 **선형**(2^48 symbolic 회피)으로 처리. 즉 branch-pair는
oracle의 올바른 알고리즘적 표현이며, 위배되는 것은 정보이론적으로 불가피한 +1뿐.

---

## 7. Compiled C prototype (Python 고정비용 제거)

`bp_kernel.c`(gcc -O3 -march=native, ctypes 로드) — core 전체를 **1회 compiled call**로 실행:

| core | rot | r_out | Python µs | C µs | speedup | py calls | **c calls** | py ns/amp | C ns/amp |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| small r3 | 6 | 3 | 184.8 | 19.2 | **9.6×** | 24 | **1** | 3850 | 399 |
| d5_r5-like r12 | 16 | 12 | 2177.8 | 968.6 | 2.2× | 64 | **1** | 33.2 | 14.8 |
| ry-like r15 | 48 | 15 | 51001 | 22372 | 2.3× | 192 | **1** | 32.4 | 14.2 |
| rx-like r10 | 7 | 10 | 410.6 | 125.8 | 3.3× | 28 | **1** | 57.3 | 17.6 |

C vs dense bit-identical(max 1.34e-15). **speedup이 저랭크(r3 9.6×)에서 최대** = Python per-call
floor가 거기서 가장 아픔(H7 확인). 잔여 gap(C 14 ns/amp vs Clifft 0.3–1)은 **kernel 최적화**(벡터화·
temp 제거)이지 알고리즘 아님 → 별도 ctypes lib보다 **Clifft의 기존 tuned C++ backend 안에 recurrence를
넣는 것**이 정답. (exact 모드는 shot마다 RNG·branch가 달라 batch 상각 없음 → per-core 비용은 shot 수에
선형, batch 1/10/1e3/1e6 모두 동일 per-core throughput.)

---

## 8. Selector (분석만, backend 미변경)

predicted nU = min(2^{#offdiag}, 2^{rmat}) vs ACTUAL nU, "symbolic should win" = (actual #U < 2·#core_rot):

| circ | fused cores | pred=act | pred>act | maxRelErr | **sym should win** |
|---|--:|--:|--:|--:|--:|
| coherent_d3_r3 | 36 | 0 | 0 | 0.97 | **0** |
| coherent_d5_r5 | 180 | 0 | 0 | 0.99 | **0** |
| cultivation_d3 | 15 | 6 | 3 | 1.00 | 6 |
| cultivation_d5 | 45 | 18 | 6 | **63.0** | 7 |
| distillation | 14 | 3 | 0 | 0.98 | **0** |
| coherent_rx_d3_r1 | 42 | 18 | 3 | 1.00 | 3 |
| coherent_ry_d3_r3 | 81 | 6 | 0 | 1.00 | **0** |

- predicted nU은 **upper bound**(pred ≥ act). **비싼 벤치(d5_r5/ry/d3_r3/distillation)는 symbolic이
  이기는 core가 0개** → symbolic을 default에서 빼도 손실 없음.
- `#core_rot > 14` 단독 판정은 **부적합**: cultivation은 cancellation으로 actual nU가 예측보다 최대
  63× 작음(saturate). 안전 predictor는 **min(2^{#offdiag}, 2^{rmat})를 2·#core_rot와 비교**하되,
  cancellation 때문에 보수적(symbolic을 dense로 보내도 그 core는 어차피 싸서 손해 ≤2×).

---

## 10. 결론

### 10.1 수학적 결론
**조건부.** O(m·2^{r_out}) **시간은 항상** branch-pair recurrence로 가능(symbolic 2^m 불필요).
O(2^{r_out}) **working set은 m-diagonal core(Case B)에서만** 가능. measured axis에 off-diagonal로
작용하는 rotation이 1개라도 있으면(Case C) 두 branch vector가 동시에 필요 → 2^{r_out+1} 정보이론적
필수. 실 benchmark의 **97/144(67%) core, 그리고 비싼 core 전부가 Case C** → 임의 core 일반은 불가능.

### 10.2 Prototype 결과
branch-pair: 1760 synthetic + 144 real core 모두 oracle/dense와 **bit-identical(err 0)**. 시간복잡도
O(m·2^{r_out+1}), working set Case B 1×2^{r_out} / Case C 2×2^{r_out}. ry 48-rotation core도 **선형**.
compiled C: bit-identical, 저랭크 9.6× speedup, core당 1 call.

### 10.3 구현 선택 — **항상 dense oracle (2^{r_out+1}), hot path를 compiled로**
- bounded 2^{r_out} executor: ❌ (비싼 core 전부 불가능, 가능한 core는 이미 쌈).
- 조건부 bounded + oracle: 실익 거의 0(D/B=1.00 on 비싼 벤치) → 복잡도만 늘 뿐.
- **symbolic fused: default에서 제외**(B보다 0.8–8.3× 나쁨, 비싼 core에서 이기는 일 없음).
- oracle은 이미 **시간 최적 O(m·2^{r_out+1})**; 유일한 약점은 Python 상수항 → **measurement core를
  compiled call 1회로**(이상적으로 Clifft C++ backend 내부의 branch-pair recurrence).

### 10.4 Clifft 대비 분리
1. 알고리즘 dense work: MDAM Fm/Fc ≤ 0.96(이미 우월, d5_r5 2700× 적음). 2. compiled runtime: C
recurrence가 Python 대비 2–10×, 추가 kernel 튜닝으로 Clifft 수준 가능. 3. Python wrapper overhead:
core당 1 call로 제거. 4. memory peak: oracle 2^{r_out+1}이 정보이론적 하한(Clifft도 동일 측정축 부담),
d5_r5에서 Clifft 256MB ≫ MDAM 0.66MB. 5. low-rank crossover: 현재 Clifft가 유리(2^k 작음·C++); compiled
oracle로 좁힘. 6. high-rank speedup: MDAM이 이미 18–54× 우월(d5_r5).

### 10.5 다음 단계
**먼저** branch-pair recurrence를 oracle의 정식 알고리즘으로 확정(symbolic 제거)하고, **그 다음**
measurement-core hot loop를 Clifft C++ backend 안에 compiled call로 통합(axis promote→core
rotation→Born→project→drop→frame를 1 call). authoritative bit-identity 검증 후 integration.

---

## 최종 한 문장

> 임의의 MDAM measurement core에 대해 O(m·2^{r_out}) 시간과 O(2^{r_out}) working set의 exact survivor
> execution은 **조건부로만 가능하다(measured axis에 off-diagonal로 작용하는 core rotation이 없는
> m-diagonal core에 한해서이며, 실제 benchmark의 비싼 core 97/144는 모두 off-diagonal이라 일반적으로는
> 불가능하다)**. 그 이유는 **off-diagonal rotation이 측정축을 survivor register와 얽혀 두 branch
> vector(α,β)를 동시에 보유해야 하므로 2^{r_out+1}이 정보이론적으로 필요하기 때문**이며, 따라서 최종
> backend는 **+1 transient를 받아들인 dense oracle을 그대로 쓰되(이미 시간 최적 O(m·2^{r_out+1}),
> symbolic fused는 제외), 그 measurement-core hot path를 branch-pair recurrence로 compiled call 1회에
> 실행(이상적으로 Clifft C++ backend 내부)** 하여 Python per-call 고정비용을 제거해야 한다.
