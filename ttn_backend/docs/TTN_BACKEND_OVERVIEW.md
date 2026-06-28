# TTN Backend — Overview (Motivation, Design, Results, Limits)

이 문서는 `clifft-paper`의 TTN backend를 **한 곳에 정리한 상위 개요**다. 동기(왜 만드는가),
목적, 설계 알고리즘, 현재 성과, 현재 한계를 모두 담는다.

## 1. 왜 TTN backend인가 — Clifft active-state 구조의 한계

Clifft는 Stim 회로를 compile해 bytecode로 실행하는 시뮬레이터다. Clifford 부분은
효율적으로 처리하되, 비-Clifford(T, rotation 등)에 닿는 identity는 **active state**로
promotion해서 실제 진폭을 들고 간다.

문제는 이 active state를 **dense 상태 벡터**로 든다는 점이다. 동시에 active인 ident 수를
`k`라 하면 메모리는

```text
M_clifft(t) = 16 * 2^(k_active(t))   bytes   (complex128)
```

즉 **활성 큐빗 수에 지수적**이다. `coherent_d5_r5`에서 peak `k=24` → `16 * 2^24 = 256 MiB`.
`k`가 더 커지는 회로(d7 계열)는 곧바로 RAM 벽에 부딪힌다. 이것이 Clifft의 **`2^k` 벽**이다.

핵심 관찰: 이 dense 벡터는 **활성 큐빗 사이의 실제 얽힘(entanglement)이 작아도 항상 `2^k`를
지불**한다. 하지만 QEC 회로의 active state는 보통 **국소적으로만 얽혀** 있다 — 한 번에 모든
큐빗이 최대로 얽히지 않는다. 그렇다면 상태를 통째 dense로 들 필요가 없다.

```text
dense        : 항상 2^k  (큐빗 수에 지수적)
저-얽힘 상태  : 훨씬 작게 표현 가능 (얽힘 구조를 따라가면)
```

이 gap이 TTN backend의 존재 이유다.

---

## 2. 목적

1. Clifft bytecode의 active-state 부분을 **tensor-tree network(TTN)**로 exact하게 실행한다.
2. 메모리를 큐빗 수가 아니라 **실제 얽힘(Schmidt rank)**에 비례하게 만들어 Clifft의
   `2^k` 벽을 넘는다.
3. 실제 tensor shape / bond / QR·transport 비용을 **계측**해서 정적 treewidth와 실행 메모리의
   gap을 정량화한다.
4. `clifft.sample`과 **bit-identical**한 측정 분포를 재현한다(검증).

현재 구현은 **exact baseline**이 기본이다. 런타임 경로에서 임의의 bond truncation/근사를
넣지 않는다(근사는 명시적 옵션으로만).

---

## 3. 설계 알고리즘

### 3.1 파이프라인

```text
Stim circuit -> clifft.compile -> bytecode
  -> export_backend_spec()        # 구조적 replay: ident lifecycle, interaction graph
  -> assign_homes_and_classify()  # 또는 carving_leaf 레이아웃: home 배정 + op class
  -> TTNBackend(spec, homing)
  -> run_shot()                   # bytecode dispatch -> 측정 record
```

### 3.2 자료구조: ident / bag / bond

| 개념 | 의미 | 차원 |
|---|---|---|
| **ident** | active state로 promotion된 논리 큐빗 (lifecycle 보유) | 물리축 = 2 |
| **bag** | TTN 트리의 노드. 여러 ident와 bond 축을 가진 tensor | — |
| **bond** | 인접 bag을 잇는 내부 index = 그 tree cut의 얽힘 굵기 | χ (가변) |

bag tensor 크기:

```text
N_B = 2^(p_B) * prod_{e~B} chi_e        bytes_B = 16 * N_B
```

- `p_B` = bag이 직접 든 물리 큐빗(own ident) 수
- `chi_e` = 인접 edge의 bond 차원 = 그 cut의 **Schmidt rank = 2^(얽힘 엔트로피)**

전체 상주 메모리 `M_store = 16 * sum_B N_B`. peak offender는 보통 `p_B + sum log2 chi_e`가 큰
hub bag이다.

### 3.3 레이아웃

- **union junction tree** (`assign_homes_and_classify`): 프로그램 전체에서 한 번이라도
  필요한 interaction을 합친 정적 트리. 단순하지만 hub bag에 bond가 몰린다.
- **carving_leaf** (`build_carving_executable_spec`): temporal-carving으로 만든 binary tree
  leaf-home 레이아웃. 벤치마크의 기본이며 union보다 peak가 작다.

home은 static이다. cross-bag 연산을 위해 ident를 임시 이동(transport)하지만 home은 안 바꾼다.

### 3.4 연산 분류와 transport

two-axis op `(u,v)`는 home 위치로 분류된다.

- **Class A**: 두 ident가 같은 bag → local 4x4 gate, transport 불필요.
- **Class B/C**: 다른 bag → **adjacent 2-bag transport sweep**으로 한 ident를 경로 따라
  옮기고(QR), gate 적용 후 되돌린다. 과거의 full-path dense contraction(`2^(sum own)`)을 피한다.

`transport_ident_across_edge`: 인접 두 bag을 shared bond로 contract(theta) → `M=Q·R` →
Q는 src, R은 dst. workspace는 path 전체가 아니라 인접 2-bag 크기로 제한.

### 3.5 메모리/연산을 줄이는 레이어 (현재 구현)

| 레이어 | 무엇을 하나 | flag |
|---|---|---|
| **persistent MULTI_CNOT window** | MULTI_CNOT+frame Clifford+ROT만 있는 window를 connected region 한 번 열어 처리. destructive-open으로 region 내 bag을 detach해 회계 | `TTN_PERSISTENT_MULTICNOT`, `TTN_DESTRUCTIVE_OPEN` |
| **staged (block-streamed) transport** | 큰 theta를 materialize하지 않고 `M=Q·R`을 블록 스트리밍 (orientation-aware Gram). workspace 병목 exact 제거 | `TTN_STAGED_TRANSPORT` |
| **MULTI_CNOT parity-gather rewrite** | 같은 subtree control을 rep control로 재귀 fold → 큰 bond를 subtree당 1회만 건넘. exact, cost-gated | `TTN_MULTICNOT_PARITY_REWRITE` |
| **exact bag fission** | 큰 bag을 저-rank bond 축으로 SVD 분해 | `TTN_BAG_FISSION_CAP_BYTES` |
| **resident streaming** | cap 초과 bag을 memmap 디스크로 spill, RAM엔 블록만 (exact out-of-core) | `TTN_RESIDENT_STREAM_CAP_BYTES` |

핵심 framing: 한 병목을 줄이면 다른 경로로 병목이 옮겨간다(**bottleneck migration**). 궁극
목표는 모든 cross-bag executor 후보를 하나의 concurrent memory cap 아래에서 고르는
memory-capped selector다(미완).

---

## 4. 현재 성과

### 4.1 정확성

작은 회로에서 `clifft.sample` 대비 측정 분포 self-sampling floor 수준(pass). 모든 exact
최적화(persistent/staged/parity rewrite)는 측정 record가 **bit-identical**임을 확인했다.

### 4.2 매-step 메모리: Clifft dense vs TTN (전체 합 비교)

`scripts/per_step_memory_compare.py`, carving_leaf, general_policy + parity rewrite.
Clifft = `16*2^(활성 ident 수)`, TTN = 실제 저장 bytes 합.

| circuit | steps | k_max | PEAK Clifft | PEAK TTN | peak비 | Σ Clifft | Σ TTN | **Σ비** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cultivation_d3 | 344(full) | 4 | 256 B | 1.16 KiB | 0.22x | 48.4 KiB | 160.9 KiB | **0.30x** |
| coherent_d5_r1 | 857(full) | 13 | 128 KiB | 46.8 KiB | 2.74x | 42.0 MiB | 12.8 MiB | **3.29x** |
| coherent_d5_r5 | 1200(prefix) | 24 | 256 MiB | 132 MiB | 1.94x | 114.0 GiB | 10.1 GiB | **11.27x** |

(모두 carving_leaf 레이아웃. 그래프: `reports/per_step_memory/{circuit}_per_step_linear.png`.)

해석:
- **Σ(시간축 메모리 합) 이득이 peak 이득보다 훨씬 크다**(d5_r5: peak 1.94x, Σ 11.27x).
  Clifft dense는 `k=24`가 되면 256 MiB에 고정되어 수백 step 유지되지만, TTN은 얽힘 peak에서만
  잠깐 132 MiB이고 대부분 구간은 ~6-10 MiB. TTN은 큐빗 수가 아니라 **실제 얽힘**을 따라간다.
- **k에 따른 crossover**: 작은 회로(k=4)는 TTN의 bag/bond 구조 오버헤드가 `2^4`보다 커서
  오히려 손해(0.30x). 이득은 활성 큐빗 k가 클수록 커진다.

### 4.3 staged transport (workspace 병목 제거)

`coherent_d5_r5` 1200-step: workspace 병목이던 121.6 MB theta를 materialize 없이 exact 회피.
general_policy와 동일 peak(138.6 MB)를 ~40% 빠르게 달성.

### 4.4 MULTI_CNOT parity-gather rewrite (연산량 절감, peak 유지)

`coherent_d5_r5` 1200-step, general_policy:

| metric | baseline | rewrite | Δ |
|---|---:|---:|---:|
| actual peak | 138.58 MB | 138.62 MB | +0.03% (유지) |
| max bond χ | 2048 | 2048 | 동일 |
| **rank-weighted path** | 1240 | 942 | **−23.8%** |
| n_transports | 854 | 770 | −9.8% |
| wall-clock | 53.4s | 31.0s | −41.9% |

exact(GF(2)/statevector/Pauli-symplectic 0 mismatch). 큰 bond를 **덜 건너서** 비싼
QR/transport를 싼 것으로 치환 → peak 불변, 연산/시간 감소.

---

## 5. Deferred affine+phase backend (boundary-free fast path)

§1의 `2^k` 벽을 **다른 방식으로** 넘는 보조 백엔드다(`affine_backend.py`). TTN이 "얽힘을
따라가는 텐서"로 dense를 대체하는 반면, 이 백엔드는 **active op 스트림이 비대각 게이트를 전혀
안 쓰는 회로**에 한해 상태를 텐서 없이 닫힌 대수 형태로 들고 간다.

### 5.1 표현 (affine + phase 정규형)

active 스트림이 `EXPAND / CNOT·MULTI_CNOT / CZ·MULTI_CZ / T·S·RZ(ROT) / Z-측정 / SWAP`
뿐이면 active 상태는 정확히 다음 형태다.

```text
|ψ⟩ = (1/2^(k/2)) · Σ_x  e^{i f(x)} |A·x ⊕ b⟩
```

- `A·x ⊕ b` : GF(2) **affine 맵** (CNOT가 누적)
- `f(x)` : 실수 **위상다항식** (T/S/RZ → 1차, CZ → 2차)
- 진폭 크기는 모든 기저에서 `1/2^(k/2)`로 **균일**(equatorial 가정)

게이트별 닫힘(증명 가능):

- **CNOT/MULTI_CNOT**: 기저 GF(2)-선형 치환 → `A,b`에 affine 합성, `f` 불변. (`bits[v]^=bits[u]`)
- **T/S/RZ(θ)**: `|y⟩→e^{iθ y_q}|y⟩`, `y_q`=x의 affine 함수 → `f`에 1차항 추가, `A` 불변.
- **CZ**: `(-1)^{y_p y_q}` → `f`에 2차항 추가.

→ 선형은 전부 `A`, 대각 위상은 전부 `f`, 크기는 균일. (CNOT+T affine+phase 정규형 / CH-form 계열.)

### 5.2 Z-측정에서 f 소거 → 샘플링 지름길

큐빗 q를 Z로 측정: `P(=1) = (1/2^k)·#{x : (A의 q행)·x ⊕ b_q = 1}`. `|e^{if}|²=1`이라 **f가 통째로
소거**된다. 0이 아닌 선형범함수는 정확히 절반에서 1 → 공평한 동전. 측정 비트들의 결합분포 = 균일
난수 `x`를 `A`의 행들로 사상한 것. 따라서:

- EXPAND마다 독립 **균일 source bit** 발행
- `A`를 `bits[ident] = source bit들의 XOR`로 심볼릭 운반
- Z-측정 = source bit들의 **parity**

거대한 위상다항식 `f`(O(k²) 항)는 **만들 필요조차 없다.** 프레임/노이즈/dormant/readout 경로는
`TTNBackend.run_shot`과 동일하게 재사용해 측정 record 분포가 일치한다.

### 5.3 메모리·연산 복잡도

| | 상태 저장 크기 | per-shot 시간 |
|---|---|---|
| clifft (dense) | `O(2^k)` (k=25 → ~512 MiB) | `Θ(2^k · G)` |
| affine | `O(k²)` 비트 (k=25 → 수백 바이트) | `Θ(G · k/64)` |

텐서·본드·QR·SVD가 **하나도 없다.** 메모리·시간 모두 **지수→다항**으로 떨어진다(G = 게이트 수).

### 5.4 적용 범위 (audited)

비대각 게이트(H / U2 / U4 / X-기저 측정)는 진폭을 간섭시켜 균일 크기를 깨므로 **boundary**다(그
지점에서 materialize 필요). 벤치마크 회로 분류:

| circuit | boundary 종류·수 | affine |
|---|---|---|
| coherent_d3_r1 / d5_r1 / d7_r1 | **없음** | ✅ 전 구간 처리 |
| cultivation_d3 | SWAP_MEAS_INTERFERE ×5 | ❌ 거부 |
| coherent_d5_r5 | U2 ×44 + U4 ×4 | ❌ 거부 |
| distillation | H ×5 + INTERFERE(SWAP_MEAS ×3 + MEAS ×1) | ❌ 거부 |

→ 진짜 boundary-free는 `coherent_*_r1` 3종뿐. 나머지는 boundary가 있어 affine이
`BoundaryEncountered`로 **거부**한다(틀린 답을 내지 않는 안전장치). 주의: 예전 deferral 분석
스크립트가 `OP_SWAP_MEAS_INTERFERE`를 boundary 목록에서 빠뜨려 cultivation_d3/distillation을
boundary-free로 오분류했었다 — 수정 완료(§9).

### 5.5 검증 결과

**correctness** (affine vs `clifft` 측정 한계분포 최대차):

| circuit | clifft N | max\|aff−clifft\| | tol(3σ) | 판정 |
|---|---:|---:|---:|---:|
| coherent_d3_r1 | 8000 | 0.0135 | 0.0268 | ✅ PASS |
| coherent_d5_r1 | 8000 | 0.0189 | 0.0268 | ✅ PASS |
| coherent_d7_r1 | 50* | 0.1756 | 0.2221 | ✅ PASS |

(*d7은 clifft가 ~3s/shot이라 120s 예산에 50 shot만 뽑혀 tol이 큼.) boundary 회로 3종은 전부 정확히 거부.

**performance** (per-shot wall-clock):

| circuit | k | clifft | affine | TTN | TTN/affine |
|---|---:|---:|---:|---:|---:|
| coherent_d3_r1 | 5 | 0.002 ms | 0.566 ms | 14.7 ms | 26× |
| coherent_d5_r1 | 13 | 0.199 ms | 1.85 ms | 169 ms | 91× |
| coherent_d7_r1 | 25 | 3006 ms | 4.38 ms | 63312 ms | **14467×** |

- affine은 TTN보다 **항상** 빠르고(26×→14467×), 배수가 k에 따라 폭발한다.
- d7에서 affine vs clifft = **687×**(3006/4.38).
- 작은 k(d3,d5)에서 clifft(컴파일 C)이 순수 파이썬 affine보다 빠른 건 **상수항(구현) 차이**지
  복잡도 차이가 아니다(§5.3). affine을 C/벡터화로 다시 짜면 모든 k에서 우위.

### 5.6 한계

- **boundary-free 회로 전용.** d5_r5 등 비대각 게이트가 있는(= TTN 프로젝트가 본래 노리는 어려운)
  회로는 통째로 다루지 못한다. boundary **사이 구간**만 부분 deferral이 가능해 *compute는 일부 절감*
  되지만 **peak 메모리는 불변**이다(§6.1 resident floor).
- 그리고 boundary-free 부류는 본질적으로 **고전적으로 쉬운 클래스**(stabilizer + 위상다항식)다 —
  affine의 거대한 이득은 "어려운 문제를 풀어서"가 아니라 clifft/TTN이 그 구조를 안 쓰고 `2^k`를
  무는 비효율을 제거해서 나온 것이다.
- **순수 파이썬** 구현이라 작은 k에서 상수항 손해(§5.5).

### 5.7 하이브리드 `(A,f)`-over-TTN 프레임 — 분석 결과와 고민 (미구현)

§5.6의 "boundary 사이만 부분 deferral"을 일반 회로로 확장한 것이 **하이브리드 프레임**이다.
첫 boundary 이후에도 affine+phase 정규형을 **버리지 않고** 일반 얽힘 상태 위에 *연산자*로 얹는다:

```text
|ψ⟩ = D_f · P_A · |φ_TTN⟩      (P_A: GF(2) affine permutation, D_f: 대각 위상)
```

active CNOT/MULTI_CNOT는 `P_A`에, 대각(T/S/RZ/CZ)은 `f`에 누적하고, 비대각 boundary(U2/U4/H/
X-기저 측정)에서만 해당 부분을 `|φ_TTN⟩`로 **materialize**한다. 이 분해는 `|φ⟩`와 무관한 순수
연산자 항등식이라 **수학적으로 정확**하다. **단, core.py에는 아직 미구현 — 아래는 전부 bytecode를
재생해 CNOT 수를 센 *모델 분석*이다**(실제 wall-clock/transport 측정 아님).

**메모리는 불변, 이득은 compute에서만.** 같은 net CNOT이 같은 tree cut을 지나므로 peak bond χ는
그대로다(§6.1 resident floor). 이득의 *유일한* 출처는 deferral 구간의 선형맵을 Gauss-Jordan/
phase-network로 **재합성**해 transport/QR **횟수**를 줄이는 것뿐이다.

**핵심 audit (d5_r5, boundary 48개 = U2×44 + U4×4):**

| 항목 | 측정값 | 함의 |
|---|---|---|
| boundary region (localize할 큐빗 수) | mean **1.08**, max 2, hist {1:44, 2:4} | boundary는 **로컬**, ~4 CNOT |
| active Z-측정 | 72회, 66/72 weight-1 | localize ~**79 CNOT**, 병목 아님 |
| boundary에 닿는 위상 (f_touch) | **244**개 → unique **187** | **진짜 비용** |
| boundary에 안 닿는 위상 (droppable) | 181개 | materialize 불필요 |

→ boundary 자체와 측정은 거의 공짜다. **비용은 boundary에 닿는 위상 244개**다.

**여러 비용 모델이 ~2×로 수렴한다(8.33× ceiling은 도달 불가):**

| 모델 | d5_r5 CNOT | 배수 | 설명 |
|---|---:|---:|---|
| A-only 프레임 | — | ≤1× | 대각이 flush 강제 → 틀림(§corr). 폐기 |
| full-defer GE ceiling | 93 | 8.33× | **선형맵만**의 하한; 위상 realize·boundary 분절 무시 → **도달 불가** |
| per-segment-reset 부분합성 | **380** | **2.04×** | boundary마다 정리 → 위상 low-weight 유지. **현실적 상한** |
| lazy-persist phase-network | 693 | 1.1× | boundary 가로질러 defer → **오히려 나쁨**(아래) |

(raw CNOT 775 기준. window: A-only meanWin 2.86 → (A,f) meas-flush 64.58로 길어짐.)

**고민 1 — local-phase fusion이 d5_r5에선 효과 0.** "비대각 게이트 앞 회전을 U2'=U2·RZ(θ)로
흡수하면 244를 줄인다"는 아이디어를 검증했더니 **fusable = 0/244**. coherent error 위상은
boundary에 도달할 때 이미 weight 2~15의 **nonlocal parity**(hist {2:17,…,10:21,…,15:2})라
boundary 큐빗 위의 weight-1 local 위상이 하나도 없다. 병합으로 244→187 unique만 줄 뿐, fuse 불가.

**고민 2 — boundary 가로질러 defer하면 오히려 손해.** lazy-persist는 boundary localize는 싸지만
(region 1.08), CNOT이 누적돼 위상 parity weight가 2~15로 **부풀어** phase-network 합성이 614 CNOT
(+측정 79 = 693, 1.1×)으로 **per-segment-reset 380(2.04×)보다 나쁘다**. → boundary마다 정리하는
보수적 모델이 더 낫다. 최적 interleaved 합성은 ≤380이겠으나 미측정.

**고민 3 — 이건 전부 CNOT 카운트 모델이다.** TTN의 실제 비용은 **bond χ 가중 경로**(rank-weighted
transport)라 "CNOT 수 최소화 ≠ transport/QR 비용 최소화". 실제 wall-clock은 구현해서 재야 안다.

**정직한 결론:** d5_r5에서 하이브리드 compute 이득은 **모든 모델에서 ~2× 상한, peak는 불변**.
boundary-free 회로(d7_r1)는 §5의 standalone affine이 이미 14467×를 내므로 하이브리드가 불필요하고,
distillation은 CNOT이 6개뿐이라 raw/GE=0.50(materialize가 더 비쌈) — **하이브리드가 의미있는 구간이
좁다**. 미해결: ~2×(peak 불변)가 core.py 구현 복잡도를 정당화하는가 — 현재는 보류.

---

## 6. 현재 한계

### 6.1 Resident entanglement floor (가장 큰 벽)

peak는 가장 큰 bag tensor가 결정하고, 그 크기는 거의 전부 **bond χ의 곱**이다.
`coherent_d5_r5` peak bag B72 = `[2,1024,64,32]` = 64 MiB에서 22비트 중 **21비트가 bond**,
물리 큐빗은 1개(factor 2)뿐. 큰 bond χ=1024~2048은 그 tree cut의 **진짜 얽힘(Schmidt rank)**
이라 exact로는 못 줄인다. 최악 cut rank가 χ면 그 junction은 `~χ^2`, 양 끝 bag 합쳐 `~2·χ^2`가
exact 하한(d5_r5에서 ~134 MB). 이걸 더 내릴 exact 수단은 **resident streaming(저장을 RAM 밖
으로)**뿐이고, 근사(bond truncation)는 측정 오차를 별도 보고해야 한다.

### 6.2 SVD / QR 연산 수와 원인

요청대로 명확히 구분한다. **exact 런타임 경로에는 SVD가 없다.**

```text
n_svd (exact path)        = 0       # transport는 QR(_thin_qr)만 사용
n_qr  (d5_r5 1200-step)   ≈ 2100~2180
```

- **QR**: 모든 adjacent transport가 1회 QR. n_qr이 n_transports(~800)보다 큰 것은 region
  contraction/split, fission 등에서도 QR이 나오기 때문. parity rewrite는 n_qr을 −2% 정도만
  바꾼다(큰 QR 몇 개가 작은 QR 여러 개로 치환 → 횟수는 비슷, 크기는 작아짐).
- **SVD가 언제 생기나 (원인)**: 기본 exact 경로엔 없음. SVD는 다음 **선택적/근사** 경로에서만
  나온다.
  1. `fission_bag_exact` (bag fission) — 큰 bag을 저-rank 축으로 분해할 때 SVD.
  2. `TTN_SVD_TRUNC_RTOL` (approximate bond truncation) — 명시적 근사.
  3. static compression / offline 실험 스크립트.
  - staged transport는 SVD가 아니라 Hermitian Gram의 `eigh`를 쓴다.
- 함의: "연산량을 줄였다"의 정체는 **SVD/QR 횟수 감소가 아니라 비싼(큰-χ) 연산을 싼 연산으로
  치환한 것**이다. 시간/면적 지표(rank-weighted path, wall-clock)에서 이득이 크고, 횟수
  지표(n_qr)에서는 작다.

### 6.3 그 밖의 한계

- **bottleneck migration**: 모든 cross-bag op를 하나의 memory cap 아래 고르는 selector가
  없어, 한 병목을 줄이면 다른 fallback/path 경로가 peak가 된다. 1200-step에서 exact
  persistent-only가 dense보다 커지는 것이 그 예.
- **static union layout**: 시간별 live graph를 못 살리고 hub bag에 bond가 몰린다. carving_leaf가
  완화하지만 최적은 아니다.
- **small-circuit overhead**: 활성 큐빗이 적은 회로(k≤~5)는 bag/bond 구조 오버헤드 때문에
  dense보다 손해(§4.2 cultivation_d3 0.30x).
- **parity rewrite 범위**: 현재 per-control fallback 경로에서만 발동(persistent window가
  흡수하는 건 이미 싸다). rep 선택은 휴리스틱(Steiner-tree 최적 routing 미구현).
- `coherent_d7_r7`는 union B0 degree 99로 numpy 64차원 한계에 걸려 초기 표현부터 실패
  (algorithmic blowup 이전의 representation 문제).

---

## 7. 다음 단계 (우선순위)

1. **Resident streaming 런타임 통합**: BlockTensorStore로 큰 bag을 out-of-core로 두고 peak
   op만 블록 스트리밍 → resident floor를 블록 한도로 낮춤 (feasibility 확인됨).
2. **Memory-capped executor selector**: 모든 cross-bag 후보(local/transport/fused/batched/
   destructive-open/SVD)를 하나의 concurrent cap 아래에서 선택.
3. **parity rewrite 일반화 심화**: Steiner-tree 최적 routing, transient-bond 반영 cost.
4. **layout/skeleton search**: selector 위에서 재평가.
5. RASL은 마지막에 resident cap을 악화시키지 않는 보조 path/refactor reducer로.

---

## 8. 주요 flag 요약

```text
# persistent MULTI_CNOT region
TTN_FUSE_MULTICNOT, TTN_PERSISTENT_MULTICNOT, TTN_DESTRUCTIVE_OPEN,
TTN_FUSE_MULTICNOT_BATCH, TTN_FUSE_MULTICNOT_CAP_BYTES, TTN_FUSE_MULTICNOT_TOTAL_CAP_BYTES
# staged transport
TTN_STAGED_TRANSPORT, TTN_STAGED_BLOCK_BYTES, TTN_EXACT_TOTAL_CAP_BYTES
# parity rewrite
TTN_MULTICNOT_PARITY_REWRITE, TTN_MULTICNOT_PARITY_MIN_GAIN
# fission / resident streaming / approximation
TTN_BAG_FISSION_CAP_BYTES, TTN_RESIDENT_STREAM_CAP_BYTES, TTN_SVD_TRUNC_RTOL
```

## 9. 핵심 검증/측정 스크립트

```text
scripts/verify_ttn.py                        # 작은 회로 correctness
scripts/verify_multicnot_parity_rewrite.py   # parity rewrite exactness (GF2/statevec/symplectic)
scripts/check_parity_rewrite_correctness.py  # rewrite on/off 측정 record bit-identical
scripts/measure_parity_rewrite.py            # QR/transport/peak/elapsed delta
scripts/per_step_memory_compare.py           # Clifft dense vs TTN 매-step + 합
scripts/run_general_policy_benchmark.py      # 정책별 벤치마크
scripts/big_edge_crossing_audit.py           # 큰-edge crossing 진단
# deferred affine+phase 백엔드 (§5)
scripts/verify_affine_backend.py             # affine vs clifft 분포 + 거부 + per-shot 속도
scripts/affine_memory_compare.py             # affine vs clifft dense per-step 메모리 (§5.3/5.5)
scripts/analyze_deferral_regime.py           # boundary 밀도/deferrable run-length (boundary set 수정됨)
# 하이브리드 (A,f)-over-TTN 프레임 분석 (§5.7, 미구현 — CNOT 카운트 모델)
scripts/measure_deferral_cnot_compression.py # raw CNOT vs net affine 맵 합성(GE) 압축비
scripts/measure_realizable_deferral.py       # flush 정책별 CNOT-batching window
scripts/boundary_region_audit.py             # boundary별 region-size + f_touch + droppable
scripts/measure_phase_network_synthesis.py   # 검증된 greedy phase-network 합성기
scripts/phase_reduction_audit.py             # f_touch 위상 weight/fuse/병합/합성 비용
```
