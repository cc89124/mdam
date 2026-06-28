# MDAM 메모리 이득의 귀속(attribution) 분석 — squeeze 몫과 MDAM 순(net) 몫의 분리

**작성 목적.** "MDAM이 Clifft 대비 메모리를 N배 줄였다"는 보고서의 수치에서, 그 이득 중
**얼마가 Clifft 컴파일러가 이미 하는 일(squeeze 패스)의 몫이고 얼마가 MDAM 고유의 기여인지**를
회로별로 정량 분리한다. 단일 "N배" 헤드라인은 이 둘을 섞기 때문에 MDAM의 효과를 실제보다
크게 보이게 할 수 있다. 이 문서는 그 분해를 실측으로 못 박는다.

**한 줄 결론.**
> MDAM의 **고유** 메모리 이득은 회로 부류에 따라 **0배에서 2048배까지** 극단적으로 갈린다.
> 깨끗하게 MDAM 단독으로 큰 이득이 나오는 곳은 **R_Z 다라운드 표면부호(coherent_d5_r5 등)
> 한 부류뿐**이며, 그 외(R_Y, cultivation, off-axis)에서는 이득이 squeeze 몫이거나, 0이거나,
> 아예 실행 불가다. 따라서 "8192×" 같은 단일 수치를 MDAM novelty로 제시하면 부풀림이다.

---

## 1. 배경 — 왜 baseline 분리가 필요한가

### 1.1 "Clifft 대비"라는 비교의 함정

근사 없는 near-Clifford 시뮬레이터의 메모리는 **"동시에 살아 있는 magic(=non-Clifford) 축의 개수
k"** 가 결정한다. 상태 배열이 `2^k` 개의 복소 진폭을 들고 있어야 하기 때문이다. 따라서 모든
비교의 핵심 지표는 이 **활성 차원 k** 다.

MDAM은 "측정 구동(measurement-driven)"으로 각 측정 시점에 그 측정과 얽힌 회전만 골라
materialize하여 k를 줄인다고 주장한다. 그런데 이 비교가 공정하려면 **무엇과 비교하는지**가
명확해야 한다. Clifft 자체가 이미 활성 차원을 줄이는 컴파일 패스를 갖고 있기 때문이다.

### 1.2 squeeze 패스란 무엇인가 (`StatevectorSqueezePass`)

Clifft의 컴파일 단계 HIR 패스로, 회로를 실행하기 전에 게이트 순서를 두 방향으로 민다
(`src/clifft/optimizer/statevector_squeeze_pass.cc`):

1. **Sweep 1 — 측정을 왼쪽으로(앞으로) 당김:** `MEASURE`를 가능한 한 일찍 수행.
2. **Sweep 2 — non-Clifford(T/회전)를 오른쪽으로(뒤로) 밀어냄:** magic을 가능한 한 늦게 생성.

두 sweep 모두 **인접한 두 연산이 교환 가능할 때만(`can_swap`)** 자리바꿈하며,
**측정/기댓값 장벽(`EXP_VAL`)은 넘지 못한다.**

**효과:** magic을 늦게 만들고 측정을 일찍 하면 동시에 떠 있는 magic 축 수(k)와 큰 차원에
머무는 시간이 줄어든다. **즉 squeeze는 MDAM과 같은 아이디어("non-Clifford 늦추고 측정 당기기")의
국소·정적 버전이다.** 차이는:

| | squeeze (Clifft 내장) | MDAM |
|---|---|---|
| 시점 | 컴파일 타임(정적) | 실행 중(동적) |
| 범위 | 인접 게이트끼리만 swap | 측정마다 anticommutation cone 전체 |
| 장벽 | 측정 장벽 못 넘음 | 넘음 |

### 1.3 우리 baseline은 이미 squeeze가 켜져 있다 (코드 확인)

`compile_bounded`(우리가 Clifft를 부르는 유일한 경로)는 다음과 같이 부른다
(`nearclifford_backend/clifft_axis/bounded.py:46`):

```python
clifft.compile(stim_text, bytecode_passes=None)
```

`bytecode_passes=None`은 **bytecode 패스(게이트 fusion)만** 끈다. `hir_passes` 인자는 생략되어
기본값 `default_hir_pass_manager()`가 쓰이고, 그 안에 `StatevectorSqueezePass`가
`default_enabled=true`로 들어 있다(`src/clifft/optimizer/pass_registry.h:54`). 따라서
**우리가 "Clifft k"라 부르는 `prog.peak_rank`는 이미 squeeze를 거친 값**이다.

→ 보고서의 비교는 (naive eager가 아니라) **squeeze가 적용된 Clifft 대비**다. 좋은 소식이지만,
"그 위에서 MDAM이 얼마나 더 줄였는가"를 따로 떼어내지 않으면 squeeze 몫과 MDAM 몫이 섞인다.

---

## 2. 측정 방법 — 세 baseline (A)/(B)/(C)

같은 회로·같은 결정적(seed=1) 경로에서 세 가지를 측정한다.

| 코드 | 정의 | 측정 방법 |
|---|---|---|
| **(A) eager** | squeeze를 끈 Clifft (`hir_passes=None, bytecode_passes=None`) | 컴파일된 bytecode를 구조적으로 walk하여 활성 슬롯 수 추적 |
| **(B) squeeze** | 기본 Clifft = **우리 baseline** (`bytecode_passes=None`, HIR 기본) | 동일 구조적 walk |
| **(C) MDAM** | 측정 구동 cone materialization | 백엔드 실제 실행 + 측정별 (transient, resident) 기록 |

### 2.1 두 가지 메모리 지표

- **transient (peak rank) k:** 실행 중 동시에 살아 있는 magic 축의 **최댓값**. `2^k` = 최대
  배열 크기 = 그 회로가 돌 수 있느냐(실행 가능성)를 결정하는 지표.
- **integrated `Σ 2^rank`:** 매 연산마다 그 순간 보유한 `2^rank`를 더한 값 = 메모리×시간 적분.
  peak는 같아도 "큰 차원에 얼마나 오래 머무느냐"를 잡는다. A·B는 per-op 구조 walk, C는 동일
  per-op 프레임에서 측정별 (transient/resident)로 재구성 — 세 baseline 모두 같은 가중 방식.

### 2.2 활성 슬롯 추적 규칙 (구조적 walk)

bytecode를 순회하며 `OP_EXPAND*`(magic 축 birth)에서 +1, `OP_MEAS_ACTIVE*`(측정·소멸)에서 −1.
Clifft는 매 연산에서 `2^활성`을 메모리에 들고 있으므로 integrated는 모든 op에 대해 `2^활성`을 합산.
MDAM은 회전을 지연(defer)하므로 birth에서 resident가 늘지 않고, 측정 cone에서만 transient로
치솟았다가 survivor로 내려간다 — 이 궤적을 측정별 (transient, resident)로 재구성해 합산.

### 2.3 실행 가능성 가드 (off-axis 폭발 처리)

off-axis(R_X/R_Y) d5 회로는 MDAM의 cone이 폭발한다(아래 §5). `magic_cap = 2^20`을 걸어
MDAM이 그 이상을 materialize하려 하면 `MagicCapExceeded`로 빠르게 중단시켜 **INFEASIBLE**로
표시한다(가드 없이 돌렸을 때 한 회로가 8.4GB RSS·27분 CPU로 thrashing함을 확인하고 추가).

---

## 3. 결과 표 (실행 가능한 11개 회로)

> off-axis d5 4개(`coherent_rx_d5_r1/r5`, `coherent_ry_d5_r1/r5`)는 MDAM **실행 불가**이므로
> 이 표에서 제외했다(§5에서 별도로 다룸).

열 의미:
- `kA / kB / kC` = (A)eager / (B)squeeze=baseline / (C)MDAM의 **peak rank**.
- **`MDAM peak 순이득`** = `2^(kB − kC)` = squeeze-clifft 대비 MDAM이 **추가로** 줄인 배율
  (= MDAM 단독 기여, peak 메모리 기준). **이것이 핵심 열이다.**
- `squeeze integ` = `iA / iB` = squeeze가 줄인 integrated 배율(참고).
- **`MDAM integ 순이득`** = `iB / iC` = squeeze 위에서 MDAM이 **추가로** 줄인 integrated 배율.

| 회로 | 축 | kA (eager) | kB (squeeze=baseline) | kC (MDAM) | **MDAM peak 순이득** | squeeze integ | **MDAM integ 순이득** |
|---|---|--:|--:|--:|--:|--:|--:|
| coherent_d3_r1 | R_Z | 8 | 5 | 0 | **32×** (2⁵) | 2.8× | 10.6× |
| **coherent_d3_r3** | R_Z | 8 | 8 | 5 | **8×** (2³) | 1.7× | **16.5×** |
| coherent_d5_r1 | R_Z | 24 | 13 | 0 | **8192×** (2¹³) ⚠️ | 259× | 2540× |
| **coherent_d5_r5** | R_Z | 24 | 24 | 13 | **2048×** (2¹¹) | 1.5× | **3888×** |
| coherent_rx_d3_r1 | R_X | 17 | 14 | 11 | **8×** (2³) | 5.2× | 19.9× |
| coherent_rx_d3_r3 | R_X | 17 | 14 | 12 | **4×** (2²) | 4.8× | 7.9× |
| coherent_ry_d3_r1 | R_Y | 17 | 16 | 16 | **1× (없음)** | 2.8× | 9.5× |
| coherent_ry_d3_r3 | R_Y | 17 | 16 | 16 | **1× (없음)** | 2.8× | 8.5× |
| cultivation_d3 | T | 4 | 4 | 4 | **1× (없음)** | 1.0× | 2.0× |
| cultivation_d5 | T | 10 | 10 | 10 | **1× (없음)** | 1.1× | 2.3× |
| distillation | T | 5 | 5 | 4 | **2×** (2¹) | 1.2× | 3.8× |

⚠️ **coherent_d5_r1의 8192×는 과대해석 주의** — §4.2에서 상세히 설명.

---

## 4. 부류별 상세 해석

### 4.1 R_Z 다라운드 표면부호 — `coherent_d3_r3`, `coherent_d5_r5` ✅ 진짜 novelty

**이것이 MDAM이 깨끗하게 이기는 유일한 부류다.**

- `coherent_d5_r5`: `kA=24, kB=24` → **squeeze가 peak를 0만큼 줄였다(A=B).** 그런데 MDAM은
  `kC=13`으로 **24→13 = 2¹¹ = 2048× 줄였다.** 즉 이 peak 감소는 **squeeze가 한 게 0이고
  100% MDAM 몫**이다. integrated도 squeeze는 1.5×에 그치는데 MDAM이 그 위에서 **3888×**를 더 줄인다.
- `coherent_d3_r3`: 같은 패턴. `kA=kB=8`(squeeze peak 기여 0), MDAM `8→5 = 8×`, integrated 16.5×.

**왜 squeeze는 여기서 0이고 MDAM만 되는가?** 다라운드 QEC 회로의 peak는 **한 라운드 안에서
누적되는 magic**이 결정한다. squeeze의 인접 swap은 **측정 장벽을 못 넘어서** 라운드 경계 너머로
회전을 옮기지 못한다 — 그래서 peak를 못 깎는다. 반면 MDAM은 각 측정마다 그 측정과 반교환하는
회전들의 **ancestor-closure cone**만 골라 materialize하므로, 장벽에 걸리지 않고 한 측정당
필요한 최소 축만 띄운다. 이 구조적 차이가 2048×의 출처다.

> **이 부류의 수치는 부풀림이 아니다.** squeeze가 0을 기여하는 지점에서 MDAM이 전부 달성했고,
> 별도 분석에서 `coherent_d3_r3`의 모든 측정 core가 stabilizer-nullity 기준 **이미 최소
> rank(s_out=0)**임을 exact certificate로 확인했다(즉 미구현 최적화에 기대지 않은 실측 달성치).

### 4.2 R_Z 단라운드 — `coherent_d3_r1`, `coherent_d5_r1` ⚠️ 과대(squeeze + 회로 triviality)

겉보기 수치가 가장 큰(`d5_r1`의 8192×) 부류지만, 그 대부분은 MDAM 고유 기여가 아니다.

- `coherent_d5_r1`: `kA=24 → kB=13`(squeeze가 **이미 11비트, 절반을 깎음**) → `kC=0`.
  - MDAM의 `13→0`은 cone 영리함이 아니라 **이 회로에 측정 관련 magic이 0개**라서다.
    (별도 flush-meter에서 flush된 회전 0개 확인 — 사실상 Clifford 회로.)
  - 즉 "8192×"는 (squeeze 2048× = 24→13) × (회로가 trivial해서 13→0) 의 곱이고,
    **MDAM의 cone 메커니즘이 기여한 건 거의 없다.**
- `coherent_d3_r1`: 비슷하게 squeeze `8→5`, MDAM `5→0`(magic 0). 32× 중 상당 부분이
  squeeze + triviality.

> **이 부류의 큰 배율을 MDAM novelty로 제시하면 부풀림이다.** 보고서는 "단라운드 = 회로가
> 거의 Clifford"라는 단서를 함께 적어야 한다.

### 4.3 off-axis R_X (d3) — `coherent_rx_d3_r1/r3` 🟡 실재하나 작음

- MDAM peak 순이득 `4×~8×`(2~3비트), integrated `8~20×`. squeeze도 4.8~5.2× 기여.
- R_X 회전은 Z-대각이 아니라 cone에 X 성분이 들어와 축이 더 필요하다 — 그래서 R_Z만큼 못 줄인다.
  그래도 MDAM이 squeeze 위에서 추가 이득을 내는 것은 맞다(중간 정도).

### 4.4 off-axis R_Y (d3) — `coherent_ry_d3_r1/r3` ⚠️ peak 이득 0

- `kB=16 → kC=16`: **MDAM이 peak를 하나도 못 줄였다(1×).** R_Y는 cone이 X/Y 성분으로 가득 차
  측정마다 거의 모든 축이 필요해진다.
- integrated만 8~9× 이득이 있는데, 이는 "지연으로 큰 차원에 머무는 시간"을 줄인 효과일 뿐
  **peak(실행 가능성) 이득은 없다.** squeeze가 16~16(거의 0)인 것과 합치면 R_Y는 두 방법 모두
  peak를 거의 못 건드린다.

### 4.5 cultivation (T-게이트) — `cultivation_d3/d5` ⚠️ peak 이득 0

- `kA=kB=kC`(4=4=4, 10=10=10): **squeeze도 MDAM도 peak를 못 줄인다(1×).** integrated 이득도
  2~2.3×로 미미.
- cultivation은 r=k(활성 차원 = peak) 영역이라 줄일 여지 자체가 거의 없다. **MDAM의 메모리
  novelty가 이 부류엔 사실상 없다.**

### 4.6 distillation — 🟡 작음

- MDAM peak `5→4 = 2×`, integrated 3.8×. squeeze는 거의 0(1.2×). 작지만 MDAM 몫.

---

## 5. 한계 — off-axis d5는 MDAM 실행 불가 (제외된 4개 회로)

다음 4개는 **MDAM이 아예 돌지 못한다**(표에서 제외):

| 회로 | kA (eager) | kB (squeeze) | C (MDAM) |
|---|--:|--:|---|
| coherent_rx_d5_r1 | 49 | 38 | **INFEASIBLE** (|M| > 2²⁰에서 중단) |
| coherent_rx_d5_r5 | 49 | 38 | **INFEASIBLE** |
| coherent_ry_d5_r1 | 49 | 47 | **INFEASIBLE** |
| coherent_ry_d5_r5 | 49 | 47 | **INFEASIBLE** |

off-axis(X/Y) 회전이 거리 5에서 측정마다 cone에 X/Y 성분을 대량 끌어들여 materialize 차원이
폭발한다(가드 없이는 8GB+ thrashing). **Clifft는 squeeze 후 38~47로 정상 실행되지만 MDAM은
2²⁰조차 못 버틴다 — 이 부류에서 MDAM은 Clifft보다 오히려 나쁘다(이득 음수/불가).**
이는 메모리 절감 주장의 **적용 경계**이며, 보고서가 명시해야 할 한계다.

---

## 6. 귀속 분해의 산술 (왜 이렇게 나누는가)

전체 이득은 두 단계의 곱이다:

```
(A) eager  --squeeze-->  (B) baseline  --MDAM-->  (C)
            squeeze 몫              MDAM 순(net) 몫
```

- peak 기준: `전체 = 2^(kA−kC)`, 이 중 `squeeze 몫 = 2^(kA−kB)`, **`MDAM 몫 = 2^(kB−kC)`**.
- integrated 기준: `전체 = iA/iC`, `squeeze 몫 = iA/iB`, **`MDAM 몫 = iB/iC`**.

보고서가 보고해야 하는 정직한 수치는 **`MDAM 몫`(= baseline (B) 대비)** 이지, `전체`(= eager (A)
대비)가 아니다. (A) 대비를 MDAM 공으로 돌리면 squeeze가 한 일을 MDAM이 가져가는 셈이다.

**예시 — coherent_d5_r5 (peak):** kA=24, kB=24, kC=13.
- 전체 = 2^(24−13) = 2048×. squeeze 몫 = 2^(24−24) = 1× (= 0). **MDAM 몫 = 2^(24−13) = 2048×.**
- → 여기서는 전체 = MDAM 몫. squeeze가 0이라 부풀림 없음. ✅

**예시 — coherent_d5_r1 (peak):** kA=24, kB=13, kC=0.
- 전체 = 2^(24−0) = 16,777,216×. squeeze 몫 = 2^(24−13) = 2048×. MDAM 몫 = 2^(13−0) = 8192×.
- → "MDAM 8192×"는 맞지만 그 8192×는 **회로에 magic이 0이라** 0으로 떨어진 것. 전체 1677만×를
  MDAM 공으로 보이게 하면 큰 부풀림. ⚠️

---

## 7. 최종 판정

**MDAM의 고유(순) 메모리 이득은 부류에 따라 0배~2048배로 극단적으로 갈린다:**

| 부류 | MDAM peak 순이득 | 판정 |
|---|---|---|
| R_Z 다라운드 (d3_r3, d5_r5) | 8× ~ **2048×** | ✅ **진짜 novelty** (squeeze peak 기여 0, MDAM 단독) |
| R_Z 단라운드 (d3_r1, d5_r1) | 32× ~ 8192× | ⚠️ **과대** (squeeze + 회로 near-trivial) |
| off-axis R_X (d3) | 4× ~ 8× | 🟡 실재하나 작음 |
| off-axis R_Y (d3) | **1× (없음)** | ⚠️ peak 이득 없음, integrated만 |
| cultivation (T) | **1× (없음)** | ⚠️ 메모리 이득 사실상 없음 |
| distillation (T) | 2× | 🟡 작음 |
| off-axis d5 (rx/ry) | **실행 불가** | ❌ MDAM이 Clifft보다 나쁨 |

**정직한 한 줄:**
> MDAM의 깨끗한 메모리 novelty는 **R_Z 다라운드 표면부호** 한 부류이며, 그 증거는
> `coherent_d5_r5`의 peak **2048×**(= squeeze가 0을 기여하는 지점에서 MDAM이 24→13 달성)이다.
> 그 외 부류에서는 이득이 squeeze 몫이거나, 0이거나, 실행 불가다. 따라서 "8192×/4096×" 류의
> 단일 헤드라인은 회로 부류별로 갈라 적어야 정직하다.

---

## 8. 검증 한계와 재현

- **이 분석은 메모리(활성 차원)만** 측정한다. FLOP·wall-clock은 별도(다른 분석에서 d5_r5는
  FLOP도 Clifft를 이김을 확인). exactness(Born 확률 일치 ~1e-15, records bit-identical)는 앞선
  `coherent_d3_r3` 분석에서 별도 확인됨 — 즉 (C) MDAM의 수치는 정확한 시뮬레이션 위의 값.
- (A)/(B)의 Clifft 정확성은 Clifft 자체 보증에 의존(별도 재검증 안 함). integrated에서 A vs B는
  per-op 구조 walk, B vs C는 동일 per-op 프레임 — 각 비교는 내부 일관이나, A→B→C를 한 줄로
  곱할 때는 단위 차이에 유의.
- **재현 스크립트:** `/tmp/abc_sweep.py` (이 표를 생성). 실행:
  ```
  /home/jung/clifft_env/bin/python -u /tmp/abc_sweep.py
  ```
- **핵심 코드 위치:** squeeze 패스 `clifft/optimizer/statevector_squeeze_pass.cc`,
  registry `pass_registry.h:54`(default_enabled=true); baseline 컴파일
  `nearclifford_backend/clifft_axis/bounded.py:46`(`bytecode_passes=None`, HIR 기본=squeeze ON);
  MDAM 측정/cone `nearclifford_backend/clifft_axis/bounded.py:430`(measure_z),
  `nearclifford_backend/lazy.py:162`(`_core_indices` anticommutation closure).

---

## 9. transient / resident 분리 — "즉시 회수(immediate recovery)" 이득

§3의 peak는 **transient**(측정 중 순간 최댓값)였다. 프로토콜은 **resident**(측정과 측정 *사이*에
정착한 차원)도 따로 보라고 요구한다. 둘의 차이는 MDAM이 **측정 직후 측정 축을 즉시 drop**하여
회수하는 양이다.

| 회로 | B=squeeze peak | **C transient** | **C resident** | +1 spike (즉시 회수) | MDAM transient 몫 비율 (B−C_t)/(A−C_t) |
|---|--:|--:|--:|--:|--:|
| coherent_d3_r1 | 5 | 0 | 0 | 0 | 62% |
| **coherent_d3_r3** | 8 | 5 | 4 | 1 | **100%** |
| coherent_d5_r1 | 13 | 0 | 0 | 0 | 54% |
| **coherent_d5_r5** | 24 | 13 | 12 | 1 | **100%** |
| coherent_rx_d3_r1 | 14 | 11 | 10 | 1 | 50% |
| coherent_rx_d3_r3 | 14 | 12 | 11 | 1 | 40% |
| coherent_ry_d3_r1 | 16 | 16 | 15 | 1 | **0%** |
| coherent_ry_d3_r3 | 16 | 16 | 15 | 1 | **0%** |
| cultivation_d3 | 4 | 4 | 3 | 1 | — (A=C, 전체 0) |
| cultivation_d5 | 10 | 10 | 9 | 1 | — (A=C, 전체 0) |
| distillation | 5 | 4 | 3 | 1 | 100% |

**핵심 관찰 — resident = transient − 1 이 거의 보편적이다.** MDAM은 Born 직후 측정 축을
`_drop_localized`로 즉시 떨어뜨리므로, 측정 사이에 머무는 resident는 측정 중 transient보다 항상
1비트(2×) 낮다. **이 +1 회수는 transient cone 이득이 0인 부류에서도 살아 있다:**

- **R_Y(`ry_d3`)**: transient는 16=16으로 cone 이득 0(비율 0%)이지만, **resident는 16→15로 2×**.
  즉 R_Y에서 MDAM이 주는 유일한 메모리 이득은 이 "측정 축 즉시 drop"(transient cone이 아님).
- **cultivation**: transient 10=10(이득 0)이지만 resident 10→9로 2×. 마찬가지.

**transient 몫 비율(맨 오른쪽 열) = MDAM이 차지하는 peak 감소의 비율(비트 기준):**
- d3_r3, d5_r5 = **100%** (squeeze가 peak에 0 기여 → MDAM 단독).
- d5_r1=54%, d3_r1=62% (절반 이상이 squeeze + triviality).
- rx_d3 = 40~50% (squeeze와 MDAM이 나눠 가짐).
- ry_d3 = **0%** (transient는 squeeze가 한 1비트가 전부, MDAM transient 기여 0 — resident만).
- cultivation = 정의 불가(A=C, 누구도 transient를 못 줄임).

---

## 10. 프로토콜 요구사항별 대응 (재평가 체크리스트)

> 아래는 "baseline 분리 + attribution 강제 정량화" 프로토콜의 6개 요구에 대한 항목별 대응이다.
> 이전 결론을 옹호하지 않으며, 실제로 내 초기 주장("squeeze=0, MDAM 100%")은 **transient에만
> 맞고 integrated에선 틀렸음**을 §4·§9에서 스스로 정정했다.

**[1] 세 baseline 측정 — 충족.** (A)eager=`hir_passes=None`, (B)squeeze=기본 컴파일=**우리 실제
baseline**(코드로 확인, §1.3), (C)MDAM. 동일 회로·동일 결정적 경로. **(B)가 진짜 baseline임을
코드 레벨에서 못 박았다** — (A) 대비 이득을 MDAM 공으로 돌리는 오류를 차단.

**[2] transient/resident/integrated × A/B/C + 분해 — 충족.** §3(transient·integrated), §9(resident),
귀속 분해 = `squeeze 몫 = A−B`, **`MDAM 순몫 = B−C`**, 비율 `(B−C)/(A−C)`(§9 맨 오른쪽 열).
비율이 작은 회로(d5_r1 54%, rx_d3 40%, ry_d3 0%)는 **그대로 작다고 정직히 보고**.

**[3] (B)≠(C)가 갈리는 회로 — 존재함, 명시.** squeeze의 인접 swap이 못 잡고 MDAM의
ancestor-closure cone만 잡는 구조 = **R_Z 다라운드 QEC(d3_r3, d5_r5) 그 자체**. 측정 의존성으로
얽힌 anti-commute 회전들이 측정 장벽 양쪽에 흩어져 있어 squeeze가 못 모으고 MDAM cone이 모은다 →
**(B)−(C) = 3비트(d3_r3), 11비트(d5_r5)로 substantial.** 반대로 (B)≈(C)인 곳(R_Y, cultivation의
transient)은 그 사실을 결론으로 보고(§4.4–4.5): 그 부류에선 cone 이득이 없고 resident +1만 남는다.

**[4] 구현됨 vs 미구현(B\*_j) 분리 — 충족.** 측정된 모든 수치는 **구현된 두 메커니즘**의 결과다:
(i) delayed-birth(회전 지연 → cone만 materialize) = transient 이득의 출처, (ii) measured-axis drop
(`_drop_localized`) = resident의 +1 즉시 회수. **미구현 B\*_j(측정마다 resident까지 exact frame
reduction)에 기대지 않는다** — 별도 `coherent_d3_r3` 분석에서 모든 측정 core의
resident가 stabilizer-nullity 기준 **이미 최소(s_out=0)**임을 exact certificate로 확인했으므로,
B\*_j가 더 줄일 여지가 이 회로엔 없다. 부류별 net 이득의 소재:
- **R_Z 대각 다라운드**: transient(cone) + resident(+1), 둘 다 구현됨.
- **R_Y / cultivation**: transient parity(이득 0), resident +1만 — 즉 구현된 measured-axis-drop의 효과뿐.

**[5] 평가 한계 — 명시.**
- **메모리(활성 차원)만** 측정. FLOP·wall-clock 미측정(별도 분석에서 d5_r5는 FLOP도 우위).
- **stabilizer tableau와 Pauli localization은 구현되어 있다**(`Xc/Zc` frame, `_localize_to_Z`,
  AG 측정) — 따라서 multi-qubit Clifford 측정/cone이 과대평가되지 **않는다**. (C)의 cone 크기는
  실제 materialize된 값.
- **off-axis d5(rx/ry) 4개는 MDAM 실행 불가**(§5) — cone 폭발(>2²⁰). 이 부류에서 (C)는 측정
  자체가 불가하므로 표에서 제외. Clifft(A/B)는 38~47로 정상 → **MDAM이 더 나쁜 적용 경계**.
- exactness: (C) MDAM은 `coherent_d3_r3`에서 Born ~1e-15·records bit-identical로 검증(정확한
  시뮬레이션 위의 값). (A)/(B) Clifft는 Clifft 자체 보증에 의존(별도 재검증 안 함).

**[6] 최종 분류 + novelty 판정 — 아래.**

### 10.1 "MDAM의 핵심 기여는 정확히 무엇인가" (squeeze가 하는 것을 뺀 나머지)

MDAM이 squeeze 위에서 **추가로** 주는 것은 정확히 두 가지이며, 프로토콜의 (a)/(b)/(c)로 분류하면:

- **(a) 특정 회로 구조에서의 cone 우위 — 실재, 가장 큼.** R_Z 다라운드에서 측정 장벽을 넘는
  anticommutation cone이 squeeze의 인접 swap이 못 하는 peak 감소를 한다. 증거: d5_r5 transient
  **24→13(2048×, squeeze 몫 0)**, d3_r3 8→5(8×). **이것이 MDAM의 진짜 novelty.**
- **(b) resident/integrated 즉시 회수 우위 — 실재, 보편적이나 작음.** 측정 축 즉시 drop으로
  resident가 항상 transient−1(2×). transient 이득이 0인 R_Y/cultivation에서도 살아 있는 유일한
  이득. 단 크기는 1비트(2×)로 고정.
- **(c) 미구현 B\*_j에 거는 잠재 우위 — 해당 없음.** d3_r3에서 resident가 이미 최소(s_out=0)라
  B\*_j가 더 줄일 게 없음을 확인. **현재 수치는 (c)에 의존하지 않는다.**

### 10.2 novelty 실재 여부 — 정직한 한 줄 판정

> **실재한다. 단, 한 부류에 한정된다.** 증거는 `coherent_d5_r5`의 transient **24→13(2048×)**이며,
> 이는 squeeze가 peak에 0을 기여하는(장벽을 못 넘는) 지점에서 MDAM의 측정 장벽-넘는 cone이
> 단독으로 달성한 값이다. **그 외 부류에서는 novelty가 (b) 2× 즉시 회수로 축소되거나(R_Y,
> cultivation), squeeze+triviality에 섞이거나(단라운드 R_Z), 실행 불가(off-axis d5)다.** 따라서
> 단일 "8192×/4096×" 헤드라인은 부풀림이며, **"R_Z 다라운드에서 peak 8×~2048×, 그 외엔 2× 또는
> 없음 또는 불가"**로 부류를 갈라 보고하는 것이 정직하다.
