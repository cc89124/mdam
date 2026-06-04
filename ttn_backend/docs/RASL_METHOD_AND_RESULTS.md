# RASL: Rank-Aware Symplectic Localization

이 문서는 현재 `clifft-paper` 저장소에 구현된 RASL 방식의 동작 원리, 구현 범위, metric 정의, 실험 결과, 한계와 다음 작업을 정리한다.

현재 결론부터 쓰면 다음과 같다.

- 현재 RASL은 기본적으로 **fixed TTN layout 위에서 Pauli localization target을 바꾸는 bounded heuristic**이다.
- RASL v1의 주된 효과는 resident memory 감소가 아니라 **localization-induced path/refactor work 감소**다.
- proxy 분석에서는 `coherent_d5_r5`에서 14개 localization target 변경이 선택됐고, refactor proxy가 `2780 -> 2710`으로 감소했다.
- 실제 TTN 실행 실험에서는 `coherent_d5_r1`에서 executable RASL change 1개가 적용됐고, actual resident peak는 변하지 않았지만 rank-weighted path work는 `128.28 -> 114.49`로 감소했다.
- `coherent_d5_r5`는 60초 timeout window 안에서 accepted RASL step까지 도달하지 못했으므로, actual RASL effect는 아직 측정되지 않았다.

중요한 용어 구분:

- `resident_bound_proxy`: layout과 separator 기반의 compile-time proxy다. 실제 tensor memory가 아니다.
- `resident_actual_*`: TTN 실행 중 실제 bag tensor shape에서 측정한 값이다.
- `workspace_proxy_score`: path/region 기반 proxy다.
- `workspace_actual_peak_bytes`: 실제 transport/refactor 중 생성된 workspace tensor bytes다.

Proxy와 actual은 절대 같은 표에서 같은 의미로 비교하면 안 된다.

## 1. 문제 배경

Clifft는 near-Clifford 회로를 frame-factored 방식으로 컴파일한다. 많은 Clifford gate는 Pauli frame update로 흡수되지만, 일부 Pauli localization 과정에서는 active state 위에 실제 CNOT/CZ sequence가 남는다.

TTN backend에서는 이런 active multi-axis operation이 cross-bag이면 다음 비용을 만든다.

- path traversal
- adjacent 2-bag transport
- QR refactor
- bond dimension growth
- transient workspace tensor
- resident bag tensor size 증가

기존 Clifft default localization은 quantum simulation 관점에서는 타당하지만, TTN layout의 bond/rank cost를 고려하지 않는다. RASL의 목적은 이 localization freedom을 사용해 TTN 경로 비용이 낮은 target axis를 선택하는 것이다.

RASL의 목표는 전역 최적화가 아니다. 현재 구현은 다음 조건을 만족하는 bounded refinement pass다.

```text
Clifft default localization L0
-> fixed TTN layout T
-> RASL candidate generation/scoring on T
-> selected localization L1
```

선택 기준은 lexicographic이다.

```text
1. resident_bound_proxy
2. workspace_proxy_score
3. refactor_path_proxy
4. path_cost_proxy
```

그리고 default candidate는 항상 포함한다.

```text
selected resident_bound_proxy <= default resident_bound_proxy
```

즉 RASL v1은 resident memory를 줄인다고 주장하지 않는다. 먼저 resident proxy를 악화시키지 않는 범위에서 path/refactor work를 줄이는지 확인하는 pass다.

## 2. 현재 구현 파일 구조

관련 파일:

- `ttn_backend/rasl/symplectic.py`
  - binary symplectic Pauli vector engine
  - H/S/CNOT/CZ conjugation
  - phase tracking
- `ttn_backend/rasl/candidate.py`
  - `CliffordOp`
  - `LocalizationCandidate`
  - candidate verification
- `ttn_backend/rasl/builders.py`
  - active-only Z-normalize routing candidate builder
  - conservative Builder B placeholder
- `ttn_backend/rasl/cost.py`
  - fixed-layout proxy cost evaluator
- `ttn_backend/rasl/select.py`
  - resident-preserving candidate selection rule
- `ttn_backend/scripts/rasl_report.py`
  - analysis/profiling pass
  - per-step CSV + summary JSON 생성
- `ttn_backend/scripts/rasl_audit.py`
  - accepted changed step audit
- `ttn_backend/scripts/actual_rasl_experiment.py`
  - actual TTN tensor/bond metric experiment
  - conservative `RASL-exec-active-only` mode
- `ttn_backend/tests/test_rasl_symplectic.py`
  - phase-aware symplectic unit tests
- `ttn_backend/core.py`
  - actual resident/workspace/refactor instrumentation
  - conservative RASL executable override hook

## 3. Symplectic Pauli 표현

Pauli는 binary symplectic vector로 표현한다.

```text
P = (x, z) in F_2^(2N)

I = (0, 0)
X = (1, 0)
Z = (0, 1)
Y = (1, 1)
```

현재 `PauliVec`는 다음 정보를 갖는다.

```python
class PauliVec:
    x: np.ndarray[bool]
    z: np.ndarray[bool]
    phase: int
```

`phase`는 `i^phase`를 의미한다.

```text
0: +1
1: +i
2: -1
3: -i
```

지원하는 Clifford conjugation:

```text
H(q):
  x_q <-> z_q

S(q):
  z_q ^= x_q

CNOT(c -> t):
  x_t ^= x_c
  z_c ^= z_t

CZ(a, b):
  z_b ^= x_a
  z_a ^= x_b
```

단, phase는 단순 bit update만으로는 안전하지 않으므로 현재 구현은 gate별 truth table을 사용해 phase delta까지 검증한다. `ttn_backend/tests/test_rasl_symplectic.py`는 n <= 4 random Clifford sequence에 대해 brute-force matrix conjugation과 Pauli type 및 phase를 비교한다.

검증 명령:

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.tests.test_rasl_symplectic
```

현재 결과:

```text
RASL symplectic tests passed
```

## 4. Candidate 모델

RASL candidate는 한 localization step에서 default sequence를 대체할 수 있는 Clifford sequence다.

주요 필드:

```python
@dataclass
class LocalizationCandidate:
    step_id: int
    kind: str
    target_axis: int | None
    ops: list[CliffordOp]
    final_pauli_type: str | None

    proxy_path_cost: float
    proxy_workspace: float
    proxy_resident_bound: float
    refactor_cost: float

    valid: bool
    reject_reason: str | None
```

Candidate는 선택 전에 반드시 symplectic engine으로 검증한다.

```python
out = apply_ops(mapped_pauli, candidate.ops)
candidate.valid = out.weight() == 1
```

즉 candidate sequence를 적용한 뒤 mapped Pauli가 single-axis Pauli가 되지 않으면 invalid다.

현재 RASL report에서 다루는 mapped Pauli는 기존 bytecode의 active CNOT / MULTI_CNOT localization window를 재구성한 것이다. 특히 `OP_ARRAY_MULTI_CNOT`은 다음 형태로 해석한다.

```text
target = axis_1
mask bits = controls
default V = product CNOT(control -> target)
```

예:

```text
support = {0, 2, 4}
default target = 0
default V = CNOT(2,0) CNOT(4,0)
```

RASL은 같은 Pauli support를 single-axis로 localize하되 target을 바꾼 sequence를 후보로 만든다.

```text
chosen target = 2
chosen V = CNOT(0,2) CNOT(4,2)
```

## 5. Candidate Builder A: active-only Z-normalize routing

Builder A는 support가 전부 active axis일 때만 사용한다.

조건:

```text
support(mapped Pauli) subset active_set
has_dormant == False
```

현재 분석 대상 bytecode에서는 주로 Z-string 형태를 다룬다. 일반적인 active-only Pauli에 대해서는 local H/S로 Z-normalization을 먼저 한다.

```text
X -> H -> Z
Y -> S, H -> Z
Z -> no-op
```

그 뒤 target axis를 하나 고르고, support의 다른 축을 target으로 fold한다.

Direct-star candidate:

```text
for q in support:
    if q != target:
        CNOT(q -> target)
```

예:

```text
support = {a, b, c}
target = b

candidate:
  CNOT(a -> b)
  CNOT(c -> b)
```

이 candidate는 Z-string localization에서는 자연스럽다. Z control/target conjugation을 보면 여러 Z support가 target 하나로 합쳐진다.

현재 구현에서는 `active_z_route_star`가 주로 accepted candidate를 만든다.

## 6. Builder B 상태

원래 계획의 Builder B는 mixed/dormant-aware symplectic pivot elimination이다. 하지만 현재 구현에서는 보수적으로 제한되어 있다.

현재 Builder B 동작:

- support가 dormant/mixed이면 executable candidate를 만들지 않는다.
- active-only이면 Builder A family로 위임한다.

이유:

- dormant axis에 대한 H/S/CNOT는 active geometry와 promotion 여부를 바꿀 수 있다.
- measurement/rotation sign까지 바꾸는 phase-aware executable rewrite가 완전히 연결되지 않으면 조용히 틀릴 수 있다.

따라서 현재 RASL 결과는 사실상 **active-only localization target retargeting** 실험이다.

## 7. Proxy cost model

RASL analysis pass는 fixed TTN layout 위에서 candidate를 scoring한다.

Layout에서 각 ident는 home bag을 갖고, bag tree edge는 separator size를 갖는다.

### 7.1 Resident bound proxy

각 bag B에 대해:

```text
resident_exp_proxy(B)
  = own_count(B) + sum_{e incident to B} separator_bits(e)
```

전역 resident proxy:

```text
resident_bound_proxy
  = max_B resident_exp_proxy(B)
```

현재 RASL candidate는 localization target만 바꾸고 fixed layout 자체를 바꾸지 않는다. 따라서 대부분의 step에서 candidate별 `resident_bound_proxy`는 동일하다.

중요:

```text
resident_bound_proxy != actual resident memory
```

실제 resident memory는 TTN 실행 중 bond dimension이 실제로 얼마나 자랐는지에 따라 결정된다.

### 7.2 Path cost proxy

2-qubit op `(u, v)`에 대해:

```text
path_cost(u, v)
  = sum separator_bits(e) over TTN path(home(u), home(v))
```

Candidate path cost:

```text
proxy_path_cost(candidate)
  = sum path_cost(op.a, op.b) over 2q ops in candidate
```

### 7.3 Workspace proxy

2-qubit op의 path region R에 대해:

```text
workspace_proxy(R)
  = sum_{B in R} own_count(B)
    + sum_{e in boundary(R)} separator_bits(e)
```

Candidate workspace:

```text
proxy_workspace(candidate)
  = max workspace_proxy(op) over 2q ops in candidate
```

### 7.4 Refactor proxy

현재 구현의 간단한 refactor proxy:

```text
refactor_cost = proxy_path_cost + num_2q_ops
```

이 값은 실제 QR time이 아니다. 실제 실행 metric은 `ttn_backend/scripts/actual_rasl_experiment.py`에서 별도로 측정한다.

## 8. Selection rule

현재 선택 rule:

```python
feasible = [
    c for c in [default] + valid_candidates
    if c.proxy_resident_bound <= default.proxy_resident_bound
]

chosen = min(
    feasible,
    key=lambda c: (
        c.proxy_resident_bound,
        c.proxy_workspace,
        c.refactor_cost,
        c.proxy_path_cost,
        c.num_2q_ops(),
        c.kind,
    )
)
```

그 후 추가 acceptance 조건:

```text
chosen != default
chosen resident proxy <= default resident proxy
chosen refactor proxy <= default refactor proxy
chosen path proxy <= default path proxy
그리고 workspace/refactor/path 중 하나는 실제 개선
```

따라서 RASL은 target 변경이 가능하더라도 path/refactor/workspace proxy가 개선되지 않으면 default를 유지한다.

## 9. Analysis-only RASL report

명령:

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.rasl_report --enable-rasl \
  distillation cultivation_d3 cultivation_d5 coherent_d3_r1 coherent_d3_r3 \
  coherent_d5_r1 coherent_d5_r5 \
  --rasl-max-steps 200 \
  --rasl-max-support 10 \
  --rasl-builder full \
  --rasl-global-rollback \
  --out-csv reports/rasl_steps_full.csv \
  --out-json reports/rasl_summary_full.json
```

결과:

| circuit | considered | changed | resident bound proxy | workspace proxy peak | refactor proxy |
|---|---:|---:|---:|---:|---:|
| distillation | 6 | 0 | 5 -> 5 | 6 -> 6 | 12 -> 12 |
| cultivation_d3 | 31 | 0 | 7 -> 7 | 8 -> 8 | 115 -> 115 |
| cultivation_d5 | 104 | 1 | 20 -> 20 | 25 -> 25 | 832 -> 778 |
| coherent_d3_r1 | 8 | 0 | 4 -> 4 | 4 -> 4 | 12 -> 12 |
| coherent_d3_r3 | 44 | 0 | 13 -> 13 | 14 -> 14 | 108 -> 108 |
| coherent_d5_r1 | 27 | 1 | 17 -> 17 | 23 -> 23 | 352 -> 339 |
| coherent_d5_r5 | 200 | 14 | 120 -> 120 | 158 -> 158 | 2780 -> 2710 |

해석:

- Resident proxy는 모든 회로에서 유지됐다.
- Workspace proxy peak도 유지됐다. 이는 global peak offender가 바뀌지 않았다는 뜻이다.
- Refactor proxy는 일부 회로에서 감소했다.
- `coherent_d5_r5`에서 14개 target 변경이 발생했다.

## 10. `coherent_d5_r5` changed-step audit

Audit 파일:

```text
reports/rasl_changed_audit_coherent_d5_r5.json
```

대표 패턴 1:

```text
step_id: 392
support: 0 2 4
default target: 0
chosen target: 2
builder: active_z_route_star

default V:
  CNOT(2,0) CNOT(4,0)

chosen V:
  CNOT(0,2) CNOT(4,2)

default edge hits:
  0-4:2

chosen edge hits:
  0-4:1

reduced edge:
  0-4:2->1

refactor proxy delta:
  +7
```

이 패턴은 여러 step에서 반복된다.

```text
392, 824, 1090, 1391, 1630, 1968, 2259 ...
```

대표 패턴 2:

```text
support: 8 10 22
default target: 8
chosen target: 10

default:
  CNOT(10,8) CNOT(22,8)

chosen:
  CNOT(8,10) CNOT(22,10)

edge hit:
  0-1:2 -> 0-1:1
```

의미:

- RASL은 큰 separator/path를 덜 여러 번 건드리는 target으로 바꾸고 있다.
- 하지만 이건 proxy edge-hit 기준이다.
- 실제 bond dimension이 줄었는지는 actual execution으로만 확인할 수 있다.

## 11. Actual TTN instrumentation

`ttn_backend/core.py`에 실제 metric을 추가했다.

각 시점마다 bag tensor에서 직접 계산한다.

```python
numel_B = bag.tensor.size
bytes_B = bag.tensor.nbytes
log2_numel_B = log2(numel_B)
```

Actual resident peak:

```text
resident_actual_peak_log2_numel
  = max_t max_B log2(numel(T_B(t)))

resident_actual_peak_bytes
  = bytes of the peak offender bag tensor
```

기록하는 offender 정보:

```text
actual_peak_offender_bag
actual_peak_offender_step
actual_peak_offender_shape
actual_peak_offender_p_B
actual_peak_offender_incident_bond_dims
actual_peak_offender_incident_edge_ids
```

Workspace actual:

```text
workspace_actual_peak_bytes
  = max transient theta tensor bytes during transport/refactor
```

Refactor/path actual:

```text
num_path_contract
num_center_move
num_qr
num_svd
num_refactor
sum_path_length
sum_rank_weighted_path_length
sum_refactor_input_numel
max_refactor_input_numel
```

Edge diagnostic:

```text
edge_max_bond_dim
edge_hit_count
edge_rank_weighted_hits
```

이제 proxy와 actual은 report에서 분리된다.

## 12. Conservative executable RASL experiment

완전한 RASL executable rewrite는 아직 없다. 대신 최소 안전 subset을 구현했다.

이름:

```text
RASL-exec-active-only
```

조건:

```text
active_only == True
has_dormant == False
builder_kind starts with active_z_route
chosen V sequence consists only of CNOTs
fallback to default on unsafe
```

현재 구현 방식:

- `ttn_backend/scripts/rasl_report.py`가 만든 accepted step의 `chosen_v_sequence`를 읽는다.
- 해당 step이 `OP_ARRAY_MULTI_CNOT`이면 default CNOT sequence 대신 chosen CNOT sequence를 실행한다.
- 그 외 op는 default dispatch를 유지한다.
- 같은 seed에서 default record와 RASL record가 일치하는지 확인한다.

주의:

- 이것은 Clifft C++ bytecode를 재컴파일하는 방식이 아니다.
- Python TTN backend dispatch 중 해당 localization window만 override하는 실험 경로다.
- RASL change가 timeout window 안에서 실제로 적용되지 않으면 actual RASL effect는 측정되지 않은 것으로 표시한다.

## 13. Actual experiment 결과

명령 예:

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.actual_rasl_experiment \
  distillation cultivation_d3 coherent_d3_r1 coherent_d5_r1 \
  --enable-rasl-exec-active-only \
  --runtime-timeout 60 \
  --out-csv reports/actual_rasl_comparison_small.csv \
  --out-json reports/actual_rasl_comparison_small.json \
  --out-md reports/actual_rasl_report_small.md

/home/jung/clifft_env/bin/python -m ttn_backend.scripts.actual_rasl_experiment coherent_d5_r5 \
  --enable-rasl-exec-active-only \
  --runtime-timeout 60 \
  --out-csv reports/actual_rasl_comparison_d5r5.csv \
  --out-json reports/actual_rasl_comparison_d5r5.json \
  --out-md reports/actual_rasl_report_d5r5.md
```

Combined result:

```text
reports/actual_rasl_comparison.csv
reports/actual_rasl_comparison.json
reports/actual_rasl_report.md
```

요약:

| circuit | RASL actual available | executable changes | analysis changes | default actual log2 | RASL actual log2 | resident delta | correctness |
|---|---:|---:|---:|---:|---:|---:|---|
| distillation | no | 0 | 0 | 4 | null | null | null |
| cultivation_d3 | no | 0 | 0 | 4 | null | null | null |
| coherent_d3_r1 | no | 0 | 0 | 4 | null | null | null |
| coherent_d5_r1 | yes | 1 | 1 | 12 | 12 | 0 | pass |
| coherent_d5_r5 | no | 0 | 14 | 23 | null | null | not measured |

`coherent_d5_r1` actual 비교:

| metric | default | RASL-exec-active-only | delta |
|---|---:|---:|---:|
| resident actual peak log2 numel | 12 | 12 | 0 |
| resident actual peak bytes | 65536 | 65536 | 0 |
| workspace actual peak bytes | 65536 | 65536 | 0 |
| QR count | 246 | 244 | -2 |
| refactor count | 43 | 42 | -1 |
| path length sum | 67 | 64 | -3 |
| rank-weighted path length | 128.276 | 114.492 | -13.785 |
| correctness | pass | pass | - |

해석:

```text
coherent_d5_r1:
  actual resident memory decreased? no
  actual workspace peak decreased? no
  actual path/refactor work decreased? yes
  actual runtime decreased? no clear claim; one-shot time was slightly slower due instrumentation/noise
```

Edge diagnostic for `coherent_d5_r1`:

```text
edge 0-1:
  default max chi = 62
  RASL max chi = 64
  hit count = 19 -> 18
  rank-weighted hits = 73.276 -> 63.492

edge 1-4:
  max chi = 8 -> 8
  hit count = 10 -> 9
  rank-weighted hits = 15 -> 13

edge 3-4:
  max chi = 8 -> 8
  hit count = 10 -> 9
  rank-weighted hits = 15 -> 13
```

즉 `coherent_d5_r1`은 다음 case다.

```text
Case A/B mixed:
  hit_count와 rank-weighted work는 줄었다.
  global resident peak는 unchanged.
  일부 max chi는 unchanged 또는 약간 증가했다.
```

따라서 이 결과만으로는 RASL이 actual memory optimization이라고 말할 수 없다.

## 14. `coherent_d5_r5` actual 상태

`coherent_d5_r5` default actual partial result:

```text
status: timeout
resident_actual_peak_log2_numel: 23
resident_actual_peak_bytes: 134217728
workspace_actual_peak_bytes: 134217728
num_qr: 432
num_refactor: 93
sum_path_length: 106
sum_rank_weighted_path_length: 137.095
```

Peak offender:

```text
bag: 0
p_B: 9
shape includes:
  physical axes: 2^9
  major bonds: 4, 16, several 2s
```

RASL actual:

```text
not measured
```

이유:

```text
60초 timeout/partial execution window 안에서 executable RASL accepted step까지 도달하지 못했다.
num_rasl_executable_changes = 0
analysis changes = 14
```

따라서 `coherent_d5_r5`에 대해서는 현재 다음 말만 가능하다.

- Proxy 분석상 RASL은 14개 target 변경을 선택한다.
- 그 변경들은 edge hit/refactor proxy를 줄인다.
- 하지만 actual resident/bond effect는 아직 측정되지 않았다.
- 이를 보려면 accepted step 주변 window replay 또는 checkpoint/restart 방식이 필요하다.

## 15. 현재 답할 수 있는 질문

### Q1. RASL이 actual resident TTN memory를 줄였나?

현재 측정된 executable case인 `coherent_d5_r1`에서는 아니다.

```text
resident_actual_peak_log2_numel: 12 -> 12
resident_actual_peak_bytes: 65536 -> 65536
```

`coherent_d5_r5`는 actual RASL effect가 측정되지 않았다.

### Q2. RASL이 어떤 edge의 actual bond dimension을 줄였나?

`coherent_d5_r1`에서는 뚜렷한 max chi 감소는 관측되지 않았다. 주요 edge의 hit/rank-weighted work는 줄었지만 max bond는 유지되거나 약간 증가했다.

예:

```text
edge 0-1:
  max chi 62 -> 64
  hit 19 -> 18
  rank-weighted 73.276 -> 63.492
```

### Q3. Local bond 변화가 global peak memory를 바꿨나?

현재 측정된 case에서는 global peak memory를 바꾸지 않았다.

### Q4. RASL이 actual path/refactor work를 줄였나?

`coherent_d5_r1`에서는 yes.

```text
QR count: 246 -> 244
refactor count: 43 -> 42
sum path length: 67 -> 64
sum rank-weighted path length: 128.276 -> 114.492
```

### Q5. RASL이 runtime을 줄였나?

현재 한 shot + instrumentation 결과로는 주장할 수 없다.

```text
default elapsed: 0.116s
RASL elapsed: 0.128s
```

이 차이는 Python overhead, instrumentation, noise 영향이 크다. Runtime claim을 하려면 다중 shot 반복과 instrumentation-off timing이 필요하다.

### Q6. 결과 correctness는 유지됐나?

`coherent_d5_r1`의 conservative executable RASL subset은 같은 seed에서 default record와 일치했다.

하지만 이것은 전체 Clifft executable rewrite correctness를 의미하지 않는다. 현재는 Python TTN backend의 일부 `OP_ARRAY_MULTI_CNOT` dispatch override에 대한 검증이다.

## 16. 현재 한계

### 16.1 Full executable RASL이 아니다

현재 `RASL-exec-active-only`는 accepted active-only CNOT sequence 일부만 실행한다.

아직 미구현:

- Clifft C++ compiler localization decision hook
- full bytecode re-emission
- rotation angle/sign rewrite
- measurement sign/parity rewrite
- dormant/mixed support symplectic elimination
- global retrace/recompile

### 16.2 Builder B가 아직 실질적으로 작동하지 않는다

현재 Builder B는 conservative placeholder다. dormant/mixed support를 실제로 최적화하지 않는다.

### 16.3 Resident memory 감소는 layout 문제에 더 가깝다

RASL은 target selection으로 path work를 줄일 수 있지만, resident peak는 다음 요인이 지배한다.

- hub bag degree
- incident bond product
- exact QR bond growth
- bag ownership layout
- active lifetime geometry

따라서 RASL만으로 resident peak가 줄어들 것을 기대하면 안 된다. 현재 paper claim은 다음처럼 제한해야 한다.

```text
RASL is a budgeted, rank-aware symplectic localization refinement that
preserves resident-memory proxy while reducing localization-induced
path/refactor work.
```

Actual resident memory improvement은 별도 결과가 나오기 전까지 주장하지 않는다.

## 17. 다음 작업

### 17.1 Windowed actual replay for `coherent_d5_r5`

`coherent_d5_r5`의 accepted RASL steps는 392, 824, 1090, ...에 있다. 60초 partial run에서는 executable change가 적용되지 않았다. 다음 중 하나가 필요하다.

1. 긴 timeout으로 full/longer run
2. checkpoint after default prefix
3. accepted step 주변 window replay
4. deterministic state snapshot/restart

목표는 다음을 실제로 측정하는 것이다.

```text
edge 0-4 hit/rank/bond before vs after selected target change
edge 0-1 hit/rank/bond before vs after selected target change
global resident peak change 여부
```

### 17.2 Instrumentation-off timing

현재 elapsed는 instrumentation 포함이다. Runtime claim을 위해서는 다음 두 모드가 필요하다.

```text
actual metric mode: detailed logging on
timing mode: minimal counters only
```

### 17.3 Real compiler integration

최종적으로는 Python dispatch override가 아니라 Clifft localization lowering 단계에 선택지를 주입해야 한다.

필요 조건:

- phase-aware final Pauli result
- rotation sign/angle handling
- measurement sign handling
- dormant support safety
- fallback on unsafe candidate
- full retrace/recompile

### 17.4 Layout + RASL combined evaluation

RASL은 resident memory 최적화가 아니라 path/refactor 최적화에 가깝다. Resident peak를 줄이려면 layout transform과 함께 봐야 한다.

권장 순서:

```text
1. baseline layout actual
2. memory-risk-aware layout actual
3. same layout + RASL actual
4. compare:
   - resident actual
   - workspace actual
   - max chi per edge
   - hit count
   - rank-weighted path work
   - QR/refactor count
```

## 18. 현재 paper-facing 해석

현재 데이터로 쓸 수 있는 정직한 문장:

```text
RASL changes Pauli localization targets in a way that reduces fixed-layout
path/refactor proxy without increasing the resident-memory proxy.
```

actual 결과까지 포함하면:

```text
In the executable active-only subset on coherent_d5_r1, RASL preserved the
actual resident peak while reducing actual refactor/path work. The experiment
did not show resident memory reduction.
```

쓰면 안 되는 문장:

```text
RASL reduces TTN memory.
RASL reduces bond dimensions globally.
resident_bound_proxy=120 means actual memory exponent is 120.
coherent_d5_r5 RASL actual improved memory.
```

아직 `coherent_d5_r5` actual RASL은 accepted change가 실행된 구간까지 측정되지 않았기 때문이다.

## 19. 산출물 목록

Analysis/proxy:

```text
reports/rasl_steps_full.csv
reports/rasl_summary_full.json
reports/rasl_changed_audit_coherent_d5_r5.json
reports/rasl_report.md
```

Actual:

```text
reports/actual_rasl_comparison.csv
reports/actual_rasl_comparison.json
reports/actual_rasl_report.md
reports/actual_default_summary_<circuit>.json
reports/actual_rasl_summary_<circuit>.json
reports/actual_edge_rank_diff_<circuit>.csv
```

Core implementation:

```text
ttn_backend/rasl/symplectic.py
ttn_backend/rasl/candidate.py
ttn_backend/rasl/builders.py
ttn_backend/rasl/cost.py
ttn_backend/rasl/select.py
ttn_backend/scripts/rasl_report.py
ttn_backend/scripts/rasl_audit.py
ttn_backend/scripts/actual_rasl_experiment.py
ttn_backend/core.py
ttn_backend/tests/test_rasl_symplectic.py
```
