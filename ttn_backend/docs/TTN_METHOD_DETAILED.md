# Current TTN Backend Method

이 문서는 현재 `clifft-paper` 저장소의 TTN backend가 어떤 방식으로
Clifft bytecode를 실행하는지, 어떤 tensor 구조를 쓰는지, 어떤 metric으로
메모리를 해석하는지, 그리고 최근 실험 결과가 무엇을 의미하는지 정리한다.

핵심 결론은 다음과 같다.

- 현재 TTN backend는 Clifft의 compiled bytecode를 Python에서 재실행하는
  prototype simulator다.
- 저장 구조는 static junction tree 위의 tensor tree network다.
- 각 active ident는 정확히 하나의 bag tensor에 physical axis로 저장된다.
- cross-bag two-axis operation은 adjacent 2-bag transport sweep으로 처리한다.
- resident memory는 `sum_B tensor(B).nbytes`이고, bottleneck은 보통
  `p_B + sum log2 chi_e`가 큰 hub bag이다.
- RASL은 resident memory 최적화의 main pass가 아니라 path/refactor work를
  줄이는 보조 pass다.
- 현재 가장 유망한 방향은 특정 회로 손튜닝이 아니라 Clifft bytecode의
  structured active Clifford localization window를 memory-cap 기반
  persistent TTN region으로 실행하는 것이다.
- 단, 최신 실험 결과는 단순 persistent MULTI_CNOT만으로 전체 시간축 memory를
  보장하지 못한다는 점도 분명히 보여준다. 968-step prefix까지는 dense 대비
  약 8배 절감이 유지되지만, 1200-step prefix에서는 다른 fallback/path-contract
  경로가 peak가 되어 exact persistent-only가 dense보다 커진다.

## 0. Latest Status and Core Problem

현재 상황을 한 문장으로 쓰면 다음과 같다.

```text
persistent MULTI_CNOT fusion은 특정 병목을 크게 줄였지만,
전체 executor가 memory-capped scheduler가 아니기 때문에
병목이 다른 fallback/path-contract 경로로 이동한다.
```

즉 문제는 "TTN이 전혀 안 된다"가 아니다. 반대로 `OP_ARRAY_MULTI_CNOT`을
per-control CNOT으로 실행하던 병목은 실제 runtime에서 크게 줄었다. 하지만 현재
정책은 모든 cross-bag operation에 대해 동일한 memory cap을 강제하지 않는다.
따라서 persistent window가 성공한 구간 뒤에서 fallback path가 큰 workspace를
열면 peak memory가 다시 dense 근처 또는 dense 이상으로 올라간다.

### Verified Numbers

`coherent_d5_r5`, temporal-carving leaf layout, `target_ratio=4`, cap = 64 MiB,
dense active-state peak = 256 MiB 기준:

| prefix | policy | actual total peak | dense / actual | QR | transport | interpretation |
|---:|---|---:|---:|---:|---:|---|
| 839 | persistent | 33,633,376 B = 32.08 MiB | 7.98x | 895 | 318 | good, memory and work both reduced |
| 968 | persistent | 33,633,520 B = 32.08 MiB | 7.98x | 1490 | 512 | still good |
| 1200 | persistent | 278,048,128 B = 265.17 MiB | 0.97x | 2183 | 860 | exact policy fails memory-vs-dense |
| 1200 | persistent_svd, rtol=1e-4 | 162,786,080 B = 155.24 MiB | 1.65x | 2154 | 836 | approximate compression helps |
| 1200 | persistent_svd, rtol=1e-3 | 133,804,608 B = 127.61 MiB | 2.01x | 2154 | 836 | stronger compression helps more |
| 1200 | persistent_svd, rtol=1e-2 | 112,915,008 B = 107.68 MiB | 2.38x | 2154 | 836 | more aggressive, needs correctness study |

Important: `persistent_svd` is an approximation/numerical-rank policy. It is not the
same claim as exact persistent execution. It can be used as a feasibility direction,
but any paper/runtime claim using it must report the threshold and output error.

### 1200-Step Peak Diagnosis

The exact persistent-only peak at 1200 steps occurs at:

```text
step = 1130
peak bag = B72
actual total peak = 278,048,128 B
resident actual peak = 123,731,968 B
workspace actual peak = 123,731,968 B
peak stored bytes = 157,475,136 B
offender tensor shape = [2, 59, 1024, 64]
incident bond dims = [59, 1024, 64]
```

With a 64 MiB exact cap trace enabled, the same run records
`cap_infeasible_exact_count = 132`. The important point is that at the peak the
stored resident state alone is already above the cap:

```text
stored_bytes   = 154,316,160 B
workspace      = 123,731,968 B
cap            =  67,108,864 B
```

Therefore a selector that only chooses a smaller workspace is not sufficient.
Once resident stored memory has crossed the cap, the executor needs an exact
resident-reducing candidate, such as a different layout/subtree structure,
exact rank cleanup, or a staged schedule that avoids creating the large bonds.
If no such exact candidate exists, the step must be reported as
`CAP_INFEASIBLE_EXACT` for that cap. Approximate SVD truncation can reduce the
peak but must be reported separately.

For the same step, the baseline shape was:

```text
[2, 128, 1024, 64]
```

Thus persistent execution did reduce one bond (`128 -> 59`), but the remaining
`1024 * 64` bond product is still too large. This is why one optimization appears to
work and then another bottleneck becomes dominant.

### What Is Actually Broken

The broken part is not the isolated MULTI_CNOT fusion. That part works on the tested
prefixes. The broken part is the global execution policy:

```text
current:
  MULTI_CNOT windows are memory-capped and destructive-open.
  fallback path-contract / transport operations are not globally memory-capped.

needed:
  every cross-bag executor candidate must be selected under one concurrent memory cap.
```

Current executor candidates are effectively chosen by opcode-specific dispatch:

```text
OP_ARRAY_MULTI_CNOT -> persistent/fused region if possible
otherwise           -> batch/fallback/per-control/path transport
other cross-bag op  -> existing class B/C transport/path logic
```

This causes bottleneck migration:

```text
per-control MULTI_CNOT bottleneck
  -> fixed by persistent fusion
  -> fallback/path-contract peak becomes bottleneck
  -> optional SVD reduces that peak but introduces approximation and runtime cost
```

The next algorithmic target is therefore not "make the MULTI_CNOT window larger".
An experiment that included `OP_ARRAY_CNOT/CZ` inside persistent windows passed
same-record smoke tests but worsened d5_r5 memory because the region support grew.
This option is kept behind:

```text
TTN_PERSISTENT_INCLUDE_ARRAY_CLIFFORD=1
```

and is disabled by default.

### Required Next Abstraction

The next executor should be a memory-capped selector, not an opcode-specific fallback
chain:

```text
for each active/cross-bag operation or window:
    build candidate executors:
      1. local/direct
      2. adjacent transport
      3. path transport
      4. fused region
      5. smaller batched region
      6. destructive-open region
      7. optional SVD compression region

    reject candidates whose concurrent live memory exceeds cap:
      stored_outside_open_regions
    + open_region_tensor
    + temporary_workspace
    <= cap

    among feasible candidates, minimize:
      QR/transport/refactor work

    if no exact candidate satisfies the cap:
      either split further, mark the step as cap-infeasible,
      or use an explicitly approximate compression policy.
```

This is the current central problem. Without this selector, each new local optimization
can simply move peak memory to another execution path.

### Big-Edge Crossing Reduction

`staged transport` 이후 남은 큰 resident bond는 exact하게 임의로 낮출 수 없다.
예를 들어 `chi=2048`인 edge는 실제 Schmidt rank가 그만큼 필요한 구간에서는 유지해야 한다.
하지만 그 큰 edge를 지나는 transport/refactor 횟수는 줄일 수 있다.

Tree edge `e`가 active axes를 `L_e | R_e`로 나눌 때, 2축 op `(u,v)`가 양쪽에
갈라져 있으면 transport path는 `e`를 지난다. 따라서 다음 값은 layout/scheduling으로
줄일 수 있는 실행 비용이다.

```text
edge_hit_count(e)
edge_rank_weighted_hits(e) = sum over path crossings log2(chi_e)
```

진단 스크립트:

```bash
/home/jung/clifft_env/bin/python ttn_backend/scripts/big_edge_crossing_audit.py \
  coherent_d5_r5 \
  --metrics-json reports/staged_bench_d5r5_1200/staged_transport/coherent_d5_r5/coherent_d5_r5/carving_leaf_metrics.json \
  --out-dir reports/big_edge_crossing_audit_d5r5_1200
```

현재 `coherent_d5_r5` 1200-step staged run에서 상위 offender는:

```text
edge 73-74: max chi 512,  hit 33, rank-hit 239.8, MULTI_CNOT crossing controls 262
edge 72-73: max chi 2048, hit 29, rank-hit 230.0, MULTI_CNOT crossing controls 204
```

즉 큰-edge work는 `OP_ARRAY_MULTI_CNOT` target/control이 큰 edge 양쪽으로 반복해서
갈라지는 구조에서 주로 나온다. 다음 후보는:

```text
1. layout local search: high-rank edge crossing MULTI_CNOT support를 같은 쪽 subtree로 cluster
2. parking/lifetime scheduling: 한 번 건넌 ident를 즉시 home으로 되돌리지 않고 window 동안 유지
3. persistent window refinement: 큰 edge를 가르는 MULTI_CNOT step들을 더 큰 window 단위로 처리
```

이 최적화는 resident floor를 직접 없애는 방법이 아니다. resident floor는 block-store
resident streaming 또는 approximation 없이는 유지된다. Big-edge crossing reduction은
QR/transport/workspace materialization 횟수를 줄이는 보조 축이다.

### Independent Follow-Up Experiments

Two independent directions are now tracked.

#### A. Region-Local Clifford Frame Lifting

Goal:

```text
reduce materialization / QR / transport caused by Clifford-only windows
```

Implemented first building block:

```text
ttn_backend/clifford_frame.py
  RegionLinearFrame
  compose_cnot
  compose_multicnot
  compose_swap
  materialize_to_tensor

ttn_backend/tests/test_clifford_frame.py
```

Validation:

```text
random bitstring CNOT/SWAP/MULTI_CNOT equivalence: pass
random tensor materialization equivalence: pass, ||seq-frame|| < 1e-10
```

This is not yet integrated into full runtime. The intended conservative runtime
policy is:

```text
accumulate Clifford-only active windows in a region-local linear frame
materialize once at non-Clifford / measurement / unsupported boundary
```

This targets QR/transport/materialization count, not directly resident memory.

#### B. Cap-Triggered Bag Fission

Goal:

```text
reduce peak resident bag tensor after persistent fusion has moved the bottleneck
```

Offline feasibility experiment implemented:

```text
ttn_backend/scripts/bag_fission_offline.py
```

Run target:

```text
coherent_d5_r5, max_steps=1200, peak bag B72
old shape = [2, 59, 1024, 64]
old bytes = 123,731,968 B
```

Offline result:

| mode | tol | old log2 | best peak log2 | peak ratio | total ratio | error |
|---|---:|---:|---:|---:|---:|---:|
| exact | 0 | 22.883 | 21.000 | 3.688x | 3.681x | 0 |
| approx | 1e-4 | 22.883 | 21.000 | 3.688x | 3.681x | 5.62e-13 |
| approx | 1e-3 | 22.883 | 20.770 | 4.325x | 2.879x | 5.62e-13 |
| approx | 1e-2 | 22.883 | 20.329 | 5.872x | 3.908x | 5.62e-13 |

Best exact split:

```text
left  = [phys:9, bond:0-72]
right = [bond:72-73, bond:72-108]
rank  = 32
old peak bytes        = 123,731,968 B
new exact peak bytes  =  33,554,432 B
new exact total bytes =  33,614,848 B
```

Interpretation:

```text
B72 is not intrinsically dense at this snapshot.
The current runtime stores it in a bad one-bag tensor structure.
Local exact fission can reduce this offender by ~3.69x.
```

Runtime fission is not implemented yet. It should only be attempted after the
offline exact split is converted into a safe local bag surgery that updates
neighbor maps, bond axes, and routing invariants.

## 1. Scope

현재 backend의 목표는 Clifft 자체를 대체하는 production simulator가 아니다.
목표는 다음 네 가지다.

1. `clifft.sample`과 같은 measurement distribution을 재현한다.
2. Clifft bytecode의 active-state 부분을 TTN으로 실행한다.
3. 실제 tensor shape, bond dimension, QR/refactor cost를 계측한다.
4. paper용으로 static treewidth metric과 executable TTN memory 사이의 gap을
   정량화한다.

현재 구현은 exact baseline을 기본으로 한다. 즉 runtime path에서는 임의의
bond truncation이나 SVD approximation을 도입하지 않는다. 다만 별도 static
compression experiment에서는 고정 peak tensor 하나에 대해 numerical-rank
tolerance를 주고 압축 가능성을 실험한다. 이 실험은 full runtime 최적화가
아니라 feasibility test다.

### Current Direction: General Persistent Region Policy

현재 backend의 최적화 방향은 특정 QEC 회로 이름에 맞춘 rule이 아니다.
일반화 대상은 Clifft compiled bytecode에서 반복적으로 나타나는 다음 구조다.

```text
frame-only Clifford ops
+ active single-axis diagonal/non-Clifford rotations
+ OP_ARRAY_MULTI_CNOT localization windows
+ occasional measurement/U4/nonlocal boundary
```

이 구조는 `coherent_d5_r5` 하나에만 있는 것이 아니라 coherent/cultivation 계열
QEC benchmark에서 공통적으로 나타난다. 따라서 최적화 단위는 회로가 아니라
bytecode window다.

일반 policy는 다음과 같다.

```text
1. bytecode stream을 왼쪽에서 오른쪽으로 scan한다.
2. OP_ARRAY_MULTI_CNOT을 만나면 window 후보를 만든다.
3. window 내부에 다음 op만 허용한다.
   - frame-only Clifford update
   - region-local OP_ARRAY_ROT
   - 추가 OP_ARRAY_MULTI_CNOT
4. measurement, U4, unsupported nonlocal active op가 나오면 window를 닫는다.
5. 후보 region에 대해 memory cap을 평가한다.
6. cap 안이면 destructive-open persistent region으로 실행한다.
7. cap 밖이면 batch split 또는 기존 path fallback을 쓴다.
```

memory cap은 회로 이름으로 정하지 않는다. 기본 실험에서는 dense active-state
peak에서 자동 산출한다.

```text
dense_peak_bytes = 16 * 2^k_peak
cap_bytes = dense_peak_bytes / target_ratio
```

예를 들어 `target_ratio = 4`이면 dense peak의 1/4까지 region/window memory를
허용한다. `target_ratio = 8`이면 더 엄격한 cap이다.

### Destructive-Open Liveness

persistent region을 열 때 memory를 다음처럼 계산해야 한다.

```text
live_total =
    stored_outside_open_regions
  + open_region_tensor
  + temporary_workspace
```

즉 open region이 기존 bag tensors를 대체해야 한다. 기존 bag tensors와 open
region tensor를 동시에 live로 세면 dense 근처까지 올라가며 memory 이점이 사라진다.
현재 `TTN_DESTRUCTIVE_OPEN=1` 모드는 region 내부 bag tensors를 scalar placeholder로
detach해서 실제 Python runtime metric도 destructive-open liveness에 맞춘다.

### Peak-Aware Compression

모든 step에서 압축을 강제할 필요는 없다. 작은 tensor 구간에서는 QR/local path가
더 빠르다. 큰 peak-risk 구간에서만 비싼 compression을 쓰는 정책이 필요하다.

현재 runtime flag는 다음과 같다.

```text
TTN_FUSE_MULTICNOT=1
TTN_PERSISTENT_MULTICNOT=1
TTN_DESTRUCTIVE_OPEN=1
TTN_FUSE_MULTICNOT_BATCH=1
TTN_FUSE_MULTICNOT_CAP_BYTES=<cap>
TTN_FUSE_MULTICNOT_TOTAL_CAP_BYTES=<cap>

# optional approximation/compression
TTN_SVD_TRUNC_RTOL=1e-4
TTN_SVD_TRUNC_MIN_MATRIX_ELEMS=1048576
```

### General Sweep Script

`ttn_backend/scripts/qec_persistent_policy_sweep.py`는 이 policy를 여러 benchmark에
같은 방식으로 적용한다.

```bash
python ttn_backend/scripts/qec_persistent_policy_sweep.py \
  distillation cultivation_d3 coherent_d3_r1 coherent_d5_r1 \
  --target-ratio 8 \
  --policies baseline,persistent \
  --out-dir reports/qec_persistent_policy_sweep_smoke
```

`coherent_d5_r5`처럼 긴 회로는 prefix budget을 지정한다.

```bash
python ttn_backend/scripts/qec_persistent_policy_sweep.py coherent_d5_r5 \
  --target-ratio 4 \
  --d5r5-max-steps 839 \
  --policies baseline,persistent \
  --out-dir reports/qec_persistent_policy_sweep_d5r5_839_ratio4
```

현재 확인된 대표 결과:

```text
coherent_d5_r1 full:
  baseline actual_total_peak = 123,024 B
  persistent actual_total_peak = 44,240 B
  dense / persistent = 2.96x
  QR 1047 -> 613
  transport 574 -> 190

coherent_d5_r5 839-step prefix, target_ratio=4:
  baseline actual_total_peak = 142,730,336 B = 136.12 MiB
  persistent actual_total_peak = 33,633,376 B = 32.08 MiB
  dense / persistent = 7.98x
  QR 2254 -> 895
  transport 1336 -> 318
  persistent windows = 5

coherent_d5_r5 968-step prefix, target_ratio=4:
  persistent actual_total_peak = 33,633,520 B = 32.08 MiB
  dense / persistent = 7.98x
  QR = 1490
  transport = 512

coherent_d5_r5 1200-step prefix, target_ratio=4:
  persistent actual_total_peak = 278,048,128 B = 265.17 MiB
  dense / persistent = 0.97x
  exact persistent-only is worse than dense at this prefix

coherent_d5_r5 1200-step prefix, target_ratio=4, persistent_svd rtol=1e-2:
  actual_total_peak = 112,915,008 B = 107.68 MiB
  dense / actual = 2.38x
  this is approximate/numerical-rank compression and needs correctness/error reporting
```

이 결과는 특정 step hand-tuning이 아니라 동일한 bytecode-structured policy와
자동 cap 산출로 얻은 것이다. 하지만 1200-step 결과가 보여주듯, 아직 full
`coherent_d5_r5` 전체 시간축 결과는 아니다. 다음 과제는 단순 window coverage
확장이 아니라 fallback/path-contract까지 포함하는 global memory-capped executor
selection이다.

## 2. Directory Layout

TTN 관련 코드는 `ttn_backend/` package 아래로 정리되어 있다.

```text
ttn_backend/
  core.py
  backend_spec.py
  frame_layer.py
  treewidth.py
  layout_transform.py
  rasl/
    symplectic.py
    candidate.py
    builders.py
    cost.py
    select.py
  scripts/
    verify_ttn.py
    metrics_report.py
    memory_risk_report.py
    memory_diagnosis_report.py
    time_graph_report.py
    static_ttn_compression_experiment.py
    rasl_report.py
    rasl_audit.py
    actual_rasl_experiment.py
  tests/
    test_ttn_transport.py
    test_rasl_symplectic.py
  docs/
    TTN_BACKEND.md
    TTN_MEMORY_DIAGNOSIS.md
    RASL_METHOD_AND_RESULTS.md
    TTN_METHOD_DETAILED.md
```

핵심 runtime은 `ttn_backend/core.py`에 있다.

- `TTNBag`: bag 하나의 tensor와 metadata
- `TTNState`: 전체 TTN state와 tensor operation
- `TTNBackend`: Clifft bytecode dispatch, frame/noise/record 관리

정적 layout과 operation classification은 `ttn_backend/backend_spec.py`가
담당한다.

## 3. End-to-End Pipeline

현재 실행 흐름은 다음과 같다.

```text
Stim circuit
-> clifft.compile(stim_src)
-> Program bytecode
-> export_backend_spec(program)
-> assign_homes_and_classify(spec)
-> TTNBackend(spec, homing)
-> run_shot(program, seed)
-> measurement record
```

좀 더 자세히 쓰면 다음 단계다.

1. Clifft가 Stim circuit을 compile해서 `Program`을 만든다.
2. `backend_spec.export_backend_spec()`가 bytecode를 structural replay한다.
3. Structural replay 중 active ident lifecycle, two-axis interaction graph,
   measurement site, swap event를 기록한다.
4. union interaction graph에서 static junction-tree layout을 만든다.
5. 각 ident의 home bag을 정한다.
6. 각 two-axis operation을 Class A/B/C로 분류한다.
7. `TTNBackend.run_shot()`이 bytecode를 순서대로 dispatch한다.
8. Active-state operation은 `TTNState`의 tensor operation으로 적용한다.
9. Frame/noise/readout/record는 `frame_layer.py`와 backend dispatch가 관리한다.
10. 실행 중 actual tensor memory, workspace, QR/refactor metric을 기록한다.

현재 layout은 static union layout이다. 즉 전체 프로그램에서 발생할 수 있는
active interaction의 union graph를 기준으로 하나의 bag tree를 만든다.
시간별 live graph에 따라 layout 자체가 바뀌지는 않는다.

## 4. Ident, Slot, Bag

Clifft bytecode의 `axis_1`, `axis_2`는 runtime slot이다. 하지만 TTN backend는
slot을 그대로 tensor axis id로 쓰지 않는다. `OP_EXPAND`류 instruction에서
새 active identity를 만들고, 그 identity를 `ident`라고 부른다.

구분은 다음과 같다.

| 개념 | 의미 |
| --- | --- |
| slot | Clifft bytecode axis index. SWAP에 따라 어떤 ident가 들어있는지 바뀐다. |
| ident | active state에 promotion된 logical identity. lifecycle이 있다. |
| home bag | ident의 canonical physical axis가 저장되는 static bag. |
| own axis | bag tensor가 직접 들고 있는 ident의 physical dimension-2 axis. |
| bond axis | adjacent bag과 연결되는 TTN internal index. |

`TTNBackend`는 shot 시작 때 `slot2id = {}`로 시작한다. `OP_EXPAND`가 나오면
해당 step에 할당된 ident를 `slot2id[axis]`에 등록하고, 그 ident를 home bag에
physical axis로 추가한다. `OP_ARRAY_SWAP`은 tensor를 움직이지 않고
`slot2id` label과 Pauli frame만 swap한다.

## 5. TTNBag Tensor Convention

각 bag tensor의 axis order는 고정 convention을 따른다.

```text
[own_idents...] + [bond axes in sorted neighbor order]
```

예를 들어 bag `B3`의 `own_idents = [5, 9]`, neighbors가 `[0, 4, 7]`이면
tensor axis 의미는 다음과 같다.

```text
axis 0: physical ident 5
axis 1: physical ident 9
axis 2: bond B3-B0
axis 3: bond B3-B4
axis 4: bond B3-B7
```

`TTNBag.bond_axis_pos(neighbor_id)`는
`len(own_idents) + neighbors.index(neighbor_id)`를 반환한다.

초기 bag tensor는 모든 bond dimension이 1인 scalar/product 상태다.
bag에 neighbor가 있으면 shape은 `(1, 1, ..., 1)`이고, isolated bag이면
scalar tensor다.

## 6. Runtime Invariants

현재 exact TTN runtime은 세 가지 핵심 invariant를 유지한다.

```text
I1. 각 active ident는 정확히 하나의 bag.own_idents에만 존재한다.
I2. active ident u는 home[u] bag의 own_idents에 존재한다.
I3. 각 tree edge (i,j)에 대해 B_i와 B_j의 bond dimension이 일치한다.
```

단, adjacent transport sweep 중에는 ident 하나를 임시로 다른 bag으로 이동하기
때문에 I2가 일시적으로 깨진다. `apply_2q_class_B_path()`는 forward transport,
local gate, reverse transport를 끝낸 뒤 `check_all_invariants()`로 I1/I2/I3를
다시 확인한다.

이 design choice가 중요하다. `home` map은 static이다. cross-bag operation을
처리하기 위해 ident의 실제 physical axis를 잠깐 움직이지만, `home_of[u]`를
영구적으로 바꾸지는 않는다. `backend_spec.assign_homes_and_classify()`가
처음에 계산한 op classification이 전체 실행 내내 유효해야 하기 때문이다.

## 7. Backend Spec Construction

`backend_spec.export_backend_spec(program)`는 Clifft bytecode를 구조적으로 replay한다.
이 replay는 amplitude를 계산하지 않고, active ident의 lifecycle과 interaction
graph만 추적한다.

주요 output은 다음과 같다.

- `union`: static union junction-tree layout
- `peak`: Clifft structural analyzer가 본 peak snapshot 정보
- `op_to_bag`: 각 single/two-axis op가 어느 bag에 covered되는지
- `measurement_spec`: measurement ident와 marginal bag
- `lifecycle`: ident별 promote/demote step
- `swap_events`: active slot swap 기록
- `invariants`: two-axis coverage, running intersection 등

### 7.1 Structural Replay

Replay는 instruction opcode에 따라 다음 정보를 만든다.

- `OP_EXPAND*`: 새 ident 생성
- two-axis active op: union graph에 edge 추가
- multi-axis star op: hub-target과 mask bit 사이 edge 추가
- `OP_ARRAY_SWAP`: slot mapping swap
- active measurement: ident demote
- single-axis active op: single op 기록
- frame/noise/detector류: active graph 구조에는 영향 없음

이 단계에서 가장 중요한 invariant는 다음이다.

```text
모든 two-axis active operation (u,v)에 대해 {u,v}를 포함하는 bag이 존재해야 한다.
```

이 조건이 깨지면 TTN layout이 해당 bytecode를 실행할 수 없다.

### 7.2 Static Union JT Layout

현재 layout은 union interaction graph에서 만든 static junction tree다.
즉 시간별 active graph가 아니라 전체 프로그램 중 한 번이라도 필요해진
interaction을 모두 합친 graph를 기준으로 bag tree를 만든다.

장점:

- 구현이 단순하다.
- 모든 operation coverage를 정적으로 보장할 수 있다.
- `home`, `op_class`, path를 한 번만 계산하면 된다.

단점:

- 실제로 동시에 live하지 않는 bond까지 같은 hub bag에 붙을 수 있다.
- high-degree hub bag은 많은 bond axis를 동시에 갖게 된다.
- numpy ndarray dimension limit에도 걸릴 수 있다.
  `coherent_d7_r7` baseline은 B0 degree가 99라 초기 shape 차원 수부터
  numpy limit 64를 넘는다.

## 8. Home Assignment

`assign_homes_and_classify(spec)`는 각 ident에 대해 home bag을 정한다.
현재 heuristic은 다음이다.

```text
각 ident가 포함될 수 있는 candidate bag 중,
그 ident가 참여하는 two-axis op를 가장 많이 cover하는 bag을 home으로 선택한다.
tie는 작은 bag id 쪽으로 해결한다.
```

이 결과로 `owned_phys`가 만들어진다.

```text
owned_phys[bag_id] = [ident ids whose home is bag_id]
```

각 ident는 정확히 하나의 home을 갖는다. Junction tree의 여러 bag에 같은 ident가
포함될 수 있지만, physical axis는 home bag 하나에만 직접 저장한다. 다른 bag의
동일 ident 포함은 structural coverage용 vertex일 뿐, runtime physical axis가
복제되는 것은 아니다.

## 9. Operation Classification

Two-axis operation `(u,v)`는 다음 세 class로 분류된다.

### 9.1 Class A

```text
home(u) == compute_bag
home(v) == compute_bag
```

두 physical axis가 같은 bag에 있으므로 local 4x4 gate를 바로 적용한다.
QR transport가 필요 없다.

### 9.2 Class B

```text
정확히 한쪽 ident의 home이 compute_bag이다.
```

한쪽 axis는 operation을 cover하는 bag에 있고, 다른 한쪽은 다른 home bag에 있다.
현재 runtime은 두 home 사이 path를 따라 한 ident를 transport해서 gate를 적용한 뒤
되돌린다.

### 9.3 Class C

```text
home(u) != compute_bag
home(v) != compute_bag
```

두 ident 모두 operation을 cover하는 bag과 다른 곳에 home을 갖는다.
현재 구현에서는 home(u)-home(v) path를 따라 한 ident를 다른 ident 쪽으로
transport한다. 즉 Class B와 같은 adjacent 2-bag sweep primitive로 처리한다.

현재 backend에서 중요한 점은, Class B/C가 full-path contraction을 하지 않는다는
것이다. 과거 방식은 path 전체를 하나의 dense tensor로 만들었고, `2^(sum own axes)`
규모의 transient memory를 만들었다. 현재 방식은 adjacent 2-bag transport만
사용한다.

## 10. Active State Initialization

`OP_EXPAND`류 instruction은 dormant/frame 상태에서 active state axis를 만든다.

현재 구현의 기본 expand는 새 ident를 home bag의 own block 끝에 추가하고
`|+>` 상태로 초기화한다.

```text
T_new[..., x, bonds...] = T_old[..., bonds...] / sqrt(2)
for x in {0,1}
```

`OP_EXPAND_T`, `OP_EXPAND_T_DAG`, `OP_EXPAND_ROT`은 expand 직후 frame bit에 따라
diagonal phase를 적용한다. 예를 들어 T gate류는 `frame.x(axis)` 값에 따라
phase 또는 conjugate phase를 선택한다.

## 11. Single-Axis Active Operations

Single-axis operation은 ident의 home bag에서 local tensor update로 처리한다.

지원되는 주요 op:

- `OP_PHASE_T`
- `OP_PHASE_T_DAG`
- `OP_PHASE_ROT`
- `OP_ARRAY_H`
- `OP_ARRAY_S`
- `OP_ARRAY_S_DAG`
- `OP_ARRAY_T`
- `OP_ARRAY_T_DAG`
- `OP_ARRAY_ROT`
- `OP_ARRAY_U2`

Diagonal op는 axis slice에 scalar를 곱한다.

```text
axis value 0 -> multiply by c0
axis value 1 -> multiply by c1
```

General 2x2 unitary는 해당 axis를 마지막으로 옮긴 뒤 matrix multiply를 수행하고
axis를 원래 위치로 되돌린다.

`OP_ARRAY_U2`는 Clifft optimizer가 single-axis run을 fused한 op다. Constant pool의
frame-state-dependent 2x2 matrix를 선택해서 active axis에 적용하고, Clifft와 같은
out frame state로 Pauli frame을 갱신한다.

## 12. Two-Axis Active Operations

지원되는 주요 two-axis op:

- `OP_ARRAY_CNOT`
- `OP_ARRAY_CZ`
- `OP_ARRAY_MULTI_CNOT`
- `OP_ARRAY_MULTI_CZ`
- `OP_ARRAY_U4`

`OP_ARRAY_CNOT`과 `OP_ARRAY_CZ`는 op class를 조회한다.

```text
Class A: apply_2q_class_A(u, v, U4)
Class B/C: apply_2q_class_B_path(u, v, U4, path)
```

`OP_ARRAY_MULTI_CNOT`은 `axis_1`을 target slot으로 보고, mask bit들을 control slot로
해석한다. 각 control-target pair를 CNOT으로 풀어서 동일한 Class A/B/C dispatch를
사용한다. 각 CNOT 후 Pauli frame도 `frame.cnot(control, target)`으로 갱신한다.

`OP_ARRAY_MULTI_CZ`도 mask bit target들에 대해 CZ를 반복 적용한다.

`OP_ARRAY_U4`는 Clifft optimizer가 two-axis tile run을 fused한 op다. Constant pool의
frame-state-dependent 4x4 matrix를 선택한다. Clifft 내부 convention은 `|hi,lo>`
basis이고 `lo`가 least-significant bit이므로 runtime에서는 axis ordering을 맞추기
위해 `(ident_hi, ident_lo)` 순서로 4x4 gate를 적용한다. 적용 후 두 axis의 frame
state를 out state로 갱신한다.

## 13. Local 2Q Gate Application

두 ident가 같은 bag 안에 있으면 `_apply_2q_local()`이 실행된다.

절차:

1. canonical center를 해당 bag으로 이동한다.
2. 두 physical axis를 tensor 끝의 두 축으로 moveaxis한다.
3. tensor를 `(-1, 4)` matrix로 reshape한다.
4. 오른쪽에 `U4.T`를 곱한다.
5. 원래 tensor shape로 되돌린 뒤 axis 위치를 복원한다.

이때 4x4 basis ordering은 호출자가 맞춰야 한다. 일반 CNOT/CZ는 `(control,
target)` 순서로 호출한다. U4는 Clifft convention 때문에 `(hi, lo)` 순서를 사용한다.

## 14. Adjacent 2-Bag Transport Sweep

현재 TTN backend의 가장 중요한 primitive는
`transport_ident_across_edge(ident, src, dst)`다.

목표:

```text
src bag에 있는 physical axis ident를 adjacent dst bag으로 옮긴다.
home map은 바꾸지 않는다.
```

절차:

1. center를 `src`로 이동한다.
2. `src.tensor`와 `dst.tensor`를 shared bond 위에서 contraction한다.
3. resulting theta에서 `ident` axis를 dst side로 재분할한다.
4. theta를 matrix `M = left | right`로 reshape한다.
5. thin QR을 수행한다.
6. `Q`를 src tensor로 reshape한다.
7. `R`을 dst tensor로 reshape한다.
8. `src.own_idents`에서 ident를 제거하고 `dst.own_idents`에 정렬 삽입한다.
9. 새 bond dimension을 양쪽 bag tensor에 반영한다.
10. center는 `dst`가 된다.

현재 QR은 `_thin_qr()`을 사용한다. 이는 QR 후 `R`의 numerically-zero row를
floating-point tolerance 기준으로 제거한다. 이것은 bond cap이나 approximate
truncation이 아니라, QR 결과에서 드러난 zero-rank row 제거다.

`apply_2q_class_B_path(u, v, U4, path)`는 이 primitive를 사용한다.

```text
for i = 0..k-1:
    transport u: B_i -> B_{i+1}

apply local U4 to u,v in B_k

for i = k-1..0:
    transport u: B_{i+1} -> B_i
```

이 방식의 장점은 path 전체를 한 번에 contract하지 않는다는 점이다. transient
workspace는 path 전체 tensor가 아니라 adjacent 2-bag merged tensor 크기로 제한된다.

다만 이 제한이 resident memory를 자동으로 줄인다는 뜻은 아니다. transport와 QR을
반복하면서 bond dimension이 커지면 hub bag의 tensor가 커진다.

## 15. Measurement

Active Z measurement는 다음 절차로 처리한다.

1. center를 measured ident의 home bag으로 이동한다.
2. 해당 axis가 1인 slab의 norm으로 `p1`을 계산한다.
3. RNG로 outcome을 sample한다.
4. outcome slice를 선택하고 `sqrt(p)`로 normalize한다.
5. measured ident axis를 tensor에서 제거한다.
6. `slot2id`에서 해당 slot을 삭제한다.
7. frame bit와 sign을 반영해 classical record에 쓴다.

`OP_MEAS_ACTIVE_INTERFERE`는 measurement 전에 H를 active axis에 적용한 뒤 Z
measurement를 수행한다.

Dormant measurement는 active tensor를 건드리지 않는다. frame bit와 RNG만으로
record와 frame state를 갱신한다.

## 16. Frame, Noise, Readout

`frame_layer.py`는 Pauli frame과 Clifft noise semantics를 맞추기 위한 helper를
담당한다.

현재 runtime에서 frame-only opcode는 tensor를 건드리지 않고 frame만 갱신한다.

- `OP_FRAME_H`
- `OP_FRAME_S`
- `OP_FRAME_S_DAG`
- `OP_FRAME_CNOT`
- `OP_FRAME_CZ`
- `OP_FRAME_SWAP`

Noise 관련 op:

- `OP_APPLY_PAULI`
- `OP_NOISE`
- `OP_NOISE_BLOCK`
- `OP_READOUT_NOISE`

Noise sampling은 Clifft의 `sample` path와 맞추기 위해 `ClifftNoiseSampler`를 사용한다.
`OP_NOISE_BLOCK`은 noise site range를 순회한다. `OP_READOUT_NOISE`는 measurement
record bit를 확률적으로 flip한다.

이 계층이 맞지 않으면 TTN tensor update가 맞아도 measurement distribution이
Clifft와 달라진다. 과거 `execute` path와 `sample` path mismatch 분석에서 이 부분이
중요했다.

## 17. Memory Model

현재 paper 방향에서 가장 중요한 식은 bag tensor 크기다.

Bag `B`의 physical own axis 수를 `p_B(t)`, incident TTN edge `e`의 실제 bond
dimension을 `chi_e(t)`라고 하자.

```text
N_B(t) = 2^p_B(t) * prod_{e incident to B} chi_e(t)
E_B(t) = log2 N_B(t)
       = p_B(t) + sum_{e incident to B} log2 chi_e(t)
```

Complex128 element를 쓰므로 byte 수는 보통 다음이다.

```text
bytes_B(t) = 16 * N_B(t)
```

전체 resident stored memory:

```text
M_store(t) = 16 * sum_B N_B(t)
M_store_peak = max_t M_store(t)
```

Peak offender bag:

```text
argmax_B N_B(t_peak)
```

Workspace는 opened connected region `R`에 대해 다음으로 해석한다.

```text
N_R(t) = 2^p_R(t) * prod_{e in boundary(R)} chi_e(t)
log2 N_R(t) = p_R(t) + sum_{e in boundary(R)} log2 chi_e(t)
```

현재 adjacent 2-bag transport에서는 workspace가 `src`와 `dst`를 합친 theta
tensor 크기로 기록된다.

## 18. Static Metrics vs Actual Metrics

현재 문서와 report에서는 proxy와 actual을 분리해야 한다.

### 18.1 Structural Lower Bound

`union["sum2"]`는 다음 값이다.

```text
sum_B 2^|V(B)|
```

이 값은 bond dimension이 모두 1이라고 가정한 structural lower bound다.
실행 메모리 예측값이 아니다.

### 18.2 Separator-Saturated Bound

`compute_memory_estimates()`는 각 edge bond가 separator Hilbert dimension까지
saturate된다고 보고 upper estimate를 만든다.

```text
M_separator_worst =
sum_B 16 * 2^own_count(B) * prod_{e incident to B} 2^|sep_e|
```

이 값은 exact QR에서 가능한 worst-case bound로, 실제 runtime보다 매우 loose할 수 있다.

### 18.3 Runtime Actual

실제 runtime에서 기록되는 값은 tensor shape에서 직접 계산한다.

- `resident_actual_peak_numel`
- `resident_actual_peak_log2_numel`
- `resident_actual_peak_bytes`
- `actual_peak_offender_bag`
- `actual_peak_offender_shape`
- `actual_peak_offender_p_B`
- `actual_peak_offender_incident_bond_dims`
- `workspace_actual_peak_bytes`
- `workspace_actual_peak_log2_numel`

이 값이 actual이다. `resident_bound_proxy`, `workspace_proxy_score`,
`refactor_path_proxy`와 섞어 해석하면 안 된다.

## 19. Runtime Instrumentation

`TTNState.metrics`는 실행 중 다음을 추적한다.

Resident/workspace:

- `peak_stored_bytes`
- `resident_actual_peak_numel`
- `resident_actual_peak_log2_numel`
- `resident_actual_peak_bytes`
- `workspace_actual_peak_bytes`
- `peak_pair_workspace_bytes`
- `top5_bag_sizes`
- `top5_pair_workspace`

Bond/rank:

- `max_bond_dim_observed`
- `max_separator_size_observed`
- `edge_max_bond_dim`
- `edge_hit_count`
- `edge_rank_weighted_hits`

Refactor:

- `n_transports`
- `n_qr`
- `n_svd`
- `num_path_contract`
- `num_center_move`
- `num_refactor`
- `sum_path_length`
- `sum_rank_weighted_path_length`
- `sum_refactor_input_numel`
- `max_refactor_input_numel`

Timeout-aware scripts는 `run_shot(..., runtime_timeout=T)`를 사용한다. Timeout이
발생하면 partial metrics를 남기고 `steps_completed < total_steps`로 report한다.

## 20. Verification Status

작은 회로에서는 TTN output이 `clifft.sample`의 self-sampling floor와 같은 수준에
들어온다.

최근 기준:

| Circuit | 5000-shot Clifft self | 5000-shot TTN vs Clifft | 해석 |
| --- | ---: | ---: | --- |
| `distillation` | 약 0.031 | 약 0.032 | pass |
| `cultivation_d3` | 약 0.010 | noise/frame fix 이후 pass 수준 | pass |
| `coherent_d3_r1` | 약 0.016 | pass 수준 | pass |

`coherent_d5_r5` 이상은 runtime memory/time 문제가 커서 full distribution 검증보다
partial runtime metric과 static compression feasibility를 먼저 보고 있다.

## 21. Known Large-Circuit Diagnosis

### 21.1 Time-Varying Graph Result

`time_graph_report.py`는 step별 live graph와 B0 incident edge simultaneity를 기록한다.

중요 결과:

| Circuit | Status | Key observation |
| --- | --- | --- |
| `coherent_d5_r5` | timeout around step 990/3228 | peak step 977, peak bag B0, `E_B=23`, B0 union load 14, max live load 14 |
| `coherent_d7_r1` | complete | peak step 905, B0, `E_B=24`, union load 13.459, max live load 12.858 |
| `coherent_d7_r7` | init error | B0 degree 99, numpy ndarray max dimension 64 초과 |

`coherent_d5_r5`의 B0는 inactive-but-allocated bond contribution이 0이다. 즉 관측된
peak에서는 B0 incident bond들이 실제로 동시에 살아 있다. Lazy allocation만으로는
그 peak를 줄일 수 없다.

`coherent_d7_r1`은 inactive contribution이 약 0.6 bit라 lazy allocation이 약간
도움될 수는 있지만, 큰 개선은 layout 자체가 필요하다.

### 21.2 Peak Tensor

`coherent_d5_r5`의 핵심 peak:

```text
step = 977
bag = B0
peak B0 tensor numel = 2^23
peak B0 tensor bytes = 134,217,728
```

이 peak는 static union layout의 B0 hub가 여러 live bond를 동시에 들면서 발생한다.
여기서 dense active-state 전체보다 극적으로 작아지는 구조가 아직 runtime layout에
반영되지 못하고 있다.

## 22. Static Peak-Bag Compression Experiment

`static_ttn_compression_experiment.py`는 full runtime을 바꾸지 않고, 저장된 peak B0
tensor 하나를 대상으로 numerical-rank TTN decomposition을 시도한다.

문제 설정:

```text
Input X = coherent_d5_r5 step 977 B0 tensor
old_numel = 2^23
old_bytes = 134,217,728
```

SVD split:

```text
X_S -> M_{A|B}
M = U Sigma Vh
rank r determined by rel_tol or energy_tol
left = U_r sqrt(Sigma_r)
right = sqrt(Sigma_r) Vh_r
```

결과:

| Mode | Rule/Tol | New peak log2 numel | New total log2 numel | Peak ratio | Total ratio | Rel error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| depth1 | rel 1e-8 | 20.000 | 20.585 | 8.0x | 5.33x | 1.31e-12 |
| recursive | rel 1e-8 initial | 19.000 | 20.285 | 16.0x | 6.56x | 1.48e-12 |
| recursive | rel 1e-8 deeper | 18.000 | 18.807 | 32.0x | 18.29x | 1.52e-12 |
| beam | rel 1e-8 | 17.000 | 18.459 | 64.0x | 23.27x | 1.53e-12 |
| recursive | energy 1e-4 initial | 16.948 | 18.504 | 66.3x | 22.57x | 9.77e-5 |
| recursive | energy 1e-4 deeper | 16.503 | 18.312 | 90.3x | 25.78x | 9.77e-5 |
| beam | energy 1e-4 | 16.533 | 18.483 | 88.44x | 22.90x | 9.78e-5 |

해석:

- 고정 peak B0 tensor는 numerical-rank TTN으로 상당히 줄어든다.
- 따라서 해당 peak tensor가 본질적으로 dense라서 못 줄이는 것이 아니다.
- 현재 static union hub layout이 tensor structure를 나쁘게 잡고 있다.
- 이 결과는 full runtime 최적화가 아니라, 더 나은 tensor structure가 존재한다는
  feasibility evidence다.

Beam search는 `rel_tol=1e-8`에서 greedy보다 더 좋은 구조를 찾았다. 기존 deeper
greedy는 peak를 `2^18` elements까지 줄였고, beam은 같은 사실상 exact tolerance에서
peak를 `2^17` elements까지 줄였다. 이는 split 순서와 leaf 선택이 중요하다는 뜻이다.

Beam tree bottleneck decomposition 결과, `2^17` peak는 open original leg product
때문이 아니라 internal bond rank product 때문이다. Peak leaf는 `rootRL`이고,
분해는 다음과 같다.

```text
log2(numel(rootRL)) = 17
open_leg_logsum = 2
internal_bond_logsum = 15
internal ranks = 256, 128
```

즉 다음에 `2^17` 아래로 줄이려면 peak leaf의 open leg를 단순히 나누는 것보다,
`root -> rootR -> rootRL`로 이어지는 internal rank를 낮추는 split choice 또는
multi-snapshot rank-aware objective가 필요하다.

반면 `energy_tol=1e-4`에서는 현재 제한 설정의 beam이 deeper greedy를 이기지 못했다.
Greedy가 peak `log2(numel)=16.503`, total `18.312`였고 beam은 peak `16.533`,
total `18.483`이었다. 즉 approximate/numerical-rank 영역에서는 더 큰 beam width,
다른 split generator, 또는 objective 조정이 필요하다.

Plateau split을 허용한 실험은 peak는 `2^18`로 유지했지만 total memory가 악화됐다.
따라서 현재까지는 peak 감소가 없으면 split을 계속하는 방식이 좋지 않다.

Fixed-topology reuse test에서는 step-977 beam topology를 다른 high-critical B0
snapshot 11개에 그대로 적용했다. 모든 step에서 구조적 적용과 reconstruction은
성공했지만, memory quality는 균일하지 않았다.

```text
steps tested = 11
successful = 11
worst fixed peak log2 = 21.169925
median peak compression ratio = 64x
min peak compression ratio = 2.67x
max reconstruction error = 1.32e-12
```

특히 step `944`가 counterexample이다.

```text
old log2 numel = 22.585
fixed topology peak log2 = 21.170
peak compression = 2.67x
max internal rank = 1152
```

따라서 step-977 topology 하나를 바로 full runtime에 patch하면 안 된다. 다음 단계는
step `944` 같은 hard case를 포함하는 multi-snapshot common-skeleton search다.

## 23. RASL의 현재 위치

RASL은 Rank-Aware Symplectic Localization이다. 현재 구현은 fixed TTN layout 위에서
Pauli localization target을 바꿔 path/refactor proxy를 줄인다.

관찰:

- `coherent_d5_r5`에서 analysis pass는 일부 step의 target을 바꿨다.
- 예: `default_path_cost 14 -> chosen_path_cost 7`
- resident proxy는 유지됐다.
- actual experiment에서도 resident memory 감소는 확인되지 않았고, path/refactor
  work 감소 쪽이 주효했다.

따라서 현재 paper framing은 다음이다.

```text
RASL = resident memory optimizer X
RASL = resident memory를 악화시키지 않으면서 path/refactor work를 줄이는 보조 pass O
```

메모리 문제의 main direction은 TTN layout과 rank-aware tensor structure다.

## 24. Current Limitations

현재 방식의 한계는 명확하다.

1. Static union layout은 시간별 live graph를 충분히 활용하지 못한다.
2. High-degree hub bag은 많은 incident bond product를 한 tensor에 곱한다.
3. Degree만 줄이는 naive hub3 transform은 large separator를 복제해서 worst-case bound를
   악화시킬 수 있다.
4. Exact runtime은 아직 모든 cross-bag operation을 하나의 global memory cap 아래에서
   선택하지 않는다. `OP_ARRAY_MULTI_CNOT`에는 persistent/destructive-open policy가
   있지만, fallback/path-contract 경로는 여전히 큰 workspace를 열 수 있다.
5. 이 때문에 한 병목을 줄이면 다른 병목이 peak가 되는 bottleneck migration이 발생한다.
   `coherent_d5_r5` 1200-step prefix에서 exact persistent-only가 dense보다 커지는
   것이 이 현상이다.
6. `_thin_qr()`은 zero row 제거 수준의 rank cleanup이 기본이다. SVD truncation은
   optional approximation으로 존재하지만, threshold/error/correctness를 별도로 보고해야 한다.
7. Exact runtime은 아직 peak bag 내부를 nested TTN으로 자동 분해하지 않는다.
8. `coherent_d7_r7` baseline은 numpy dimension limit 때문에 초기 tensor 표현부터
   실패한다. 이는 algorithmic memory blowup 이전의 representation failure다.
9. Static compression experiment는 고정 snapshot에서만 수행된다. 아직 full circuit
   execution에 integrated layout으로 반영되지 않았다.

## 25. What To Optimize Next

현재 evidence에 따르면 다음 순서가 맞다.

```text
1. 현재 executor의 모든 peak를 concurrent memory 식으로 분해한다.
2. OP_ARRAY_MULTI_CNOT persistent/destructive-open은 유지한다.
3. fallback/path-contract/Class B/C 경로에도 같은 memory cap을 적용한다.
4. 각 cross-bag op/window마다 executor 후보를 만들고 cap 안에서 선택한다.
5. cap을 넘는 후보는 batch split, smaller region, destructive-open, 또는 fallback으로 낮춘다.
6. exact 후보가 cap을 만족하지 못하는 step만 optional SVD/numerical-rank compression 대상으로 표시한다.
7. approximation을 쓰는 경우 threshold, output error, measurement distribution drift를 별도 검증한다.
8. layout/static skeleton search는 이 executor selector 위에서 다시 평가한다.
9. RASL은 마지막에 resident cap을 악화시키지 않는 보조 path/refactor reducer로 붙인다.
```

가장 직접적인 다음 실험은 다음이다.

- `coherent_d5_r5` 1200-step prefix의 `step=1130, B72` peak를 기준으로
  fallback/path-contract가 왜 cap을 넘는지 trace한다.
- fallback path에서도 live memory를 다음 식으로 강제한다.

```text
stored_outside_open_regions
+ open_region_tensor
+ temporary_workspace
<= cap
```

- 같은 operation에 대해 다음 후보를 모두 비교한다.

```text
local/direct
path transport
single-step fused region
persistent region
batched region
destructive-open region
optional SVD-compressed region
```

- 최종 선택은 다음 lexicographic rule을 따른다.

```text
1. exact 후보 중 cap 만족 여부
2. concurrent total memory
3. QR/transport/refactor work
4. runtime proxy
```

- exact 후보가 없을 때만 approximation 후보를 별도 label로 허용한다.

## 26. Useful Commands

작업 디렉터리는 `/home/jung/clifft-paper`다.

Small correctness:

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.verify_ttn
```

Transport unit test:

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.tests.test_ttn_transport
```

Runtime metrics:

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.metrics_report \
  --runtime-timeout 60 \
  --variants baseline \
  --out-csv reports/baseline.csv \
  --out-json reports/baseline.json
```

Memory risk:

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.memory_risk_report \
  --variants baseline,hub3 \
  --include-runtime \
  --out-csv reports/memory_risk.csv \
  --out-json reports/memory_risk.json \
  --out-md reports/memory_risk_summary.md
```

Actual peak decomposition:

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.memory_diagnosis_report \
  distillation cultivation_d3 coherent_d3_r1 coherent_d5_r1 coherent_d5_r5 \
  --variants baseline \
  --runtime-timeout 60
```

Time-varying graph:

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.time_graph_report \
  --runtime-timeout 60 \
  --variants baseline
```

Static B0 compression:

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.static_ttn_compression_experiment \
  --circuit coherent_d5_r5 \
  --step 977 \
  --rank-rules rel energy \
  --tols 1e-8 1e-6 1e-4 \
  --mode depth1 recursive \
  --random-candidates 300 \
  --top-svd 8 \
  --max-depth 6 \
  --min-gain 1.01 \
  --snapshot-cache-dir reports \
  --out-dir reports/static_rel1e8_deep
```

Static B0 compression with beam search:

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.static_ttn_compression_experiment \
  --circuit coherent_d5_r5 \
  --step 977 \
  --bag B0 \
  --rank-rules rel \
  --tols 1e-8 \
  --mode beam \
  --random-candidates 120 \
  --top-svd 6 \
  --max-depth 6 \
  --min-node-numel 1024 \
  --min-gain 1.01 \
  --beam-width 4 \
  --beam-node-splits 2 \
  --beam-max-rounds 12 \
  --snapshot-cache-dir reports \
  --out-dir reports/static_rel1e8_beam
```

## 27. Multi-Snapshot Global Skeleton Search

단일 step 977에서 찾은 beam tree는 `peak_log2=17`까지 B0 tensor를 줄였지만,
fixed-topology reuse test에서 step 944가 hard counterexample로 남았다. 따라서
다음 offline 단계는 step마다 다른 topology를 찾는 것이 아니라, 여러 critical
snapshot에 대해 하나의 공통 binary skeleton을 찾는 것이다.

새 스크립트:

```text
ttn_backend/scripts/multisnapshot_global_tree_search.py
```

수학적 objective는 다음 lexicographic min-max score다.

```text
Score(T) =
(
  max_t peak_log2(T, t),
  max_t total_log2(T, t),
  max_t error(T, t),
  num_tensors(T),
  tree_depth(T)
)
```

여기서 `T`는 global leaf universe `U = union_t L(t)` 위의 하나의 binary tree다.
각 snapshot `t`에서는 live leaf `L(t)`만 materialize한다. 어떤 split `A|B`에서
`A ∩ L(t)` 또는 `B ∩ L(t)`가 비면 해당 split은 그 step에서 inactive이며, 높은
rank bond를 만들지 않는다. 이것이 lazy live allocation 규칙이다.

후보 split 생성기는 다음을 사용하지만, 이들은 최종 목적함수가 아니다.

```text
1. live-logdim balanced split
2. co-live spectral bisection
3. random balanced splits
4. previous static tree-derived splits
5. large-leg separation / current-layout repair split
```

최종 선택은 항상 실제 SVD factorization을 여러 snapshot에 적용한 actual
multi-snapshot memory score로만 정한다. `alpha/beta/gamma` 형태의 weighted sum은
사용하지 않는다.

현재 smoke/최소-budget 실행:

```bash
OPENBLAS_NUM_THREADS=1 /home/jung/clifft_env/bin/python \
  -m ttn_backend.scripts.multisnapshot_global_tree_search \
  --circuit coherent_d5_r5 \
  --bag B0 \
  --steps 903 930 935 944 949 967 977 978 979 988 989 \
  --rank-rule rel \
  --rel-tol 1e-8 \
  --beam-width 2 \
  --beam-rounds 1 \
  --top-svd 1 \
  --random-candidates 5 \
  --beam-node-splits 1 \
  --previous-tree reports/static_rel1e8_beam/static_ttn_b0_compression_tree_beam_rel_1em08.json \
  --snapshot-cache-dir reports/fixed_topology_reuse_rel1e8/snapshots \
  --out-dir reports/multisnapshot_global_rel1e8
```

결과 요약:

```text
current_hub:
  worst_peak_log2 = 23.000 at step 977

fixed_T977 baseline:
  worst_peak_log2 = 21.170 at step 944
  min peak compression = 2.667x
  max reconstruction error = 1.32e-12

common_global_tree, minimal budget:
  worst_peak_log2 = 22.170 at step 944
  max reconstruction error = 9.89e-9
  num_tensors = 2
  tree_depth = 1
```

해석은 조심해야 한다. 최소 budget common tree는 current B0 hub보다는 낫지만,
기존 step-977 fixed topology보다 worst-case가 더 나쁘다. 즉 공통 skeleton
문제 자체는 구현됐고 actual objective로 평가되지만, 현재 실행 budget에서는
step 944를 `21.17` 아래로 낮추는 더 좋은 공통 topology를 아직 찾지 못했다.
더 큰 beam budget, 더 나은 admissible lower-bound pruning, 또는 multi-snapshot
common-skeleton search의 branch-and-bound 강화가 다음 개선 방향이다.

알고리즘적으로 이 문제는 작은 leaf 수를 제외하면 exact global optimum을 보장하기
어렵다. binary tree topology 수가 super-exponential이고, 각 split rank가 tensor
값과 tolerance에 의존하기 때문이다. 따라서 현재 구현은 전역 최적 알고리즘이 아니라
수학적으로 명확한 min-max actual memory objective를 직접 평가하는 bounded
anytime heuristic이다.

### 27.1 Depth-1 Failure Debug

초기 common search 결과가 `num_tensors=2`, `tree_depth=1`에서 멈춘 이유를
추가로 디버그했다.

추가 구현:

```text
1. search_debug.json / search_debug.md 출력
2. round별 candidate count, eligible nodes, beam state 기록
3. recursive split에서 parent internal bond를 child partition에 전달하는 boundary-leg 처리
4. assignment-beam split generator 추가
5. depth-diverse beam 옵션 추가
```

중요한 버그 수정:

```text
recursive split 시 current subtree tensor에는 parent internal bond가 존재하지만,
global skeleton leaf set에는 original leg만 존재한다.

따라서 child split을 평가할 때 parent bond를 A/B 중 한쪽 boundary leg로 같이
넘겨야 한다. 기존에는 이 leg를 partition에서 누락해 "axes don't match array"
에러가 발생했다.
```

수정 후 `step=944` 단독으로 depth-diverse beam을 실행했다.

```bash
OPENBLAS_NUM_THREADS=1 /home/jung/clifft_env/bin/python \
  -m ttn_backend.scripts.multisnapshot_global_tree_search \
  --circuit coherent_d5_r5 \
  --bag B0 \
  --steps 944 \
  --rank-rule rel \
  --rel-tol 1e-8 \
  --beam-width 6 \
  --beam-rounds 4 \
  --top-svd 2 \
  --random-candidates 8 \
  --beam-node-splits 3 \
  --assignment-beam-width 16 \
  --assignment-beam-outputs 2 \
  --depth-diverse-beam \
  --previous-tree reports/static_rel1e8_beam/static_ttn_b0_compression_tree_beam_rel_1em08.json \
  --snapshot-cache-dir reports/fixed_topology_reuse_rel1e8/snapshots \
  --out-dir reports/multisnapshot_global_rel1e8_step944_depthdiverse
```

결과:

```text
current_hub, step 944:
  peak_log2 = 22.585

fixed_T977, step 944:
  peak_log2 = 21.170
  num_tensors = 7
  depth = 4

common_global_tree, depth-diverse beam:
  peak_log2 = 21.907
  num_tensors = 2
  depth = 1
```

디버그 요약:

```text
generated candidate splits = 68
evaluated candidate trees = 68
candidates with peak_log2 < 21.9069 = 0
deeper candidates were evaluated, but all had the same peak and larger total.
```

따라서 현재 상태는 다음과 같이 해석해야 한다.

```text
1. evaluator는 동작한다.
2. recursive boundary-leg handling도 동작한다.
3. shallow-only pruning 문제는 완화했다.
4. 그래도 현재 후보 생성기는 step 944 peak offender를 직접 깨는 split을 찾지 못한다.
```

즉 다음 개선은 단순히 `beam_rounds`를 늘리는 것이 아니라, peak offender node를
target으로 하는 split search가 필요하다. 구체적으로는:

```text
1. 현재 best tree의 peak node를 식별한다.
2. 그 node의 open legs + incident internal bonds를 기준으로 local bipartition을 탐색한다.
3. |S|가 작으면 exact bipartition enumeration을 사용한다.
4. |S|가 크면 assignment beam을 peak-node-local objective로 돌린다.
5. 이 local split이 global score를 실제로 낮추는지 multi-snapshot evaluator로 검증한다.
```

현재 common search는 “global root split generator”로는 충분하지 않다. 다음 버전은
`peak-node refinement` 또는 `branch-and-bound over peak offender splits`로 가야 한다.

## 28. Summary

현재 TTN 방식은 다음 한 줄로 요약된다.

```text
Clifft bytecode의 active ident를 static union JT 위의 bag tensor들에 저장하고,
cross-bag active op는 adjacent 2-bag QR transport sweep으로 exact하게 처리한다.
```

이 방식은 작은 회로에서는 Clifft sampling distribution을 재현한다. 하지만 큰 회로에서는
static union layout의 hub bag이 많은 live bond를 동시에 들면서 resident memory가
커진다.

따라서 현재 연구 방향은 Pauli localization을 더 밀어붙이는 것이 아니라,
actual tensor size objective:

```text
N_B(t) = 2^p_B(t) * prod_e chi_e(t)
```

를 직접 줄이는 layout/rank-aware tensor structure를 찾는 것이다. Static B0
compression 실험은 더 작은 tensor structure가 존재한다는 강한 evidence를 이미
보여준다. 다음 단계는 이 static feasibility를 executable TTN runtime layout으로
연결하는 것이다.

## 29. Temporal Carving Runtime Bridge

기존 `qec_temporal_carving_report.py`는 compile-time proxy evaluator였다. 이
adapter는 Clifft bytecode에서 active live trace와 active two-axis event만 추출해서
temporal-live carving objective를 평가했다. 이 값은 layout 후보를 만드는 데는 쓸 수
있지만, 실제 TTN backend가 실행하며 만드는 bond dimension이나 tensor shape를
측정하지 않는다.

따라서 runtime 검증용 bridge를 별도로 추가했다.

```text
ttn_backend/scripts/qec_temporal_carving_runtime.py
```

이 script의 흐름은 다음과 같다.

```text
1. Clifft program compile
2. 기존 backend_spec 생성
3. temporal_carving pipeline으로 binary carving tree 생성
4. carving tree를 실행 가능한 leaf-home TTN bag tree로 변환
5. 기존 TTNBackend.run_shot()으로 Clifft bytecode 직접 실행
6. TTNState.metrics에서 actual tensor/bond/QR metric 수집
```

여기서 `carving_leaf` layout은 pure binary tree다.

```text
leaf bag     : active ident 1개 소유
internal bag : active ident 0개 소유
edge         : static TTN bond, 초기 dim = 1
```

기존 `assign_homes_and_classify()`는 junction-tree layout을 전제로 하므로,
모든 2-axis op에 대해 두 ident를 같이 포함하는 compute bag을 요구한다. Pure carving
tree는 이 조건을 만족하지 않는다. 그래서 runtime bridge는 별도의 homing/classification을
쓴다.

```text
home(ident) = corresponding leaf bag
single-axis op = leaf bag local update
two-axis op = path between two leaf homes, Class C transport sweep
```

즉 이 bridge는 proxy trace를 다시 해석하는 것이 아니라, 기존 `TTNBackend`가 실제
Clifft bytecode를 dispatch한다. 따라서 frame/noise/measurement/fused U2/U4 처리는
기존 runtime backend와 동일하다. 차이는 static bag tree와 home map뿐이다.

### 29.1 Actual Runtime Results

다음 명령으로 측정했다.

```bash
/home/jung/clifft_env/bin/python ttn_backend/scripts/qec_temporal_carving_runtime.py distillation \
  --runtime-timeout 30 --out-dir reports/qec_temporal_carving_runtime_smoke

/home/jung/clifft_env/bin/python ttn_backend/scripts/qec_temporal_carving_runtime.py coherent_d5_r1 \
  --runtime-timeout 60 --out-dir reports/qec_temporal_carving_runtime_d5r1

/home/jung/clifft_env/bin/python ttn_backend/scripts/qec_temporal_carving_runtime.py coherent_d5_r5 \
  --runtime-timeout 60 --out-dir reports/qec_temporal_carving_runtime_d5r5
```

결과:

| circuit | mode | steps | actual peak log2 | peak stored bytes | workspace bytes | max bond | QR |
|---|---|---:|---:|---:|---:|---:|---:|
| distillation | baseline_jt | 2040/2040 | 4 | 384 | 256 | 2 | 14 |
| distillation | carving_leaf | 2040/2040 | 3 | 656 | 128 | 2 | 79 |
| coherent_d5_r1 | baseline_jt | 857/857 | 12 | 129408 | 65536 | 62 | 246 |
| coherent_d5_r1 | carving_leaf | 857/857 | 11 | 91216 | 32768 | 32 | 1047 |
| coherent_d5_r5 | baseline_jt | 1006/3228 timeout | 23 | 134223584 | 134217728 | 16 | 432 |
| coherent_d5_r5 | carving_leaf | 968/3228 timeout | 22 | 114510096 | 67108864 | 1681 | 3862 |

### 29.2 Interpretation

이 결과는 proxy가 아니라 actual tensor execution 결과다.

첫째, lazy allocation이 실제 메모리에는 효과가 있다.

```text
coherent_d5_r1:
  actual peak tensor log2  12 -> 11
  workspace bytes          65536 -> 32768
  peak stored bytes        129408 -> 91216

coherent_d5_r5 partial:
  actual peak tensor log2  23 -> 22
  workspace bytes          134217728 -> 67108864
  peak stored bytes        134223584 -> 114510096
```

둘째, pure leaf-home carving tree를 그대로 runtime layout으로 쓰면 QR/transport 비용이
크게 증가한다.

```text
distillation:
  QR 14 -> 79

coherent_d5_r1:
  QR 246 -> 1047

coherent_d5_r5 partial:
  QR 432 -> 3862
```

이는 leaf-only tree가 모든 cross-active operation을 긴 path transport로 처리하기
때문이다. 따라서 temporal carving 자체는 resident/workspace memory를 줄이는 방향을
보여주지만, 최종 backend layout은 다음을 함께 만족해야 한다.

```text
1. lazy allocation이 가능한 degree-capped binary/tree-like structure
2. 자주 상호작용하는 active ident 쌍은 path가 너무 길어지지 않도록 배치
3. peak offender bond product를 낮추면서 QR/transport count를 폭증시키지 않음
```

즉 이번 runtime bridge의 결론은:

```text
lazy allocation은 actual memory에서 효과가 있다.
하지만 pure carving leaf-home layout은 refactor work가 너무 커서 그대로 최종안은 아니다.
다음 단계는 memory objective와 path/refactor objective를 분리해서,
memory를 보존하면서 transport 비용을 줄이는 executable layout refinement다.
```

## 30. Persistent MULTI_CNOT 이후 현재 병목

`OP_ARRAY_MULTI_CNOT`을 control별 CNOT으로 쪼개 실행하던 기존 경로는 실제 병목이었다.
이를 persistent fused window executor로 바꾸면 839/968-step prefix에서는 dense 대비
약 8배 수준의 메모리 절감이 유지되고, QR/transport도 함께 줄었다.

하지만 `coherent_d5_r5` 1200-step prefix에서는 병목이 이동했다.

```text
새 병목:
  OP_ARRAY_MULTI_CNOT 자체의 per-control 처리
  -> 해결됨

남은 병목:
  Class B/C path transport fallback
  -> 큰 pair workspace와 큰 resident bag tensor를 동시에 만든다
```

대표 offender는 다음과 같다.

```text
circuit = coherent_d5_r5
prefix = 1200 steps
peak step = 1130
opcode = OP_ARRAY_MULTI_CNOT
selected_executor = class_BC_path_transport
offender bag = B72 계열
offender shape = [2, 59, 1024, 64]
```

이 shape의 원인은 한 bag에 여러 큰 incident bond가 동시에 곱해지는 것이다.

```text
numel = 2 * 59 * 1024 * 64
bytes = 123,731,968
```

따라서 다음 최적화는 `MULTI_CNOT window를 더 키우기`가 아니라,
모든 cross-bag executor에서 다음 두 가지를 같이 적용하는 것이다.

```text
1. destructive-open liveness:
   region/pair workspace가 열릴 때 absorbed bag tensor를 live stored에서 제거한다.

2. cap-triggered local bag fission:
   cap을 넘는 resident bag tensor를 exact local SVD/QR factorization으로 microtree화한다.
```

## 31. Destructive Pair-Open 계측 수정

기존 metric은 adjacent transport workspace를 다음처럼 계산했다.

```text
old total = stored + pair_workspace
```

하지만 transport에서 실제로 열리는 pair region은 source/destination bag tensor를 대체한다.
논리적 destructive liveness는 다음이 맞다.

```text
new total =
  stored_before_transport
  - bytes(src_bag)
  - bytes(dst_bag)
  + pair_workspace
```

이를 runtime metric에 반영했다. 이 변경은 objective를 바꾼 것이 아니라,
이미 persistent region에서 쓰던 destructive-open liveness를 adjacent pair transport에도
일관되게 적용한 것이다.

1200-step prefix에서 fission 없이 destructive pair-open만 적용한 결과:

| policy | steps | actual total peak | dense 대비 | peak step | QR | transport |
|---|---:|---:|---:|---:|---:|---:|
| persistent, no fission, destructive pair-open | 1200 | 157,475,136 B | 1.70x 절감 | 1130 | 2183 | 860 |

Dense active state peak는 `268,435,456 B = 256 MiB`이다.

## 32. Cap-Triggered Bag Fission Prototype

Offline feasibility에서 B72 offender tensor는 exact하게 잘 쪼개졌다.

```text
old shape = [2, 59, 1024, 64]
old bytes = 123,731,968

best exact split:
  left  = physical axis + bond 0-72
  right = bonds 72-73, 72-108
  internal rank = 32

new peak child bytes = 33,554,432
new total child bytes = 33,614,848
peak reduction = 3.69x
reconstruction error = 0
```

이 결과는 B72 peak tensor가 intrinsically dense하지 않다는 뜻이다.
현재 하나의 bag에 bond product가 몰려 있어서 큰 것이고, local microtree fission으로
resident tensor peak를 줄일 수 있다.

Runtime prototype은 다음 제약으로 구현했다.

```text
1. exact SVD only
2. physical own axes는 기존 bag에 유지
3. incident bond axes만 split
4. home map 전체 재생성 없음
5. fission 후 current tree path를 매 transport마다 다시 계산
```

1200-step prefix 결과:

| policy | steps | total peak | dense 대비 | peak stored | workspace peak | fissions | QR | transport | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| persistent + destructive pair-open | 1200 | 157,475,136 B | 1.70x | 157,475,136 B | 123,731,968 B | 0 | 2183 | 860 | 45.6s |
| + exact runtime bag fission | 1200 | 141,928,608 B | 1.89x | 141,928,608 B | 123,731,968 B | 5 | 2206 | 879 | 177.5s |

해석:

```text
bag fission은 resident peak를 추가로 낮춘다.
하지만 현재 prototype은 exact SVD 후보 평가 비용이 커서 runtime overhead가 크다.
따라서 항상 켜는 정책이 아니라, peak-triggered / critical-only / cached fission으로 제한해야 한다.
```

또한 fission temp peak는 다음과 같이 기록된다.

```text
bag_fission_temp_peak_bytes = 123,731,968
```

즉 exact fission 자체도 큰 SVD workspace를 요구한다. runtime claim에서는
`steady-state live tensor peak`와 `fission temporary peak`를 분리해서 보고해야 한다.

## 33. Region-Local Clifford Frame Status

`RegionLinearFrame` foundation을 추가했다.

지원 범위:

```text
OP_ARRAY_CNOT
OP_ARRAY_MULTI_CNOT
OP_ARRAY_SWAP
```

표현:

```text
x' = A x xor b
```

검증:

```text
1. random bitstring에서 sequential CNOT/SWAP == frame apply
2. 작은 random tensor에서 sequential tensor apply == frame materialization
3. 허용 오차 < 1e-10
```

현재 상태:

```text
frame 자료구조와 unit test는 통과했다.
persistent MULTI_CNOT window 내부에 v1 runtime integration을 적용했다.
```

v1 integration은 다음처럼 동작한다.

```text
1. persistent window는 기존처럼 open/close한다.
2. window 내부 OP_ARRAY_MULTI_CNOT / OP_ARRAY_CNOT은 tensor에 즉시 적용하지 않고
   RegionLinearFrame에 누적한다.
3. OP_ARRAY_ROT 같은 non-Clifford boundary 또는 window close에서 frame을 한 번
   materialize한다.
```

1200-step prefix 결과:

| policy | total peak | dense 대비 | QR | transport | frame updates | materializations | avoided tensor applies | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| persistent, no frame | 157,475,136 B | 1.70x | 2183 | 860 | 0 | 0 | 0 | 45.6s |
| frame-lift v1 | 157,475,136 B | 1.70x | 2183 | 860 | 20 | 20 | 53 | 45.7s |
| frame-lift + include active Clifford | 340,633,088 B | 0.79x | 2739 | 1350 | 21 | 21 | 53 | 142.8s |

해석:

```text
1. frame-lift v1은 tensor CNOT apply 53개를 제거했다.
2. 하지만 persistent window의 open/close/refactor 구조는 그대로라 QR/transport는 줄지 않았다.
3. materialization reason은 대부분 non_clifford_rot이다.
4. active Clifford까지 window에 단순히 포함하면 support/region이 커져 memory와 QR이 악화된다.
```

따라서 현재 v1은 correctness foundation이다. QR/SVD를 실제로 줄이려면 다음 단계가
필요하다.

```text
Clifford-only window를 open하지 말고 frame만 누적
→ non-Clifford/measurement boundary에서 필요한 최소 region만 materialize
```

즉 단순히 persistent window 내부 apply를 frame으로 바꾸는 것으로는 부족하고,
open/close 자체를 생략하는 executor selector가 필요하다.

## 34. 현재 결론

현재까지의 일반화 가능한 정책은 다음과 같다.

```text
1. structured bytecode fusion:
   MULTI_CNOT per-control 실행을 persistent fused window로 바꾼다.

2. destructive-open liveness:
   open region/pair workspace는 absorbed tensors를 대체한다.

3. cap-triggered resident repair:
   cap을 넘는 bag은 local exact fission 후보로 microtree화한다.

4. frame-lifted Clifford execution:
   Clifford-only 구간은 tensor에 즉시 materialize하지 않고 pending frame으로 늦춘다.
```

현재 수치상 가장 중요한 결과는:

```text
coherent_d5_r5 1200-step prefix:
  dense peak                         = 268,435,456 B
  persistent + destructive pair-open = 157,475,136 B  (1.70x)
  + exact runtime bag fission        = 141,928,608 B  (1.89x)
```

즉 1200-step prefix에서도 dense보다 작은 exact 실행은 다시 확보했다.
다만 아직 충분히 압도적이지 않고, fission runtime overhead가 크다.

다음 우선순위:

```text
1. fission 후보 caching / critical-only triggering
2. fission temporary workspace를 포함한 full peak 분리 보고
3. RegionLinearFrame runtime integration
4. cap-aware executor selector로 fallback/path transport 전부 통합
```

## 35. B72 Fission Maximum Search

`coherent_d5_r5` 1200-step prefix의 B72 offender tensor에 대해 더 강한 offline
fission 탐색을 수행했다.

입력 tensor:

```text
step = 1130
bag = B72
shape = [2, 59, 1024, 64]
old bytes = 123,731,968
old log2(numel) = 22.883
```

### 35.1 Greedy recursive depth sweep

`max_depth=8`까지 recursive fission을 허용했다.

결과:

| mode | tol | best peak bytes | best peak log2 | peak ratio | total ratio |
|---|---:|---:|---:|---:|---:|
| exact | 0 | 33,554,432 | 21.000 | 3.688x | 3.681x |
| approx | 1e-4 | 33,554,432 | 21.000 | 3.688x | 3.681x |
| approx | 1e-3 | 28,606,464 | 20.770 | 4.325x | 2.879x |
| approx | 1e-2 | 21,069,824 | 20.329 | 5.872x | 3.908x |

### 35.2 Full binary tree enumeration

B72는 original open leg가 4개뿐이므로 가능한 full binary tree topology 15개를 전부
평가했다.

결과:

| mode | tol | best tree | best peak bytes | best peak log2 | max rank | peak ratio |
|---|---:|---|---:|---:|---:|---:|
| exact | 0 | `((0, 1), (2, 3))` | 33,554,432 | 21.000 | 64 | 3.688x |
| approx | 1e-4 | `((0, 1), (2, 3))` | 33,554,432 | 21.000 | 64 | 3.688x |
| approx | 1e-3 | `((0, 1), (2, 3))` | 33,554,432 | 21.000 | 64 | 3.688x |
| approx | 1e-2 | `((0, 1), (2, 3))` | 33,554,432 | 21.000 | 64 | 3.688x |

해석:

```text
1. B72 단일 tensor의 exact fission 한계는 약 32 MiB이다.
2. full binary topology를 전부 봐도 exact peak는 2^21 elements 아래로 내려가지 않았다.
3. 따라서 1200-step 전체 peak를 더 낮추려면 B72 하나를 더 쪼개는 것만으로는 부족하다.
4. 남은 peak는 pair workspace / neighboring fission / path transport policy까지 같이 줄여야 한다.
```

주의:

```text
greedy approximate depth sweep은 1e-3, 1e-2에서 32 MiB보다 낮은 child peak를
찾았지만, 이는 truncated SVD approximation이다. exact claim과 섞으면 안 된다.
```

## 36. Compile-Time SVD 가능성

SVD를 compile-time에 전부 미리 하는 것은 일반적으로 불가능하다.

이유:

```text
1. fission 대상 tensor 값은 앞선 active operations, noise sample, measurement branch,
   frame state에 의존한다.
2. 같은 static bag/shape라도 shot마다 tensor entries가 달라질 수 있다.
3. 따라서 U/S/V numerical factors 자체는 compile-time circuit structure만으로 고정되지 않는다.
```

하지만 compile/profile-time에 미리 할 수 있는 것은 많다.

```text
가능:
  1. fission 대상 bag/window 선정
  2. split topology 선정
  3. parent/internal bond placement 선정
  4. cap-trigger rule 선정
  5. 어떤 SVD matrix shape가 나올지 예측
  6. rank/spectrum 통계 profile
  7. repeated branch에서 같은 frame-equivalent tensor라면 SVD basis reuse certificate

불가능하거나 조건부:
  1. arbitrary shot/branch의 exact U/S/V를 compile-time에 고정
  2. measurement collapse 이후 tensor의 SVD를 미리 고정
  3. non-Clifford/noise branch가 split을 가로지르면 reference SVD 재사용
```

따라서 올바른 방향은:

```text
compile/profile-time:
  critical offender와 fission topology를 미리 찾는다.

runtime:
  cap을 넘는 순간, 미리 정한 split topology로만 SVD를 수행한다.
  가능한 경우 SVD cache/reuse certificate를 사용한다.
```

즉 SVD의 값 자체를 무조건 미리 저장하는 접근이 아니라,
`where/how to fission`을 compile-time에 고정하고, runtime search를 제거하는 방식이
필요하다.

## 37. Transport Prefission: Peak Workspace를 만들기 전에 쪼개기

기존 runtime fission은 transport workspace가 이미 만들어진 뒤에 resident bag을
쪼개는 구조였다. 따라서 `workspace_actual_peak_bytes`가 그대로 남았다.

이를 보완하기 위해 transport 직전 workspace를 예측하고, cap을 넘으면 adjacent edge
양끝 bag 중 fission 가능한 bag을 먼저 exact fission한 뒤 current tree path를 다시
계산하는 prototype을 추가했다.

```text
before transport Bsrc -> Bdst:
  estimate theta workspace
  if workspace > cap:
      exact-fission src or dst
      recompute current tree path
      retry
```

### 37.1 coherent_d5_r5 1200-step 결과

비교 기준:

```text
dense peak = 268,435,456 B
```

| policy | total peak | dense 대비 | workspace peak | resident peak | fissions | QR | transport | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| persistent + destructive pair-open | 157,475,136 B | 1.70x | 123,731,968 B | 123,731,968 B | 0 | 2183 | 860 | 45.6s |
| transport prefission cap64 | 138,578,656 B | 1.94x | 67,108,864 B | 67,108,864 B | 4 | 2176 | 854 | 46.2s |
| transport prefission cap32 | 156,782,704 B | 1.71x | 88,080,384 B | 88,080,384 B | 14 | 2279 | 937 | 147.5s |
| post-hoc exact bag fission cap64 | 141,928,608 B | 1.89x | 123,731,968 B | 123,731,968 B | 5 | 2206 | 879 | 177.5s |

해석:

```text
1. transport prefission cap64가 현재 exact best이다.
2. 1200-step exact peak는 157.5 MB -> 138.6 MB로 내려갔다.
3. workspace peak도 123.7 MB -> 64 MB로 내려갔다.
4. QR/transport/runtime overhead는 거의 증가하지 않았다.
5. cap32는 더 많이 쪼개지만 topology가 복잡해져 stored peak와 runtime이 악화됐다.
```

즉 단순히 cap을 낮춘다고 항상 좋은 것이 아니다.

```text
too weak:
  큰 workspace가 남는다.

too aggressive:
  microtree가 길어지고 transport path가 복잡해져 resident/work가 다시 증가한다.

best observed:
  cap64 transport prefission
```

현재 1200-step에서 가장 좋은 exact 수치는:

```text
actual total peak = 138,578,656 B
dense ratio = 268,435,456 / 138,578,656 = 1.94x
memory reduction = 48.4%
```

아직 압도적이지 않다. 하지만 중요한 변화는:

```text
post-hoc fission:
  큰 workspace를 만든 뒤 쪼갬

transport prefission:
  큰 workspace를 만들기 전에 쪼갬
```

으로 바뀌었다는 점이다.

다음 과제:

```text
1. prefission split을 runtime에서 탐색하지 말고 profile-time plan으로 고정
2. cap64 근처에서 per-edge adaptive cap 선택
3. step 1130뿐 아니라 다른 critical transport edge에도 같은 rule 적용
4. path 전체를 대상으로 한 staged region transport selector 구현
```

## 38. Generalized Policy Benchmark Across QEC Circuits

지금까지의 policy를 회로 전용 rule이 아니라 일반 executor 후보로 묶어 전체 benchmark
prefix에서 검증했다.

정책:

```text
carving_base:
  no structured executor optimization

fuse_only:
  cap-aware OP_ARRAY_MULTI_CNOT region/batch fusion
  destructive-open liveness

general_policy:
  fuse_only
  + persistent MULTI_CNOT windows
  + transport prefission cap64
```

모든 decision은 opcode/window/support/workspace/bag-shape/cap 기반이다.
`coherent_d5_r5`, `step=1130`, `B72` 같은 hardcoded rule은 없다.

### 38.1 1200-step prefix 결과

Dense 기준은 `16 * 2^flat_peak_k`로 계산했다.

| circuit | best policy | steps | actual peak bytes | dense/peak | vs carving_base | QR | transport | runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| distillation | fuse_only | 1200/2040 | 656 | 0.780 | 1.195 | 70 | 62 | 0.02s |
| cultivation_d3 | fuse_only | 344/344 | 1,184 | 0.216 | 1.297 | 320 | 176 | 0.13s |
| coherent_d3_r1 | fuse_only | 255/255 | 1,168 | 0.438 | 1.205 | 100 | 48 | 0.04s |
| coherent_d5_r1 | fuse_only | 857/857 | 51,360 | 2.552 | 2.395 | 492 | 84 | 0.37s |
| coherent_d5_r5 | general_policy | 1200/3228 | 138,578,656 | 1.937 | 4.372 | 2176 | 854 | 46.20s |
| cultivation_d5 | fuse_only | 1200/1784 | 27,136 | 0.604 | 1.784 | 291 | 64 | 0.19s |
| coherent_d7_r1 | fuse_only | 1200/1905 | 86,517,648 | 6.205 | 1.606 | 851 | 134 | 10.71s |

전체 CSV/JSON:

```text
reports/general_policy_benchmark_1200_v2/summary.csv
reports/general_policy_benchmark_1200_v2/summary.json
reports/general_policy_benchmark_1200_v2/report.md
```

### 38.2 Interpretation

결과는 명확하다.

```text
1. executor optimization은 d5/d7 계열에서 의미 있는 memory/work 감소를 보인다.
2. 하지만 단일 policy가 모든 회로에서 best는 아니다.
3. persistent + prefission은 coherent_d5_r5에서 best이다.
4. coherent_d7_r1에서는 persistent window가 resident bond를 키워 memory가 악화되고,
   fuse_only가 best이다.
5. 작은 회로에서는 dense active state가 너무 작아서 TTN 자체가 dense보다 메모리에서
   불리하다.
```

따라서 일반화된 알고리즘은 다음 형태여야 한다.

```text
profile-time executor policy selector:
  candidates = {
    dense fallback,
    carving_base,
    fuse_only,
    persistent_prefission,
    later: staged_region_transport,
  }

  evaluate actual/profiled memory objective
  choose lowest peak subject to correctness and runtime budget
```

즉 최종 claim은 `general_policy 하나를 항상 켠다`가 아니다.

```text
올바른 일반화:
  structured executor candidates를 만들고,
  각 QEC trace의 memory/work profile에 따라 policy를 선택한다.
```

현재 관측상:

```text
coherent_d5_r5:
  persistent_prefission 필요

coherent_d7_r1:
  fuse_only가 더 안정적

small circuits:
  dense fallback 고려 필요
```
