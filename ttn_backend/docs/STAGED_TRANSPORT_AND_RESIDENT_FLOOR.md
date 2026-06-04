# Staged Transport and the Resident Entanglement Floor

이 문서는 "peak를 더 압축하지 말고, peak일 때 작은 메모리로 순차 연산한다"는
전략(streaming)을 실제 TTN runtime에 구현한 결과와, 그 결과가 드러낸 다음 병목
(resident entanglement floor)을 정리한다.

## 1. 전략

```text
1. peak를 더 압축하는 것은 그만한다 (단순 TTN/fission은 2배 이상 어렵다).
2. peak 연산을 블록 단위로 순차(streamed) 실행해 동시 상주 메모리를 작게 유지한다.
3. 메모리가 작은 구간에서는 불필요한 압축/QR을 줄여 시간을 벌고,
   그 시간을 메모리가 큰 peak 구간의 느린 블록 연산에 쓴다.
```

## 2. Feasibility 확인 (synthetic)

모든 peak 연산은 `theta = src @ dst` (GEMM) + `QR(theta)`로 환원된다. 둘 다 exact한
블록/스트리밍 형태가 존재한다. `scripts/streaming_transport_feasibility.py`:

| regime | workspace 절감 | 속도 | 정확도 |
|---|---|---|---|
| tall-skinny (TTN transport 전형, bond rank 작음) | 38x | 0.21x (4.7배 빠름) | 4.6e-16 |
| medium (rank 1024) | 7.8x | 1.30x 느림 | 1.1e-15 |
| near-square (rank 4000) | 1.0x (효과 없음) | 2.94x 느림 | 1.2e-15 |

스트리밍은 bond rank가 작을 때(=TTN이 의미 있는 영역) 큰 효과를 내고, near-square
(=TTN 자체가 무력한 영역)에서만 효과가 없다. 즉 스트리밍이 손해 보는 구간과 TTN이
무력한 구간이 일치한다.

## 3. Staged Transport 구현

`core.py`의 `transport_ident_across_edge`에 통합했다.

- `TTN_STAGED_TRANSPORT=1`: theta를 materialize하기 전에 `full_theta_bytes`를 추정하고,
  cap(`TTN_EXACT_TOTAL_CAP_BYTES`)을 넘으면 staged, 아니면 기존 dense.
- `_staged_factor_blocks`: 전체 theta/M을 만들지 않고 `M = Q·R`을 블록 스트리밍으로 계산.
  - orientation-aware: tall(left≥right)이면 `MᴴM`(row stream + 필요시 CholeskyQR2
    reorthonormalize), wide(left<right)이면 `MMᴴ`(source-column stream, Q는 고유벡터 직접).
  - **항상 작은 쪽 Gram을 만든다.** 첫 버전은 항상 row-stream해서 wide M에서 right²
    Gram이 1957MB로 폭발했다. 이것이 핵심 버그였고, orientation 분기로 해결.
  - rank는 `8·sqrt(eps)·smax`로 floor (Gram이 특이값을 제곱하므로). 조건수가 나쁘면
    (`cond > 1e6`) 예외를 던져 dense fallback (정확성 우선).
- `TTN_STAGED_OUTPUT_FISSION=1`: Q/R 출력 bag이 cap을 넘으면 기존 exact fission으로
  microtree화.

### 정확성

- 단위검증(tall/wide/perm-2D/rank-deficient): recon·orth ~1e-14, rank 정확 복원.
- 회로 검증(`scripts/check_staged_transport_correctness.py`): distillation,
  coherent_d3_r1에서 forced-staged 측정 레코드가 dense와 **bit-identical**.

staged의 peak는 dense와 동일한 destructive-open liveness로 계산한다
(`stored − absorbed + open_region`). 회계를 맞추지 않으면 staged가 거짓으로 나빠 보인다.

## 4. 벤치마크 결과

`scripts/run_general_policy_benchmark.py`, policies: fuse_only, general_policy,
staged_transport, staged_transport_fission. cap = 64 MiB.

### coherent_d5_r5, 1200-step (dense active peak 256 MiB)

| policy | peak | dense/peak | workspace | resident | QR | transport | 시간 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fuse_only | 159.9 MB | 1.68x | 125.8 | 125.8 | 2326 | 860 | 96.9s |
| general_policy | 138.6 MB | 1.94x | 67.1 | 67.1 | 2176 | 854 | 74.2s |
| **staged_transport** | **138.6 MB** | **1.94x** | 67.1 | 67.1 | 2154 | 836 | **43.3s** |
| staged_transport_fission | 138.6 MB | 1.94x | 67.1 | 67.1 | 2154 | 836 | 43.2s |

### coherent_d5_r5, 1395-step

| policy | peak | dense/peak | 시간 |
|---|---:|---:|---:|
| general_policy | 138.6 MB | 1.94x | 83.3s |
| **staged_transport** | **138.6 MB** | **1.94x** | **69.9s** |

### coherent_d7_r1, 1200-step (dense peak 537 MB)

| policy | peak | dense/peak | staged 발동 |
|---|---:|---:|---:|
| **fuse_only** | **88.5 MB** | **6.06x** | - (best) |
| general_policy | 175.9 MB | 3.05x | - |
| staged_transport | 175.9 MB | 3.05x | 0 (미발동) |

### 해석

- staged_transport은 **transport workspace가 병목인 d5_r5에서 general_policy와 동일한
  peak을 ~40% 빠르게** 달성한다. 121.6 MB theta를 materialize하지 않고 exact하게 회피하며,
  prefission의 bag-surgery 없이 같은 메모리 한도에 도달한다.
- d7_r1에서는 단일 transport theta가 cap 미만이라 staged가 발동하지 않는다. 그 peak는
  persistent-region workspace / resident에서 오며, fuse_only가 best다. 정책은 회로별로
  선택해야 한다는 기존 결론과 일치.

## 5. 다음 병목: Resident Entanglement Floor

staged가 workspace를 제거한 뒤, d5_r5 1200-step의 peak(138.6 MB)을 분해하면:

```text
kind = record_metrics (steady-state stored, open_region = 0)
step = 1130, opcode = OP_ARRAY_MULTI_CNOT
peak_stored = 138.58 MB
  B72  [32, 2048, 64]  = 67.1 MB   (internal bag, physical 0)
  B73  [2048, 512, 4]  = 67.1 MB   (internal bag, physical 0)
  ...
shared bond = 2048 = 2^11,  saturated:  32×64 = 512×4 = 2048
```

즉 peak는 transport workspace가 아니라 **연산 사이에 저장된 state**이고, B72–B73 사이의
Schmidt cut이 `chi = 2^11`이다. 24-qubit 상태(dense 2^24 = 256 MB)에서 11-ebit cut을
**exact**하게 표현하는 최소 비용은 양쪽 텐서 합 `~2 × 2^22 ≈ 134 MB`이고, 이는 관측된
138 MB와 일치한다.

## 5.1 큰 bond를 "줄이는" 것과 "덜 여는" 것은 다르다

현재 exact 실행에서 `chi=2048` 같은 큰 bond를 임의로 `512`로 낮추는 것은 일반적으로
불가능하다. 그것은 Schmidt rank 자체를 줄이는 문제이고, exact baseline에서는 실제
entanglement가 있으면 유지해야 한다.

하지만 큰 bond를 지나는 transport/refactor 호출 수는 layout과 scheduling에 따라 달라진다.
tree edge `e`가 active axes를 `L_e | R_e`로 나눌 때, 2축 op `(u,v)`가 다음 조건을
만족하면 그 op는 edge `e`를 건넌다.

```text
u in L_e, v in R_e   or   u in R_e, v in L_e
```

따라서 다음 값은 줄일 수 있는 실행 비용이다.

```text
N_cross(e) = #{2-axis active ops whose operands are separated by e}
rank_weighted_cross(e) = sum over crossings log2(chi_e at crossing time)
```

이 최적화의 목표는:

```text
not:  chi_e = 2048 -> 512
but:  high-rank edge e를 여는 transport/refactor 횟수 감소
```

이다. resident memory floor를 없애지는 못하지만, 큰 bond를 반복해서 여는 QR/transport
work와 workspace materialization을 줄인다.

### Big-edge crossing audit

이를 실제 trace에서 확인하기 위해 다음 진단 스크립트를 추가했다.

```text
ttn_backend/scripts/big_edge_crossing_audit.py
```

예시 실행:

```bash
/home/jung/clifft_env/bin/python ttn_backend/scripts/big_edge_crossing_audit.py \
  coherent_d5_r5 \
  --metrics-json reports/staged_bench_d5r5_1200/staged_transport/coherent_d5_r5/coherent_d5_r5/carving_leaf_metrics.json \
  --out-dir reports/big_edge_crossing_audit_d5r5_1200
```

출력:

```text
reports/big_edge_crossing_audit_d5r5_1200/
  coherent_d5_r5_big_edge_crossing_edges.csv
  coherent_d5_r5_big_edge_crossing_ops.csv
  coherent_d5_r5_big_edge_crossing_windows.csv
  coherent_d5_r5_big_edge_crossing_report.md
  coherent_d5_r5_big_edge_crossing_summary.json
```

`coherent_d5_r5`, 1200-step staged run의 상위 edge는 다음과 같다.

| edge | max chi | runtime hits | rank-hit | static crossing ops | MULTI_CNOT crossing controls | diagnosis |
|---|---:|---:|---:|---:|---:|---|
| 73-74 | 512 | 33 | 239.8 | 319 | 262 | cluster/window candidate |
| 72-73 | 2048 | 29 | 230.0 | 243 | 204 | cluster/window candidate |
| 74-84 | 32 | 22 | 85.0 | 261 | 225 | cluster/window candidate |
| 20-21 | 16 | 23 | 75.3 | 172 | 150 | cluster/window candidate |
| 2-20 | 16 | 23 | 75.3 | 161 | 144 | cluster/window candidate |

해석:

```text
1. d5_r5의 큰-edge work offender는 일반 CNOT보다 OP_ARRAY_MULTI_CNOT 비중이 크다.
2. 따라서 "큰 edge를 덜 열기"의 1차 대상은 MULTI_CNOT target/control clustering,
   persistent window, parking/lifetime scheduling이다.
3. resident peak floor 자체는 여전히 존재하므로 resident streaming과 병행해야 한다.
```

이 결과는 다음 최적화가 회로/step/bag hardcoding이 아니라, runtime trace에서 top
rank-weighted edge를 찾고 그 edge를 반복해서 가르는 op/window를 줄이는 일반 정책으로
정의될 수 있음을 보여준다.

### Static clustering 실험 결과

위 audit에 따라 작은 `OP_ARRAY_MULTI_CNOT` support를 target home bag에 정적으로 모으는
실험을 구현했다.

```text
TTN_CLUSTER_MULTICNOT_TOP
TTN_CLUSTER_MULTICNOT_MIN_CONTROLS
TTN_CLUSTER_MULTICNOT_MAX_SUPPORT
TTN_CLUSTER_MULTICNOT_MAX_BAG_OWN
```

이 정책은 회로/step/bag hardcoding 없이 `base_spec["op_to_bag"]`의 MULTI_CNOT
window를 보고 target/control을 같은 home bag으로 옮긴다. 그러나 actual runtime 결과는
나빴다.

| policy | status | steps | total peak | stored peak | workspace | QR | transport | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| staged baseline | complete | 1200 | 138,578,656 B | 138,578,656 B | 67,108,864 B | 2154 | 836 | 43.3s |
| cluster top1 | timeout | 750 | 92,464,592 B | 92,464,592 B | 50,331,648 B | 669 | 316 | 151.7s |
| cluster top3 | timeout | 750 | 92,465,216 B | 92,465,216 B | 50,331,648 B | 649 | 308 | 153.1s |
| cluster top20 | timeout | 990 | 356,666,896 B | 218,273,296 B | 150,994,944 B | 2740 | 1576 | 255.2s |

해석:

```text
static clustering은 특정 MULTI_CNOT window에는 유리하지만,
1. local p_B를 키우고,
2. 다른 op/window path를 길게 만들며,
3. resident entanglement floor를 낮추지 못한다.
```

따라서 big-edge crossing reduction은 static homing rewrite가 아니라 runtime-local 정책으로
가야 한다.

```text
next:
  - parking/lifetime scheduling: 한 번 큰 edge를 건넌 ident를 즉시 복귀시키지 않음
  - edge-aware persistent window selector: top rank-hit edge를 기준으로 window를 유지
  - resident streaming: 큰 bond 자체의 RAM resident를 block-store로 낮춤
```

### Fission은 resident를 줄이지 못한다 (오히려 악화)

fission cap을 50 MB로 낮춰 강제 발동시킨 실험(`reports/staged_fission_probe`):

```text
fission 10회 발동
  B72 [2,24,1024,64] 50MB -> child 16.8MB (rank 16)    # 일부 축은 저rank
  B148[2,1024,1536]  50MB -> child 33.6MB (rank 1024)  # 1024 bond은 irreducible
결과 peak_stored: 138.6MB -> 156.6MB (악화), total peak 160.9MB
fission temp peak: 67.1MB
```

개별 bag은 일부 축에서 저rank라 쪼개지지만, 큰 bond(1024~2048)는 rank가 그대로다
(genuine entanglement). fission은 bag 수와 총 stored를 늘려 `sum_B |B|`를 악화시키고,
fission SVD 자체가 67 MB temp peak를 만든다.

### 결론

```text
1. workspace overhead: staged transport으로 exact하게 제거됨.
   (1200-prefix에서 TTN을 dense보다 나쁘게 만들던 항이 사라짐.)
2. resident state: max Schmidt cut에 대해 ~2^(2·cut)이 exact 하한이다.
   genuinely entangled 구간에서는 fission/압축으로 이 값을 내릴 수 없다.
3. 따라서 exact TTN의 peak는 이제 회로의 실제 entanglement(2^(2·maxcut))를 따른다.
   max cut < k/2인 회로에서는 dense(2^k)보다 작아 clifft의 k=24 벽을 넘을 수 있다.
```

이 하한을 더 내리는 lever는 두 가지뿐이다.

```text
A. out-of-core / blocked resident streaming:
   큰 bag을 디스크/블록에 두고, 연산 때만 큰 bond를 블록 단위로 스트리밍한다.
   RAM peak = 블록 크기, state = 디스크. 시간을 대가로 더 큰 k를 가능하게 한다.
   (이것이 streaming 전략의 resident 버전이다.)

B. approximation (bond truncation):
   2^11 bond를 잘라 resident를 줄이되, 측정 분포 오차를 별도 보고해야 한다.
   (현재 연구가 피하려는 방향.)
```

## 7. Resident Tensor-Bag Streaming (방향 A) — Feasibility 확정

중요한 구분:

```text
staged transport = "계산용" 임시 tensor(theta)를 스트리밍한다 (workspace). 완료됨.
resident streaming = "저장용" tensor-bag 자체를 block-store 상태로 유지한다. 신규.
```

핵심은, 큰 bag을 블록으로 계산한 뒤 다시 하나의 dense ndarray로 RAM에 합치면
peak가 그대로라는 점이다. 따라서 큰 bag은 연산 후에도 block-store 상태를 유지해야 한다.

```python
bag.tensor = np.ndarray          # 기존: 항상 RAM-resident dense
bag.tensor = BlockTensorStore    # 신규: 큰 bond 축으로 블록, out-of-core(memmap)
```

### 7.1 BlockTensorStore

`ttn_backend/block_tensor_store.py`:

- 한 축(보통 가장 큰 bond)으로 블록화하고, 전체 tensor는 단일 memmap 파일(디스크)에 둔다.
- `ram_bytes` = 블록 하나(cache), `ooc_bytes` = 전체(디스크). peak 회계는 블록만 센다.
- exact block-wise 연산: `squared_norm`, `apply_diagonal_on_axis`,
  `apply_matrix_on_axis`, `axis1_squared_norm`(측정 marginal), block 축을
  contract하는 transport(블록 스트리밍 + 누적).

연산은 block 축 `a`에 대해 세 종류로 나뉜다.

```text
1. non-block 축에 작용 (diagonal/local 1-axis, b != a 측정): 블록마다 적용. RAM=1블록.
2. block 축을 contract (인접 transport가 그 bond를 가로지름): 블록 스트림+누적. RAM=1블록.
3. block 축을 다른 축과 묶어 matrix로 reshape: 블록 유지 불가 → 재블록/materialize.
```

### 7.2 Feasibility 결과

`ttn_backend/scripts/resident_streaming_feasibility.py`:

B72 규모 tensor `[2,2048,1024]`(67 MB)를 2048 축으로 block-store(블록 256, 8블록):

| 연산 | error | block RAM |
|---|---:|---:|
| squared_norm | 1.3e-15 | 8.39 MB |
| diagonal (T) on axis 0 | 0 | 8.39 MB |
| local 2x2 (H) on axis 0 | 0 | 8.39 MB |
| Z-marginal P(x0=1) | 3.9e-16 | 8.39 MB |
| contract OVER 2048 bond | 6.8e-16 | 8.54 MB |

```text
dense 67.1 MB -> 블록 1개 8.39 MB resident = 8.0x, 모두 exact
```

실제 d5_r5 peak 시나리오(B72 `[32,2048,64]` + B73 `[2048,512,4]`)를 직접 모사:

```text
dense pair resident          = 134.2 MB
block-store 결합 resident      =  16.78 MB  (8.0x 작음)
2^11 bond 가로지르는 streamed transport: error 6.9e-16 (exact)
-> 134 MB 쌍이 절대 동시 상주하지 않는다.
```

즉 resident streaming은 exact하며, 134 MB peak를 ~2 블록(약 17 MB)까지 내릴 수 있음이
확인됐다. (단, transport의 contraction *결과* 자체도 커질 수 있으므로 출력은 staged
transport이 즉시 QR-split로 스트리밍해야 한다. 즉 resident-streaming(저장) +
staged-transport(연산)이 합쳐져야 양쪽 모두 작게 유지된다.)

### 7.3 dense Clifft와의 차이

```text
dense Clifft       : full active state vector(2^k)가 RAM에 있어야 함.
TTN + resident stream: full vector도, 큰 bag 전체도 RAM에 없음. 필요한 블록만 RAM.
```

이것이 exact한 이유: 정보를 버리지 않고 저장 위치만 RAM 밖으로 옮긴다.
compression이 아니라 exact out-of-core / blocked resident execution이다.

### 7.4 Runtime 통합 로드맵 (다음 단계)

저장층(BlockTensorStore)과 exact·RAM-bounded feasibility는 검증됐다. run_shot 전체를
한 번에 바꾸지 않고 다음 순서로 붙인다.

```text
1. 회계: _stored_bytes가 block-store bag을 블록 cache로 세고, out_of_core_bytes를 따로 기록.
   spill hook: bag_bytes > TTN_RESIDENT_STREAM_CAP_BYTES면 BlockTensorStore로 spill.
2. block 축을 안 섞는 op부터 block-aware로:
   norm/normalize, diagonal/local 1-axis, 측정(marginal+projection, 측정축 != block축).
3. peak op(B72-B73 transport)을 block-aware staged transport로:
   입력을 block-store에서 스트림, 출력 Q/R을 block-store로 직접 write
   (staged factor가 이미 블록 스트리밍이므로 입력 소스/출력 싱크만 store로 교체).
4. block 축을 섞는 op는 재블록 또는 일시 materialize로 안전 fallback (정확성 우선).
5. d5_r5 1200 prefix에서 resident_actual_peak가 ~134 MB -> 블록 한도로 내려가는지 실측.
```

산출물:

```text
ttn_backend/block_tensor_store.py                    # out-of-core blocked storage
ttn_backend/scripts/resident_streaming_feasibility.py # exact + 8x resident 검증
```

## 6. 산출물

```text
core.py
  _staged_factor_blocks, _staged_rank_from_gram   # 블록 스트리밍 exact 인수분해
  _transport_ident_across_edge_method             # staged/dense 분기 + 회계
scripts/streaming_transport_feasibility.py        # synthetic feasibility
scripts/check_staged_transport_correctness.py     # dense vs staged bit-identical
scripts/run_general_policy_benchmark.py           # +staged_transport(_fission) 정책

reports/staged_bench_d5r5_1200/                    # 1200-step 4정책
reports/staged_bench_d5r5_1395/                    # 1395-step 4정책
reports/staged_bench_d7r1_1200/                    # d7r1 4정책
reports/staged_fission_probe/                      # resident fission 비효과 실험

env flags:
  TTN_STAGED_TRANSPORT, TTN_STAGED_TRANSPORT_FORCE, TTN_STAGED_BLOCK_BYTES,
  TTN_STAGED_MIN_BYTES, TTN_STAGED_COND_MAX, TTN_STAGED_FORCE_REORTH,
  TTN_STAGED_OUTPUT_FISSION
```
