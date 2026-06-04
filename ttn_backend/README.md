# TTN Backend Directory Layout

이 디렉터리는 Clifft paper용 TTN backend 구현, RASL 분석 코드, metric 수집 스크립트, 검증 테스트를 한곳에 모은 package다.

## Current Status

최신 결론은 다음과 같다.

```text
persistent MULTI_CNOT fusion은 실제 runtime에서 QR/transport와 memory peak를 줄인다.
staged transport는 큰 theta workspace를 exact block-streaming으로 제거한다.
coherent_d5_r5 1200-step prefix의 현재 best exact peak는 약 138.6 MB이며,
dense 256 MiB 대비 약 1.94x 절감이다.
```

대표 수치:

```text
coherent_d5_r5, dense peak = 256 MiB

839-step persistent exact:
  actual peak = 32.08 MiB
  dense / actual = 7.98x

968-step persistent exact:
  actual peak = 32.08 MiB
  dense / actual = 7.98x

1200-step staged_transport exact:
  actual peak = 138.58 MB
  dense / actual = 1.94x
  QR = 2154, transport = 836
  => workspace 병목은 제거됐고, 남은 병목은 resident entanglement floor.

1200-step persistent_svd rtol=1e-2:
  actual peak = 107.68 MiB
  dense / actual = 2.38x
  => approximate/numerical-rank compression. 별도 correctness/error 보고 필요.
```

따라서 현재 핵심 문제는 layout 하나나 MULTI_CNOT 하나가 아니라, 모든 cross-bag
operation/fallback 경로를 같은 concurrent memory cap 아래에서 선택하는
executor-selection 문제다. 또한 `chi=2048` 같은 큰 bond 자체를 exact하게 줄이기
어려운 구간에서는, 그 bond를 지나는 transport/refactor 횟수를 줄이는
big-edge crossing reduction과 resident streaming을 병행해야 한다. 자세한 정리는
`docs/STAGED_TRANSPORT_AND_RESIDENT_FLOOR.md`와 `docs/TTN_METHOD_DETAILED.md`를 본다.

루트 디렉터리에 흩어져 있던 TTN 관련 파일을 다음 구조로 정리했다.

```text
ttn_backend/
  __init__.py
  core.py
  backend_spec.py
  frame_layer.py
  layout_transform.py
  treewidth.py
  rasl/
    __init__.py
    symplectic.py
    candidate.py
    builders.py
    cost.py
    select.py
  scripts/
    actual_rasl_experiment.py
    metrics_report.py
    memory_risk_report.py
    memory_diagnosis_report.py
    time_graph_report.py
    static_ttn_compression_experiment.py
    analyze_static_ttn_tree.py
    fixed_topology_reuse_experiment.py
    multisnapshot_global_tree_search.py
    rasl_report.py
    rasl_audit.py
    measure_paths.py
    verify_ttn.py
    run_ttn.py
    run_clifft.py
    compute_treewidth.py
    debug_bytecode.py
    compare.py
  tests/
    test_rasl_symplectic.py
    test_ttn_transport.py
  docs/
    TTN_BACKEND.md
    TTN_METHOD_DETAILED.md
    RASL_METHOD_AND_RESULTS.md
    TTN_MEMORY_DIAGNOSIS.md
```

## Core Modules

- `core.py`: `TTNState`, `TTNBag`, `TTNBackend` runtime. Adjacent 2-bag transport, QR refactor, bytecode dispatch, actual metric instrumentation을 담당한다.
- `backend_spec.py`: Clifft bytecode structural replay, ident lifecycle, interaction graph, JT/union bag layout, home assignment, op classification.
- `treewidth.py`: active interaction graph와 treewidth/JT layout 계산.
- `frame_layer.py`: Pauli frame, noise site sampling, constant-pool helper.
- `layout_transform.py`: hub-degree reduction 등 layout transform 실험 코드.

## RASL Modules

- `rasl/symplectic.py`: binary symplectic Pauli vector engine, phase-aware H/S/CNOT/CZ conjugation.
- `rasl/candidate.py`: localization candidate 자료구조와 verification.
- `rasl/builders.py`: active-only Z-normalize routing candidate 생성.
- `rasl/cost.py`: fixed-layout path/workspace/resident proxy cost 계산.
- `rasl/select.py`: resident proxy를 악화시키지 않는 candidate 선택.

## Scripts

작업 디렉터리는 `/home/jung/clifft-paper`로 두고 module 실행을 권장한다.

```bash
cd /home/jung/clifft-paper
```

RASL analysis/proxy report:

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

RASL changed-step audit:

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.rasl_audit \
  --csv reports/rasl_steps_full.csv \
  --circuit coherent_d5_r5 \
  --json reports/rasl_changed_audit_coherent_d5_r5.json
```

Actual RASL experiment:

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.actual_rasl_experiment \
  distillation cultivation_d3 coherent_d3_r1 coherent_d5_r1 \
  --enable-rasl-exec-active-only \
  --runtime-timeout 60 \
  --out-csv reports/actual_rasl_comparison_small.csv \
  --out-json reports/actual_rasl_comparison_small.json \
  --out-md reports/actual_rasl_report_small.md
```

Static/runtime memory metrics:

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.metrics_report \
  --runtime-timeout 60 \
  --variants baseline \
  --out-csv reports/baseline.csv \
  --out-json reports/baseline.json
```

Memory-risk offender report:

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.memory_risk_report \
  --variants baseline,hub3 \
  --include-runtime \
  --out-csv reports/memory_risk.csv \
  --out-json reports/memory_risk.json \
  --out-md reports/memory_risk_summary.md
```

Actual peak-memory decomposition:

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.memory_diagnosis_report \
  distillation cultivation_d3 coherent_d3_r1 coherent_d5_r1 \
  --variants baseline \
  --runtime-timeout 60 \
  --out-summary-csv reports/ttn_memory_diagnosis_summary.csv \
  --out-edges-csv reports/ttn_memory_diagnosis_edges.csv \
  --out-json reports/ttn_memory_diagnosis.json \
  --out-md reports/ttn_memory_diagnosis.md
```

Time-varying live-graph evolution:

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.time_graph_report \
  --runtime-timeout 60 \
  --variants baseline \
  --out-summary-csv reports/time_graph_summary.csv \
  --out-steps-csv reports/time_graph_steps.csv \
  --out-critical-csv reports/time_graph_critical.csv \
  --out-b0-edges-csv reports/time_graph_b0_edges.csv \
  --out-overlap-csv reports/time_graph_b0_overlap.csv \
  --out-json reports/time_graph_report.json \
  --out-md reports/time_graph_report.md
```

이 리포트는 step별 live axes/bags/TTN edges, peak/critical snapshot,
B0 incident edge simultaneity, union-vs-live bond load gap을 기록한다.

Big-edge crossing audit:

```bash
/home/jung/clifft_env/bin/python ttn_backend/scripts/big_edge_crossing_audit.py \
  coherent_d5_r5 \
  --metrics-json reports/staged_bench_d5r5_1200/staged_transport/coherent_d5_r5/coherent_d5_r5/carving_leaf_metrics.json \
  --out-dir reports/big_edge_crossing_audit_d5r5_1200
```

이 리포트는 실제 runtime의 `edge_hit_count`, `edge_rank_weighted_hits`,
`edge_max_bond_dim`을 Clifft bytecode op/window와 연결한다. 목적은 큰 bond 자체를
줄이는 것이 아니라, 큰 bond를 지나는 transport/refactor 호출 수를 줄일 수 있는
layout clustering, persistent window, parking/lifetime scheduling 후보를 찾는 것이다.

주의: 이 audit 결과를 바탕으로 `TTN_CLUSTER_MULTICNOT_TOP` 정적 clustering을
구현해 시험했지만, `coherent_d5_r5` 1200-step에서는 악화됐다. top1/top3는 750-step에서
timeout, top20은 990-step에서 356 MB peak를 만들었다. 따라서 이 knob은 기본 off이고,
현재 결론은 static homing rewrite가 아니라 runtime-local parking/window selector가
필요하다는 것이다. 상세 결과는
`reports/big_edge_crossing_audit_d5r5_1200_v2/static_clustering_experiment.md`.

Static peak-bag TTN compression feasibility:

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.static_ttn_compression_experiment \
  --circuit coherent_d5_r5 \
  --step 977 \
  --bag B0 \
  --rank-rules rel energy \
  --tols 1e-8 1e-6 1e-4 \
  --mode depth1 recursive \
  --random-candidates 100 \
  --top-svd 6 \
  --max-depth 4 \
  --out-dir reports
```

이 실험은 full runtime 변경 없이 고정 peak tensor 하나를 SVD 기반
numerical-rank TTN으로 다시 분해해, 더 작은 정적 tensor structure가
존재하는지 확인한다.

Beam-search version:

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

현재 `coherent_d5_r5` B0 peak tensor에서는 greedy recursive `rel_tol=1e-8`의
peak `log2(numel)=18.0`보다 beam search가 더 작은 `17.0` 구조를 찾았다.

Beam tree bottleneck decomposition:

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.analyze_static_ttn_tree \
  --out-dir reports/static_rel1e8_beam
```

Fixed-topology reuse test:

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.scripts.fixed_topology_reuse_experiment \
  --circuit coherent_d5_r5 \
  --bag B0 \
  --tree-json reports/static_rel1e8_beam/static_ttn_b0_compression_tree_beam_rel_1em08.json \
  --steps 977 978 979 988 989 903 930 935 944 949 967 \
  --rel-tol 1e-8 \
  --snapshot-cache-dir reports/fixed_topology_reuse_rel1e8/snapshots \
  --out-dir reports/fixed_topology_reuse_rel1e8
```

현재 reuse 결과는 구조적으로는 11/11 step에서 성공했지만, step `944`에서
peak compression이 `2.67x`까지 떨어졌다. 따라서 단일 step-977 topology를 바로
runtime에 patch하기보다 multi-snapshot common-skeleton search가 다음 단계다.

## Tests

```bash
/home/jung/clifft_env/bin/python -m ttn_backend.tests.test_rasl_symplectic
/home/jung/clifft_env/bin/python -m ttn_backend.tests.test_ttn_transport
```

Both pass after the reorganization.

## Import Compatibility

기존 핵심 import는 유지된다.

```python
from ttn_backend import TTNBackend, TTNState
```

세부 모듈은 package 경로를 사용한다.

```python
from ttn_backend.backend_spec import export_backend_spec, assign_homes_and_classify
from ttn_backend.layout_transform import reduce_hub_degree
from ttn_backend.rasl.candidate import CliffordOp
```

이제 루트의 `ttn_backend.py`, `backend_spec.py`, `frame_layer.py`, `treewidth.py`, `layout_transform.py`, `ttn_rasl/`는 package 내부로 이동했다.

## Reports

실험 산출물은 루트의 `reports/` 아래에 둔다. 중간 smoke/debug/tiny/deep 실행
결과와 대형 snapshot cache는 정리했고, 현재는 재현과 의사결정에 필요한 canonical
결과만 남긴다.

```text
reports/
  static_rel1e8_beam/
    static_ttn_b0_compression_summary.csv
    static_ttn_b0_compression_tree_beam_rel_1em08.json
    beam_tree_bottleneck_report.md

  fixed_topology_reuse_rel1e8/
    reuse_summary.csv
    reuse_report.md
    reuse_per_step_tree_stats.json

  multisnapshot_global_rel1e8/
    summary.csv
    per_step.csv
    best_tree.json
    report.md
```

즉 코드와 검증 스크립트는 `ttn_backend/`로 모으고, 실험 출력 데이터는 `reports/`에
유지하되, 중간 산출물은 기본적으로 보관하지 않는다. snapshot cache는 필요하면 각
script가 다시 생성한다.

## Detailed Method Notes

현재 TTN backend의 전체 실행 방식, tensor convention, Class A/B/C dispatch,
adjacent 2-bag transport sweep, actual/proxy memory metric 구분, 큰 회로 진단과
static peak-bag compression 결과는 다음 문서에 자세히 정리했다.

```text
ttn_backend/docs/TTN_METHOD_DETAILED.md
```

최근 상태 요약:

```text
coherent_d5_r5 1200-step prefix:
  dense peak                         = 268,435,456 B
  persistent MULTI_CNOT + destructive pair-open
                                     = 157,475,136 B  (1.70x dense 대비 절감)
  + exact runtime bag fission        = 141,928,608 B  (1.89x dense 대비 절감)
```

해석:

```text
1. MULTI_CNOT per-control 병목은 persistent fused executor로 줄었다.
2. 1200-step 병목은 Class B/C path transport와 큰 resident bag tensor로 이동했다.
3. destructive pair-open은 workspace와 absorbed pair tensor를 동시에 세지 않게 해서
   dense 이하 exact memory를 회복한다.
4. exact bag fission은 resident peak를 추가로 낮추지만, 현재 prototype은 SVD 비용이
   크므로 critical-only/cached 정책이 필요하다.
```
