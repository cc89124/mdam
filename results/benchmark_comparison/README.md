# benchmark_comparison

11개 QEC 벤치마크에 대한 MDAM vs clifft 비교 데이터. 측정 규약(모든 표 공통): 단일 코어 고정
(taskset), single-thread BLAS, cold-amortized total_wall/N. clifft는 기본 컴파일 설정(squeeze
최적화 포함)이며 비교 기준으로만 쓰인다.

## 파일

| 파일 | 내용 |
|---|---|
| **wall_table.csv** | **최종 wall 표** (11 벤치). 열 설명은 아래. |
| wall_table.tsv | 같은 데이터 + 측정 프로토콜·판정식·검증 내역 상세 주석. |
| analysis.md | 비용 구조 분해(제어 vs 산술), 포화 진단, walk 컴파일 회계, state-캐시 ablation 등 결과를 뒷받침하는 분석 모음. |
| adaptive_algorithm_results/ | 벤치별 adaptive 판정 기록 그림: (A) 누적 distinct BoundaryKey U(n) 포화 곡선, (B) window hit ratio h_w vs 실측 손익분기 h_be, (C) 실측 LEAN vs AUTH runtime 교차. |
| active_state_results/ | active magic rank(maxM) 관련 그림. |
| flop_table.md / flop_table_native.csv | FLOP 축 비교표 (wall과 별개 축). |
| saturation_curve.png / lean_warm_long.png | 포화 진단: marginal miss m(N), distinct key 성장 기울기 β, lean walk 장기 학습 실측. |

## wall_table.csv 열

- `clifft_ns` — baseline shot당 ns.
- `mdam_ns` / `mdam_vs_clifft` — **전체 알고리즘**(probe → LEAN/AUTH 라우팅 → walk 컴파일
  게이트/경주 → cruise, cold 단일호출)의 shot당 ns와 배수. 표의 대표 숫자.
- `route` — 실행이 최종적으로 안착한 경로: `LEAN(compiled)` / `LEAN(interp)` / `AUTH`.
- `shots` — 측정 배치 크기 N.
- `bitexact` — authoritative 기준 경로 대비 record 일치 (spot 2×2000).
- `auth_only_ns` / `auth_only_vs_clifft` — ablation: 전 구간 AUTH 강제. 캐시 층의 기여를 분리.
- `lean_only_ns` / `lean_only_vs_clifft` — ablation: LEAN 강제(AUTH 라우팅 비활성, 컴파일
  게이트/경주는 동일 기준·동일 N). AUTH 옵션의 기여를 분리. `lean_fb_pct` = 그 실행의 전체 miss%.
- `nocompile_ns` — ablation: 판정은 전부 수행하되 walk 컴파일만 금지. 컴파일 층의 기여를 분리.
- `regime` — 회로 분류: BoundaryKey 포화 / localization(r≪k) / 비포화.

읽는 법: LEAN 회로에서는 `mdam ≈ lean_only`(라우팅 오버헤드는 오차 수준), AUTH 회로에서는
`mdam ≈ auth_only`이고 `mdam − lean_only` 격차가 AUTH 옵션의 가치다. `lean_only − nocompile`
비교는 컴파일 층의 가치를 보여준다.
