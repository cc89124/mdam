# benchmark_comparison — 현재 유효한 결과

측정 규약(모든 표 공통): taskset -c 2, single-thread BLAS, cold-amortized total_wall/N,
python=/home/jung/clifft_env/bin/python. Clifft은 외부 baseline으로만 사용(알고리즘 내부 결정에 불개입).

**2026-07-07 기준선: MDAM은 state를 캐시하지 않는다.** 유일한 캐시는 BoundaryKey→transition
(자동자 테이블). dense state 스냅샷 풀(mc_pool)은 원리 위배로 알고리즘에서 제거(컴파일 기본값
`mc_pool_off=1`), 모든 표는 pool-free 빌드로 재측정. 근거 ablation은 analysis.md §6.

| 파일 | 내용 |
|---|---|
| **wall_table.csv** | **최종 wall 표** (11 벤치). `mdam_ns`/`mdam_vs_clifft` = 전체 알고리즘(v2 adaptive 판정 + walk 컴파일 게이트/경주, cold 단일호출). route는 딱 셋: `LEAN(compiled)` / `LEAN(interp)` / `AUTH`. ablation 열: `auth_only_ns`+`auth_only_vs_clifft`(전부 AUTH 강제) / `lean_only_ns`+`lean_only_vs_clifft`(**LEAN 강제 — AUTH 라우팅만 비활성, 컴파일 게이트/경주는 프로덕션과 동일 기준·동일 N**; `lean_fb_pct`=그 실행의 전체 miss%) / `nocompile_ns`(판정만, 컴파일 금지). 따라서 LEAN 회로에선 mdam≈lean_only이고, AUTH 회로에서 (mdam−lean_only) 차이가 곧 AUTH 옵션의 가치다. |
| wall_table.tsv | 같은 데이터 + 상세 문서 주석(측정 프로토콜, v2 판정식, pool-free 기준선, 검증 내역). 기존 스크립트 호환용 구 열 순서 유지. |
| analysis.md | 과정 리포트 통합본: 비용 구조, d3_r3 규명, 포화 진단, codegen 회계, state-pool 제거 ablation. |
| adaptive_algorithm_results/ | 벤치별 3-panel 논문 그림: (A) 누적 distinct BoundaryKey U(n) 포화, (B) window hit ratio h_w vs 실측 손익분기 h_be, (C) 실측 LEAN vs AUTH runtime 교차. |
| flop_table.md / flop_table_native.csv | FLOP 축 비교표 (wall과 별개 축). |
| saturation_curve.png | 포화 진단 (bcap local-key, 구현 독립): marginal miss m(N), distinct keys D(N) 기울기 β, 투영 E[T]. |
| lean_warm_long.png | 실제 lean walk의 장기 학습 실측 (fb 감쇠, 자동자 성장, 학습 중 wall). |
