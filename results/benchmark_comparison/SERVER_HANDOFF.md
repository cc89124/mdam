# 서버 측정 지시문 (SNU Xeon Gold 6246R)

목표: 이 서버에서 clifft와 MDAM의 wall time을 전체 11개 벤치마크에 대해 측정하고
`results/benchmark_comparison/wall_table.csv` 를 이 환경 기준으로 갱신한다.

## 상황
- 이 저장소가 작업 대상. 측정 스크립트: `results/benchmark_comparison/scripts/`
  (`clifft_wall.py`, `tier_wall_row.py`, `measure_all512.py` 등 — 경로 자동감지, 수정 불필요).
- conda env `mdam` (python 3.12, `$CXX` = conda gcc 14.3). clifft 0.5.0 소스빌드 완료, `avx512` 확인됨.
- native VM을 `$CXX -O3 -march=native -std=c++17 -DNDEBUG -shared -fPIC native_mdam_vm.cpp
  ../backend/clifft_axis/cpp/mdm_core_executor.cpp -o native_mdam_vm.so` 로 컴파일했더니
  `scripts/measure_all512.py coherent_d3_r1 100000` 실행 시 native 배치 호출에서 segfault.

## 0단계 — segfault 해결 (최우선)
1. `-march=native` 대신 `-mavx2` 로 VM 재컴파일 후 같은 명령 재실행. 통과하면 avx2 빌드로 확정
   (VM은 비트연산 위주라 avx512 이득 거의 없음. clifft는 avx512 유지 — baseline에 유리한 보수적 조건).
2. 그래도 죽으면 clifft 버전 문제 가능성 (원측정은 0.4.2.dev2+g2655e48c6):
   `git clone https://github.com/unitaryfoundation/clifft ~/clifft-src && cd ~/clifft-src &&
   git checkout 2655e48c6 && pip install .` 후 재실행. (commit이 upstream에 없으면 보고하고 중단)
3. 필요하면 gdb backtrace로 원인 파악 (`mdam/native_vm/*.hpp` 소스 있음).
   무엇을 바꿨고 왜 결과에 안전한지 보고에 명시할 것.

## 측정 환경 규칙 (전 측정 공통)
- 물리코어 1개 고정: `cat /sys/devices/system/cpu/cpu2/topology/thread_siblings_list` 로 HT 짝 확인
  후 `taskset -c 2`. `top`으로 다른 사용자 부하 확인 — 부하 크면 한가한 물리코어로 바꾸고 기록.
- OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1 (스크립트가 자동 설정).
- 측정은 순차 실행 (동시 실행 금지).

## 1단계 — clifft baseline
`scripts/clifft_wall.py <bench> [target_s=20]` : 기본 컴파일 설정(squeeze 포함) + `clifft.sample`
레코드 샘플링, cold-amortized wall/N. 벤치 11개:
`coherent_d3_r1 cultivation_d3 coherent_d3_r3 distillation surface_d7_r7 coherent_d5_r1
coherent_d7_r1 coherent_rx_d3_r1 cultivation_d5 coherent_rx_d3_r3 coherent_d5_r5`
- `coherent_d7_r1`, `coherent_d5_r5` 는 clifft가 shot당 수 초 → target_s=60, shot 수 10개 미만이면 명시.

## 2단계 — MDAM production + ablations
`scripts/tier_wall_row.py <bench> <lean_ns> <clifft_ns> <resfile> [flags]` 가 홈 머신 프로토콜 원본:
- **mdam_ns**: 프로덕션 엔트리 `mdam_run.run_batch` (probe→route→gate→race→cruise), .so 캐시
  prewarm 후 cold 단일 호출. N 정책 = `max(150000, 18e9/lean_ns)`, 상한 10M shots & 400MB 버퍼.
  예외: `cultivation_d3` N=1,000,000 / `coherent_d5_r5` N=8,000.
  (`lean_ns` 인자는 N 결정용 추정치 — 가벼운 프로브로 구해 넣으면 됨)
- **lean_only**: `force_lean` 플래그 (AUTH 라우팅 비활성, 컴파일 게이트/레이스는 동일).
- **auth_only**: `nvm_mdam_sample_batch` 직접 호출을 동일 N에서 cold-amortized
  (`measure_all512.py`의 `auth_rep` 참고).
- **nocompile_ns**: `noprewarm` 플래그 (.so 캐시 없이 콜드).
- **비트 정확성**: run_batch(2000) vs sample_batch(2000), seed 11·22, mismatch 0 필수
  (tier_wall_row.py에 구현). 실패 시 그 벤치는 FAIL로 표기하고 계속 진행.
- reduce_full 재시도: 대규모 N에서 ~1/2M shot 확률의 RuntimeError → seed 바꿔 재시도가 정상
  (구현돼 있음). 재시도 횟수 보고.

## 3단계 — wall_table.csv 갱신
기존 컬럼 구조 유지:
`bench,k,maxM,nmeas,clifft_ns,mdam_ns,mdam_vs_clifft,route,shots,bitexact,auth_only_ns,
auth_only_vs_clifft,lean_only_ns,lean_only_vs_clifft,lean_fb_pct,nocompile_ns,regime`
- `k, maxM, nmeas, regime` 은 회로 속성 → 기존 값 유지.
- `X_vs_clifft` = clifft_ns / X_ns ("56.95x" 형식).
- csv 옆 README 또는 csv 헤더 주석에 측정 환경 한 줄: CPU 모델, 핀 코어, clifft 버전+svm_backend,
  VM 컴파일 플래그, 날짜.

## 보고 규칙 (중요)
- 홈 머신(i7-8700K, clifft avx2)과 배수가 달라지는 것은 정상 (여긴 clifft가 avx512로 빨라짐).
  달라진 배수를 있는 그대로 기록 — "이긴다/진다" 프레이밍 금지, 배수만.
- 측정 중 이상(타 사용자 부하, 스왑, 재시도 다발) 발견 시 해당 벤치 재측정.
- **git push 금지** (인증 없음). 갱신된 csv와 요약을 작업트리에 남기고 보고만.

## 최종 보고 형식
1. segfault 원인과 해결책 (0단계).
2. 벤치별 clifft_ns / mdam_ns / mdam_vs_clifft / bitexact 요약표.
3. 홈 머신 표 대비 배수가 크게 달라진 벤치 목록과 원인 추정 (avx512 baseline 가속 등).
