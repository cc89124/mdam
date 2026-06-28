# Compiled MDAM branch-pair core executor vs Clifft — FLOP & wall-time

Numerical hot path of one MDAM measurement core moved to C++ (single call/core, direct general-Pauli,
no per-rotation localization), compared to Clifft in the same compiled/complex128 environment.
Authoritative Python oracle UNCHANGED; new path is feature-flagged (`compiled_core`, default OFF).

Artifacts: `artifacts/compiled_core_vs_clifft/` (flop_summary, walltime_sample, uid_cost_comparison,
correctness_core, environment.json). Code: `nearclifford_backend/clifft_axis/cpp/mdm_core_executor.cpp`,
`compiled_core.py`. Verification scripts: `/tmp/validate_compiled.py`, `/tmp/brute_clean.py`,
`/tmp/measure_flop_wall.py`.

---

## 23.1 구현 요약
- **추가 파일:** `cpp/mdm_core_executor.cpp` (+`.so` RELEASE `-O3 -march=native -DNDEBUG` / PROFILE
  `-DMDAM_COST_PROFILE=1`), `compiled_core.py` (ctypes binding + `CompiledCoreExecutor`).
- **수정한 기존 코드:** 없음 (authoritative `bounded.py` 그대로; 신규 경로는 별도 모듈 + flag).
- **C++ 호출 경계:** `mdm_execute_core(phi_in, joint, survivor, ...)` — measurement core당 **1회**.
  control plane(core 선택, L-M plan, pullback)은 Python; data plane(joint build → core rotations →
  L-M → Born → outcome → normalize → drop)은 C++.
- **state layout:** joint = `[2^{r_work}]` contiguous (r_work = r_mat = r_out+1, 측정축 포함 두 branch);
  survivor `[2^{r_out}]`. caller 할당, 재사용. rotation loop 내부 allocation 0.
- **kernel:** direct diagonal phase sweep (x=0) / off-diagonal butterfly (x≠0), pair canonical
  `j<j^x`; rotation UID당 dense 1회; symbolic term 0.

## 23.2 백업
- branch `backup/mdam-pre-compiled-core-20260625_231009`, tag 동일, archive
  `/home/jung/mdam_precompiled_backup_20260625_231009/repository_snapshot.tar.gz` (+sha256),
  working_tree.diff(1.5MB), original commit `a05843e`.

## 23.3 정확성 (전부 PASS)
| test | cases | FAIL | max p0 err | max survivor err | reference |
|---|--:|--:|--:|--:|---|
| synthetic (Hermitian gen, r_out 1–6, m 1–32, I/X/Y/Z meas, both outcomes) | 1438 | **0** | 3.3e-16 | 1.7e-15 | brute-force Kronecker |
| real cores (9 benches, both outcomes) | 573×2 | **0** | 2.3e-15 | 3.1e-14 | python dense oracle |
| real cores → authoritative oracle p0 link | 573 | **0** | 2.0e-15 | — | oracle core_log |

(초기 synthetic의 비물리적 fail은 **non-Hermitian generator**(random pp → 비유니터리 "rotation",
state를 ~1e-5로 소멸)로 인한 ill-conditioned Born이었고, C++ 버그가 아님 — 물리적 Hermitian
generator(pp≡popc(x&z) mod2)에서 FAIL=0. 실제 core는 항상 Hermitian.)

## 23.4 FLOP 비교 (PROFILE counters: complex add=2, mul=6 real-FLOP; Clifft = instruction-trace 재구성)

| benchmark | Clifft rank | MDAM work | Clifft FLOP | MDAM FLOP | **MDAM/Clifft** | touch MDAM/Clifft |
|---|--:|--:|--:|--:|--:|--:|
| distillation | 5 | 4 | 2.32e3 | 8.34e3 | 3.60 | 3.36 |
| cultivation_d3 | 4 | 4 | 1.59e3 | 2.78e4 | 17.5 | 4.34 |
| cultivation_d5 | 10 | 10 | 1.99e5 | 3.84e6 | 19.3 | 2.35 |
| coherent_d3_r3 | 8 | 5 | 5.86e4 | 9.59e4 | 1.64 | 0.68 |
| **coherent_d5_r5** | **24** | **13** | **1.88e10** | **1.43e8** | **0.008** | **0.0016** |
| coherent_rx_d3_r1 | 14 | 11 | 7.36e5 | 1.61e6 | 2.19 | 0.72 |
| coherent_ry_d3_r3 | 16 | 16 | 4.44e7 | 5.15e8 | 11.6 | 6.18 |

**d5_r5: MDAM은 Clifft 대비 FLOP을 125× 적게**(cmul 1.99e7 vs 3.09e9=155×, amplitude-touch
2.39e7 vs 1.52e10=636×) 한다 — rank 24→13 절감의 직접 결과.
**저·중 rank(같은 rank): MDAM이 11–19× 더 많은 FLOP** — direct general-Pauli butterfly가 joint
2^{r_work}=2^{r_out+1}에서 4 cmul/amp인 반면 Clifft는 rotation을 single-axis **diagonal half-array**
(2^{k-1}, 1 cmul/amp)로 localize하기 때문. cultivation/ry core는 전부 butterfly(diag=0).

## 23.5 Wall-time 비교 (RELEASE; MDAM sample-only = core당 C++ call 합/shot; Clifft = sample-only warmed median; compile 분리)

| benchmark | ΔRank | Clifft sample ms | MDAM core ms | **MDAM/Clifft** | Clifft compile ms | py→C++ calls |
|---|--:|--:|--:|--:|--:|--:|
| distillation | 1 | 0.018 | 0.290 | 16.5 | 3.20 | 18 |
| cultivation_d3 | 0 | 0.008 | 0.398 | 47.5 | 0.91 | 20 |
| cultivation_d5 | 0 | 0.117 | 1.853 | 15.8 | 5.42 | 59 |
| coherent_d3_r3 | 3 | 0.019 | 0.919 | 48.4 | 0.73 | 48 |
| **coherent_d5_r5** | **11** | **13973.4** | **33.4** | **0.0024** | 2.62 | 240 |
| coherent_rx_d3_r1 | 3 | 0.139 | 1.269 | 9.1 | 0.28 | 56 |
| coherent_ry_d3_r3 | 0 | 4.900 | 97.2 | 19.8 | 0.48 | 132 |

**d5_r5: 동일 compiled 환경에서 MDAM이 Clifft보다 418× 빠르다**(33.4ms vs 14.0초). 다른 모든
benchmark에서는 Clifft가 9–48× 빠르다.

## 23.6 병목 분석
- MDAM hot kernel: off-diagonal general-Pauli butterfly (cultivation/ry는 100% butterfly). joint
  2^{r_work}에서 rotation당 4 cmul/amp. py→C++ 호출은 measurement core당 1회(=cores).
- Clifft hot kernel: localized single-axis diagonal half-array rotation(1 cmul/amp, 2^{k-1}) +
  active-array 관리. 성숙한 C++.
- d5_r5에서 Clifft가 14초인 이유: peak rank 24(2^24≈1.7e7 amplitudes)에서 rotation 328개(rank24).
  MDAM은 measurement마다 localize-drop으로 rank 13 유지 → 2^11=2048× 작은 배열.

## 23.7 Localization 판정 재확인 (§16, compiled A/B, direct vs MDAM-localized)
이전 단계(`MDAM_localization_necessity.md`) 재확인: **같은 compiled kernel**에서 direct general-Pauli
vs per-rotation localized(H/CNOT/diag)를 212개 ry localized rotation에 대해 비교 — localized 27462µs
vs **direct 15239µs → direct 1.80× 빠름**, amplitude 3.5× 적음. 즉 **MDAM 내부에서는 direct가 정답**
(L-R 제거). 단 direct는 Clifft의 localized-diagonal보다는 FLOP이 많다(위 23.4): MDAM-direct는
"MDAM-localized보다 빠른" 것이지 "Clifft-localized보다 적은 FLOP"이 아니다.

## 23.8 Crossover
> **MDAM compiled은 ΔRank = (Clifft active rank − MDAM work rank)가 충분히 클 때만 Clifft를 이긴다.**

측정: ΔRank=11(d5_r5) → MDAM 418× 우세. ΔRank 0–3(나머지 전부) → Clifft 9–48× 우세. direct
general-Pauli의 rotation당 FLOP 오버헤드(같은 rank에서 ~10–19×)와 Clifft C++ 성숙도를 2^{ΔRank}가
넘어서야 하므로, **임계값은 ΔRank ≈ 4–5**(2^{ΔRank} ≳ 16–32 > FLOP 오버헤드). 실벤치 중 이를 넘는
것은 d5_r5뿐이고, 그것이 유일하게 메모리·시간이 진짜 문제인 회로다.

## 23.9 최종 결정
- **compiled MDAM core executor를 authoritative default로 전면 전환하지 않는다.** 저·중 rank
  (ΔRank 작음)에서 Clifft가 빠르므로.
- **고-rank(ΔRank 큰) 회로에서 선택적으로 채택**: d5_r5류에서 418× 우세 + FLOP 125× 절감.
- 현 단계: 신규 경로는 **검증 완료 + 기본 OFF**. ΔRank 기반 선택 정책(또는 Clifft가 OOM/초저속인
  high-rank에서만 MDAM)으로 후속 통합. authoritative oracle은 변경 없음.

---

## §24 최종 답변

> 기존 authoritative MDAM은 `backup/mdam-pre-compiled-core-20260625_231009` (tag 동일,
> archive `/home/jung/mdam_precompiled_backup_20260625_231009/repository_snapshot.tar.gz`)에
> 백업했으며, 신규 compiled branch-pair executor는 measurement core당 C++ 호출 1회, rotation UID당
> dense 적용 1회, symbolic term 0으로 구현했다. 정확성 검증 결과는 **synthetic 1438 + 실제 573×2
> core 모두 FAIL=0 (max p0 err 2.3e-15, survivor err 3.1e-14, 오라클 p0 link 2.0e-15)** 이다.
> 동일 compiled 조건에서 MDAM의 총 FLOP은 Clifft 대비 **고-rank(d5_r5)에서 0.008× (125× 적음),
> 저·중 rank에서 1.6–19× (많음)** 이며, sample-only wall time은 **d5_r5에서 0.0024× (418× 빠름),
> 나머지에서 9–48× (느림)** 이다.

1. `coherent_d5_r5`에서 Clifft와 MDAM의 FLOP 비율은 **MDAM/Clifft = 0.008 (Clifft가 125× 많음;
   cmul 155×, amplitude-touch 636×)** 이다.
2. 저-rank benchmark에서 MDAM이 Clifft보다 느린 직접 원인은 **(a) MDAM이 같은 rank에서 rotation당
   10–19× 더 많은 FLOP을 한다(direct general-Pauli butterfly가 joint 2^{r_out+1}에서 4 cmul/amp인데
   Clifft는 single-axis diagonal half-array 1 cmul/amp로 localize), (b) ΔRank≈0이라 상쇄할 rank 이득이
   없다, (c) Clifft C++가 더 성숙하다** 이다.
3. compiled direct와 현재 per-rotation localization의 wall-time 비율은 **direct/localized = 0.56
   (direct가 1.80× 빠름; 212 ry core, 15239µs vs 27462µs)** 이다.
4. 최종 authoritative 경로로 채택해야 할 실행 구조는 **dense oracle + compiled branch-pair core
   executor(측정당 C++ 1회, direct general-Pauli, L-R 없음), feature-flag 기본 OFF, ΔRank가 큰
   고-rank 회로(d5_r5류)에서 선택적 채택** 이다 — 저-rank에서는 Clifft가 더 빠르므로 전면 전환은
   하지 않는다.
