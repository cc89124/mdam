# MDAM vs Clifft — operation-by-operation forensic profiling

**측정 환경:** `/home/jung/clifft_env/bin/python`, 동일 회로·동일 seed(7, warmed)·동일 measurement
branch. Clifft는 C++ `clifft.sample` (median of 9). MDAM oracle(`_fused_measure=False`) /
fused(`_fused_measure=True`)는 `nearclifford_backend.clifft_axis.bounded`. 계측: `budget.charge`
(oracle pass별 dense element-touch = resident 합), `_apply_xz`/`_fmul` (fused dense / symbolic
term-pair), per-measurement wall, tracemalloc peak, clifft `active_k_history`로 rotation 실행 rank
재구성. 스크립트: `/tmp/forensic.py`, `/tmp/forensic_analyze.py`, `/tmp/forensic_ry.py`.

---

## 9.1 한 문장 결론

> **MDAM의 rank 절감은 정상이고(모든 rotation을 Clifft와 같거나 더 작은 rank에서 1회만 dense
> 실행, Fm/Fc ≤ 0.96), Clifft 대비 "느림"의 주원인은 FLOP도 rank도 아니라 MDAM이 Python/NumPy
> 구현이라 작은 2^k에서 호출당 ~1µs 고정비용이 지배하기 때문이다(H7, ns/amp overhead 78–5108×).
> 그 위에 얹히는 fused 전용 추가비용(oracle 대비 1.2–6.7×)은 measurement core의 Pauli-sum이
> rotation 수에 대해 ~2^m로 커지는 symbolic build이며(H1), frame-awareness/Z-localization은
> 전체의 ~2–7%만 차지하므로 우선순위가 아니다.**

그리고 전제 자체가 rank 의존적이다: **유일하게 진짜 큰 회로인 coherent_d5_r5(Clifft rank 24)에서는
MDAM이 Clifft보다 18–54× 빠르고 메모리는 400–750× 작다.** "MDAM이 Clifft보다 느리다"는 것은
저-rank(2^k가 작아 C++가 µs 안에 끝나는) 벤치마크에만 해당한다.

---

## 9.2 비용 분해 표

### (A) 전체 회로 3-way: Clifft(C++) / MDAM oracle(dense) / MDAM fused(symbolic)

| benchmark | Clifft rank | MDAM rank | Clifft ms | oracle ms | fused ms | oracle/Clifft | fused/Clifft | fused/oracle | Clifft peak | oracle peak | fused peak |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| coherent_d3_r3 | 8 | 5→4 | 0.020 | 9.6 | 19.8 | 468× | 968× | 2.07× | 4 KB | 33 KB | 47 KB |
| **coherent_d5_r5** | **24** | **13→12** | **10545** | **194.7** | **592.7** | **0.018×** | **0.056×** | 3.04× | **256 MB** | **347 KB** | **664 KB** |
| cultivation_d3 | 4 | 4→3 | 0.009 | 5.8 | 7.0 | 630× | 758× | 1.20× | <1 KB | 1285 KB | 1285 KB |
| cultivation_d5 | 10 | 10→9 | 0.118 | 98.3 | 656.3 | 833× | 5563× | 6.68× | 16 KB | 8604 KB | 8608 KB |
| distillation | 5 | 4→3 | 0.018 | 16.3 | 19.5 | 884× | 1053× | 1.19× | <1 KB | 1212 KB | 1213 KB |
| coherent_rx_d3_r1 | 14 | 11→10 | 0.149 | 8.0 | 14.8 | 54× | 100× | 1.86× | 256 KB | 136 KB | 160 KB |
| coherent_ry_d3_r3 | 16 | 16 | 4.973 | 477.3 | 781.7 | 96× | 157× | 1.64× | 1024 KB | 426 KB | **4142 KB** |

- Clifft peak = 2^rank × 16 byte (active array). MDAM peak = tracemalloc.
- d5_r5: MDAM peak가 Clifft의 256 MB 대비 347 KB(750× 작음). **MDAM이 fused에서도 Clifft보다 peak
  bytes가 큰 경우는 ry 단 하나(4142 vs 1024 KB)** — rank 16에서 symbolic dict + apply_xz 2^r 임시가
  Clifft의 단일 1 MB 배열을 넘기 때문(H2). 나머지는 Clifft와 같거나 작다.

### (B) H7 — dense amplitude-touch 1개당 ns (Python/NumPy vs C++)

| benchmark | Clifft touch | Clifft ns/amp | oracle touch | oracle ns/amp | overhead |
|---|--:|--:|--:|--:|--:|
| coherent_d3_r3 | 3.1e4 | 0.66 | 2.9e3 | 3366 | **5108×** |
| coherent_d5_r5 | 1.8e10 | 0.58 | 4.3e6 | 45.9 | **78×** |
| cultivation_d3 | 1.3e3 | 7.15 | 9.2e2 | 6321 | 884× |
| cultivation_d5 | 2.9e5 | 0.40 | 9.9e4 | 996 | 2469× |
| distillation | 7.2e2 | 25.8 | 5.0e2 | 32469 | 1258× |
| coherent_rx_d3_r1 | 4.8e5 | 0.31 | 6.3e4 | 126 | 407× |
| coherent_ry_d3_r3 | 1.5e7 | 0.34 | 1.5e7 | 32.2 | 94× |

**overhead가 rank↑에 따라 5108×(rank8)→78×(rank24)로 단조 감소** = 고정 per-call 비용의 지문.
NumPy in-place op 1회 floor 측정: 4-element 1170 ns, 256-element 721 ns, 65536-element 0.45 ns/amp
(`a *= s` median). 즉 호출당 ~0.7–1.2 µs는 배열 크기와 무관한 Python+dispatch 고정비. Clifft(C++)는
모든 크기에서 0.3–1 ns/amp. 작은 2^k에서 이 floor가 전부를 지배한다.

### (C) oracle dense-touch를 pass별로 분해 (resident 합 = 읽고/쓴 amplitude)

오라클은 측정마다 ~7개의 분리된 full-array sweep을 돈다. 대표값:

| pass | d5_r5 % | cult_d5 % | ry % | 역할 |
|---|--:|--:|--:|---|
| rot | 48.4 | 57.8 | 24.9 | core rotation flush (butterfly/half-array) |
| purge (H/CNOT) | 7.9 | 19.2 | **60.5** | off-diag rotation/측정축 localization |
| sqnorm | 19.8 | 10.5 | 6.6 | Born 양쪽 branch 노름 (1 sweep) |
| promote | 7.9 | 4.2 | 2.7 | +1축 kron 성장 |
| normalize | 7.9 | 4.2 | 2.7 | 사영 후 정규화 |
| drop | 4.0 | 2.1 | 1.3 | 측정축 memmove |
| post-reduce | 4.0 | 2.1 | 1.3 | resident note |

ry는 **purge(localization 버터플라이)가 60.5%** — off-diagonal R_Y를 rank 16에서 H/CNOT로 단축하는
dense 비용이며 Clifft도 동일하게 부담(OP_ARRAY_H 120개). 나머지 벤치는 rot가 최대.

### (D) fused 추가비용: symbolic build(fmul) vs dense(apply_xz) vs survivor

| benchmark | fmul pair-mul Σ | apply_xz touch | survivor touch | 지배 core (wall) |
|---|--:|--:|--:|---|
| coherent_d5_r5 | 4.6e5 | **1.3e7** | 3.4e5 | 분산 (m=21 50ms, dense 위주) |
| cultivation_d5 | **1.1e6** | 3.0e4 | 4.1e3 | **m=4 561ms = 회로의 86%, 순수 symbolic** |
| coherent_ry_d3_r3 | 2.7e5 | 3.2e6 | 1.5e5 | m=22 191ms (fmul 2.6e5 + apply_xz 1.0e6) |

- cultivation_d5: 단일 core m=4가 561ms. fmul pair-mul 1.06e6, nU=1024, apply_xz는 2050뿐 →
  **dense가 아니라 Python dict 곱(symbolic build)이 전부.**
- d5_r5: apply_xz 1.3e7(oracle rot 2.06e6의 6×) — #U(avg 60)가 #core_rot를 대체하며 dense touch가 늘어남.

### (E) Path A/B/C/D — 측정 core별 dense-equiv 비용

D = #core_rot·2^r_out + 2^r_out (각 rotation을 survivor에 1회) · C = #U·2^r_out (현 fused) · B = #core_rot·2^r_mat (oracle)

| benchmark | D | C(fused) | B(oracle) | C/D | B/D |
|---|--:|--:|--:|--:|--:|
| coherent_d3_r3 | 9.7e2 | 4.2e3 | 1.7e3 | 4.3 | 1.7 |
| coherent_d5_r5 | 1.5e6 | 1.1e7 | 2.6e6 | 7.9 | 1.8 |
| cultivation_d3 | 2.5e2 | 3.4e2 | 4.4e2 | 1.4 | 1.8 |
| cultivation_d5 | 3.5e4 | 5.4e5 | 6.6e4 | 15.6 | 1.9 |
| distillation | 7.5e1 | 8.1e2 | 1.1e2 | 10.7 | 1.5 |
| coherent_rx_d3_r1 | 1.5e4 | 6.3e4 | 2.4e4 | 4.1 | 1.6 |
| coherent_ry_d3_r3 | 8.7e4 | 1.5e6 | 2.7e4 | 17.3 | 0.3 |

**핵심:** dense-localized flush(D)는 oracle(B/D≈1.8, 그 1.8×가 곧 r_mat vs r_out의 +1 transient)
보다도, 현 fused(C/D=4–17×)보다도 싸다. **transient를 없애는 가장 싼 방법은 symbolic이 아니라
2^r_out에서의 dense-localized flush.** 즉 느린 건 MDAM 알고리즘이 아니라 현재 symbolic branch-map 구현.

---

## H1–H7 판정 (실측 근거)

- **H1 Pauli-sum explosion — 확정(단, fused-vs-oracle 한정).** ry meas=1 core(r_in 2→r_mat 16,
  rotation 48개)의 U term 성장은 정확히 1,2,4,…,262144,1097790 — **2^(#rotation), cancellation 사실상
  없음**(1024→1023). cap 2^14는 rotation 14에서 발동, 그 전까지 build+alloc 낭비(2.7e5 pair-mul, 4 MB).
  cultivation_d5 m=4는 19 rotation에서 **2^r_mat=1024로 포화**하지만 build는 1.06e6 dict 곱(=561ms).
  oracle dense(48×2^16=3.15e6 / 19×2^10=1.9e4)가 압도적으로 싸다. **MDAM-vs-Clifft 느림의 원인은 아님.**
- **H2 survivor intermediate가 factor-2 절감 상쇄 — 확정(ry).** fused는 U-term마다
  `vec = cc·_apply_xz(...)`로 2^r_in 새 배열을 할당. ry rank 16에서 1 MB×수개가 동시 live → peak
  426→4142 KB(10×, Clifft 1 MB도 초과). d5_r5는 347→664 KB로 완만.
- **H3 다중 full-array pass — 부분.** oracle은 측정당 rot/purge/sqnorm/promote/normalize/drop/
  post-reduce ~7 sweep(B/D≈1.8×가 그 누적). 큰 비용은 아니나 각 sweep이 별도 NumPy 호출이라 H7와 곱해짐.
- **H4 rotation을 더 큰 rank에서 실행 — 반증.** Fm/Fc = 0.08, 0.00, 0.93, 0.96, 0.38, 0.11, 0.69 —
  **모두 ≤ 1.** MDAM은 어떤 rotation도 Clifft보다 큰 rank에서 실행하지 않는다. d5_r5는 2.06e6 vs Clifft
  5.64e9(2700× 적음). rank 히스토그램: Clifft는 rank 24에 328개, MDAM 최대 rank 13(187개).
- **H5 allocation/copy — H2/H7에 포함.** fused per-term 2^r 할당과 dict 컨테이너가 실주범. oracle은
  in-place kernel로 transient를 chunk(2^11)에 가둬 copy 최소.
- **H6 localization/frame transform — dense는 ry에서 큼, frame bit-tracking은 무시 가능.** ry oracle
  purge(H 버터플라이) 60.5%. 단 이는 off-diag rotation의 dense 단축으로 Clifft도 동일 부담. frame의
  bitwise pullback은 dense로 안 잡힘(O(weight) lookup).
- **H7 구현 언어·자료구조 — 확정, MDAM-vs-Clifft의 지배 원인.** ns/amp overhead 78–5108×, rank에
  반비례. NumPy 호출당 ~1 µs 고정 floor 측정. 동일 dense-arithmetic을 C++(Clifft)는 0.3–1 ns/amp로 끝냄.

---

## 7. 최초 divergence point

- **저-rank 벤치(d3_r3, cult, distill, rx, ry):** Clifft 전체 회로가 0.15 ms 미만 = MDAM의 **첫
  measurement 한 번의 NumPy 호출 오버헤드보다 작다.** 따라서 MDAM 누적시간은 사실상 **연산 1–2개째에서
  즉시 Clifft 전체를 추월**한다(구조적·Python 기인, 특정 core 아님).
- **coherent_d5_r5:** MDAM은 전 구간에서 Clifft보다 싸다 — **추월 지점 없음**(Clifft 10.5s vs MDAM 0.2–0.6s).
- **fused 내부 divergence(oracle 대비):** cultivation_d5는 core **m=4** 하나가 561 ms로 회로의 86%;
  ry는 **m=22**가 191 ms로 24%; d5_r5는 60개 core에 고르게 분산(최대 m=21 50 ms).

---

## 8. ry 별도 분석

- **rank 16→16인 이유:** 측정 core(meas 1/9/17)가 r_in 2→r_mat 16, rotation 43–48개(전부 off-diag).
  symbolic이 cap을 넘어 fallback → oracle이 2^16 dense materialize → rank 16 유지.
- **2^#rot 폭발:** U term이 rotation마다 정확히 doubling(측정값 위 1,2,…,1.1e6). 48-rotation core면 2^48≈2.8e14.
- **localizer on/off가 term을 못 줄임:** 폭발 동인은 diagonal 여부가 아니라 **rotation 개수**. localize는
  off-diag→diag만 바꿀 뿐 곱 횟수는 동일.
- **oracle 2^16 vs symbolic 2^48:** oracle dense = 48×2^16 = 3.15e6, symbolic = 2^48 — 약 10^8× 차이.
- **cap 발동 시점/낭비:** cap 2^14는 rotation 14에서 발동. 그 전까지 ~14회 fmul(2.7e5 pair-mul) + 최대
  16k-term dict(≈1 MB) 할당이 이미 소모됨(헛수고).
- **사전 판정 가능 여부 — 가능.** core의 off-diag rotation 수(#core_rot)는 _fused_setup 시작 시 이미
  안다. 성장이 ~2^min(#rot, r_mat·2)이므로 `#core_rot > log2(cap)=14`면 cap 초과가 확정 → build 전에
  바로 dense-localized로 보내면 헛수고 0. (이번 작업에선 구현하지 않고 판정 가능성만 확인.)

---

## 9.3 원인 순위 (벤치마크별, runtime 기준)

```
coherent_d5_r5  (MDAM이 Clifft보다 18× 빠름; 비용은 fused-vs-oracle 3.0× 안에서)
  1. fused apply_xz dense (#U·2^r)            ~60%   (oracle 대비 6× dense)
  2. symbolic fmul build                       ~25%
  3. survivor + 7-pass sweeps                  ~15%
cultivation_d5  (fused/oracle 6.7×)
  1. core m=4 symbolic Pauli-sum build(H1)     ~86%   (1.06e6 dict 곱, Python)
  2. 나머지 14 core dense                       ~14%
coherent_ry_d3_r3  (fused/oracle 1.6×)
  1. off-diag localization purge(H 버터플라이)  ~55%   (Clifft 공유 dense; H6)
  2. dense rot flush                           ~25%
  3. 헛수고 symbolic build + apply_xz 임시(H1/H2) ~15%
저-rank 전반 (MDAM-vs-Clifft)
  1. Python/NumPy per-call 고정비용(H7)         지배   (ns/amp 400–5000×)
  2. 7-pass sweep 다중 호출(H3)                 부차
```

## 9.4 구현 오류 vs 알고리즘 한계

**구현(implementation) 문제 — 고치면 됨:**
- fused가 transient 제거를 #U·2^r_out symbolic으로 함 → dense-localized면 #core_rot·2^r_out(C/D=4–17×↓).
- ry: cap 초과를 **사후**에 발견해 build를 이미 낭비. #core_rot로 **사전** 판정 가능.
- fused per-term `cc·_apply_xz`가 매 term 2^r 배열을 새로 할당(H2/H5). in-place 누산으로 제거 가능.
- oracle 7-pass(sqnorm/normalize/drop/post-reduce 분리)가 NumPy 호출을 늘려 H7를 증폭.
- Python dict 기반 Pauli-sum(자료구조 overhead).

**알고리즘 표현(algorithmic) 한계 — 표현을 바꿔야 함:**
- branch-map의 symbolic Pauli-sum은 core rotation 수에 대해 2^m로 커진다(rank 지수 절감과 무관). 고-rotation
  core(ry)에는 표현 자체가 부적합. → bounded dense(2^r) 표현이어야 함.

**근본적으로 Python인 점(H7)** 은 알고리즘 결함이 아니라 구현 선택. 큰 rank(d5_r5)에서는 이미 Clifft를
이긴다. 저-rank의 "느림"은 C 확장/배치/벡터화로 닫히는 상수항이지 점근(asymptotic) 문제 아님.

## 9.5 다음 구현이 만족해야 할 불변식

1. working set ≤ O(2^r_out) (상수 배 buffer만; 2^r_mat = 2^(r_out+1) 별도 배열 금지).
2. 비용이 2^(#core_rot)에 비례하면 안 됨. 측정 core 비용 = O(#core_rot · 2^r_out).
3. 각 rotation UID는 dense에 **정확히 1회** 적용(현 oracle/Clifft가 이미 만족; fused는 #U회로 위반).
4. 측정될 축을 별도 2^(r_out+1) buffer로 만들지 않음(transient 제거 — 현 fused가 만족하는 유일한 항목).
5. survivor를 2^r_out에 **직접** 생성(per-term 2^r_in 임시 누산 금지).
6. cap 도달 후 fallback이 아니라, #core_rot 사전 비용모델로 symbolic을 아예 선택하지 않음.
7. NumPy 호출 수를 측정당 O(1)~O(#core_rot)로 — sweep 융합(Born+normalize+drop 1-pass)으로 H7 floor 상각.
8. Clifft 대비 동일 UID의 dense rank·kernel·호출 수를 설명 가능해야 함(현재 Fm/Fc≤1로 이미 우월, 유지).

---

## 최종 목표 문장 (실측 완성)

> MDAM은 Clifft 대비 dense rank를 **24**에서 **12**로 줄였고(coherent_d5_r5), 순수 dense rotation 비용은
> **약 2700배 감소**했다(2.06e6 vs 5.64e9 touch, Fm/Fc=3.6e-4; 모든 벤치 Fm/Fc≤0.96). 그러나
> **cultivation_d5의 core m=4** measurement core에서 Pauli-sum term 수가 **1024**개까지 증가하면서
> symbolic build가 **561 ms**(회로의 86%)를 썼고, ry에서는 per-term 임시가 peak live memory **4.1 MB**
> (Clifft 1 MB 초과)를 썼다. 따라서 **저-rank에서의 성능 역전 주원인은 Python/NumPy 호출당 ~1µs 고정
> 비용(H7, ns/amp 78–5108×)** 이고, **fused 전용 추가비용의 주원인은 rotation 수에 2^m로 커지는 symbolic
> Pauli-sum build(H1)** 이며, frame-awareness 또는 Z-localization은 전체 비용의 **~2–7%**(ry의 dense
> localization은 Clifft와 공유)만 건드리므로 우선순위가 아니다. 다음 backend는 **working-set O(2^r_out),
> 비용 ∝ #core_rot·2^r_out (2^#rot 아님), rotation당 dense 1회, 측정당 NumPy 호출 O(#core_rot)** 를
> 만족하는 dense-localized/factorized execution이어야 한다.
