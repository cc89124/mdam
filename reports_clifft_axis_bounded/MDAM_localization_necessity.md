# Per-rotation localization necessity — dense branch-pair oracle forensic

**확정 기준:** (+1) transient 제거·symbolic·Z-max DP·beam은 범위 밖. authoritative 경로 = rotation
수에 선형인 dense branch-pair oracle(`_fused_measure=False`). 두 질문만 닫는다: (1) dense
branch-pair의 FLOP/wall 병목은 어디인가, (2) per-rotation Pauli localization이 실제로 필요한가.

**핵심 사실(코드):** 엔진은 이미 두 경로를 모두 가짐 — `_flush_offdiag_localized`(L-R, `_loc_min_size
=2^14`로 rank≥14에서만 발동) vs `_pauli_lincomb_inplace` off-diagonal butterfly(direct). 따라서
**Path A=`_loc_min_size=2^14`, Path B=`_loc_min_size=2^62`**로 같은 harness·자료형에서 정확 비교됨.

**스크립트:** `/tmp/loc_forensic.py`(Path A/B 전체+per-kernel), `/tmp/loc_kernel.c`+
`/tmp/loc_compiled.py`(compiled stage-2).

---

## 1. Localization 3분류 + 발동 위치

| | 정의 | 발동 |
|---|---|---|
| **L-M** | 측정 Pauli M을 Z_m으로 보내 branch-pair 형태 + measured-axis drop | 모든 magic 측정(필수, gate 없음) |
| **L-R** | 각 core rotation의 reduced Pauli를 H/CNOT로 single-axis로 만든 뒤 실행 | **rank≥14에서만 — 실측상 coherent_ry_d3_r3에서만 212회** |
| L-Z | core 전체 Z-max(DP/enum) | **미검증(지시상 제외)** |

**L-R 발동 횟수(Path A):** distillation/cultivation/d3_r3/d5_r5/rx = **0회**, ry = **212회**.
→ **6/7 벤치는 Path A ≡ Path B**(이미 direct butterfly 사용). L-R 필요성은 전적으로 ry에서 결정됨.

---

## 3. 정확성 — Path A == Path B (bit-identical)

3 seed(7,8,9) measurement outcome 시퀀스가 **7개 벤치 모두 A==B 일치**(outID=True). frame
**표현**(Xc/Zc)은 다름(L-R이 localizer를 frame에 fold) — 물리 상태는 동일. ⇒ **L-R은 정확성에
불필요**, 순수 성능 선택지.

---

## 8. 전체 회로 Path A vs B (Python, median of 7)

| benchmark | rank | **A ms (L-R on)** | **B ms (direct)** | A/B | A peak KB | B peak KB | L-R |
|---|--:|--:|--:|--:|--:|--:|:--:|
| distillation | 4 | 12.2 | 13.4 | 0.91 | 1211 | 1211 | no(noise) |
| cultivation_d3 | 4 | 5.5 | 5.5 | 1.00 | 1284 | 1284 | no |
| cultivation_d5 | 10 | 89.1 | 87.4 | 1.02 | 8602 | 8602 | no |
| coherent_d3_r3 | 5 | 8.5 | 9.0 | 0.94 | 29 | 29 | no |
| coherent_d5_r5 | 13 | 179.9 | 180.2 | 1.00 | 333 | 333 | no |
| coherent_rx_d3_r1 | 11 | 8.1 | 7.9 | 1.02 | 132 | 132 | no |
| **coherent_ry_d3_r3** | 16 | **467** | **643** | **0.73** | 419 | 415 | **yes** |

ry만 차이. L-R가 **Python wall을 27% 줄임**.

---

## 4·11. ry per-kernel 분해 (병목)

**Path A (L-R on):** rot_offdiag 92call **376ms(90%)** / cnot 171call 20ms(4.7%) / h 110call
15ms(3.6%) / diaghalf 56call 2ms / L-R synth 33ms / L-M synth 16ms. **amps touched ≈ 12.6e6.**

**Path B (direct):** rot_offdiag **153call 606ms(98.8%)** / 나머지 L-M 소량. **amps touched ≈ 4.7e6.**

**FLOP 병목 = off-diagonal butterfly rotation kernel**(양 경로 90–99%). L-R은 153→92 butterfly로
줄이는 대신 H/CNOT/diag 4.4 pass/rot를 추가 → **Path A가 amplitude를 2.7× 더 touch**(12.6e6 vs 4.7e6).
Born/normalize/drop <1%, L-M ~3%.

**wall 병목 ≠ FLOP 병목 이유:** off-diagonal butterfly가 **Python에서 ~165 ns/amp**(NumPy
fancy-index: `np.arange`+boolean mask+gather). L-R의 H/CNOT/diag는 contiguous strided **view**라
NumPy-friendly(빠름). 즉 wall 병목은 **butterfly의 NumPy dispatch**이지 butterfly의 산술 FLOP이
아니다. 그래서 Python에서 "pass를 4.4배 늘려도" 총 wall이 줄어든다.

---

## 9. Compiled stage-2 — L-R의 Python 이득이 사라지는가

per-pass C kernel 비용(`loc_kernel.c`, gcc -O3 -march=native), rank16: butterfly 125627 ns /
h_pass 48322 / cnot 46539 / diag 48295. C direct_rot은 reference와 bit-exact(0.0e+00).

**212개 실제 ry localized rotation에 대해 compiled 비용 합성:**

| | compiled wall(합) | amplitude touch |
|---|--:|--:|
| **Path A (localized)** | **27462 µs** | 2.59e7 |
| **Path B (direct)** | **15239 µs** | 7.41e6 |
| **A/B** | **1.80×** | **3.50×** |

**Compiled에서는 direct(B)가 1.80× 빠르고, localized(A)는 amplitude를 3.5× 더 touch.**
butterfly가 pass당 2.6× 비싸도, L-R가 4.4 pass를 쓰므로 총합은 direct가 이긴다.

**결론:** L-R의 Python 이득(0.73×)은 **NumPy fancy-index artifact**. compiled에서는 **direct가
1.80× 우월**하고 traffic도 3.5× 적다. → per-rotation localization은 compiled 경로에서 **제거**.

---

## 6. L-R 순이득 Δ (rotation 특성별)

localized ry rotation 212개: X-weight 1(128)/2(64)/3(16)/4(4), rank 14(20)/15(168)/16(24),
평균 4.4 array-pass/rot. Δ = C_loc − C_direct:
- **Python:** Δ<0 (localized 유리, fancy-index 회피). weight↑일수록 이득 감소(pass↑).
- **Compiled:** Δ>0 모든 weight (direct가 pass 1개). weight-1조차 localized 4.4 pass > 1 butterfly.
- **amortization 없음:** 각 rotation이 자기 axis로 localize+frame fold라 후속 rotation에 재사용 안 됨.

---

## 13. 결론

### 13.1 한 문장
> Dense branch-pair MDAM의 FLOP 병목은 **off-diagonal general-Pauli rotation butterfly kernel**이고,
> wall-time 병목은 **(저rank) NumPy per-call dispatch + (ry) 그 butterfly의 NumPy fancy-index 비용**
> 이다. Per-rotation Pauli localization(L-R)은 direct general-Pauli 실행 대비 amplitude touch(FLOP/
> memory traffic)를 **+250%(3.5×)**, compiled wall time을 **+80%(1.80×)** 증가시키므로(Python에서는
> NumPy fancy-index 때문에 ry wall을 −27% 감소시키지만 그건 구현 artifact) **compiled 경로에서
> 제거**해야 한다. Measurement-axis localization(L-M)은 전체의 ~3%이며 **유지**한다.

### 13.2 정확성
synthetic(이전 단계 1760) + real 7벤치×3seed: **outcome bit-identical, FAIL 0.** C direct_rot err 0.0.

### 13.5 판정
- **L-M: 선택→유지** (모든 측정에 필요, ~3% 비용, 병목 아님; 제거하려면 Path C probe 필요하나 불요).
- **L-R: 불필요(정확성) / Python-only 성능 / compiled에서 제거 대상.**
- **L-Z: 미검증**(L-R이 이득 아님으로 판정됐으므로 추진 근거 없음).

### 13.6 최종 실행 구조
**measurement-axis localization(L-M) 1회 → direct general-Pauli branch-pair rotations → Born →
normalize/drop.** per-rotation localization 없음.

### 13.7 integration 계획(이번엔 미실행)
- 현 Python 백엔드: L-R 게이트(`_loc_min_size=2^14`)는 ry에서 Python-최적이므로 **Python 단계에선
  그대로 둠**(6/7 벤치엔 no-op). 단독으로 끄면 ry가 467→643ms로 느려짐.
- compiled 백엔드(직전 단계 결정 = oracle hot path를 Clifft C++로): **L-R 제외**, direct general-Pauli
  butterfly + 측정당 1 compiled call. compiled에서 direct 1.80× 우월 + traffic 3.5× 절감.
- 즉 **L-R 제거는 "compile" 작업과 한 묶음**: Python에선 유지, 컴파일하면서 제거.

---

## 최종 질문 답

1. **FLOP 최다 발생:** rotation kernel(off-diagonal general-Pauli butterfly). Born/normalize/drop은
   각 <1%, localization은 ry에서만 ~8%(H/CNOT) + L-M ~3%.
2. **wall 지배:** 저rank=NumPy dispatch(이전 forensic, 78–5108× over C++); ry=offdiag butterfly의
   NumPy fancy-index(~165 ns/amp). 실제 산술 FLOP이 아님.
3. **L-R 필요성:** **성능 최적화일 뿐, 정확성엔 불필요**(Path B bit-identical). 그나마도 Python-전용.
4. **localization 제거 시:** compiled에선 **H/CNOT pass 감소(제거)가 butterfly 증가보다 큼** → direct가
   1.80× 빠름, amplitude 3.5× 적음. Python에선 반대(fancy-index 때문에 L-R 유리).
5. **비싼 core 최종 최속:** **compiled direct general-Pauli branch-pair butterfly**(측정당 1 call).
