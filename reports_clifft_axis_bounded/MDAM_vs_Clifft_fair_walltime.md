# MDAM vs Clifft — fair full-sample wall-time (scope-corrected)

기존 비교는 MDAM의 C++ numerical-core 시간 합과 Clifft의 full-sample 시간을 비교하여 timer scope가
일치하지 않았다. 이번에는 양쪽 모두 compile/plan 완료 후 public sampling path 전체를 outer timer
(`perf_counter_ns`)로 측정했다. 그 결과 `coherent_d5_r5`의 full-sample MDAM/Clifft wall-time 비율은
**0.015 (MDAM 66× 빠름)** 이고, 저·중 rank benchmark의 비율은 **27×–2741× (MDAM 느림)** 이다. MDAM이
더 적은 state work 또는 amplitude touch에도 느린 직접 원인은 **MDAM full-shot의 93–99.9%가 Python
control plane이고 C++ numerical kernel은 0.1–6.9%에 불과하기 때문** 이다 (arithmetic FLOP도, kernel
ns/touch도 아님 — high-rank에서는 C++ kernel이 Clifft보다 amplitude당 오히려 빠르다).

**환경:** taskset core 2 고정, `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`, complex128, gcc -O3
-march=native -DNDEBUG (RELEASE timing), warmed, shots-per-block calibrated, median. MDAM =
`backend.sample` (compiled_core=True, fused OFF) public API. Clifft = `clifft.sample`. 백업
`backup/mdam-before-fair-walltime-20260625_235209`. 통합: `compiled_core.try_compiled_measure` +
`bounded.measure_z` 4-line default-OFF dispatch (authoritative oracle 경로 불변).
Artifacts: `artifacts/mdam_vs_clifft_fair_walltime/`.

---

## 비교 A — Full sample 대 Full sample (HEADLINE, §18.1)

| benchmark | shots/block | Clifft ms/shot | MDAM ms/shot | **MDAM/Clifft** | Clifft compile ms | MDAM plan ms | reps c/m |
|---|--:|--:|--:|--:|--:|--:|--:|
| distillation | 101 | 0.0114 | 11.66 | **1022×** | 3.27 | 57.3 | 400/11 |
| cultivation_d3 | 161 | 0.0021 | 5.89 | **2741×** | 0.86 | 23.2 | 400/11 |
| cultivation_d5 | 16 | 0.0811 | 71.54 | **882×** | 5.01 | 452.9 | 400/11 |
| coherent_d3_r1 | 612 | 0.0013 | 1.33 | **1008×** | 0.27 | 5.76 | 400/11 |
| coherent_d3_r3 | 132 | 0.0140 | 6.10 | **437×** | 0.51 | 34.7 | 400/11 |
| coherent_d5_r1 | 148 | 0.2052 | 5.54 | **27×** | 0.99 | 29.0 | 82/11 |
| **coherent_d5_r5** | 1 | **8616.95** | **129.90** | **0.015 (66× 빠름)** | 2.90 | 707 | 3/19 |
| coherent_rx_d3_r1 | 115 | 0.1302 | 7.04 | **54×** | 0.27 | 30.7 | 166/11 |
| coherent_ry_d3_r3 | 1 | 5.41 | 471.84 | **87×** | 0.71 | 1907 | 400/11 |

**유일하게 MDAM이 이기는 회로는 `coherent_d5_r5` (Clifft rank 24): 66× 빠름.** 나머지 전부 MDAM이
27–2741× 느림. (MDAM plan time(structure pass)은 one-time이며 ry 1907ms·d5_r5 707ms로 큼.)

## 비교 B — Numerical kernel 대 Numerical kernel (§13.2)

MDAM C++ executor 시간 합 vs Clifft full-sample(≈전부 C++ data-plane). high-rank에서 C++ kernel은
Clifft보다 amplitude당 빠르다:

| benchmark | MDAM C++ core ms | Clifft full ms | MDAM ns/touch | Clifft ns/touch | **md/cli ns/touch** |
|---|--:|--:|--:|--:|--:|
| coherent_d5_r5 | 9.02 | 8617 | 0.38 | 0.57 | **0.67 (MDAM 빠름)** |
| coherent_ry_d3_r3 | 0.66 | 5.41 | 0.01 | 0.43 | **0.02 (MDAM 빠름)** |
| cultivation_d5 | 0.34 | 0.081 | 0.54 | 0.31 | 1.77 |
| coherent_d3_r3 | 0.33 | 0.014 | 20.4 | 0.59 | 34.6 |
| cultivation_d3 | 0.13 | 0.002 | 27.3 | 1.98 | 13.8 |

high-rank(d5_r5, ry)에서 C++ numerical kernel은 Clifft보다 amplitude-touch당 **빠르다**. 저-rank에서
느린 것은 tiny array(2^k 작음)에서 ctypes/setup 고정비가 touch당 비용을 키우기 때문(여전히 full-shot
비중은 작음).

---

## §18.5 MDAM full-shot breakdown — **Python control plane이 지배**

| benchmark | full ms | C++ executor ms | **C++ %** | Python control ms | **control %** | C++ calls |
|---|--:|--:|--:|--:|--:|--:|
| distillation | 11.19 | 0.113 | 1.0% | 11.07 | **99.0%** | 4 |
| cultivation_d3 | 5.00 | 0.129 | 2.6% | 4.87 | **97.4%** | 4 |
| cultivation_d5 | 69.60 | 0.337 | 0.5% | 69.27 | **99.5%** | 5 |
| coherent_d3_r3 | 5.95 | 0.329 | 5.5% | 5.62 | **94.5%** | 11 |
| **coherent_d5_r5** | 130.34 | 9.02 | 6.9% | 121.32 | **93.1%** | 55 |
| coherent_rx_d3_r1 | 6.84 | 0.207 | 3.0% | 6.64 | **97.0%** | 5 |
| coherent_ry_d3_r3 | 465.28 | 0.662 | 0.1% | 464.62 | **99.9%** | 9 |

**MDAM full-shot의 93–99.9%는 Python control plane**(run_shot orchestration, lazy-pending 관리, core
discovery/lookup, pullback, frame/ledger update, stabilizer(Gottesman-Knill) 측정, measurement record
생성). 내가 C++로 옮긴 numerical kernel은 **0.1–6.9%**. 즉 C++ 최적화는 full-shot에 거의 영향이 없다.
(unattributed: full − C++ − control = 0; control에 stabilizer 측정·frame·record 모두 포함.)

## §19 d5_r5 three-number scope

| d5_r5 metric | time |
|---|--:|
| MDAM C++ numerical cores only | 9.02 ms |
| MDAM full sample | 130.34 ms |
| MDAM non-core overhead (Python) | 121.32 ms (93.1%) |
| Clifft full sample | 8600 ms |
| **Full-sample speedup (MDAM)** | **66×** |
| (참고) 기존 core-only "speedup" | 953× / 418× |

**기존 418×는 scope 불일치였고, 공정한 full-sample speedup은 66×다.** Clifft가 14초나 걸려도 MDAM은
130ms — rank 24→13 절감이 압도적이라, Python 93% 오버헤드에도 불구하고 66× 우세.

## §15 효율 정규화 (full-sample 기준)

| benchmark | cli ns/FLOP | md ns/FLOP | md/cli | md/cli FLOP | md/cli touch | md/cli full-wall |
|---|--:|--:|--:|--:|--:|--:|
| coherent_d5_r5 | 0.46 | 0.91 | 2.0 | 0.008 | 0.0016 | **0.015** |
| cultivation_d5 | 0.41 | 18.7 | 46 | 19.3 | 2.35 | 882 |
| coherent_d3_r3 | 0.24 | 63.6 | 267 | 1.64 | 0.68 | 437 |
| coherent_rx_d3_r1 | 0.18 | 4.37 | 25 | 2.19 | 0.72 | 54 |
| coherent_ry_d3_r3 | 0.12 | 0.92 | 7.5 | 11.6 | 6.18 | 87 |

## §16 더 적은 touch인데 느린 benchmark (핵심 진단)

**coherent_d3_r3 (touch 0.68× = MDAM이 더 적음), 그런데 437× 느림:**
- 원인 split: C++ numerical 0.33ms(**5%**), Python control 5.62ms(**92%**). → arithmetic도, touch도
  아니고 **Python control plane**.
- C++ kernel ns/touch 20.4 vs Clifft 0.6 (tiny-array call 고정비); 그러나 그조차 full-shot의 5%뿐.

**coherent_rx_d3_r1 (touch 0.72×), 54× 느림:** C++ 3%, **Python 94%**.

**coherent_d5_r5 (touch 0.0016× = 636× 적음):** 이 거대한 touch 절감이 Python 93% 오버헤드까지 이기고
MDAM이 66× **승리**.

→ **MDAM이 더 적은 touch에도 느린 직접 원인은 software(Python control plane) overhead이지 arithmetic
overhead가 아니다.** (cultivation/ry처럼 같은 rank에서 direct general-Pauli가 FLOP을 더 쓰는 것도
사실이나, full-wall에서는 그조차 C++ 6.9% 이내라 부차적.)

---

## §20 Crossover (실측만, 추측 금지)

실측 full-sample 승패:

| ΔRank (Clifft − MDAM work) | benchmark | full-wall MDAM/Clifft | winner |
|---|---|--:|---|
| 11 | coherent_d5_r5 | 0.015 | **MDAM (66×)** |
| ~? (rank0 magic) | coherent_d5_r1 | 27 | Clifft |
| 0 | coherent_ry_d3_r3 | 87 | Clifft |
| 0 | cultivation_d5 | 882 | Clifft |
| 3 | coherent_d3_r3 | 437 | Clifft |
| 3 | coherent_rx_d3_r1 | 54 | Clifft |
| 0–1 | distillation/cultivation_d3/d3_r1 | 1008–2741 | Clifft |

**측정된 회로 중 MDAM이 full-sample로 이기는 것은 ΔRank=11인 d5_r5 단 하나다.** 정확한 crossover
ΔRank 임계값은 이 데이터만으로 확정할 수 없다(full-wall은 Python control plane이 지배하므로
ΔRank만의 함수가 아님) — controlled rank-sweep이 필요하다. 따라서 "ΔRank≥N이면 MDAM 승"이라고
단정하지 않는다.

---

## §24 최종 답변

1. **d5_r5 기존 418×는 full-sample 기준 66×다** (8600ms vs 130ms). 기존 418×/953×는 MDAM C++ core-only
   대 Clifft full-sample의 scope 불일치였다.
2. **MDAM full sample 중 C++ numerical core 비율: 0.1–6.9%** (d5_r5 6.9%, ry 0.1%, cult_d5 0.5%).
3. **Python control plane + state build/copy 비율: 93.1–99.9%** (full-shot의 거의 전부).
4. **같은 rank에서 MDAM/Clifft FLOP 비율: cultivation_d5(rank10) 19.3×, ry(rank16) 11.6×** (MDAM이
   많음 — direct general-Pauli butterfly가 2^{r_out+1} joint에서 4 cmul/amp인데 Clifft는 single-axis
   diagonal half-array 1 cmul/amp). 단 이는 full-wall의 ≤6.9%이라 주원인 아님.
5. **같은 amplitude touch당 numerical wall(C++) MDAM/Clifft: high-rank d5_r5 0.67×·ry 0.02× (MDAM이
   빠름), low-rank 2.2–35× (tiny-array call 고정비로 느림).** full-sample 기준 per-touch는 Python이 지배.
6. **MDAM이 더 적은 touch에도 느린 benchmark: coherent_d3_r3(0.68×), coherent_rx_d3_r1(0.72×)** —
   둘 다 full-wall 437×/54× 느림.
7. **그 주원인은 software overhead(Python control plane 92–94%)이지 arithmetic overhead가 아니다.**
   C++ numerical은 5%·3%뿐.
8. **현재 구현 기준 MDAM이 실제로 유리한 benchmark는 `coherent_d5_r5`(rank 24) 하나, 66× full-sample.**
   다른 모든 회로에서 Clifft가 빠르다.

### 결론 / 다음 작업 (이번 범위 밖)
- C++ numerical executor는 정상(고-rank에서 amplitude당 Clifft보다 빠름). **full-shot 병목은
  Python control plane(93–99.9%)** 이므로, 의미 있는 wall 개선은 measurement loop / lazy-pending /
  core discovery / frame-ledger / record 생성을 compiled/batch화하는 것이다(이번 작업 범위 아님).
- correctness regression 0 (9 benches × 5 seeds bit-identical), authoritative default OFF 유지.
