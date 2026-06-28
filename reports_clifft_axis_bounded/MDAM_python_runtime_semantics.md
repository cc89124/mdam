# MDAM full-shot Python runtime — semantic inventory (Phase 1)

목적: "MDAM full-shot의 93.1–99.9%를 차지하는 Python control plane"을 C++로 옮기기 전에, 그
control plane이 **정확히 어떤 함수들로 구성되고 어디에 시간이 가는지**를 코드 분석 + 실측으로 규명한다.
이 문서는 추측이 아니라 (a) 전체 코드 경로 추적과 (b) warmed `run_shot` 함수단위 cProfile + 비-cProfile
실측에 근거한다. 산출 데이터: `artifacts/mdam_fullshot_cpp_runtime/phase1_profile.json`,
`dispatch_overhead_confirm.json`.

기준 구성 = 직전 fair-walltime와 동일: `compiled_core=True`, `_fused_measure=False`,
`clifft_axis_bounded=True, enforce=True`, `taskset -c 2`, threads=1, warmed.

---

## 0. 한 줄 요약 (먼저)

"93–99.9% Python control plane"은 **단일 덩어리가 아니다.** 실측 결과 세 개의 성격이 전혀 다른 비용으로
분해된다:

1. **Opcode-dispatch 인터프리터 오버헤드** (`_opname`+`enum.__str__`/`__get__`, `frame_layer._d`,
   `run_shot`의 `if name==...` 문자열 체인) — stabilizer-heavy 저-rank 벤치(distillation, d5_r1)의
   지배 비용. `_opname` 메모이즈만으로 **full-shot wall의 10–14%**가 사라진다(실측). **C++ VM이 아니라
   Python 단계에서 opcode를 한 번 precompile하면 제거된다.**
2. **Clifford frame + pending 켤레(conjugation)** (`_apply_clifford_to_all`,
   `lazy._conj_cx/_conj_h`) — d5_r5(유일한 승리 벤치)의 지배 비용. C++ frame/ledger가 노릴 **진짜
   대상**. d5_r5 cProfile: frame+pending 켤레 ≈115 ms vs C++ numerical `execute` 8.5 ms.
3. **Python 스칼라 dense rotation kernel** (`_pauli_lincomb_inplace`, `_h_axis`, `_cnot_axes`,
   `bit_count`) — ry_d3_r3(가장 느린 벤치, 463 ms)의 90%. compiled core가 off-diagonal R_Y core를
   **거절**하고 oracle Born 경로가 Python 스칼라 루프로 회전을 적용하기 때문. 이는 spec이 명시적으로
   **다음 별도 작업**으로 분리한 numerical(같은-rank FLOP) 영역이다.

따라서 spec의 전제("control plane을 C++ VM으로 옮기면 wall이 떨어진다")는 **부분적으로만** 맞다.
실측이 가리키는 최적 순서는 §8에 있다.

---

## 1. full-shot call graph

```
Backend.sample(prog, shots, seed)                         # public batch API
  master = default_rng(seed)
  for sh in range(shots):
     sd = master.integers(0, 2**63-1)                     # per-shot seed draw
     run_shot(prog, sd)                                   # ← 전체 control plane
        _structure_for(prog)         [shot-INVARIANT, 캐시됨: dead uids + fast_cores]
        rng = default_rng(sd)                             # PCG64, per shot
        _reset(prog):  frame=PauliFrame(); nc=BoundedEngine(n); slot2id={}
        noise_sampler = ClifftNoiseSampler(prog, rng)
        for step in prog:                                 # EVENT LOOP
           name = _opname(inst.opcode)                    # ← dispatch (enum→str)
           ┌ OP_FRAME_*        → self.frame.{h,s,cnot,cz,swap}      (dormant Clifford)
           ├ OP_NOISE/_BLOCK   → ds._apply_noise_site(.., frame, rng, sampler)  [rng]
           ├ OP_APPLY_PAULI    → ds._apply_cp_mask(.., frame, rng) if record[c]==1 [rng,feedback]
           ├ OP_READOUT_NOISE  → if rng.random()<p: record[m]^=1               [rng]
           ├ OP_MEAS_DORMANT_* → record[c]=frame.xb(a)^sign  | rng.integers(0,2)[rng]
           ├ OP_EXPAND[_T/_ROT]→ _birth(slot)=nc.h(q); _rot(slot,θ)
           ├ OP_PHASE/ARRAY_T/ROT → nc.apply_rotation(0,1<<q,θ)   (lazy pending append)
           ├ OP_ARRAY_{H,S,CNOT,CZ,MULTI_*} → nc.{h,s,cx,cz}(..)  + frame.{..}
           │      └ nc.h/s/cx/cz: _apply_clifford_to_all(fn)  AND  pending 전체 _conj_*
           ├ OP_ARRAY_U2/U4    → _apply_u2/u4: ttn_backend.core matrix de-fusion → ZXZ rot
           └ OP_MEAS_ACTIVE_*  → nc.measure_z(q); record[c]; frame.set_xz       [rng]
                 └ measure_z: _flush_core (pending core flush) →
                       stabilizer(_ag_measure)[rng.integers]  |  magic(compiled/fused/oracle)[rng.random]
        return self.record                                # {classical_idx: bit}
```

핵심: control plane은 **두 개의 Clifford 레이어**를 동시에 굴린다.
- `self.frame = PauliFrame()` — 전체 n-slot dormant Pauli frame(읽기/쓰기 bit, readout, dormant 측정).
- `self.nc` — near-Clifford 엔진의 **자체 tableau** `Xc/Zc`(active/magic register) + dense `phi` +
  lazy `pending` ledger + budget.
그리고 단일 numpy PCG64 `rng`가 noise·dormant·measurement 전반에 **인터리브**되어 소비된다.

---

## 2. 함수 단위 인벤토리 (shot-static vs shot-dynamic vs C++-이동)

`호출/shot`은 d5_r5 warmed shot 실측(대표값). "C++ 이동"은 spec의 full-VM 목표 기준 분류.

| 함수 | 역할 | 호출/shot(d5_r5) | static/dynamic | C++ 이동 | 비고 |
|---|---|---:|---|---|---|
| `run_shot` | 이벤트 루프 | 1 | dynamic | 전체 | 루프 본체가 dispatch 체인 |
| `_opname`/`enum.__str__` | opcode→이름 | 12054 | **static** | **불필요** | precompile로 제거(메모이즈만 10–14%) |
| `frame_layer._d` | inst→dict 디코드 | 1222 | static | precompile | 매 step dict 생성 |
| `PauliFrame.{h,s,cnot,cz,swap,set_xz,xb,zb}` | dormant frame | ~수백 | dynamic(bit) | 이동 | 순수 int bit 연산 |
| `_apply_clifford_to_all(fn)` | nc tableau 켤레 | 899 | dynamic(support static) | **이동(핵심)** | 매 Clifford마다 2n Pauli 켤레 |
| `lazy.cx/h/s/cz` | tableau+pending 켤레 | 779(cx) | dynamic | **이동(핵심)** | pending 전체 재작성 |
| `lazy._conj_cx/_conj_h/_conj_s` | pending 1개 켤레 | 66037 | dynamic(support static) | **이동(핵심)** | O(gates×pending) |
| `apply_rotation` | pending append | 450 | static(목록) | 이동 | uid 단조증가 |
| `_flush_core`/`_fast_cores` lookup | core 선택 | 72 | **static(seed-invariant)** | offline plan | core uid는 shot 불변(검증됨) |
| `_pullback`(inv-frame) | U_C^† P U_C | 다수 | dynamic(support static) | 이동 | O(weight) lookup |
| `_ag_measure` | GK stabilizer 측정 | 12 | dynamic | 이동 | `rng.integers`, frame 갱신 |
| `try_compiled_measure` | magic core (C++ dispatch) | 72 | 혼합 | 부분완료 | localizer/M_mat plan은 static |
| `compiled_core.execute` | **C++ numerical core** | 55 | dynamic | **이미 C++** | full-shot의 6.9%만 |
| `_pauli_lincomb_inplace` | Python dense rotation | (ry 145) | dynamic | numerical(별도) | off-diagonal oracle 경로 |
| `_apply_noise_site`/sampler | noise 주입 | ~수백 | dynamic | 이동 | `rng` 소비 |
| `record[...]=` / 출력 | measurement record | n_meas | dynamic | 이동 | dict→buffer |

**shot-static의 핵심 사실**(이미 `_structure_for`가 3-seed로 검증): active-gate stream과 per-measurement
core(uid)와 dead-uid는 **outcome-독립 = seed-invariant**. 회로의 feedback은 전부 Pauli-frame으로만
라우팅되고 active tableau/rotation은 절대 조건부가 아니기 때문. → core membership, rotation UID 순서,
M_mat 레이아웃, localizer **support**, pullback **support**는 offline plan으로 hoist 가능. **shot마다
달라지는 것은 frame의 phase/sign bit, RNG, outcome, dense phi뿐.**

---

## 3. RNG 소비 맵 (bit-identity의 핵심)

C++에서 bit-identical(replay)을 보장하려면 **모든 draw 지점과 순서**가 일치해야 한다. 실측한 draw 종류:

| 지점 | API | 종류 |
|---|---|---|
| per-shot seed | `master.integers(0,2**63-1)` | 1 int/shot |
| noise site | `ClifftNoiseSampler` + `_apply_noise_site` | site별 다수 |
| conditional Pauli | `_apply_cp_mask(.., rng)` | feedback 시 |
| readout noise | `rng.random()` | site별 1 double |
| dormant random meas | `rng.integers(0,2)` | 1 bit |
| magic measurement | `rng.random()` | core별 1 double |
| stabilizer measurement(`_ag_measure`) | `rng.integers(0,2)` | 1 bit |

draw가 noise·dormant·measurement에 **인터리브**되므로, native-RNG C++가 bit-identical하려면 numpy
**PCG64의 `.random()`/`.integers()`를 draw-for-draw 재현**해야 한다. 이는 매우 비용이 크다. spec(§11,
§14.3)은 이를 인정하고 **replay 모드**(동일 uniform stream 공급)로 정합성을 검증하도록 한다 — 단
replay조차 C++가 **동일 순서로** draw를 소비해야 하므로 noise sampler·dormant·measurement 제어흐름을
충실히 재현해야 한다.

---

## 4. 측정된 Python-time 분해 (cProfile, warmed, ms/shot 환산)

| 벤치 | 지배 함수 (tottime 상위) | 성격 | 최적 처방 |
|---|---|---|---|
| distillation | `_opname`+enum, getattr, `_d`, frame `cnot`, `pauli_commute`, `_ag_measure` | **dispatch+frame+GK** | precompiled dispatch(Python) |
| coherent_d5_r1 | `_opname`, `_apply_clifford_to_all`+`fn`, `pauli_commute`, `_ag_measure` | dispatch+frame+GK | precompiled dispatch + C++ frame |
| coherent_d3_r3 | `_opname`, `fn`(cx 켤레), `lazy.cx`, `_apply_clifford_to_all`, `_conj_cx` | dispatch+**conjugation** | C++ frame+ledger |
| **coherent_d5_r5** | `fn`(44ms), `_conj_cx`(27ms), `_apply_clifford_to_all`(23ms), `cx`(21ms); `execute`(C++) 8.5ms | **conjugation 지배** | **C++ frame+ledger (핵심 대상)** |
| coherent_ry_d3_r3 | `_pauli_lincomb_inplace`(444ms), `bit_count`(74ms), `_cnot_axes`, `_h_axis` | **numerical kernel** | off-diagonal core(**별도 작업**) |
| coherent_rx_d3_r1 | `_apply_clifford_to_all`, `_pauli_lincomb_inplace`, `_pullback_via_basis` | 혼합 | 혼합 |

비-cProfile 실측(`_opname` 메모이즈 delta):

| 벤치 | base ms | memo ms | dispatch% |
|---|---:|---:|---:|
| distillation | 11.57 | 9.99 | **13.7%** |
| coherent_d3_r1 | 1.31 | 1.12 | **14.4%** |
| cultivation_d3 | 5.25 | 4.58 | 12.8% |
| coherent_d5_r1 | 5.42 | 4.77 | 12.0% |
| coherent_d3_r3 | 6.09 | 5.47 | 10.1% |
| cultivation_d5 | 69.5 | 65.0 | 6.5% |
| coherent_rx_d3_r1 | 6.87 | 6.63 | 3.5% |
| coherent_d5_r5 | 128.8 | 125.3 | 2.7% |
| coherent_ry_d3_r3 | 463.2 | 468.8 | -1.2% |

`_opname` 메모이즈는 dispatch 오버헤드의 **일부**(enum 문자열화)일 뿐이다. opcode를 offline에서 정수
dispatch-id + 사전 디코드 인자로 precompile하면 `if name==...` 체인과 `_d`까지 제거되어 dispatch-bound
벤치에서 20–40%까지 회수 가능(미실측, 다음 단계).

---

## 5. C++ full-VM의 hard blockers (실측 기반)

1. **numpy PCG64 RNG 재현 / replay-stream 인터리브.** draw가 noise·dormant·measurement에 섞여 있어,
   native bit-identity는 PCG64를 draw-for-draw 재현해야 하고, replay조차 C++가 동일 제어흐름으로 draw를
   소비해야 한다. → noise sampler·dormant·measurement를 **전부** C++로 옮겨야 비로소 replay가 성립.
2. **ClifftNoiseSampler + `_apply_noise_site`.** clifft 내부 noise 모델 의존. d5_r5에서 noise/pauli op
   483개/shot. C++ 포팅 범위에 포함되며 RNG 소비 순서를 지배.
3. **U2/U4 de-fusion(`ttn_backend.core`).** 행렬 → ZXZ/KAK 분해(LAPACK-class). ry/rx 등 off-axis에서
   등장(active_clifford 360/shot @ ry). C++ 포팅 난도 높음.
4. **두 Clifford 레이어 동기화**(PauliFrame ↔ nc tableau)와 lazy pending 켤레의 정확한 순서/부호.
5. **ry의 지배 비용은 control plane이 아니라 numerical kernel** — C++ control-plane VM으로는 ry의
   463 ms 중 거의 못 줄인다(메모이즈 delta −1.2%가 방증).

---

## 6. 규모/타당성 판정

full C++ VM은 event loop + PauliFrame + ClifftNoiseSampler + U2/U4 분해 + lazy ledger(per-Clifford
켤레) + GK `_ag_measure` + magic measurement + record 생성 + PCG64 replay를 **bit-identical로** 재구현해야
한다. 이는 수천 줄 규모이며 한 세션에서 완결·검증 불가능하다. 또한 실측상 **벤치마다 지배 비용이 달라**
단일 VM이 일률적 이득을 주지 않는다(ry는 범위 밖 numerical, 저-rank는 dispatch).

---

## 7. 권장 staged 계획 (실측이 가리키는 순서)

- **S1 — precompiled dispatch (Python, feature-flag, authoritative 보존).** prog를 offline에서
  (dispatch-id, 사전 디코드 인자) 리스트로 1회 변환 → `run_shot` hot loop에서 `_opname`/enum/`_d`/문자열
  체인 제거. dispatch-bound 벤치 10–40% 회수 예상. **저위험·고ROI·C++ 불필요.** record-bit-identical 검증.
- **S2 — C++ frame+ledger 커널.** `_apply_clifford_to_all` + pending `_conj_*`를 C++로(또는 우선
  알고리즘적으로 active-column만 켤레). d5_r5의 지배 비용(≈115 ms) 직격. 단 RNG·measurement는 Python에
  남겨도 됨(이 두 함수는 RNG를 소비하지 않으므로 bit-identity 무관).
- **S3 — (spec의 full VM) batch C++ runtime.** S1/S2로 줄지 않는 잔여(noise sampler, GK, record,
  per-shot 오케스트레이션)를 C++ batch로. PCG64 replay·noise sampler 포팅이 전제. 가장 큰 작업.
- **별도 — off-diagonal numerical core(ry).** spec이 명시한 다음 단계. control plane과 무관.

이 순서는 spec의 "Python runtime overhead 제거" 목표를 **측정된 wall 감소/위험 비율이 큰 것부터** 달성한다.
S3(완전 VM)는 여전히 목표지만, S1·S2가 먼저 대부분의 dispatch/conjugation 비용을 회수한다.
