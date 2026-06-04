# MULTI_CNOT Parity-Gather Rewrite

이 문서는 `OP_ARRAY_MULTI_CNOT`을 **exact CNOT-network로 다시 합성**해서 큰 bond를
건너는 transport/QR 횟수를 줄이는 방법의 검증 결과를 정리한다. 목표는 peak 메모리를
유지(또는 절감)하면서 연산량을 줄이는 것이다. (resident entanglement floor 자체를
없애는 방법이 아니다 — 그건 resident streaming/approximation의 몫이다.)

## 1. 아이디어

`MULTI_CNOT(target=t, controls=C)`는 parity 연산이다.

```text
t <- t XOR (XOR_{c in C} c)
```

기본 실행은 control마다 `CNOT(c, t)`이고, control들이 어떤 tree edge `e`의 반대편
subtree에 모여 있고 target이 다른 쪽이면 모든 CNOT이 `e`를 건넌다 (e를 `2m`번 왕복).

Rewrite: 같은 subtree의 control들을 **하나의 accumulator control로 local하게 접고**,
큰 edge는 **accumulator 하나만 한 번 건넌 뒤** 다시 펴서 복구한다.

```text
for c in subtree_controls \ {acc}:  CNOT(c, acc)     # subtree 내부, e 안 건넘
CNOT(acc, t)                                          # e를 한 번만 건넘
for c in reversed(...):             CNOT(c, acc)      # uncompute (acc, c 복구)
```

큰 edge `e` 왕복 횟수: `2m -> 2`. 추가 비용은 subtree 내부 local CNOT `2(m-1)`개.

## 2. Exactness (증명됨)

`scripts/verify_multicnot_parity_rewrite.py` (런타임 무관, 순수 수치):

```text
trials=300
EXACT variants (lin / statevec / symplectic mismatches):
  single_acc        : 0 / 0 / 0
  control_reduction : 0 / 0 / 0
```

- GF(2) 선형 map 동일 (`clifford_frame.RegionLinearFrame`).
- 임의 복소 statevector 적용 결과 동일 (~1e-15).
- Pauli frame symplectic conjugation 동일 (X_i, Z_i 생성자 명시 비교). CNOT-only
  회로의 symplectic은 GF(2) 행렬로 완전히 결정되므로, A가 같으면 frame도 같다.

### 경계 조건 (중요)

control이 아닌 **Steiner ident를 parity 경유지로 XOR하면 틀린다.** TTN의 모든
ident는 scratch 0이 아니라 살아있는 데이터라서 그 원래 값이 parity에 섞인다.
테스트에서 Steiner 경유 변형은 Steiner 노드가 경로에 있을 때 **115/115 모두 실패**했다.
따라서 런타임은 **control끼리만 접고 accumulator를 transport로 건너야 한다** — 마침
현재 `transport_ident_across_edge`가 정확히 그 동작(축 이동, XOR 아님)이다.

## 3. 런타임 구현 (재귀 Steiner-subtree gather)

`core.py`:

- `TTNBackend._build_steiner_children(h_t, control_bags)`
  - control home bag들과 target home `h_t`의 union path로 `h_t`-rooted Steiner tree를 만든다.
  - `children[node]` = `h_t`에서 멀어지는 방향 이웃, `parent[node]` = `h_t` 방향 이웃.
- `TTNBackend._execute_multicnot_parity_gather(target_slot, target_ident, ctrl_pairs, step)`
  - `h_t`에 이미 있는 control은 바로 `CNOT(c, target)` (Class A).
  - `h_t`의 immediate subtree(gateway)마다 **재귀적으로** control을 하나의 representative
    control로 접는다. 각 노드에서 child subtree의 rep들과 그 노드의 local control들을
    모아, `h_t`에 rank-weighted로 가장 가까운 rep로 fold한다 (control-into-control).
  - subtree마다 cost gate로 fold vs direct를 독립 결정 → 비싼 edge가 없는 subtree는
    regression 없이 direct로 둔다.
  - rep을 target으로 한 번 carry(gateway edge 1회 crossing), 그 뒤 fold를 역순 uncompute해
    모든 control 복구.
  - 각 emit은 기존 `_apply_cnot_idents`(Class A / path transport) + `frame.cnot`로 실행 →
    state와 frame이 동일한 net Clifford로 갱신된다.
- `TTNBackend._rank_weighted_path_cost(src, dst)` = `sum_e (1 + log2 chi_e)` (현재 bond dim).
- dispatch: `OP_ARRAY_MULTI_CNOT`의 per-control fallback loop 직전에서, 남은 control들이
  모두 같은 target을 가리키고 `TTN_MULTICNOT_PARITY_REWRITE=1`이면 rewrite로 실행.

immediate-neighbor grouping은 이 재귀의 depth-1 특수 경우다. 깊은 subtree에서는 재귀가
deep edge까지 fold한다 (`coherent_d5_r5`에서 관측된 max fold depth = 13).

### Cost-gated selector (필수)

무조건 fold하면 안 된다. fold용 local CNOT도 transport 왕복이고 compute+uncompute로
2배 든다. 비싼 gateway edge가 없으면 fold는 순손해다 (아래 §4의 700-step regression).
그래서 group마다 proxy cost를 비교해 **fold가 실제로 쌀 때만** 적용한다.

```text
naive_cost = sum_{c in group} 2 * cost(home_c, h_t)
fold_cost  = 2 * cost(home_acc, h_t) + sum_{c in rest} 4 * cost(home_c, home_acc)
fold iff  naive_cost >= min_gain * fold_cost  and naive_cost > fold_cost
```

`TTN_MULTICNOT_PARITY_MIN_GAIN` (기본 1.0).

## 4. 측정 결과 (`scripts/measure_parity_rewrite.py`)

`coherent_d5_r5`, carving_leaf layout, cap 64 MiB.

### selector가 regression을 막는다

cost-gate 없이 "같은 subtree면 무조건 fold"한 첫 prototype은 700-step pure-fallback에서
`transports 494 -> 834 (+69%)`로 **악화**됐다 (그 구간엔 χ=2048 같은 비싼 edge가 아직
없어서 fold 왕복만 추가됨). cost-gate를 넣은 뒤 같은 구간은 `494 -> 488`로 무해해졌다.
이는 사용자가 제안에서 명시한 "selector가 필요하다"를 그대로 실증한다.

### realistic 결과 (1200-step, 둘 다 완주)

재귀 일반화 버전 (max fold depth 13, gateway 12개 중 4개 fold / 8개 direct):

| policy | metric | baseline | rewrite | delta |
|---|---|---:|---:|---:|
| general_policy | actual peak | 138,578,656 | 138,618,368 | +0.03% (유지) |
| general_policy | max_bond_dim | 2048 | 2048 | 동일 |
| general_policy | elapsed | 53.4s | 31.0s | -41.9% |
| general_policy | n_transports | 854 | 770 | -9.8% |
| general_policy | **rank-weighted path** | 1240 | 942 | **-23.8%** |
| general_policy | n_qr | 2176 | 2129 | -2.2% |
| staged_transport | actual peak | 138,578,656 | 138,618,368 | +0.03% (유지) |
| staged_transport | elapsed | 45.1s | 31.5s | -30.1% |
| staged_transport | n_transports | 836 | 778 | -6.9% |
| staged_transport | **rank-weighted path** | 1210 | 952 | **-21.0%** |

해석:

```text
1. peak 메모리는 사실상 불변(+0.03%). 큰 bond를 "덜 열" 뿐 Schmidt rank(chi)를 줄이는 게
   아니므로 resident entanglement floor는 그대로다. (예상된 결과.)
2. 줄어든 건 SVD/QR "횟수"가 아니라 "비싼 연산"이다:
     - n_svd: 0 -> 0 (exact 경로엔 SVD 없음. 줄일 게 없음)
     - n_qr (횟수): -2% 수준 (큰 QR 몇 개가 작은 QR 여러 개로 바뀌어 총개수는 거의 그대로)
     - sum_rank_weighted_path = sum_crossings log2(chi): -21~24% (큰 bond crossing 횟수 감소)
   즉 "연산을 덜 한다"기보다 "비싼 연산(큰 theta+큰 QR)을 싼 연산으로 바꿨다"가 정확하다.
3. wall-clock 30~42% 감소. (noisy하지만 rank-weighted 감소와 방향 일치 — 시간은 횟수가
   아니라 행렬 크기에 지배되므로 큰 연산이 싼 연산으로 바뀌면 횟수가 같아도 빨라진다.)
4. 단 12개 window만 발동했지만 비싼 edge에 정확히 맞아 효과가 크다. 재귀가 max depth 13까지
   작동하지만 cost-gate가 12개 중 4개 gateway만 fold하고 나머지는 direct로 둔다.
5. correctness: 작은 회로(distillation, coherent_d3_r1, coherent_d5_r1, cultivation_d3,
   coherent_d3_r3)에서 rewrite on/off 측정 레코드가 pure/fused 정책 모두 bit-identical.
```

주의: "본드가 줄었다"가 아니다. 본드 차원 chi(다리 폭)는 그대로(2048)이고, 그 다리를 건너는
횟수만 control 수(m)에서 1로 줄였다. peak는 chi가 결정하므로 유지된다.

## 5. 한계와 다음 단계

- 현재 rewrite는 per-control **fallback** loop에서만 발동한다. persistent window가
  흡수하는 MULTI_CNOT은 이미 frame-composition으로 싸므로 영향 없음 — 비싼 fallback
  crossing만 정확히 공략한다.
- 재귀 Steiner-subtree gather로 일반화 완료 (immediate-neighbor는 depth-1 특수 경우).
  관측된 max fold depth = 13. control-only fold 불변식 유지.
- accumulator 선택과 fold 순서는 현재 휴리스틱(rank-weighted 최근접). Steiner-tree 기반
  최적 routing으로 더 줄일 여지가 있다.
- proxy cost는 현재 bond dim 기준 정적 추정이다. fold가 transient bond를 키울 수 있으니
  더 정밀한 cost가 필요하면 boundary χ 변화를 반영해야 한다.

## 6. Flags

```text
TTN_MULTICNOT_PARITY_REWRITE=1     # enable
TTN_MULTICNOT_PARITY_MIN_GAIN=1.0  # fold only if >= this much cheaper (proxy)
```

## 7. 산출물

```text
core.py
  TTNBackend._rank_weighted_path_cost
  TTNBackend._apply_cnot_idents
  TTNBackend._execute_multicnot_parity_gather
  dispatch hook in OP_ARRAY_MULTI_CNOT (per-control fallback)
scripts/verify_multicnot_parity_rewrite.py     # exactness (GF2/statevec/symplectic)
scripts/check_parity_rewrite_correctness.py    # bit-identical records on/off
scripts/measure_parity_rewrite.py              # QR/transport/peak/elapsed deltas
```
