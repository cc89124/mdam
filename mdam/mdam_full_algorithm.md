# Measurement-Driven Active Core Simulator: 전체 알고리즘 수학적 설명

> display math는 GitHub 렌더링 호환을 위해 ```math 코드펜스, inline math는 `$...$` 형식으로 작성했다.

> 목적: 이 문서는 **measurement-driven dense core**를 중심으로 한 near-Clifford/QEC 회로 시뮬레이터의 전체 알고리즘을 수학적으로 정리한다.  
> 포함 범위: Pauli/Clifford 표현, 상태 불변식, `G` 구간 연산, `M` 구간 연산, dense core 생성 원리, AUTH/LEAN 비용 모델과 adaptive 선택, 전체 실행 알고리즘, 구현 관점의 연산량.

---

## 0. 핵심 요약

전체 알고리즘의 핵심은 다음 한 문장이다.

```math
\boxed{
\text{전체 }2^N\text{ 상태를 만들지 않고, 각 측정 }M_j\text{가 실제로 필요로 하는 독립 Pauli 자유도만 모아 }2^{r_j}\text{ dense core를 만든다.}
}
```

여기서:

- $N$: 전체 물리/가상 qubit 수
- $r_j$: measurement boundary $M_j$에서 필요한 dense core rank
- $r_j \ll N$이면 exponential cost가 $2^N$이 아니라 $2^{r_j}$가 된다.

회로는 measurement boundary 기준으로 나뉜다.

```math
\mathcal C
=
G_0, M_0, G_1, M_1,\dots,G_{B-1},M_{B-1}.
```

- $G_j$: measurement $M_j$ 전까지의 gate/noise/feedback 구간
- $M_j$: 실제 measurement boundary

실행 상태는 shot $i$, boundary $j$ 직전에 다음처럼 둔다.

```math
S_i^j
=
\left(
F_i^j,
\mathcal L_i^j,
v_i^j,
m_i^j,
\rho_i^j
\right).
```

| 기호 | 의미 | 구현 객체 |
|---|---|---|
| $F_i^j$ | Clifford/Pauli frame | tableau, inverse frame, frame bitset |
| $\mathcal L_i^j$ | pending non-Clifford rotations | pending rotation ledger |
| $v_i^j$ | 현재 reduced core/cache node id | 내용-주소화 node id (상태 저장 없음) |
| $m_i^j$ | measurement record | record buffer |
| $\rho_i^j$ | RNG state | per-shot RNG / noise sampler |

중요한 구분:

```math
\boxed{m_i^j = \text{측정 결과 record}}
```

```math
\boxed{\mathcal L_i^j = \text{아직 dense core에 적용하지 않은 non-Clifford rotation list}}
```

둘은 절대 같은 것이 아니다.

---

# 1. Pauli와 Clifford의 수학적 표현

## 1.1 Pauli string의 binary symplectic 표현

$N$-qubit Pauli string은 phase를 제외하면 두 bit vector로 표현된다.

```math
P(x,z)
=
X^x Z^z
=
\bigotimes_{q=0}^{N-1} X_q^{x_q} Z_q^{z_q},
\quad
x,z\in\mathbb F_2^N.
```

즉 Pauli 하나는 binary vector 하나다.

```math
p=(x\mid z)\in\mathbb F_2^{2N}.
```

각 qubit에서:

| $(x_q,z_q)$ | Pauli |
|---:|---|
| $(0,0)$ | $I$ |
| $(1,0)$ | $X$ |
| $(0,1)$ | $Z$ |
| $(1,1)$ | $Y$, phase convention 포함 |

두 Pauli의 곱은 phase를 제외하면 XOR이다.

```math
(x_1\mid z_1)(x_2\mid z_2)
\sim
(x_1\oplus x_2\mid z_1\oplus z_2).
```

## 1.2 Commutation test

두 Pauli vector

```math
p=(x\mid z),\quad q=(x'\mid z')
```

의 symplectic inner product를 다음처럼 정의한다.

```math
[p,q]
=
x\cdot z' + z\cdot x'
\pmod 2.
```

그러면:

```math
P(p)P(q)=(-1)^{[p,q]}P(q)P(p).
```

즉:

```math
[p,q]=0 \Rightarrow \text{commute},
```

```math
[p,q]=1 \Rightarrow \text{anticommute}.
```

구현에서는 bitset AND + popcount parity다.

```cpp
bool anticommutes(Pauli p, Pauli q) {
    uint64_t acc = 0;
    for (int w = 0; w < words; w++) {
        acc ^= popcount((p.x[w] & q.z[w]) ^ (p.z[w] & q.x[w])) & 1;
    }
    return acc & 1;
}
```

비용:

```math
C_{\mathrm{comm}}=O(N/w),
```

where $w=64$ machine word width.

---

# 2. 상태 표현의 기본 불변식

## 2.1 전체 상태를 직접 저장하지 않는다

일반 state-vector simulator는

```math
|\psi\rangle\in\mathbb C^{2^N}
```

를 저장해야 한다. 그러나 이 알고리즘은 전체 상태를 저장하지 않는다.

대신 frame과 pending rotation, 그리고 작은 dense core node를 사용한다.

개념적으로 shot $i$, boundary $j$의 물리 상태는 다음처럼 표현된다.

```math
|\psi_i^j\rangle
=
\Gamma_i^j\,
U(F_i^j)\,
U(\mathcal L_i^j)\,
\mathcal I_{v_i^j}
|\chi_{v_i^j}\rangle.
```

각 항의 의미는 다음과 같다.

| 항 | 의미 |
|---|---|
| $\Gamma_i^j$ | global phase / scalar |
| $U(F_i^j)$ | Clifford frame이 나타내는 unitary |
| $U(\mathcal L_i^j)$ | pending non-Clifford rotations의 곱 |
| $\mathcal I_{v_i^j}$ | reduced core를 전체 Hilbert space에 embed하는 map |
| $|\chi_{v_i^j}\rangle$ | node $v_i^j$에 저장된 dense core state |

pending list는 다음과 같다.

```math
\mathcal L_i^j
=
\left[
(q_1,\theta_1),
(q_2,\theta_2),
\dots,
(q_{\ell},\theta_{\ell})
\right],
\quad
q_a\in\mathbb F_2^{2N}.
```

따라서:

```math
U(\mathcal L_i^j)
=
R_{q_\ell}(\theta_\ell)
\cdots
R_{q_2}(\theta_2)
R_{q_1}(\theta_1),
```

where:

```math
R_q(\theta)=e^{-i\theta P(q)/2}.
```

## 2.2 Core node의 의미

node $v$는 단순한 물리 qubit subset이 아니다. node는 다음 데이터를 가진다.

```math
v = (B_v, \chi_v, \sigma_v).
```

| 항 | 의미 |
|---|---|
| $B_v$ | local core basis, global Pauli axes를 local axes로 바꾸는 symplectic basis |
| $\chi_v\in\mathbb C^{2^{r_v}}$ | dense core vector |
| $\sigma_v$ | signs/stabilizer metadata |

$B_v$는 global Pauli subspace를 local $r_v$-qubit Pauli algebra로 표현하는 좌표계다.

```math
\lambda_{B_v}: \mathcal W_v \subseteq \mathbb F_2^{2N}
\rightarrow
\mathbb F_2^{2r_v}
```

이며, 이 map은 다음 성질을 만족해야 한다.

```math
P(p)\,\mathcal I_v |\xi\rangle
=
\eta_v(p)\,\mathcal I_v\,P(\lambda_{B_v}(p))|\xi\rangle,
\quad
p\in\mathcal W_v.
```

여기서:

- $\mathcal W_v$: node가 표현할 수 있는 global Pauli subspace
- $\eta_v(p)\in\{\pm 1,\pm i\}$: stabilizer/fixed-axis 때문에 생기는 scalar phase
- $P(\lambda_{B_v}(p))$: local $r_v$-qubit Pauli

이 식이 dense core의 정확성 조건이다.

즉 global Pauli를 full $2^N$ 공간에 적용하는 대신, local Pauli를 $2^{r_v}$ dense core에 적용해도 같은 결과가 나와야 한다.

---

# 3. 회로 분해: G 구간과 M 구간

회로는 measurement boundary 기준으로 분해한다.

```math
\mathcal C
=
G_0,M_0,G_1,M_1,\dots,G_{B-1},M_{B-1}.
```

각 $G_j$는 measurement $M_j$ 이전의 연산 묶음이다.

```math
G_j=(o_{j,1},o_{j,2},\dots,o_{j,g_j}).
```

각 operation은 다음 중 하나다.

1. Clifford gate
2. Non-Clifford Pauli rotation
3. Stochastic Pauli noise
4. Classically controlled Pauli correction
5. Record write/read
6. Detector/postselection/output bookkeeping

$M_j$는 active measurement boundary다.

```math
M_j=(p_j^{\mathrm{phys}},\mathrm{slot}_j,\text{reset/update rule}).
```

---

# 4. G 구간의 수학적 연산

## 4.1 Clifford gate

physical Clifford gate $C$가 적용된다고 하자.

현재 상태:

```math
|\psi\rangle=U(F)U(\mathcal L)\mathcal I_v|\chi_v\rangle.
```

Clifford 적용 후:

```math
|\psi'\rangle=C|\psi\rangle
=C U(F)U(\mathcal L)\mathcal I_v|\chi_v\rangle.
```

따라서 frame을 갱신한다.

```math
U(F')=C U(F).
```

즉:

```math
F\leftarrow C\circ F.
```

구현:

```cpp
frame.apply_clifford(C);
```

비용:

- tableau 직접 갱신이면 $O(N/w)$ 또는 gate 종류에 따라 $O(1)\sim O(N/w)$
- offline compile에서 처리하면 runtime 비용 0
- shot-dependent Pauli frame만 있으면 bit XOR 수준

## 4.2 Non-Clifford Pauli rotation

physical non-Clifford rotation:

```math
R_{p}^{\mathrm{phys}}(\theta)=e^{-i\theta P(p)/2}
```

가 들어왔다.

현재 상태:

```math
|\psi\rangle=U(F)U(\mathcal L)\mathcal I_v|\chi_v\rangle.
```

적용 후:

```math
|\psi'\rangle=R_p(\theta)U(F)U(\mathcal L)\mathcal I_v|\chi_v\rangle.
```

frame 뒤로 넘긴다.

```math
|\psi'\rangle
=U(F)
\left[U(F)^\dagger R_p(\theta)U(F)\right]
U(\mathcal L)\mathcal I_v|\chi_v\rangle.
```

Clifford conjugation은 Pauli를 Pauli로 보낸다.

```math
\tilde p = F^{-1}p,
\quad
U(F)^\dagger P(p)U(F)=sP(\tilde p),
\quad s\in\{+1,-1\}.
```

sign은 angle에 흡수한다.

```math
U(F)^\dagger R_p(\theta)U(F)
=R_{\tilde p}(s\theta).
```

따라서 pending list에 append한다.

```math
\mathcal L\leftarrow \mathcal L\cdot[(\tilde p,s\theta)].
```

구현:

```cpp
Pauli p_tilde = inverse_frame.pullback(p_phys);
if (sign < 0) theta = -theta;
pending.push({p_tilde, theta});
```

비용:

```math
C_{\mathrm{pullback}}=O(cN/w),
```

where $c$는 physical Pauli support와 tableau overlap. pending append는 mask copy라서

```math
C_{\mathrm{append}}=O(N/w).
```

중요:

```math
\boxed{\text{이 단계에서는 dense vector }\chi\text{를 건드리지 않는다.}}
```

## 4.3 Stochastic Pauli noise

noise site $e$에서 Pauli error $a$가 확률 $p_e(a)$로 발생한다고 하자.

```math
a_i\sim p_e(a).
```

physical error Pauli $P(a_i)$를 frame 뒤로 넘긴다.

```math
\tilde a_i=F^{-1}a_i.
```

이것은 non-Clifford가 아니므로 pending에 넣지 않고 Pauli frame에 곱한다.

```math
F_i^{\mathrm{Pauli}}
\leftarrow
\tilde a_i F_i^{\mathrm{Pauli}}.
```

구현:

```cpp
Pauli sampled = noise.sample(rng);
Pauli a_tilde = inverse_frame.pullback(sampled);
pauli_frame ^= a_tilde;
```

저확률 noise에서는 Bernoulli를 모든 site마다 뽑지 않고 gap/hazard sampling을 쓸 수 있다.

naive 비용:

```math
C_{\mathrm{noise,naive}}=O(E_j) + O(f_jN/w),
```

where $E_j$는 noise site 수, $f_j$는 실제 fault 수.

sparse fault sampling이면 기대 비용:

```math
\mathbb E[C_{\mathrm{noise,sparse}}]=O(f_j\log E_j)+O(f_jN/w).
```

## 4.4 Feedback Pauli correction

record bit $m[k]$에 따라 Pauli correction $c$를 적용한다고 하자.

```math
\text{if }m[k]=1:\quad P(c)\text{ apply}.
```

frame 표현에서는:

```math
F_i^{\mathrm{Pauli}}
\leftarrow
\left(F^{-1}c\right)^{m[k]}F_i^{\mathrm{Pauli}}.
```

구현:

```cpp
if (record[k]) {
    pauli_frame ^= precompiled_feedback_pauli;
}
```

비용:

```math
C_{\mathrm{feedback}}=O(N/w)
```

또는 support가 작으면 $O(\mathrm{supp}/w)$.

## 4.5 G 구간 전체 비용

shot 하나, boundary $j$의 G 구간 비용을:

```math
A_j = A_j^{\mathrm{frame}}+A_j^{\mathrm{pending}}+A_j^{\mathrm{noise}}+A_j^{\mathrm{record}}.
```

대략:

```math
A_j
=
O\left(
(c_j+t_j+f_j+h_j)\frac{N}{w}
+ E_j^{\mathrm{sample}}
+ g_j^{\mathrm{dispatch}}
\right),
```

where:

| 기호 | 의미 |
|---|---|
| $c_j$ | runtime frame update 수 |
| $t_j$ | non-Clifford rotation pullback/append 수 |
| $f_j$ | realized fault 수 |
| $h_j$ | feedback correction 수 |
| $E_j^{\mathrm{sample}}$ | noise sampling cost |
| $g_j^{\mathrm{dispatch}}$ | instruction dispatch overhead |

중요하게, G 구간은 일반적으로 dense $2^r$ vector traversal을 하지 않는다.

---

# 5. M 구간: measurement boundary의 수학

measurement boundary $M_j$에서 physical measurement Pauli를 $p_j^{\mathrm{phys}}$라고 하자.

현재 상태:

```math
|\psi_i^j\rangle
=\Gamma_i^j U(F_i^j)U(\mathcal L_i^j)\mathcal I_{v_i^j}|\chi_{v_i^j}\rangle.
```

## 5.1 Measurement axis pullback

measurement Pauli를 frame 뒤로 넘긴다.

```math
U(F_i^j)^\dagger P(p_j^{\mathrm{phys}})U(F_i^j)
=(-1)^{\delta_{i,j}}P(\tilde p_{i,j}).
```

여기서:

- $\tilde p_{i,j}$: core/virtual basis 기준 measurement Pauli
- $\delta_{i,j}\in\{0,1\}$: sign flip

physical outcome bit $m$과 core raw outcome $b$의 관계는:

```math
\Pi_m(P(p_j^{\mathrm{phys}}))
\quad\Longleftrightarrow\quad
\Pi_{m\oplus\delta_{i,j}}(P(\tilde p_{i,j})).
```

따라서:

```math
\boxed{m_{i,j}=b_{i,j}\oplus\delta_{i,j}.}
```

## 5.2 Measurement projector

Pauli $P$ 측정의 projector는:

```math
\Pi_b(P)=\frac{I+(-1)^bP}{2},
\quad b\in\{0,1\}.
```

- $b=0$: $+1$ eigenspace
- $b=1$: $-1$ eigenspace

측정 확률:

```math
p_b=\|\Pi_b(P)|\phi\rangle\|^2.
```

측정 후 상태:

```math
|\phi'\rangle
=
\frac{\Pi_b(P)|\phi\rangle}{\sqrt{p_b}}.
```

---

# 6. Dense core 생성 원리

이 부분이 논문의 핵심이다.

## 6.1 목표

우리가 계산해야 하는 것은 full state에서의 측정 확률이다.

```math
p_b
=
\left\|
\Pi_b(P(\tilde p_{i,j}))
U(\mathcal L_i^j)
\mathcal I_{v_i^j}|\chi_{v_i^j}\rangle
\right\|^2.
```

full $N$-qubit 공간에서 계산하면 불가능하다. 대신 어떤 작은 local core basis $B_{i,j}$와 embedding $\mathcal I_{B_{i,j}}$를 만들어서:

```math
U(\mathcal L_i^j)\mathcal I_{v_i^j}|\chi_{v_i^j}\rangle
=
\mathcal I_{B_{i,j}}|\phi_{i,j}\rangle
```

가 되도록 만든다. 그러면:

```math
p_b
=
\left\|
\Pi_b(P(\hat p_{i,j}))|\phi_{i,j}\rangle
\right\|^2,
```

where:

```math
\hat p_{i,j}=\lambda_{B_{i,j}}(\tilde p_{i,j})\in\mathbb F_2^{2r_{i,j}}.
```

즉 full measurement를 작은 $r_{i,j}$-qubit core measurement로 바꾼다.

```math
\boxed{
\mathbb C^{2^N}\text{에서의 문제}
\quad\rightarrow\quad
\mathbb C^{2^{r_{i,j}}}\text{에서의 문제}
}
```

## 6.2 Relevant Pauli subspace

pending list:

```math
\mathcal L_i^j=[(q_1,\theta_1),\dots,(q_\ell,\theta_\ell)].
```

측정 $\tilde p$의 확률과 post-measurement state에 영향을 줄 수 있는 Pauli 축들의 subspace를 만든다.

먼저 현재 node가 이미 표현하는 subspace를 $\mathcal W_{v}$라고 하자.

```math
\mathcal W_v=\mathrm{span}_{\mathbb F_2}(B_v).
```

measurement boundary에서 필요한 초기 set은:

```math
\mathcal A^{(0)}_{i,j}
=
\mathcal W_{v_i^j}+\mathrm{span}\{\tilde p_{i,j}\}.
```

이제 pending rotations를 시간 순서대로 또는 backward dependency 순서로 검사한다.

rotation axis $q_a$가 현재 relevant subspace와 모두 commute하면, 현재 measurement 계산에는 직접적인 dense freedom을 추가하지 않는다.

```math
\forall r\in \mathcal A^{(s)}_{i,j},\quad [q_a,r]=0
\quad\Rightarrow\quad
q_a\text{ is not added at this step.}
```

반대로 어떤 relevant axis와 anticommute하면, 그 rotation은 측정 확률 또는 projected state를 바꿀 수 있으므로 core subspace에 포함한다.

```math
\exists r\in\mathcal A^{(s)}_{i,j}: [q_a,r]=1
\quad\Rightarrow\quad
\mathcal A^{(s+1)}_{i,j}
=
\mathcal A^{(s)}_{i,j}+\mathrm{span}\{q_a\}.
```

fixed point까지 반복한다.

```math
\mathcal A_{i,j}
=
\mathrm{Fix}
\left(
\mathcal W_{v_i^j},\tilde p_{i,j},\{q_a\}_{a=1}^{\ell}
\right).
```

이 $\mathcal A_{i,j}$가 measurement-driven dense core의 algebraic closure다.

직관:

- measurement와 완전히 독립인 pending rotation은 현재 boundary dense core에 들어오지 않는다.
- measurement와 anticommute하거나, 이미 들어온 axis와 nontrivial하게 얽히는 rotation만 들어온다.
- 따라서 dense rank는 전체 pending 수가 아니라 measurement-relevant independent rank가 된다.

## 6.3 Stabilizer/fixed-axis quotient

모든 axis가 dense freedom을 요구하는 것은 아니다. stabilizer나 dormant $|0\rangle$에 의해 고정된 방향은 scalar로 제거된다.

현재 고정된 Pauli subspace를 $\mathcal S_{i,j}$라고 하자. 예를 들어 virtual dormant $|0\rangle_D$는 $Z_d$ stabilizer를 가진다.

```math
Z_d|0\rangle_D=|0\rangle_D.
```

따라서 어떤 Pauli가 dormant part에서 Z-only이면 scalar로 처리된다.

```math
P(p_A\oplus z_D)
\left(|\xi\rangle_A\otimes |0\rangle_D\right)
=
P(p_A)|\xi\rangle_A\otimes |0\rangle_D.
```

즉 $Z_D$ 성분은 dense rank를 늘리지 않는다.

수학적으로는 다음 quotient/localization 문제를 푼다.

```math
\lambda_B:
\mathcal A_{i,j}
\rightarrow
\mathbb F_2^{2r_{i,j}}
```

such that for all $p\in\mathcal A_{i,j}$:

```math
P(p)\mathcal I_B
=
\eta_B(p)\mathcal I_B P(\lambda_B(p)).
```

여기서 $r_{i,j}$는 이 조건을 만족하는 local representation의 qubit 수다.

구현적으로는 symplectic Gaussian elimination으로 다음을 찾는다.

```math
S_{i,j}\in Sp(2N,\mathbb F_2)
```

such that:

```math
pS_{i,j}
=
(\hat x_p,\hat z_p \mid z_D(p)),
```

where:

- \((\hat x_p,\hat z_p)\in\mathbb F_2^{2r_{i,j}}
- $z_D(p)$는 dormant/fixed Z-only part
- dormant X component는 제거되어야 하며, 제거되지 않으면 해당 axis를 active/core에 promote한다.

정의:

```math
\boxed{
 r_{i,j}
 =
 \mathrm{CoreRank}(\mathcal A_{i,j},\mathcal S_{i,j})
}
```

여기서 $\mathrm{CoreRank}$는 stabilizer quotient 후 남는 non-scalar local Pauli algebra를 표현하는 데 필요한 최소 또는 구현상 선택된 virtual qubit 수다.

## 6.4 Dense workspace 생성

current node $v$의 core가 이미 $r_v$-dimensional dense vector를 가진다고 하자.

```math
\chi_v\in\mathbb C^{2^{r_v}}.
```

measurement workspace basis $B$의 rank가 $r$라면:

```math
\chi_B^{(0)}=E_{v\rightarrow B}\chi_v,
\quad
E_{v\rightarrow B}:\mathbb C^{2^{r_v}}\rightarrow\mathbb C^{2^r}.
```

일반적으로 $E_{v\rightarrow B}$는 다음 조합이다.

1. 기존 local axes 재배열
2. 새 dormant axes를 $|0\rangle$으로 tensor product
3. 필요한 local Clifford basis change
4. stabilizer scalar phase 반영

간단한 경우:

```math
\chi_B^{(0)}[x,y]
=
\chi_v[x]\cdot \mathbf 1[y=0],
```

where $y$는 새로 추가된 virtual axes의 basis bits.

즉 새 core axes는 처음에 $|0\rangle$으로 들어오고, 이후 pending rotations가 이 축을 superposition으로 만든다.

## 6.5 Pending rotations localize

각 relevant pending axis $q_a\in\mathcal A_{i,j}$를 local Pauli로 바꾼다.

```math
\hat q_a=\lambda_B(q_a)
=(\hat x_a\mid \hat z_a)\in\mathbb F_2^{2r}.
```

그 후 dense vector에 순서대로 적용한다.

```math
\chi_B^{(a)}
=
R_{\hat q_a}(\theta_a)\chi_B^{(a-1)}.
```

전체 materialization:

```math
\phi_{i,j}
=
\chi_B^{(\ell)}
=
\left(\prod_{a\in\mathrm{Rel}_{i,j}} R_{\hat q_a}(\theta_a)\right)
E_{v\rightarrow B}\chi_v.
```

여기서 절대 하지 않는 것:

```math
\boxed{
\prod_a R_{\hat q_a}(\theta_a)\text{를 }2^r\times2^r\text{ dense matrix로 만들지 않는다.}
}
```

또한 Pauli-sum으로 확장하지 않는다.

```math
\boxed{
\prod_a (c_a I+s_a P_a)\text{를 }\sum_u c_u P_u\text{로 전개하지 않는다.}
}
```

항상 factorized product를 순차 apply한다.

---

# 7. Dense rotation apply의 정확한 수식

local Pauli $\hat q=(a\mid b)\in\mathbb F_2^{2r}$가 있다고 하자.

state:

```math
|\chi\rangle
=
\sum_{x=0}^{2^r-1}\chi_x|x\rangle.
```

Pauli action은:

```math
P(\hat q)|x\rangle
=
\omega_{\hat q}(x)|x\oplus a\rangle.
```

여기서:

```math
\omega_{\hat q}(x)=i^{\kappa(\hat q)}(-1)^{b\cdot x}
```

이며 $\kappa$는 Y convention/sign phase를 포함한다.

Pauli rotation:

```math
R_{\hat q}(\theta)
=e^{-i\theta P(\hat q)/2}
=
\cos\frac\theta2 I
-i\sin\frac\theta2 P(\hat q).
```

```math
c=\cos\frac\theta2,
\quad
s=\sin\frac\theta2.
```

그러면 output amplitude $y$는:

```math
\boxed{
\chi'_y
=
c\chi_y
-i s\,\omega_{\hat q}(y\oplus a)\,\chi_{y\oplus a}.
}
```

이 식을 모든 \(y\in\{0,
\dots,2^r-1\}\)에 대해 계산하는 것이 one sweep이다.

구현:

```cpp
for (uint64_t y = 0; y < (1ull << r); y++) {
    uint64_t x = y ^ q.xmask;
    complex phase = pauli_phase(q, x);
    out[y] = c * in[y] + complex(0, -s) * phase * in[x];
}
```

비용:

```math
C_{\mathrm{rot}}(r)=\Theta(2^r).
```

pending relevant rotation 수가 $q_{\mathrm{rel}}$이면:

```math
C_{\mathrm{materialize}}(r,q_{\mathrm{rel}})
=
\Theta(q_{\mathrm{rel}}2^r).
```

---

# 8. Dense measurement apply의 정확한 수식

measurement axis도 localize한다.

```math
\hat p=\lambda_B(\tilde p)=(a_m\mid b_m)\in\mathbb F_2^{2r}.
```

현재 dense state:

```math
|\phi\rangle=\sum_y\phi_y|y\rangle.
```

## 8.1 Expectation value

```math
\langle \hat P\rangle
=\langle\phi|P(\hat p)|\phi\rangle.
```

Pauli action을 쓰면:

```math
\boxed{
\langle \hat P\rangle
=
\sum_y
\overline{\phi_y}
\,\omega_{\hat p}(y\oplus a_m)
\phi_{y\oplus a_m}.
}
```

비용:

```math
C_{\mathrm{born}}(r)=\Theta(2^r).
```

## 8.2 Outcome probability

```math
p_0=\frac{1+\langle \hat P\rangle}{2},
\quad
p_1=\frac{1-\langle \hat P\rangle}{2}.
```

수치 오차 때문에 구현에서는 clamp한다.

```cpp
p0 = clamp(0.5 * (1.0 + real(expval)), 0.0, 1.0);
```

## 8.3 Sampling

```math
u\sim\mathrm{Uniform}(0,1).
```

```math
b=
\begin{cases}
0,&u \lt p_0,\\
1,&u\ge p_0.
\end{cases}
```

## 8.4 Projection and normalization

projector:

```math
\Pi_b(\hat P)=\frac{I+(-1)^bP(\hat p)}{2}.
```

post-measurement dense state:

```math
\phi'_y
=
\frac{
\phi_y+(-1)^b\omega_{\hat p}(y\oplus a_m)\phi_{y\oplus a_m}
}{2\sqrt{p_b}}.
```

비용:

```math
C_{\mathrm{project}}(r)=\Theta(2^r).
```

## 8.5 Localize measurement to Z and drop measured axis

구현에서는 measurement Pauli $\hat p$를 local Clifford $W$로 single Z axis로 바꾸는 것이 편하다.

```math
W\hat P W^\dagger=s_m Z_{r_*},
\quad s_m\in\{+1,-1\}.
```

dense state도 같이 변환한다.

```math
|\phi^W\rangle=W|\phi\rangle.
```

그러면 branch norm은 단순 partial norm이다.

```math
s_b
=
\sum_{y:\, y_{r_*}=b} |\phi^W_y|^2.
```

sign이 있으면:

```math
p_0=
\begin{cases}
\dfrac{s_0}{s_0+s_1},&s_m=+1,\\
\dfrac{s_1}{s_0+s_1},&s_m=-1.
\end{cases}
```

projection은 killed branch를 zero하고 kept branch를 normalize한다.

결과 $b$가 나오면 measured coordinate는 고정된다. 따라서 dense rank를 줄인다.

```math
r\leftarrow r-1.
```

예를 들어 measured bit가 마지막 coordinate이고 kept bit가 $b$이면:

```math
\chi_{\mathrm{next}}[u]
=
\phi^W[(u,b)]/\sqrt{s_b}.
```

이 rank contraction이 장기적으로 dense core를 작게 유지하는 핵심이다.

---

# 9. M 구간 slow path / cache miss 알고리즘

cache miss이면 boundary transition을 실제로 만든다.

## 9.1 수학적 transition

key $k$가 주어졌다고 하자.

```math
k=(j,v,F\text{-sig},\mathcal L\text{-sig},\tilde p\text{-sig}).
```

slow path는 다음 transition을 계산한다.

```math
\mathrm{BuildEdge}(k)
=
\left(p_0(k),v_0'(k),v_1'(k),a(k)\right).
```

여기서:

- $p_0(k)$: outcome 0 확률
- $v_0'(k)$: raw outcome $b=0$일 때 next core node
- $v_1'(k)$: raw outcome $b=1$일 때 next core node
- $a(k)$: sign, local basis, auxiliary metadata

## 9.2 구현 절차

```cpp
Edge build_edge(Key k) {
    // 1. Collect relevant Pauli axes.
    AxisSet A = closure(k.current_node_basis,
                        k.measurement_axis,
                        k.pending_axes);

    // 2. Remove stabilizer/fixed-axis redundancy.
    CoreBasis B = symplectic_reduce_and_build_basis(A, stabilizers);
    int r = B.rank;

    // 3. Embed current node dense vector into workspace basis.
    Vector phi = embed_node_state(k.node, B);  // length 2^r

    // 4. Apply relevant pending rotations one by one.
    for (Rotation rot : relevant_pending) {
        LocalPauli qhat = B.localize(rot.axis);
        apply_pauli_rotation(phi, qhat, rot.theta); // Θ(2^r)
    }

    // 5. Localize measurement.
    LocalPauli phat = B.localize(k.measurement_axis);

    // 6. Compute Born probability.
    double p0 = born_probability(phi, phat); // Θ(2^r)

    // 7. Compute both projected successors.
    Vector phi0 = project(phi, phat, 0);     // Θ(2^r)
    Vector phi1 = project(phi, phat, 1);     // Θ(2^r)

    // 8. Drop measured axis if possible and intern nodes.
    NodeId n0 = intern(drop_axis_if_possible(phi0, phat, 0));
    NodeId n1 = intern(drop_axis_if_possible(phi1, phat, 1));

    return Edge{p0, n0, n1, aux};
}
```

## 9.3 slow path 연산량

axis closure cost:

```math
C_{\mathrm{closure}}
=O(q_{\mathrm{pend}}\,a\,N/w)
```

where $a$는 closure 중 axis 수.

symplectic reduction/local basis construction:

```math
C_{\mathrm{basis}}
=O(a^2N/w+a^3)
```

localization of relevant axes:

```math
C_{\mathrm{localize}}
=O(q_{\mathrm{rel}}\,r\,N/w)
```

embed/coordinate transform:

```math
C_{\mathrm{embed}}=O(2^r)\text{ to }O(r2^r),
```

구현 방식에 따라 달라진다.

rotation materialization:

```math
C_{\mathrm{rot}}=\Theta(q_{\mathrm{rel}}2^r).
```

Born/projection/two successors:

```math
C_{\mathrm{meas}}=\Theta(2^r).
```

node intern/hash:

```math
C_{\mathrm{intern}}=O(2^r)\quad(\text{dense 내용의 content-hash로 node id를 만든다; 상태 자체는 저장하지 않는다}).
```

따라서 miss cost는 대략:

```math
\boxed{
C_{\mathrm{miss}}
=
O(a^2N/w+a^3+q_{\mathrm{rel}}rN/w)
+
\Theta((q_{\mathrm{rel}}+c_m)2^r)
}
```

where $c_m$는 Born/project/intern 상수이다.

보통 큰 병목은:

```math
\Theta(q_{\mathrm{rel}}2^r).
```

---

# 10. M 구간 cache hit 알고리즘

cache hit이면 dense core를 만들지 않는다.

## 10.1 Cache key

```math
k_{i,j}=K
\left(
 j,
 v_i^j,
 \sigma(F_i^j),
 \sigma(\mathcal L_i^j),
 \sigma(\tilde p_{i,j})
\right).
```

cache entry:

```math
E(k)=\left(p_0(k),v_0(k),v_1(k),a(k)\right).
```

## 10.2 Hit execution

```math
b_{i,j}\sim\mathrm{Bernoulli}(p_0(k_{i,j})).
```

```math
v_i^{j+1}=
\begin{cases}
v_0(k_{i,j}),&b_{i,j}=0,\\
v_1(k_{i,j}),&b_{i,j}=1.
\end{cases}
```

record update:

```math
m_i[\mathrm{slot}_j]
=b_{i,j}\oplus\delta_{i,j}.
```

pending consumed or advanced according to edge metadata:

```math
\mathcal L_i^{j+1}=\mathrm{UpdatePending}(\mathcal L_i^j,E(k),b_{i,j}).
```

구현:

```cpp
Edge e = cache.lookup(key);
int b = rng.bernoulli(e.p0);
node = b ? e.next1 : e.next0;
record[slot] = b ^ delta;
apply_measurement_frame_update(frame, record[slot]);
```

비용:

```math
C_{\mathrm{hit}}
=O(C_{\mathrm{key}}+C_{\mathrm{hash}}+C_{\mathrm{rng}}+C_{\mathrm{record}}+C_{\mathrm{frame-update}}).
```

여기에는 $2^r$ dense traversal이 없다.

$v_0/v_1$은 내용-주소화 node **id**다 — cache는 상태를 저장하지 않는다. 캐시 entry는
BoundaryKey$\to$transition(위의 $E(k)$: $p_0$, successor id, anti-commuting 플래그, record 메타)이
전부다. 이 절의 무-dense hit은 LEAN walk(§11.2)에서 성립한다. miss 복구 재실행(§12.1의 CACHE 계층)
중의 boundary hit은 live core를 갖고 있으므로 cached $p_0$로 기대값 계산만 생략하고, 사영
$\Theta(2^r)$은 엔진이 그대로 수행한다.

---

# 11. AUTH와 LEAN의 비용 모델

MDAM 런타임의 production 정책은 두 개다.

```math
\boxed{\text{정책}=\{\text{AUTH},\ \text{LEAN}\}}
```

| 정책 | 의미 |
|---|---|
| AUTH | 모든 boundary를 §9 slow path로 직접 계산하는 authoritative 경로 |
| LEAN | boundary automaton을 학습해 두고, hit이면 G 구간 실행 자체를 건너뛰는 walk 경로 |

§10의 boundary cache 실행은 독립 정책이 아니다. LEAN이 miss를 복구할 때 통과하는 **내부 계층**이며, 그 자체로 최종 선택되는 일은 없다.

## 11.1 AUTH

AUTH는 모든 shot의 모든 measurement boundary에서 직접 slow path를 수행한다.

```math
T_{\mathrm{AUTH}}(N_{\mathrm{shot}})
=
\sum_{i=1}^{N_{\mathrm{shot}}}\sum_{j=0}^{B-1}
\left(A_{i,j}+C_{\mathrm{miss}}(i,j)+R_{i,j}+O_{i,j}\right)
\approx
N_{\mathrm{shot}}B(A+D+R+O),
```

where:

```math
D\approx \Theta((q_{\mathrm{rel}}+c_m)2^r).
```

특징:

- per-shot 비용이 shot 수와 무관하다 (학습이 없다).
- 메모리는 상수다: $M_{\mathrm{AUTH}}=O(\text{frame}+\text{pending}+\text{core})$.
- 이 비용 자체가 이미 $r\ll k$ localization의 산물이다. $2^k$가 큰 회로에서는 AUTH만으로도 dense-sweep 방식 대비 지수적 이득이 난다.

## 11.2 LEAN: boundary automaton walk

측정으로 검증된 separability 관찰: 고정 회로에서 shot마다 달라지는 것은 (i) noise/measurement RNG가 뽑는 비트와 (ii) 그 비트가 만드는 회전 부호·분기뿐이다. G 구간의 나머지 심볼릭 계산 전부는 (직전 boundary 상태, 그 구간의 부호 비트)의 **결정적 함수**다. LEAN은 이 함수를 자동자(automaton)로 실체화한다.

**노드** — dense core 상태의 내용 주소화(interning):

```math
v=\mathrm{intern}\left(j,\ \text{core 내용},\ \sigma(F),\ \sigma(\mathcal L)\right).
```

**Segment edge** — 두 boundary 사이의 G 구간까지 통째로 접은 전이:

```math
e:\ (v,\ b_{\mathrm{prev}},\ \sigma_{\mathrm{seg}})
\ \mapsto\
\left(p_0,\ v'_0,\ v'_1,\ \text{record 메타}\right).
```

$\sigma_{\mathrm{seg}}$는 그 구간에서 뽑힌 noise가 회전 부호와 분기 선택(예: outcome 조건부 1-qubit unitary의 입력 상태)에 미친 영향을 순서 보존 해시로 접은 값이다. measurement feedback도 frame을 거쳐 부호에 반영되므로 자동으로 key에 들어간다.

**Hit (walk)** — boundary당:

```math
W=O\!\left(n_{\mathrm{noise}}C_{\mathrm{rng}}+C_{\mathrm{hash}}+C_{\mathrm{rng}}+C_{\mathrm{record}}\right).
```

noise RNG를 소비해 $\sigma_{\mathrm{seg}}$를 만들고, edge를 찾고, $\mathrm{Bernoulli}(p_0)$를 뽑고, record를 쓴다. **G 구간 opcode 실행(frame 갱신·pullback·pending 유지 = $A$)과 dense work($D$)를 모두 하지 않는다.** miss 복구 계층(§10)의 boundary hit이 dense 기대값 계산만 덜었다면, walk는 $A$와 $D$ 전부를 없앤다.

**Miss (fallback)** — 처음 보는 edge를 만나면 그 shot 전체를 **같은 per-shot seed로** §10 cache 실행(miss 시 §9 BuildEdge 경유)으로 재실행한다. 이 재실행이 record를 복원하는 동시에 automaton에 새 노드/edge를 채운다.

```math
T_{\mathrm{slow}}
\approx
B\left(A+H+R+O\right)
+\left(\text{이 shot이 새로 만든 edge들의 }C_{\mathrm{miss}}\right)
+C_{\mathrm{intern}}.
```

bit-exactness는 구성적으로 성립한다: 모든 shot은 walk가 완주했거나, authoritative 의미론과 동일한 slow 경로로 전부 재실행되었거나 둘 중 하나다.

**집계.** miss shot 비율을 $\mathrm{fb}(N)$이라 하면:

```math
\bar T_{\mathrm{LEAN}}(N)
=
\mathrm{fb}(N)\,T_{\mathrm{slow}}
+\left(1-\mathrm{fb}(N)\right)BW .
```

$\mathrm{fb}(N)$의 감쇠는 unique edge 수 $E_N$의 성장이 멈추는가에 달려 있다:

```math
E_N\sim N^{\beta},\qquad
\beta\rightarrow0:\ \text{포화(saturating)},\qquad
\beta\approx1:\ \text{비포화}.
```

**메모리:**

```math
M_{\mathrm{LEAN}}
=V_Nm_{\mathrm{node}}+E_Nm_{\mathrm{edge}}+M_{\mathrm{hash}}.
```

자동자 테이블이 전부다. miss 복구는 상태를 복원하는 게 아니라 같은 seed로 재계산하므로, 캐시
메모리는 노드당 수십 바이트 수준으로 유계이고(전 벤치 실측 peak $\le 63$ MB), 메모리 예산이 개입할
일 자체가 사실상 없다.

## 11.3 승부 조건과 regime

LEAN이 AUTH를 이기는 조건:

```math
\boxed{
\mathrm{fb}(N)\left(T_{\mathrm{slow}}-BW\right)+BW
<
B(A+D+R+O).
}
```

$BW\ll B(A+D+R+O)$이므로 실질 조건은 $\mathrm{fb}(N)$이 충분히 작아지는가, 즉 **자동자 포화**다.

이로부터 2×2 regime 지도가 나온다.

| | boundary 통계 포화 ($\beta\to0$) | 비포화 ($\beta$ 큼) |
|---|---|---|
| **$2^k$ 절대 dense work 큼** | LEAN 또는 AUTH 어느 쪽도 큰 승리 | AUTH 승리 ($r\ll k$ localization) |
| **$2^k$ 작음** | LEAN 승리 (walk가 상수 비용) | 손해 regime — 어느 내부 경로도 per-shot 전체 재계산의 하한을 넘지 못함 |

오른쪽 아래 코너가 정직한 한계다: 상태가 작아 회피할 dense work도 없고, boundary가 재귀하지 않아 재사용할 것도 없다.

# 12. 전체 실행 알고리즘

## 12.1 실행 계층

세 계층이 있고, 위 계층이 아래 계층을 fallback으로 쓴다.

```text
LEAN walk   : sigma 해시 -> edge lookup -> Bernoulli(p0) -> record      (G 구간 실행 없음)
    | miss: 같은 per-shot seed로 shot 전체 재실행 + automaton 채움
    v
CACHE 실행  : G 구간 해석 실행 + boundary key로 edge 재사용(§10), miss면 BuildEdge(§9)
    v
AUTH        : G 구간 해석 실행 + 매 boundary BuildEdge(§9)              (강등 시의 최종 정책)
```

walk의 shot 루프:

```cpp
for (shot i) {
    node v = v_init; int b_prev = NONE;
    for (int j = 0; j < B; j++) {
        sig = fold_segment_signs(rng);          // noise draw만 소비해 부호/분기 비트를 접는다
        Edge* e = lookup(v, b_prev, sig);
        if (!e) { replay_shot_slow(seed_i);     // miss: 같은 seed로 CACHE 계층 재실행(+학습)
                  goto next_shot; }
        int b = bernoulli(rng, e->p0);
        record[slot_j] = b ^ e->delta;
        v = b ? e->v1 : e->v0;  b_prev = b;
    }
}
```

## 12.2 배치 오케스트레이터

하나의 진입점이 $N$ shots을 구간(segment)으로 나눠 실행한다. 모든 구간은 유효한 샘플을 내고, per-shot seed 규약이 동일하므로 **어떤 경로 조합이든 최종 record 스트림은 같다.**

```text
run_batch(circuit, N):
  1 PROBE : 처음 P shots을 §13의 adaptive 실행기로 돈다 — 내부적으로 T_auth 캘리브레이션
            (~512 shots, authoritative, 정상 출력) 후 LEAN + 강등 감시.
            (여기서 T_auth/T_walk/T_slow 앵커와 fb, node_rate가 실측된다)
  2 ROUTE : probe가 AUTH로 강등했으면            -> 남은 shots 전부 AUTH.
            fb > fb_max (아직 비포화)이면        -> 강등 감시를 켠 LEAN으로 계속 진행하며
                                                    chunk마다 fb를 재평가: 내려오면 3으로,
                                                    도중 강등되면 AUTH로.
  3 GATE  : 회로가 고정이므로 walk 본문(12.1)의 opcode dispatch를 회로별 straight-line
            native 함수로 펼쳐 둘 수 있다(준비는 회로 수명당 1회, 산출물은 내용 해시로
            캐시). 남은 작업이 준비 비용을 넘는지 판단:
                N_rem * T_walk * S_min > T_prep    (캐시 hit이면 T_prep ~ 0)
            아니면 해석 walk로 그대로 CRUISE.
  4 RACE  : 해석 walk와 펼친 walk를 실제 출력 chunk 하나씩으로 경주시켜 이긴 쪽을 고정.
            (펼친 쪽이 지면 후회 비용은 T_prep + chunk 하나로 유계)
  5 CRUISE: 남은 shots을 승자로 chunk 단위 실행. 펼친 walk의 miss도 같은 seed
            fallback을 그대로 쓰므로 bit-exactness 불변.
```

# 13. Adaptive selection: 어떤 기준으로 판단하는가

```math
\boxed{
\text{Adaptive는 runtime 실측으로 판단한다.}
}
```

$\mathrm{fb}(N)$과 $E_N$의 성장은 회로 구조만으로 신뢰성 있게 예측되지 않는다 — 포화 여부는 noise 통계와 boundary 재귀 구조에 의존하며, 같은 구조 계열 안에서도 갈린다. probe의 비용은 $O(P)$로 유계이고 probe shots도 전부 유효한 샘플이다.

## 13.1 측정: 비용 앵커와 곡선

배치 시작 시 처음 $\min(512,\ N/4,\ \sim\!1\text{s})$ shots을 authoritative 경로로 실행해 $\hat T_{\mathrm{auth}}$를 **캘리브레이션**한다 (이 shots도 정상 출력이다 — per-shot seed 규약 덕에 어떤 경로 조합이든 record 스트림은 동일). 이후 LEAN으로 돌며 다음이 쌓인다.

| 측정값 | 방법 |
|---|---|
| $T_{\mathrm{walk}}$ | hit shot 시간 (8-shot당 1회 샘플링 평균) |
| $T_{\mathrm{slow}}$ | miss shot(같은 seed 재실행) 시간 평균 |
| $\mathrm{fb}_w$ | window(4096 shots)별 miss shot 비율 |
| $D_w$ | 누적 distinct 노드 수 |
| $B_{\mathrm{tab}}$ | 자동자 테이블 바이트 (LEAN의 유일한 캐시 메모리) |
| magic\_ever | non-Clifford 축이 한 번이라도 materialize되었는가 |

## 13.2 판정: 항등식 + now-or-never 정리 + 검증되는 가정 1개

**항등식.** miss율 fb에서의 LEAN shot당 비용은 $c_L(\mathrm{fb})=\mathrm{fb}\,T_{\mathrm{slow}}+(1-\mathrm{fb})T_{\mathrm{walk}}$이고, 손익분기 miss율은

```math
\mathrm{fb}_{be}=\frac{\hat T_{\mathrm{auth}}-T_{\mathrm{walk}}}{T_{\mathrm{slow}}-T_{\mathrm{walk}}}.
```

$\mathrm{fb}_w \lt \mathrm{fb}_{be}$면 LEAN이 **지금** AUTH보다 빠르다 — 판정 끝, 유지. ($T_{\mathrm{slow}} \lt \hat T_{\mathrm{auth}}$인 회로는 $\mathrm{fb}_{be}>1$이라 어떤 miss율에서도 유지된다.)

**정리 (now-or-never).** 강등은 비가역이고 $\mathrm{fb}(n)$은 기대값에서 비증가(자동자는 단조 증가, shots는 i.i.d.)이므로, 전환 시점 $m$의 총비용 $C(m)$은 $\partial C/\partial m=c_L(m)-\hat T_{\mathrm{auth}}$가 감소함수인 단봉 함수다. 따라서 최적 전환은 내부에 없고 **"지금 아니면 끝까지"** 뿐이며, 판정은 남은 구간의 적분 비교로 정확히 환원된다:

```math
\text{LEAN 유지}\iff\int_{n}^{N}\!\big[\mathrm{fb}(s)(T_{\mathrm{slow}}-T_{\mathrm{walk}})+T_{\mathrm{walk}}\big]ds<(N-n)\,\hat T_{\mathrm{auth}}.
```

**가정 (H), 유일한 모델.** Heaps 법칙 $D(M)=cM^{\beta}$. Good–Turing 항등식 $m(M)=\mathbb E[\Delta D]$과 결합하면 $\mathrm{fb}(s)=\varphi\,s^{\beta-1}$이고, 적분 기준이 닫힌 식이 된다:

```math
\boxed{\ \varphi\,\frac{N^{\beta}-n^{\beta}}{\beta}\,(T_{\mathrm{slow}}-T_{\mathrm{walk}})\ \le\ (\hat T_{\mathrm{auth}}-T_{\mathrm{walk}})\,(N-n)\ }
```

$(\hat\varphi,\hat\beta)$는 측정된 $\mathrm{fb}_w$ 시계열의 log-log OLS로 추정하며, (H)는 상시 반증 시도된다: $R^2$ 문턱 + 독립 추정치 $\hat\beta_{fb}$와 $\hat\beta_D$의 일치 검사. fit이 유효하지 않은데 지고 있으면($\mathrm{fb}_w>\mathrm{fb}_{be}$) 보수적으로 강등한다. 두 개의 즉발 출구는 항등식 수준이다: **identity fast path** — 완주한 walk가 0이면서 실측 $T_{\mathrm{slow}}\ge\hat T_{\mathrm{auth}}$ (지금 엄밀히 지배당함 + 학습 증거 없음, heavy-core all-miss); **early-localization** — magic\_ever 거짓 $\wedge$ miss $\approx1$ (학습할 대상 자체가 없음).

## 13.2b 메모리: regime을 결정하지 않는다

LEAN의 캐시는 자동자 테이블뿐이므로(§11.2; state는 캐시하지 않음, 전 벤치 실측 peak $\le63$ MB)
**메모리 예산은 알고리즘의 요소가 아니다** — bounded-resource 실행을 위한 *선택적* 가드일 뿐이고,
기본 실행에서는 어떤 벤치도 가드에 닿지 않는다. 가드를 걸었고 테이블이 그것을 넘는 경우의 행동은
강등이 아니라 **freeze 검사**다:

- 현재 fb를 동결한 비용 $\mathrm{fb}_w T_{\mathrm{slow}}+(1-\mathrm{fb}_w)T_{\mathrm{walk}} \lt \hat T_{\mathrm{auth}}$이
  성립하면 **학습(insert)만 중단**하고 walk를 계속한다 (메모리 성장 0, record 불변). 성립하지
  않을 때만 강등.

즉 가드가 있어도 그것은 캐시 관리 정책에만 영향을 주고, LEAN/AUTH regime과 correctness에는 관여하지
않는다. 재승격은 없다(sticky). 단 12.2의 배치 루프는 chunk 경계에서 fb를 재평가해 GATE를 다시 열 수 있다.

## 13.3 walk 전개 게이트 (12.2의 3–4단계)

준비 비용이 있는 유일한 선택이므로 별도의 게이트를 둔다.

```math
\boxed{
N_{\mathrm{rem}}\cdot T_{\mathrm{walk}}\cdot S_{\min}\ >\ T_{\mathrm{prep}}
}
```

- $T_{\mathrm{walk}}$: probe에서 실측된 해석 walk의 per-shot 비용
- $S_{\min}$: 보수적 기대 절감률(펼친 walk가 최소 이만큼은 아낀다고 가정하는 하한; 과대평가로 marginal한 배치가 순손해 보는 일을 막는다)
- $T_{\mathrm{prep}}$: 회로별 native 함수의 생성+컴파일+로드 비용. 산출물은 내용 해시로 캐시되므로 같은 회로의 재실행에서는 $T_{\mathrm{prep}}\approx0$

게이트를 통과해도 바로 믿지 않는다: RACE(12.2의 4)가 실측으로 승자를 정하고, 지면 해석 walk로 남는다. 펼친 walk가 지는 경우도 실제로 있다 — 예컨대 긴 pure-frame 구간은 해석기의 superinstruction(경계 검사를 밖으로 빼낸 tight loop)이 거대한 unrolled 본문(I-cache 압박)보다 빠르다. race가 이를 자동으로 걸러낸다.

## 13.4 판단 기준 요약표

| 비교 | 지위 | 행동 |
|---|---|---|
| $\mathrm{fb}_w$ vs $\mathrm{fb}_{be}$ | 항등식 | 아래면 LEAN 유지 (지금 이기는 중) |
| 적분 기준 $\varphi\frac{N^\beta-n^\beta}{\beta}(T_{\mathrm{slow}}{-}T_{\mathrm{walk}})$ vs $(\hat T_{\mathrm{auth}}{-}T_{\mathrm{walk}})(N{-}n)$ | 정리 + 가정(H) | 지는 중일 때: 회수 가능하면 유지, 아니면 AUTH |
| 완주 0 $\wedge$ $T_{\mathrm{slow}}\ge\hat T_{\mathrm{auth}}$ | 항등식 | 즉시 AUTH (heavy-core all-miss) |
| magic\_ever 거짓 $\wedge$ miss $\approx1$ | 구조적 | 즉시 AUTH (학습 대상 없음) |
| 테이블 바이트 $>$ (선택적) 가드 | 안전장치 | freeze 검사 (동결 LEAN이 이기면 insert만 중단) → 아니면 강등 |
| $N_{\mathrm{rem}}T_{\mathrm{walk}}S_{\min}$ vs $T_{\mathrm{prep}}$ | 게이트 | 전개 준비 + race |
| race 실측 | 실측 | 이긴 쪽으로 cruise |

남는 상수(회귀 표본 수, $R^2$ 문턱, 캘리브레이션 예산, 메모리 캡)는 판정의 **방향**을 정하지 않는 잡음 처리·안전장치이며, 어느 회로가 유지/강등되는지는 전부 측정값과 위 식에서 나온다.

# 14. 경로별로 어떤 비용이 사라지는가

| 구간 | AUTH 비용 | CACHE 계층 효과 | LEAN walk 효과 | 펼친 walk 추가 효과 |
|---|---:|---|---|---|
| $G_j$ frame/record/noise bookkeeping | $A_j$ | 없음 | **제거** (부호 fold만 남음) | dispatch/decode 상수 제거 |
| $G_j$ pending append/pullback | $O(t_jN/w)$ | 없음 | **제거** | — |
| $M_j$ dense core build | $\Theta(q2^r)$ | $E_N$회로 축소 | $E_N$회로 축소 (fallback 경유) | — |
| $M_j$ Born/project | $\Theta(2^r)$ | $E_N$회로 축소 | $E_N$회로 축소 | — |
| $M_j$ hit lookup/RNG | — | $H+R$ 잔존 | $C_{\mathrm{hash}}+R$ 잔존 | key serializer 상수 감소 |
| 준비 비용 | 없음 | 없음 | 자동자 학습($T_{\mathrm{slow}}$ miss분) | $T_{\mathrm{prep}}$ 1회 (캐시됨) |

어느 경로도 없애지 못하고 남는 것:

- unique edge $E_N$개의 BuildEdge (자동자/캐시 학습의 원가)
- RNG (noise/measurement는 runtime 확률)
- record 쓰기
- miss shot의 slow 재실행 (bit-exactness의 대가)

따라서 두 질문이 분리된다:

```math
\boxed{
\text{boundary 통계가 포화하는가? (LEAN의 성립 조건)}
}
```

```math
\boxed{
\text{남은 배치가 준비 비용을 넘는가? (walk 전개의 성립 조건)}
}
```

# 15. 논문에서 강조해야 할 핵심 불변식

## Invariant 1. Full state never materialized

```math
|\psi\rangle\in\mathbb C^{2^N}
```

를 직접 만들지 않는다.

## Invariant 2. Dense work is bounded by measurement-relevant rank

각 boundary $j$에서 dense work는:

```math
\Theta((q_{\mathrm{rel},j}+c)2^{r_j})
```

이고 $r_j$는:

```math
r_j=\mathrm{CoreRank}(\mathcal A_j,\mathcal S_j).
```

즉 전체 qubit 수 $N$이 아니라 measurement-relevant independent axis rank가 exponential factor다.

## Invariant 3. Pending rotations are factorized

```math
U(\mathcal L)=\prod_a R_{q_a}(\theta_a)
```

로 유지하고, 절대 다음처럼 전개하지 않는다.

```math
\prod_a(c_aI+s_aP_a)
\ne
\sum_u c_uP_u\quad\text{as an explicit object.}
```

## Invariant 4. Measurement contracts the core

measurement가 active axis를 고정하면:

```math
r\leftarrow r-1.
```

이 contraction이 장기 회로에서 dense core를 bounded하게 유지하는 핵심이다.

## Invariant 5. Cache is exact transition reuse

cache edge는 approximate reuse가 아니라 같은 key에 대한 exact boundary transition reuse다.

```math
E(k)=\left(p_0,next_0,next_1,aux\right).
```

같은 key면 same Born probability와 same projected successor nodes를 재사용한다.

---

# 16. 최종 알고리즘 한 줄

```math
\boxed{
\begin{aligned}
&\text{G 구간: }\text{Clifford/Pauli/record/pending을 symbolic-bit 상태로 갱신한다.}\\
&\text{M 구간: }\text{현재 measurement와 연결된 독립 Pauli 축만 symplectic reduction으로 뽑아 }2^r\text{ dense core를 만든다.}\\
&\text{Dense work: }\text{relevant pending rotations를 factorized sweep으로 적용하고 Born/project/drop을 수행한다.}\\
&\text{LEAN: }\text{boundary automaton이 포화하면 G 구간 실행 자체를 walk로 건너뛰고, miss는 같은 seed 재실행으로 bit-exact하게 복구한다.}\\
&\text{Adaptive: }\text{probe 실측(fb, node rate, 메모리)으로 LEAN/AUTH를 고르고, 남은 배치가 준비 비용을 넘는 고정 회로에서는 walk 본문을 회로별 native code로 펼친다.}
\end{aligned}
}
```

---

# 17. 짧은 구현 체크리스트

- [ ] Pauli는 $(x|z)$ bitmask로 저장한다.
- [ ] commutation은 symplectic parity $[p,q]$로 계산한다.
- [ ] Clifford frame은 tableau/inverse tableau로 저장한다.
- [ ] non-Clifford gate는 pullback 후 pending list에 append한다.
- [ ] record $m$과 pending $\mathcal L$는 분리한다.
- [ ] measurement boundary에서 physical Pauli를 pullback하고 sign $\delta$를 얻는다.
- [ ] closure $\mathcal A_j$를 만들고 stabilizer/fixed-axis quotient를 수행한다.
- [ ] local basis $B_j$를 만들고 $r_j$를 얻는다.
- [ ] current node dense state를 workspace에 embed한다.
- [ ] relevant pending rotations를 localize 후 sequential sweep한다.
- [ ] measurement Pauli를 localize한다.
- [ ] Born probability, sampling, projection, normalization을 수행한다.
- [ ] measured axis가 고정되면 drop하여 rank를 줄인다.
- [ ] edge를 cache에 저장한다.
- [ ] cache hit에서는 dense build 없이 edge transition만 사용한다.
- [ ] segment edge key에 noise 부호·분기 비트($\sigma_{\mathrm{seg}}$)를 접어 walk를 결정적으로 만든다.
- [ ] walk miss는 같은 per-shot seed의 slow 재실행으로 복구한다 (bit-exact).
- [ ] Adaptive: $T_{\mathrm{auth}}$ 캘리브레이션 후 LEAN으로 돌며, 실측 앵커의 손익분기 $\mathrm{fb}_{be}$와 적분 기준(§13.2)으로 AUTH 강등을 판단한다.
- [ ] 큰 배치에서는 walk 전개(회로별 native 함수)를 self-relative 게이트 + race로 결정한다.

