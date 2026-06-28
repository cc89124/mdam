# 측정 구동 활성 상태 구성 기반 정확 근클리퍼드 시뮬레이션

## Measurement-Driven Active-State Materialization (MDAM)

**문서 유형:** QEC 연구 결과 보고서  
**작성 시점:** 2026년 6월

# 0. 이 문서의 핵심

이 연구의 목적은 **Clifft의 exact near-Clifford 시뮬레이션 구조를 유지하면서, dense active state의 크기를 더 줄이는 것**이다.

Clifft는 비-Clifford 회전이 등장할 때 그 효과를 virtual axis에 즉시 localize하고, 필요한 axis를 active state에 바로 추가한다. 따라서 시간 $t$의 active state 크기는

$$2^{k_{t}},\quad\quad k_{t} = \left| A_{t}^{C} \right|$$

으로 결정된다. 여기서 $A_{t}^{C}$는 Clifft가 그 시점까지 materialize한 active virtual axis 집합이다. 비-Clifford 회전이 여러 번 등장하고 해당 axis들이 아직 측정으로 제거되지 않았다면, active state는 과거의 비-Clifford 효과를 계속 누적한 상태로 커진다.

본 연구의 핵심은 이 eager materialization을 다음과 같이 바꾸는 것이다.

**핵심:** 비-Clifford 회전 도착 시 axis를 즉시 active state에 넣지 않고, 측정 시 실제로 필요한 axis만 dense state로 구현한다.

즉, 회전은 ordered pending ledger에 정확히 보존하고, 측정 $j$가 도착했을 때 그 측정 결과에 영향을 줄 수 있는 rotation core와 axis만 찾아 dense workspace를 구성한다. 목표 dense rank는 Clifft의 누적 active rank $k_{t}$가 아니라, 각 측정에서 필요한 axis 수

$$b_{j} = \left| B_{j} \right|$$

로 결정된다. 따라서 지배적인 메모리 비용을

$$O\left( 2^{k_{\max}} \right)$$

에서

$$O\left( 2^{b_{\max}} \right),\quad\quad b_{\max} = \max_{j}b_{j}$$

로 줄이는 것이 논문의 중심 주장이다.

이 방식은 회전을 버리거나 근사하는 방식이 아니다. 현재 측정과 commute하여 측정 뒤로 정확히 이동할 수 있는 회전만 symbolic pending 상태로 남긴다. 따라서 **핵심은 measurement-driven materialization이며, approximation이나 truncation이 아니다.**

# 1. Clifft의 active state와 메모리 병목

## 1.1 Clifft의 frame-factored state

Clifft는 물리 상태를 다음과 같이 분해한다.

$$\psi_{t} = \gamma_{t}U_{C,t}{\widetilde{P}}_{t}\left( \phi_{t,A_{t}^{C}} \otimes e_{0}^{\otimes \left| D_{t}^{C} \right|} \right).$$

각 항의 의미는 다음과 같다. 여기서 dormant basis vector는

$$e_{0} = (1,0)^{\mathsf{T}}$$

로 둔다.

- $U_{C,t}$: 물리 Clifford 연산을 흡수한 deterministic Clifford frame

- ${\widetilde{P}}_{t}$: shot별 Pauli noise와 correction을 저장하는 Pauli frame

- $A_{t}^{C}$: 현재 active한 virtual axis 집합

- $D_{t}^{C}$: dormant virtual axis 집합

- $\phi_{t}$: $k_{t} = \left| A_{t}^{C} \right|$개 virtual axis의 joint dense state vector

- $\gamma_{t}$: 전역 phase와 normalization scalar

Clifft에서 dense state의 복소 amplitude 수는

$$N_{t}^{C} = 2^{k_{t}}$$

이고, peak memory는

$$M_{C} = \Theta\left( 2^{k_{\max}} \right),\quad\quad k_{\max} = \max_{t}k_{t}$$

로 결정된다.

## 1.2 Clifft의 eager materialization

물리 Pauli rotation

$$R_{Q}(\theta) = e^{-i\theta Q}$$

이 시간 $t$에 등장하면, Clifft는 먼저 Clifford frame을 통해 virtual Pauli로 변환한다.

$${\widetilde{Q}}_{t} = U_{C,t}^{\dagger}QU_{C,t}.$$

그다음 virtual Clifford localization을 통해 ${\widetilde{Q}}_{t}$를 하나의 virtual axis $a$에 작용하는 Pauli로 만든다.

$$V_{t}{\widetilde{Q}}_{t}V_{t}^{\dagger} \in \{ \pm X_{a}, \pm Y_{a}, \pm Z_{a}\}.$$

해당 axis가 dormant이고 회전이 superposition을 만들면 Clifft는 즉시

$$A_{t}^{C} \leftarrow A_{t}^{C} \cup \{ a\}$$

로 확장하고,

$$\phi_{t} \leftarrow \phi_{t} \otimes e_{0}$$

를 수행한다. 이후 localized rotation을 dense state에 적용한다.

따라서 Clifft에서 axis의 생명주기는 대략 다음과 같다.

Clifft에서 axis의 생명주기는 다음 순서로 진행된다.

비-Clifford 회전 도착 → 즉시 active로 승격 → 후속 측정까지 resident로 유지 → 측정 후 제거

## 1.3 문제점

비-Clifford 회전 $R_{1},R_{2},\ldots$가 서로 다른 axis를 차례로 활성화하지만, 각 axis가 실제로 필요한 측정 시점은 서로 다를 수 있다. Clifft는 이러한 미래의 측정 필요성과 무관하게 회전 도착 시점에 axis를 즉시 materialize한다.

따라서 시점 $t$에서 Clifft active set은 개념적으로

시점 $t$의 Clifft active set $A_{t}^{C}$는 과거에 materialize되었고 아직 제거되지 않은 모든 axis의 집합이다.

가 된다.

그러나 측정 $M_{j}$를 계산할 때 실제로 필요한 것은 그중 일부일 수 있다. 즉 일반적으로

$$B_{j} \subseteq A_{t_{j}}^{C}$$

일 수 있으며, $B_{j}$가 현재 측정에 필요한 axis 집합이다.

본 연구는 이 차이를 이용한다.

# 2. MDAM의 중심 아이디어: measurement-driven materialization

## 2.1 비-Clifford 효과와 dense state를 분리

MDAM은 시간 $t$의 상태를 다음과 같이 표현한다.

$$\psi_{t} = \gamma_{t}U_{C,t}{\widetilde{P}}_{t}U_{pend,t}\left( \phi_{t,A_{t}} \otimes e_{0}^{\otimes \left| D_{t} \right|} \right).$$

Clifft 표현과 비교했을 때 새롭게 추가된 핵심 항은

$$U_{pend,t}$$

이다. 이는 아직 dense state에 materialize하지 않은 비-Clifford Pauli rotation의 ordered product이다.

$$U_{pend,t} = R_{L}R_{L-1}\cdots R_{2}R_{1},$$

$$R_{i} = R_{P_{i}}\left( \theta_{i} \right) = e^{-i\theta_{i}P_{i}}.$$

$R_{1}$이 시간상 가장 먼저 등장한 rotation이며, $R_{L}$이 가장 최근 rotation이다. 각 entry는 적어도 다음 정보를 가진다.

$$\left( P_{i},\theta_{i},\mathrm{id}_{i},t_{i} \right).$$

여기서 $P_{i}$는 현재 virtual coordinate에서의 Pauli generator이고, $\mathrm{id}_{i}$는 동일 rotation의 중복 적용을 방지하는 고유 ID이다.

## 2.2 회전 도착 시 dense state를 확장하지 않음

새로운 물리 rotation $R_{Q}(\theta)$가 등장하면 virtual generator는

$$P = U_{C}^{\dagger}QU_{C}$$

이다. 현재 Pauli frame $\widetilde{P}$와 anti-commute하면 effective angle의 부호가 바뀐다.

$$\theta\prime = (-1)^{\langle P,\widetilde{P}\rangle_{s}}\theta,$$

여기서 $\langle \cdot , \cdot \rangle_{s}$는 binary symplectic product이다.

MDAM은 이 회전을 dense state에 즉시 적용하지 않고

$$U_{pend} \leftarrow R_{P}(\theta\prime)U_{pend}$$

로 저장한다.

따라서 회전 도착 시점에는

따라서 회전 도착 시점에는 active set $A_{t}$와 dense state $\phi_{t}$가 변하지 않는다. 이 차이가 Clifft와 MDAM의 가장 본질적인 차이이다.

## 2.3 무엇을 줄이는가

Clifft는 비-Clifford 회전의 도착 시점에 axis를 active state에 넣는다. MDAM은 그 rotation이 어떤 측정에 실제로 필요해질 때까지 axis의 birth를 미룬다.

**Clifft는 rotation-driven axis birth를 사용하고, MDAM은 measurement-driven axis birth를 사용한다.**

따라서 dense state는 과거의 모든 비-Clifford axis가 아니라, 현재 측정에 필요한 axis만 포함하도록 설계한다.

# 3. 측정 시 어떤 rotation이 필요한가

## 3.1 측정 observable

시간 $t_{j}$에 물리 Pauli observable $M_{j}^{\mathrm{phys}}$를 측정한다고 하자. virtual observable은

$$M_{j} = U_{C,t_{j}}^{\dagger}M_{j}^{\mathrm{phys}}U_{C,t_{j}}$$

이다.

현재 Pauli frame이 측정 observable과 anti-commute하면 물리 outcome과 dense-state branch가 뒤집힌다.

$$p_{j} = \langle M_{j},{\widetilde{P}}_{t_{j}}\rangle_{s},$$

$$b_{j}^{\mathrm{virt}} = m_{j}^{\mathrm{phys}} \oplus p_{j}.$$

이 frame shift는 axis 선택을 바꾸지 않고, 선택되는 branch 또는 rotation angle sign만 바꾼다.

## 3.2 직접 필요한 rotation

pending rotation $R_{i} = R_{P_{i}}\left( \theta_{i} \right)$가 측정 observable $M_{j}$와 commute하면

$$\left\lbrack P_{i},M_{j} \right\rbrack = 0$$

이고, 해당 rotation은 projector를 통과하여 측정 뒤로 이동할 수 있다.

반대로

$$\{ P_{i},M_{j}\} = 0$$

이면 rotation은 현재 측정의 Born probability와 post-measurement state에 직접 영향을 줄 수 있으므로 측정 전에 materialize해야 한다.

따라서 직접 관련 집합은

$$K_{j}^{(0)} = \left\{ i\; \middle| \;\langle P_{i},M_{j}\rangle_{s} = 1 \right\}$$

이다.

## 3.3 시간 순서 때문에 필요한 간접 rotation

직접 anti-commute하는 rotation만 선택하면 충분하지 않을 수 있다. 예를 들어 시간 순서가

$$R_{i}\; \rightarrow \; R_{\ell}\; \rightarrow \; M_{j}$$

이고 $R_{\ell}$이 측정과 anti-commute하여 core에 들어가는데, earlier rotation $R_{i}$가 $R_{\ell}$과 anti-commute한다고 하자.

$$\langle P_{i},P_{\ell}\rangle_{s} = 1.$$

이 경우 $R_{i}$를 $R_{\ell}$의 뒤로 commute시켜 측정 이후로 이동할 수 없다. 따라서 $R_{i}$도 함께 materialize해야 한다.

이를 반영하여 다음 fixed-point closure를 정의한다.

$$K_{j}^{(r + 1)} = K_{j}^{(r)} \cup \left\{ i\; \middle| \;\exists\ell \in K_{j}^{(r)},\; i < \ell,\;\langle P_{i},P_{\ell}\rangle_{s} = 1 \right\}.$$

유한한 ledger이므로 어느 단계에서 수렴한다. 최종 집합을

$$K_{j} = closure\left( K_{j}^{(0)} \right)$$

라 한다.

$K_{j}$는 현재 측정에 필요한 **ordered rotation core**이다.

## 3.4 이 core가 의미하는 것

$K_{j}$는 단순히 측정 observable과 직접 겹치는 rotation 집합이 아니다. 다음 두 조건을 모두 만족하는 최소 집합이다.

1.  측정과 직접 anti-commute하는 모든 rotation을 포함한다.

2.  선택된 rotation을 시간 순서를 보존한 채 측정 앞으로 유지하기 위해 필요한 모든 anti-commuting ancestor를 포함한다.

즉 $K_{j}$는 현재 측정까지 실제로 전달되는 비-Clifford causal cone이다.

# 4. Core 외 rotation을 왜 계산하지 않아도 되는가

## 4.1 Pending unitary의 분해

core index 집합을 $K_{j}$, 나머지 deferred index 집합을 $D_{j}$라 하자.

$$D_{j} = \{ 1,\ldots,L\}\backslash K_{j}.$$

각 집합 내부의 원래 시간 순서를 유지한 product를

$$U_{K_{j}} = \prod_{i \in K_{j}}^{\leftarrow}R_{i},\quad\quad U_{D_{j}} = \prod_{i \in D_{j}}^{\leftarrow}R_{i}$$

라고 정의한다.

closure 정의에 의해 deferred rotation은 자신보다 뒤에 있는 core rotation과 commute한다. 그렇지 않다면 deferred rotation도 ancestor closure에 포함되었어야 한다.

따라서 commuting swap만 사용하여

$$U_{pend} = U_{D_{j}}U_{K_{j}}$$

로 정확히 재배열할 수 있다.

## 4.2 Deferred block과 측정의 commute

모든 $i \in D_{j}$는 측정과 직접 anti-commute하지 않으므로

$$\left\lbrack P_{i},M_{j} \right\rbrack = 0.$$

따라서

$$\left\lbrack U_{D_{j}},\Pi_{b}\left( M_{j} \right) \right\rbrack = 0,$$

여기서

$$\Pi_{b}\left( M_{j} \right) = \frac{I + (-1)^{b}M_{j}}{2}$$

이다.

그러므로

$$\Pi_{b}\left( M_{j} \right)U_{pend} = \Pi_{b}\left( M_{j} \right)U_{D_{j}}U_{K_{j}} = U_{D_{j}}\Pi_{b}\left( M_{j} \right)U_{K_{j}}.$$

이 식이 MDAM의 정확성을 지탱하는 핵심 식이다.

## 4.3 Born probability

측정 직전 base state를 $\Phi$라 하자.

$$\Phi = \phi_{A} \otimes e_{0}^{\otimes |D|}.$$

전체 pending rotation을 모두 적용한 경우의 branch norm은

$$\left. \parallel\Pi_{b}\left( M_{j} \right)U_{pend}\Phi \right.\parallel^{2}$$

이다. 위의 분해를 사용하면

$$\left. \parallel U_{D_{j}}\Pi_{b}\left( M_{j} \right)U_{K_{j}}\Phi \right.\parallel^{2}.$$

$U_{D_{j}}$는 unitary이므로 norm을 바꾸지 않는다. 따라서

$$\boxed{\left. \parallel\Pi_{b}\left( M_{j} \right)U_{pend}\Phi \right.\parallel^{2} = \left. \parallel\Pi_{b}\left( M_{j} \right)U_{K_{j}}\Phi \right.\parallel^{2}.}$$

즉 현재 측정 확률을 계산하기 위해 deferred rotation을 dense state에 적용할 필요가 없다.

이것이 **측정에 필요한 rotation만 계산해도 exact한 이유**이다.

# 5. 측정 시 어떤 axis만 dense state로 구현하는가

## 5.1 Rotation core와 axis core의 구분

$K_{j}$는 rotation ID의 집합이다. 실제 dense state의 크기를 결정하는 것은 이 rotation들이 사용하는 virtual axis의 집합이다.

각 core rotation이 Pauli localization 후 virtual axis $a_{i}$에 작용하고, 측정 observable이 axis $a_{M_{j}}$에 작용한다고 하자.

현재 측정이 요구하는 신규 axis 집합을

$$A_{j}^{\mathrm{new}} = \left( \{ a_{M_{j}}\} \cup \{ a_{i} \mid i \in K_{j}\} \right)\backslash A_{j}^{\mathrm{res}}$$

로 정의한다.

여기서 $A_{j}^{\mathrm{res}}$는 측정 직전에 이미 dense state에 남아 있는 resident axis 집합이다.

실제로 measurement workspace에 들어가는 axis는

$$B_{j}^{\mathrm{impl}} = A_{j}^{\mathrm{res}} \cup A_{j}^{\mathrm{new}}$$

이며, transient dense rank는

$$b_{j}^{\mathrm{impl}} = \left| B_{j}^{\mathrm{impl}} \right|.$$

## 5.2 논문이 목표로 하는 이상적인 measurement-only set

연구의 최종 목표는 단순히 신규 axis의 birth만 늦추는 것이 아니다. 각 측정 전에 현재 측정과 미래 상태를 보존하는 데 필요하지 않은 resident axis까지 exact frame reduction으로 제거하여, dense state를 현재 측정에 필요한 최소 axis 집합으로 다시 구성하는 것이다.

이를 이상적인 measurement-required axis set

$$B_{j}^{\star}$$

라 하자. $B_{j}^{\star}$는 다음 정보를 정확히 표현하는 최소 virtual axis 집합이다.

1.  현재 측정 core $U_{K_{j}}$

2.  측정 projector $\Pi_{b}\left( M_{j} \right)$

3.  측정 이후에도 dense amplitude로 남아야 하는 비분리 상태

목표 rank는

$$b_{j}^{\star} = \left| B_{j}^{\star} \right|.$$

연구의 핵심 메모리 지표는

$$\boxed{b_{\max}^{\star} = \max_{j}b_{j}^{\star}}$$

이다.

## 5.3 현재 구현과 최종 알고리즘의 차이

현재 구현은 다음을 이미 수행한다.

- rotation의 즉시 materialization을 피한다.

- 측정별 core를 찾아 필요한 신규 axis만 promote한다.

- 측정된 axis와 exact하게 product임이 확인된 axis를 drop한다.

따라서 **delayed birth**는 구현되어 있다.

그러나 현재 resident axis가 다음 측정에 필요하지 않더라도, frame 및 pending ledger에서 완전히 decouple되었다는 증명이 없으면 dense state에 남는다. 따라서 현재 구현의 rank는 일반적으로

$$b_{j}^{\mathrm{impl}} = \left| A_{j}^{\mathrm{res}} \cup A_{j}^{\mathrm{new}} \right|$$

이며, 항상 최소값 $b_{j}^{\star}$와 같지는 않다.

즉 현재 구현은

**현재 구현:** delayed materialization + safe measured-axis drop

이고, 논문의 완성된 목표는

**완성된 목표:** measurement-specific exact re-carving

이다.

이 구분은 매우 중요하다. “측정에 필요한 axis만 state로 구현한다”는 논문의 핵심은 $B_{j}^{\star}$를 목표로 하며, 현재 backend는 그 방향의 첫 번째 정확한 구현 단계이다.

# 6. Measurement workspace의 구성과 실행

## 6.1 Axis promotion

새로 필요한 axis $q \in A_{j}^{\mathrm{new}}$는 dormant 상태 $e_{0}$에서 시작한다.

$$\phi_{A_{j}^{\mathrm{res}}} \mapsto \phi_{A_{j}^{\mathrm{res}}} \otimes e_{0}^{\otimes \left| A_{j}^{\mathrm{new}} \right|}.$$

이 연산은 새로운 물리 정보를 만드는 것이 아니라, pending ledger에 있던 비-Clifford 효과를 계산하기 위한 coordinate dimension을 명시적으로 여는 과정이다.

## 6.2 Core 적용

workspace에서 core rotations를 원래 순서대로 적용한다.

$$\eta_{j} = U_{K_{j}}\Phi_{j}.$$

그다음 shifted measurement projector를 적용한다.

$${\overline{\chi}}_{j,b} = \Pi_{b}\left( M_{j} \right)\eta_{j}.$$

Born probability는

$$p_{j,b} = \frac{\parallel {\overline{\chi}}_{j,b} \parallel_{2}^{2}}{\parallel \Phi_{j} \parallel_{2}^{2}}.$$

outcome을 sample한 뒤 정규화한다.

$$\chi_{j,b} = \frac{{\overline{\chi}}_{j,b}}{\sqrt{p_{j,b}}}.$$

## 6.3 Ledger 갱신

현재 core에 포함된 rotation은 dense state에 정확히 한 번 적용되었으므로 ledger에서 제거한다.

$$U_{pend}^{+} = U_{D_{j}}.$$

Deferred rotation은 삭제되지 않고 원래 상대 순서를 유지한 채 다음 측정까지 pending으로 남는다.

## 6.4 측정 후 axis 제거

측정된 axis는 projector에 의해 eigenstate로 분리된다. 적절한 virtual Clifford coordinate update를 통해 해당 eigenstate를 dormant basis vector $e_{0}$로 보내고 active set에서 제거한다.

추가로 다른 axis가 product state가 되었음이 정확히 증명되면 함께 제거할 수 있다.

완전한 measurement-only 구현에서는 다음 측정에 불필요한 resident axis도 frame과 pending operator에서 decouple하여 제거해야 한다. 이 과정이 exact frame reduction이다.

# 7. 전체 정확성

## 7.1 유지해야 할 상태 불변식

모든 시점에서 내부 표현은 다음 물리 상태와 정확히 같아야 한다.

$$\psi_{t} = \gamma_{t}U_{C,t}{\widetilde{P}}_{t}U_{pend,t}\left( \phi_{t,A_{t}} \otimes e_{0}^{\otimes \left| D_{t} \right|} \right).$$

다음 조건을 유지한다.

1.  pending rotation의 상대 시간 순서를 보존한다.

2.  각 rotation ID는 정확히 한 번만 dense state에 적용된다.

3.  commute가 증명된 경우에만 rotation의 순서를 교환한다.

4.  Clifford coordinate를 바꿀 때 $U_{C}$, $\widetilde{P}$, pending generators, active state를 일관되게 변환한다.

5.  dormant axis는 virtual base state에서 $e_{0}$를 유지한다.

6.  Pauli phase와 rotation angle sign을 보존한다.

## 7.2 Exactness 정리

**정리.** 각 측정 $M_{j}$에서 $K_{j}$를 위의 anti-commutation ancestor closure로 선택하고, $B_{j}$ 위에서 $U_{K_{j}}$와 measurement projector를 정확히 적용하며, core 외 rotation을 $U_{D_{j}}$에 순서대로 보존하면, 제안 simulator가 생성하는 measurement outcome의 joint distribution은 원본 회로와 동일하다.

**증명 개요.** Clifford와 Pauli 연산은 각각 Clifford frame과 Pauli frame에 정확히 흡수된다. Non-Clifford rotation은 dense state에 즉시 적용되지는 않지만 $U_{pend}$ 안에 정확히 존재한다. 측정 시 closure 정의로 인해

$$U_{pend} = U_{D_{j}}U_{K_{j}}$$

이고

$$\left\lbrack U_{D_{j}},\Pi_{b}\left( M_{j} \right) \right\rbrack = 0$$

이다. 따라서

$$\Pi_{b}\left( M_{j} \right)U_{pend} = U_{D_{j}}\Pi_{b}\left( M_{j} \right)U_{K_{j}}.$$

$U_{D_{j}}$는 unitary이므로 Born probability를 바꾸지 않으며, post-measurement state에는 정확히 pending operator로 남는다. 따라서 각 conditional probability와 post-state가 원본 회로와 동일하다. 이를 측정 순서에 따라 귀납하면 전체 joint distribution이 동일하다.

# 8. Clifft보다 active state가 작아지는 이유

## 8.1 두 방식의 active set 정의가 다름

Clifft의 active set은

Clifft의 active set $A_{t}^{C}$는 과거 회전에 의해 즉시 materialize되었고 아직 측정으로 제거되지 않은 axis의 집합이다.

이다.

MDAM의 measurement workspace는

MDAM의 measurement workspace $B_{j}$는 현재 측정의 core와 post-state에 필요한 axis의 집합이다.

이다.

따라서 MDAM은 과거에 등장했지만 현재 측정에 영향을 주지 않는 rotation의 axis를 dense state에 포함하지 않는다. 해당 효과는 $U_{pend}$에 symbolic하게 남는다.

## 8.2 Rank 비교

Clifft와 동일한 virtual coordinate 및 Pauli localization을 사용하고, 동일 axis를 중복 생성하지 않으며, exact frame reduction이 수행된다고 하자.

현재 측정에 필요한 모든 axis는 Clifft가 해당 시점까지 이미 materialize했을 axis의 부분집합이다. 따라서

$$B_{j}^{\star} \subseteq A_{t_{j}}^{C}$$

이고

$$\boxed{b_{j}^{\star} \leq k_{t_{j}}^{C}.}$$

peak에 대해서는

$$\boxed{b_{\max}^{\star} \leq k_{\max}^{C}.}$$

따라서 dense amplitude storage는

$$2^{b_{\max}^{\star}} \leq 2^{k_{\max}^{C}}$$

로 상한이 정해진다.

## 8.3 이 bound가 성립하기 위한 구현 조건

위 포함관계는 다음을 반드시 요구한다.

1.  Clifft와 동일한 virtual axis coordinate를 기준으로 비교할 것

2.  하나의 독립 자유도를 여러 axis로 중복 표현하지 않을 것

3.  measurement projector 때문에 별도의 임시 axis를 추가하지 않을 것

4.  측정 후 불필요 axis를 frame과 pending ledger에서 exact하게 decouple할 것

5.  resident state뿐 아니라 live amplitude workspace까지 bound에 포함할 것

즉 strict memory condition은

strict memory condition은 다음과 같다.

$$M_{\mathrm{resident},t} + M_{\mathrm{workspace},t} \leq s_{c}2^{k_{t}^{C}},$$

여기서 $s_{c}$는 complex amplitude 하나의 저장 byte 수이다.

이다.

## 8.4 메모리 절감률

복소 amplitude당 저장 비용이 $s_{c}$ byte라면

$$M_{C} \approx s_{c}2^{k_{\max}^{C}},$$

$$M_{B} \approx s_{c}2^{b_{\max}} + M_{\mathrm{symbolic}}.$$

지수항 기준 메모리 절감률은

$$\frac{M_{C}}{M_{B}} \approx 2^{k_{\max}^{C} - b_{\max}}.$$

예를 들어

$$k_{\max}^{C} = 24,\quad\quad b_{\max} = 13$$

이면 dense state 크기만 기준으로

$$2^{24 - 13} = 2^{11} = 2048$$

배 줄어든다.

# 9. Offline 분석과 shot runtime의 분리

## 9.1 Core schedule은 왜 shot-independent한가

Pauli noise와 feed-forward Pauli는 ${\widetilde{P}}_{t}$만 바꾼다. 이는 rotation angle의 부호와 measurement branch를 바꾸지만, Pauli generator 사이의 commutation relation은 바꾸지 않는다.

$$\langle P_{i},P_{\ell}\rangle_{s},\quad\quad\langle P_{i},M_{j}\rangle_{s}$$

는 circuit structure만으로 결정된다.

따라서 measurement마다 다음을 오프라인에서 계산할 수 있다.

- direct anti-commuting rotation

- temporal ancestor closure

- core rotation ID

- deferred rotation ID

- required axis set

- promotion 및 drop 계획

- 실행 kernel template

shot runtime에는 다음만 남는다.

- stochastic Pauli fault sample

- Pauli frame bit update

- rotation angle sign 선택

- Born probability 계산

- measurement outcome sample

- projection 및 state update

## 9.2 Symbolic memory

pending rotation 수를 $L$이라 하고, 각 $N$-qubit Pauli를 packed bit vector로 저장하면 symbolic ledger 비용은 대략

$$O\left( L\frac{N}{w} \right)$$

이다. 여기서 $w$는 machine word bit 수이다.

Dense state의 지수적 비용과 비교하면 symbolic ledger는 polynomial memory이다.

# 10. 상태 벡터 크기 평가 범위와 지표

본 절부터는 MDAM의 **상태 벡터 크기 및 메모리 성취**를 정리한다. 이 평가는 FLOP이나 실행시간이 아니라, 실행 중 유지되는 exponential dense object의 크기에 초점을 둔다.

## 10.1 평가 대상

MDAM의 active-state는 $b$개의 비안정자 자유도에 대한 복소 상태 벡터

$$
\phi \in \mathbb{C}^{2^b}
$$

이다. 이 벡터가 실행 시스템에서 유일한 지수 크기 객체이다. complex128을 사용할 경우 amplitude당 저장 비용은 16 byte이므로 dense state 메모리는

$$
M(b)=16\cdot 2^b\ \text{bytes}
$$

로 계산한다.

Clifft와 MDAM이 공통으로 사용하는 stabilizer tableau는 $O(N^2)$ bit의 다항식 객체이므로 본 비교에서는 제외한다. 본 절의 비교 대상은 두 방식에서 차이가 발생하는 exponential active-state뿐이다.

## 10.2 Resident와 transient의 구분

- **Resident rank $b_{\mathrm{res}}$:** 측정 사이에 지속적으로 유지되는 active-state rank
- **Transient rank $b_{\mathrm{tr}}$:** 측정 core를 처리하는 동안 일시적으로 도달하는 최대 rank
- **Clifft rank $k$:** Clifft가 해당 회로에서 유지하는 active-state rank
- **Peak state size:** 전체 실행에서 가장 큰 $2^b$
- **Integrated state size:** 전체 단계에 대한 $\sum_t 2^{b_t}$

Peak는 순간 최대 메모리 용량을 나타내고, integrated state size는 큰 상태를 얼마나 오래 유지했는지를 나타낸다. 두 지표는 서로 다른 의미를 가지므로 분리하여 보고한다.

## 10.3 절감률 정의

Peak resident 절감률은

$$
S_{\mathrm{peak}}
=
2^{k_{\max}-b_{\mathrm{res,max}}}
$$

로 정의한다.

시간 누적 절감률은 Clifft와 MDAM의 실제 단계별 rank를 사용하여

$$
S_{\mathrm{int}}
=
\frac{\sum_t 2^{k_t^{C}}}
{\sum_t 2^{b_{t}^{\mathrm{res}}}}
$$

로 정의한다.

# 11. 구현 상한과 평가 조건

실행 엔진은 회로별 Clifft peak rank $k_{\max}^{C}$를 dense-memory hard cap으로 설정한다. 평가된 모든 실행에서 resident state와 측정 workspace를 포함한 live amplitude rank는 이 cap을 넘지 않도록 검사하였다.

즉, 평가 조건은 다음과 같다.

$$
b_{\mathrm{res},t}\leq k_{\max}^{C},
\qquad
b_{\mathrm{tr},t}\leq k_{\max}^{C}.
$$

이 조건은 단순히 결과 파일에서 사후 확인한 것이 아니라, 실행 중 메모리 예산 검사를 활성화하여 위반 시 실행이 중단되도록 구성한 것이다.

다만 이 구현상 guard와 제8절의 수학적 포함관계는 구분해야 한다. 수학적 상한은 동일 virtual coordinate, 중복 axis 제거, 측정 후 exact decoupling 등의 조건에서 성립한다. 구현 guard는 실제 backend가 그 상한을 위반하지 않았음을 확인하는 안전 장치이다.

# 12. 최대 active-state 크기 성취

표 1은 각 회로에서 Clifft와 MDAM의 최대 active-state 크기를 비교한 결과이다.

| 회로 | non-cliford gate | Clifft $k$ | MDAM transient | MDAM resident | Resident peak 절감 | Resident 메모리 비교 |
|---|---:|---:|---:|---:|---:|---:|
| coherent_d5_r1 | $R_Z$ | $2^{13}$ | $2^0$ | $2^0$ | **8,192×** | 16 B vs 128 KiB |
| coherent_d5_r5 | $R_Z$ | $2^{24}$ | $2^{13}$ | $2^{12}$ | **4,096×** | 64 KiB vs 256 MiB |
| coherent_d3_r1 | $R_Z$ | $2^5$ | $2^0$ | $2^0$ | 32× | 16 B vs 512 B |
| coherent_d3_r3 | $R_Z$ | $2^8$ | $2^5$ | $2^4$ | 16× | 256 B vs 4 KiB |
| coherent_rx_d3_r1 | $R_X$ | $2^{14}$ | $2^{11}$ | $2^{10}$ | 16× | 16 KiB vs 256 KiB |
| coherent_rx_d3_r3 | $R_X$ | $2^{14}$ | $2^{12}$ | $2^{11}$ | 8× | 32 KiB vs 256 KiB |
| coherent_ry_d3_r1 | $R_Y$ | $2^{16}$ | $2^{16}$ | $2^{15}$ | 2× | 512 KiB vs 1 MiB |
| coherent_ry_d3_r3 | $R_Y$ | $2^{16}$ | $2^{16}$ | $2^{15}$ | 2× | 512 KiB vs 1 MiB |
| distillation | T | $2^5$ | $2^4$ | $2^3$ | 4× | 128 B vs 512 B |
| cultivation_d3 | T | $2^4$ | $2^4$ | $2^3$ | 2× | 128 B vs 256 B |
| cultivation_d5 | T | $2^{10}$ | $2^{10}$ | $2^9$ | 2× | 8 KiB vs 16 KiB |
| surface_d7_r7 | Clifford | $2^0$ | $2^0$ | $2^0$ | 1× | 16 B vs 16 B |

> **〔그림 1 — 별도 첨부〕** *11개 회로의 단계별 active-state 크기 (전체 한눈에 보기).*
> 그림에서 가로축 = 실행 단계, 세로축 = active-state 크기(= log₂ 차원, qubit 단위).
> **빨간 실선** = Clifft가 그 단계에서 들고 있는 dense rank `2^k`. **녹색 실선** = MDAM(ours)의
> resident rank, **녹색 점선** = MDAM의 transient(측정 순간) rank. **붉은 음영** = 두 방식의 차이,
> 즉 MDAM이 절감한 메모리 영역이다. 붉은 음영이 클수록(특히 `coherent_d5_*`) MDAM의 이득이 크다.

**대표 결과.** `coherent_d5_r5`에서 Clifft는 $2^{24}$개의 amplitude, 즉 256 MiB의 dense state를 요구한다. MDAM은 resident $2^{12}$와 transient $2^{13}$을 사용하여 각각 64 KiB와 128 KiB를 요구한다. 이에 따라 resident peak 기준 4,096×, 실제 transient peak 기준 2,048×의 절감이 확인되었다.

`coherent_d5_r1`에서는 $R_Z$ 회전이 dense 비안정자 자유도를 남기지 않아 resident와 transient가 모두 $2^0$에 머물렀다. Clifft의 $2^{13}$과 비교하면 8,192×의 peak 절감이다.

# 13. 누적 active-state 크기 성취

Peak 결과만으로는 큰 상태를 얼마나 오래 유지하는지 알 수 없다. 따라서 실제 단계별 rank를 적분한 state-time 지표를 함께 계산하였다.

| 회로 | Clifft $\sum_t 2^{k_t^C}$ | MDAM $\sum_t 2^{b_t}$ | 누적 절감률 | Clifft state-time bytes | MDAM state-time bytes |
|---|---:|---:|---:|---:|---:|
| coherent_d5_r5 | $2^{34.78}$ | $2^{22.81}$ | **4,007×** | 440.4 GiB | 112.5 MiB |
| coherent_d5_r1 | $2^{21.05}$ | $2^{9.74}$ | **2,535×** | 33.2 MiB | 13.4 KiB |
| coherent_rx_d3_r1 | $2^{19.31}$ | $2^{14.85}$ | 22.0× | 9.9 MiB | 460.8 KiB |
| coherent_d3_r3 | $2^{15.85}$ | $2^{11.69}$ | 17.9× | 922 KiB | 51.6 KiB |
| coherent_ry_d3_r1 | $2^{22.36}$ | $2^{18.94}$ | 10.7× | 82.1 MiB | 7.7 MiB |
| coherent_d3_r1 | $2^{11.42}$ | $2^{8.00}$ | 10.7× | 42.8 KiB | 4 KiB |
| coherent_ry_d3_r3 | $2^{23.94}$ | $2^{20.69}$ | 9.52× | 245.4 MiB | 25.8 MiB |
| coherent_rx_d3_r3 | $2^{21.04}$ | $2^{17.96}$ | 8.45× | 32.8 MiB | 3.9 MiB |
| cultivation_d5 | $2^{18.97}$ | $2^{17.79}$ | 2.27× | 7.8 MiB | 3.5 MiB |
| distillation | $2^{14.15}$ | $2^{13.04}$ | 2.15× | 283.1 KiB | 131.6 KiB |
| cultivation_d3 | $2^{11.52}$ | $2^{10.44}$ | 2.11× | 45.9 KiB | 21.7 KiB |
| surface_d7_r7 | $2^{11.43}$ | $2^{11.43}$ | 1.0× | 43 KiB | 43 KiB |

> **〔그림 2 — 별도 첨부〕** *대표 회로 `coherent_d5_r5`의 단계별 active-state.*
> Clifft(빨강)는 각 측정 라운드마다 `2^{24}`까지 네 번 치솟지만, MDAM(녹색)은 `2^{12}` resident에
> 평평하게 머문다(측정 순간만 `2^{13}` transient로 짧게 +1). 붉은 음영 전체가 절감된 메모리이며,
> 이 면적이 §13의 누적 4,007× 절감으로 적분된다. 측정-구동 localize-and-drop이 매 라운드 직후
> rank를 즉시 회수하기 때문에 MDAM 곡선이 낮게 유지된다.

`coherent_d5_r5`의 누적 state-time은 Clifft 440.4 GiB 대비 MDAM 112.5 MiB로 4,007× 감소하였다. 이 결과는 MDAM이 peak만 순간적으로 줄인 것이 아니라 실행의 상당 구간에서 작은 state를 유지했음을 의미한다.

`coherent_d3_r3`에서는 resident peak 절감이 16×인 반면 누적 절감은 17.9×이다. 이는 Clifft가 높은 rank에 더 오래 머무르는 반면 MDAM의 높은 rank는 측정 시점에 국소적으로 나타나기 때문이다.

# 14. 회로 유형별 결과 해석

## 14.1 대각 $R_Z$ 회로: 큰 quotient가 형성되는 영역

`coherent_d3_*`와 `coherent_d5_*`의 $R_Z$ 회로에서는 측정과 commute하는 회전을 장기간 pending 상태로 유지할 수 있고, 측정 후 axis를 빠르게 제거할 수 있다.

대표적으로

$$
k_{\max}^{C}=24,\qquad
b_{\mathrm{res,max}}=12
$$

이므로 `coherent_d5_r5`에서 resident state는 4,096× 작다. 이 영역이 MDAM의 구조적 장점이 가장 크게 나타나는 적용 대상이다.

## 14.2 T-gate cultivation 및 $R_Y$ 회로: transient parity와 resident 감소

`cultivation_d3`, `cultivation_d5`, `coherent_ry_d3_*`에서는 측정 시 실제로 많은 비안정자 자유도가 동시에 필요하다. 따라서 transient rank는 Clifft peak와 동일한 수준까지 도달할 수 있다.

그러나 측정 직후 working axis를 제거하므로 resident rank는 일반적으로 한 단계 낮다.

$$
b_{\mathrm{res,max}}=k_{\max}^{C}-1
$$

인 경우 resident 메모리는 2× 줄어든다. 또한 높은 rank의 유지 시간이 짧아 $R_Y$ d3 회로에서는 누적 절감이 9.52–10.7×까지 증가한다.

## 14.3 $R_X$ 회로: 부분적 절감

d3의 $R_X$ 회로에서는 off-axis support가 존재하지만 일부 axis는 여전히 측정별로 분리된다. 이에 따라 peak resident 기준 8–16×의 절감이 나타났다.

이는 MDAM의 이득이 단순히 diagonal 여부만으로 결정되는 것이 아니라, 측정별 non-Clifford dependency의 실제 rank에 의해 결정됨을 보여준다.

## 14.4 Off-axis d5 회로: 적용 경계

다음 회로에서는 active-state가 $2^{26}$을 넘어 1 GiB 실행 상한을 초과하므로 평가를 완료하지 못했다.

| 회로 | 노이즈 | Clifft $k$ | 상태 |
|---|---:|---:|---|
| coherent_rx_d5_r1 | $R_X$ | 38 | $2^{26}$ 초과로 실행 불가 |
| coherent_rx_d5_r5 | $R_X$ | 38 | $2^{26}$ 초과로 실행 불가 |
| coherent_ry_d5_r1 | $R_Y$ | 47 | $2^{26}$ 초과로 실행 불가 |
| coherent_ry_d5_r5 | $R_Y$ | 47 | $2^{26}$ 초과로 실행 불가 |

이 결과는 MDAM이 모든 회로에서 state size를 크게 줄인다는 의미가 아님을 보여준다. X/Y support가 다수의 data qubit에 걸쳐 동시에 유지되면 measurement-visible core 자체가 커지고, exact representation도 큰 dense state를 필요로 한다.

# 15. 단계별 active-state 특성

표 4는 resident peak에 머문 단계 수와 transient spike 횟수를 정리한 것이다.

| 회로 | 전체 단계 | Resident peak rank | Transient peak rank | Resident peak 유지 단계 | Transient spike 수 |
|---|---:|---:|---:|---:|---:|
| coherent_d5_r5 | 3,229 | 12 | 13 | 1,783 | 60 |
| coherent_d5_r1 | 858 | 0 | 0 | 858 | 0 |
| coherent_d3_r3 | 565 | 4 | 5 | 171 | 12 |
| coherent_d3_r1 | 256 | 0 | 0 | 256 | 0 |
| coherent_rx_d3_r1 | 283 | 10 | 11 | 6 | 14 |
| coherent_rx_d3_r3 | 660 | 11 | 12 | 6 | 30 |
| coherent_ry_d3_r1 | 366 | 15 | 16 | 4 | 17 |
| coherent_ry_d3_r3 | 851 | 15 | 16 | 12 | 33 |
| cultivation_d3 | 345 | 3 | 4 | 143 | 5 |
| cultivation_d5 | 1,785 | 9 | 10 | 417 | 15 |
| distillation | 2,041 | 3 | 4 | 406 | 5 |
| surface_d7_r7 | 2,750 | 0 | 0 | 2,750 | 0 |

`coherent_d5_r5`는 3,229단계 중 60개의 측정 core에서만 transient $2^{13}$에 도달한다. 이는 transient와 resident를 구분하지 않고 단일 peak 값만 보고할 경우 실제 지속 메모리 특성을 충분히 설명하지 못한다는 점을 보여준다.

# 16. 정확성 검증 근거

상태 크기 감소는 amplitude truncation이나 근사가 아니라, 동일한 물리 상태를 더 작은 virtual representation으로 표현한 결과이다.

보고된 검증 결과는 다음과 같다.

1. **Off-axis $R_Y$ 검증**
   - 측정별 Born probability 차이:
     $$
     |\Delta p|\leq 2.6\times 10^{-15}
     $$
   - joint trajectory 차이:
     $$
     \leq 1.4\times 10^{-13}
     $$
   - 독립 $2^{17}$ dense statevector 및 Clifft probability 계산과 교차 검증

2. **대각 및 cultivation 검증**
   - 다수 seed에서 measurement record, peak rank, Born probability 일치
   - 측정 instrument와 dense oracle의 기계 정밀도 수준 일치
   - 첫 측정 전 unitary prefix의 physical statevector residual:
     $$
     2.22\times 10^{-16}
     $$

따라서 표에 제시한 $2^b$는 fidelity를 낮춰 얻은 압축 결과가 아니라, 동일한 exact state를 표현하는 데 실제로 사용된 dense state 크기이다.

# 17. 결과 해석 시 유의사항

1. **Resident와 transient를 구분해야 한다.**  
   Resident는 측정 사이의 지속 메모리이고, transient는 측정 core 내부의 순간 최대값이다. 실제 peak allocation을 주장할 때는 transient를 사용해야 하며, 지속 메모리 특성을 설명할 때는 resident를 함께 보고해야 한다.

2. **본 비교는 exponential active-state에 한정된다.**  
   Stabilizer tableau, record buffer, symbolic pending ledger 등의 다항식 메모리는 별도로 존재한다. 다만 Clifft도 tableau를 유지하므로 active-state 비교에서는 동일 기준으로 제외하였다.

3. **Peak와 integrated 결과는 서로 대체할 수 없다.**  
   Peak는 장비 용량 한계를, integrated state size는 메모리 점유 지속 시간을 설명한다.

4. **Off-axis d5는 현재 실행 상한을 초과한다.**  
   이는 단순 구현 실패가 아니라 measurement-visible non-Clifford rank가 큰 회로에서 exact dense representation이 여전히 지수적으로 증가한다는 적용 경계이다.

5. **메모리 절감은 FLOP 절감을 자동으로 의미하지 않는다.**  
   본 보고서의 성취는 state-vector dimension과 memory footprint에 관한 결과이다. Kernel 비용, 반복 norm 계산, 데이터 이동 비용은 별도의 성능 평가가 필요하다.

# 18. 종합 결론

MDAM은 Clifft의 rotation-driven active-state expansion을 measurement-driven materialization으로 변경한다. Non-Clifford rotation은 ordered pending ledger에 exact하게 보존되고, 각 측정에서 Born probability와 post-measurement state에 필요한 core만 dense state로 구현된다.

평가 결과는 다음과 같이 요약된다.

- `coherent_d5_r5`: Clifft 256 MiB 대비 MDAM resident 64 KiB, transient 128 KiB
- Resident peak 절감: 최대 **8,192×**
- 누적 state-time 절감: 최대 **4,007×**
- Cultivation: transient peak는 Clifft와 동일할 수 있으나 resident는 2× 감소
- Pure Clifford 회로: Clifft와 동일한 $2^0$
- Off-axis d5 회로: 큰 measurement-visible rank로 인해 현재 1 GiB 실행 상한 초과

따라서 MDAM의 핵심 성취는 다음과 같다.

> 비-Clifford 효과를 회전 발생 시점부터 누적 dense state로 유지하지 않고, 각 측정이 실제로 요구하는 virtual freedom만 선택적으로 materialize함으로써 exact near-Clifford simulation의 exponential state-vector 크기를 회로 구조에 따라 크게 감소시켰다.

# 부록 A. 회로 그룹별 단계당 active-state 그림 목록 (전체 11개 — 별도 첨부)

아래 그림들은 **본 문서에 임베드하지 않고 별도 파일로 첨부**한다. 각 그림에서 **빨간 실선** =
Clifft `2^k`, **녹색 실선** = MDAM(ours) resident, **녹색 점선** = MDAM transient, **붉은 음영** =
MDAM이 절감한 메모리 영역이며, 제목에 `Clifft 2^k → MDAM 2^b resident, N× smaller` 형식으로 회로별
peak 절감 배수를 표기하였다.

## A.1 대각 $R_Z$ coherent 회로 — 큰 quotient가 형성되는 영역
측정과 commute하는 회전을 길게 보류했다가 측정 후 즉시 회수하므로 MDAM 곡선이 Clifft보다 크게 낮다.

- 그림 A.1.1 `coherent_d3_r1`
- 그림 A.1.2 `coherent_d3_r3`
- 그림 A.1.3 `coherent_d5_r1`
- 그림 A.1.4 `coherent_d5_r5`

## A.2 Off-axis $R_X$ / $R_Y$ coherent 회로 — 부분 절감
X/Y-support로 측정별 core가 커져 transient가 Clifft peak에 근접한다($R_Y$). 이득은 주로 resident
drop과 시간 누적에서 나온다(본문 §14.2–14.3). (d=5 off-axis는 $2^{26}$ 초과로 미수록 — 본문 §14.4)

- 그림 A.2.1 `coherent_rx_d3_r1`
- 그림 A.2.2 `coherent_rx_d3_r3`
- 그림 A.2.3 `coherent_ry_d3_r1`
- 그림 A.2.4 `coherent_ry_d3_r3`

## A.3 T-게이트 마법상태 회로 (cultivation / distillation)
측정 시 다수 자유도가 동시에 필요해 transient는 Clifft와 parity일 수 있으나, 측정 직후 working
axis를 회수하여 resident는 한 단계 낮다(본문 §14.2).

- 그림 A.3.1 `cultivation_d3`
- 그림 A.3.2 `cultivation_d5`
- 그림 A.3.3 `distillation`

---

# 참고문헌

[1] B. Chase and F. Labib, “Clifft: Fast Exact Simulation of Near-Clifford Quantum Circuits,” arXiv:2604.27058, 2026.
