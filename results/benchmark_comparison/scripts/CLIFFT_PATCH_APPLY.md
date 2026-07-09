# 패치된 clifft 재구성 (서버용)

홈 머신의 clifft는 stock이 아니라 **mdam translate가 읽는 constant pool
(pauli_masks / noise_sites)을 파이썬에 노출하도록 패치된 빌드**다
(`0.4.2.dev2+g2655e48c6.d20260623`의 `d` = dirty tree). 이 디렉토리의 세 파일로 재구성한다:

- `clifft_mdam_export.patch` — upstream commit 2655e48c6 기준 git diff
  (bindings.cc 노이즈/마스크 노출 + svm cost-meter 훅 + CMakeLists 1줄)
- `cost_meter.cc`, `cost_meter.h` — 신규 파일, `src/clifft/util/` 에 복사

## 적용 절차

```bash
git clone https://github.com/unitaryfoundation/clifft ~/clifft-src
cd ~/clifft-src
git checkout 2655e48c6
git apply <이 디렉토리>/clifft_mdam_export.patch
cp <이 디렉토리>/cost_meter.{cc,h} src/clifft/util/
pip install . 2>&1 | tail -3
python -c "import clifft; print(clifft.version(), clifft.svm_backend())"
# 기대: 0.4.2.dev2+g2655e48c6.d<날짜>  avx512
```

## 설치 후 검증 (측정 전 필수)

```bash
cd <repo>/mdam/native_vm
python -c "
import sys,os; sys.path.insert(0,'.'); sys.path.insert(0,'..'); sys.path.insert(0,'../..')
import clifft
from verify_mdam_oneshot import translate
t=translate(clifft.compile(open('../../qec_bench/circuits/coherent_d3_r1.stim').read()))
print('noise sites:', len(t['site_nchan']), ' pauli masks:', len(t['mmask']))
assert len(t['site_nchan'])>0, 'noise export STILL empty'
print('translate OK')"
```

그 다음 bit-exact 정합 (`verify_mdam_batch.py` 또는 tier_wall_row.py 내장 spot check)을
1개 벤치에서 통과시킨 후 본 측정 시작.
