"""
QEC 회로 survey: k_max vs EXACT tau, dense vs tensor cost 비교.

tau는 더 이상 inline min-fill(상한)로 구하지 않는다. 옆의 treewidth.py 모듈을
호출해서, 컴파일된 Program을 replay -> per-snapshot active interaction graph ->
EXACT treewidth(subset-DP, 전 snapshot UB-pruning)로 구한다. SWAP relabel,
slot->identity 추적, strict 검증도 전부 모듈이 처리한다.

전제: treewidth.py 가 같은 디렉터리(또는 PYTHONPATH)에 있어야 한다.

실행:
  python3 compute_treewidth.py
"""

import sys
from pathlib import Path

sys.path.insert(0, "/home/jung/clifft/src/python/clifft")

import clifft
import stim
import re
from ttn_backend import treewidth as twx

CIRCUITS_DIR = Path("~/clifft-paper/qec_bench/circuits").expanduser()

KNOWN_PHYS = {"distillation": 85, "cultivation_d3": 42, "cultivation_d5": 118}
KNOWN_LOG = {
    "distillation":   "1 (15->1 MSD)",
    "cultivation_d3": "1 (MSC d=3)",
    "cultivation_d5": "1 (MSC d=5)",
    "surface_d7_r7":  "1 (surface d=7)",
}

# exact subset-DP 한계. snapshot 그래프 노드 수가 이보다 크면 그 row의 tau는
# min-fill 상한으로 fallback 되고 exact? 컬럼이 N 으로 찍힌다.
EXACT_MAX_N = 22


def get_phys(name, text):
    if name in KNOWN_PHYS:
        return KNOWN_PHYS[name]
    t2 = re.sub(r"R_Z\([^)]+\)", "DEPOLARIZE1(0.001)", text)
    t2 = re.sub(r"^EXP_VAL\b.*", "", t2, flags=re.MULTILINE)
    t2 = re.sub(r"^T_DAG\b", "I", t2, flags=re.MULTILINE)
    t2 = re.sub(r"^T\b", "I", t2, flags=re.MULTILINE)
    t2 = re.sub(r"^R_X\b", "R", t2, flags=re.MULTILINE)
    t2 = re.sub(r"^DEPOLARIZE\b", "DEPOLARIZE1", t2, flags=re.MULTILINE)
    try:
        return stim.Circuit(t2).num_qubits
    except Exception:
        return -1


def get_log(name):
    if name in KNOWN_LOG:
        return KNOWN_LOG[name]
    return "1"


def analyze_circuit(name, text):
    prog = clifft.compile(
        text,
        hir_passes=clifft.default_hir_pass_manager(),
        bytecode_passes=clifft.default_bytecode_pass_manager(),
    )
    k = prog.peak_rank

    # === 핵심: exact tau 를 모듈에서 ===
    # strict=False 로 두어 survey 도중 한 회로의 이상이 전체를 중단시키지 않게 하고,
    # peak_rank 를 넘겨 replay 재구성 peak_k 와 대조(=node-add 커버리지 무결성 체크).
    rep = twx.analyze(prog, strict=False, exact_max_n=EXACT_MAX_N,
                      peak_rank=k, verbose=False)

    tau       = rep["peak_tau"]
    tau_exact = rep["peak_tau_exact"]
    L         = rep["peak_k"]        # active-node 수 (bag 수의 O(k) 상한 proxy)
    k_ok      = rep.get("replay_peak_k_matches", True)
    struct    = rep.get("peak_struct")

    dense = f"2^{k}"
    if L > 0 and k > 0:
        tensor = f"{L} x 2^{tau}"
        gain_val = int(round(2 ** k / max(L * (2 ** tau), 1)))
        # tau 가 상한(fallback)이면 gain 은 진짜 gain 의 하한(보수적)이다.
        gain = (f"{gain_val:,}x" if gain_val >= 2 else "-")
    else:
        tensor, gain = "trivial", "-"

    # 해석: tau << k 면 GO 확실(상한이 작으니 진짜 비용도 작음).
    #       tau ~ k 는 NO-GO 가 아니라, 이 proxy 에서 go 신호가 약하다는 뜻.
    if k == 0:
        verdict = "-"
    elif tau <= max(1, k // 2):
        verdict = "GO"
    else:
        verdict = "weak"

    issues = []
    if not k_ok:
        issues.append(f"peak_k mismatch (replay {rep['peak_k']} != peak_rank {k})")

    return dict(name=name, phys=get_phys(name, text), log=get_log(name),
                k=k, tau=tau, exact=tau_exact, L=L,
                dense=dense, tensor=tensor, gain=gain,
                verdict=verdict, issues=issues, struct=struct)


def main():
    circuits = {f.stem: f.read_text() for f in sorted(CIRCUITS_DIR.glob("*.stim"))}
    if not circuits:
        print(f"No .stim circuits found in {CIRCUITS_DIR}")
        return

    W = 116
    print("=" * W)
    print(f"{'Circuit':<22} {'physQ':>5} {'logQ':>15} {'k_max':>5} "
          f"{'tau':>4} {'ex':>3} {'L':>4}  {'dense':>7}  {'tensor':>14}  "
          f"{'gain':>10}  {'verdict':>7}")
    print("=" * W)

    all_issues = []
    rows = []
    for name, text in circuits.items():
        try:
            r = analyze_circuit(name, text)
            rows.append(r)
            phys = str(r["phys"]) if r["phys"] > 0 else "?"
            ex = "Y" if r["exact"] else "N"
            print(f"{r['name']:<22} {phys:>5} {r['log']:>15} {r['k']:>5} "
                  f"{r['tau']:>4} {ex:>3} {r['L']:>4}  {r['dense']:>7}  "
                  f"{r['tensor']:>14}  {r['gain']:>10}  {r['verdict']:>7}")
            for iss in r["issues"]:
                all_issues.append(f"{name}: {iss}")
        except Exception as e:
            print(f"{name:<22}  ERROR: {type(e).__name__}: {e}")

    # ---- STEP 0: peak-snapshot junction-tree structure (backend blueprint) ----
    print()
    print("=" * W)
    print("STEP 0: peak-snapshot junction-tree structure (backend blueprint)")
    print("=" * W)
    print(f"{'Circuit':<22} {'n':>3} {'tau':>4} {'bags':>5} {'maxbag':>7} "
          f"{'sum2^|B|':>9} {'maxsep':>7} {'shape':>10}   backend")
    print("-" * W)
    for r in rows:
        st = r.get("struct")
        if not st or st["n"] == 0:
            print(f"{r['name']:<22} {'-':>3} {'-':>4} {'-':>5} {'-':>7} "
                  f"{'-':>9} {'-':>7} {'-':>10}   pure Clifford / no active TN")
            continue
        print(f"{r['name']:<22} {st['n']:>3} {st['tau']:>4} {st['n_bags']:>5} "
              f"2^{st['max_bag']:<5} {st['sum2']:>9,} {st['max_sep']:>7} "
              f"{st['shape']:>10}   {twx.backend_hint(st)}")
    print("=" * W)
    print("max tensor = 2^maxbag (=2^(tau+1)) | bond dim = 2^maxsep | total mem ~ sum2^|B|")
    print("junction-tree TN is ALWAYS a tree -> canonical form -> local measurement.")
    print("path shape -> MPS directly; branching -> TTN (MPS not ruled out).")
    print()

    print("=" * W)
    print("ex=Y: tau is EXACT treewidth of the active interaction graph (subset-DP).")
    print("ex=N: snapshot exceeded EXACT_MAX_N; tau is min-fill UPPER bound (gain is then a LOWER bound).")
    print("Interpretation: tau << k  -> GO is solid (graph tw upper-bounds true cost).")
    print("                tau ~ k   -> NOT a no-go; verify with cancellation-aware")
    print("                             contraction width (cotengra/quimb) before rejecting.")
    print("L = peak active-node count (O(k) upper proxy for tree-decomposition bag count).")
    if all_issues:
        print("\nINTEGRITY WARNINGS (suspect rows -- investigate opcode coverage):")
        for iss in all_issues:
            print("  - " + iss)


if __name__ == "__main__":
    main()
