"""Per-step active-state size: Clifft (2^k) vs live clifft_axis_bounded, one plot per circuit + a grid.
y = active-state size in qubits (= log2 dense-equivalent dimension). Reads the
bounded_<circuit>_per_step.csv files (n_active = Clifft active rank k; bounded_*_qubits)."""
import os, csv
os.chdir("/home/jung/clifft-paper")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "reports_clifft_axis_bounded"
# 11 circuits the report figures over (user-specified): R_Z / R_X / R_Y coherent + T-gate
CIRCS = ["coherent_d3_r1", "coherent_d3_r3", "coherent_d5_r1", "coherent_d5_r5",
         "coherent_rx_d3_r1", "coherent_rx_d3_r3", "coherent_ry_d3_r1", "coherent_ry_d3_r3",
         "cultivation_d3", "cultivation_d5", "distillation"]
CL = "#c0392b"      # clifft crimson
FG = "#1e8449"      # MDAM (ours) green

def load(c):
    rows = list(csv.DictReader(open(f"{OUT}/bounded_{c}_per_step.csv")))
    step = [int(r["step"]) for r in rows]
    k    = [int(r["n_active"]) for r in rows]
    tr   = [int(r["bounded_transient_qubits"]) for r in rows]
    res  = [int(r["bounded_resident_qubits"]) for r in rows]
    return step, k, tr, res

def draw(ax, c):
    step, k, tr, res = load(c)
    kpk, trpk, respk = max(k), max(tr), max(res)
    save = (2 ** (kpk - respk)) if kpk >= respk else 1
    ax.fill_between(step, res, k, color=CL, alpha=0.12)          # saving (Clifft over MDAM)
    ax.plot(step, k,  color=CL, lw=1.6, label=f"Clifft  2^k  (peak 2^{kpk})")
    ax.plot(step, tr, color=FG, lw=1.4, ls="--", label=f"MDAM (ours) transient (peak 2^{trpk})")
    ax.plot(step, res,color=FG, lw=1.8, label=f"MDAM (ours) resident (peak 2^{respk})")
    ax.set_title(f"{c}   (Clifft 2^{kpk}  ->  MDAM (ours) 2^{respk} resident,  {save:,}x smaller)",
                 fontsize=10)
    ax.set_xlabel("runtime step"); ax.set_ylabel("active-state size  (qubits = log2 dim)")
    ax.set_ylim(bottom=0); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25)

# per-circuit PNGs
for c in CIRCS:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    draw(ax, c)
    fig.tight_layout(); fig.savefig(f"{OUT}/bounded_vs_clifft_{c}_qubits.png", dpi=130)
    plt.close(fig)

# overview grid (11 circuits -> 4x3, last cell blank)
fig, axes = plt.subplots(4, 3, figsize=(21, 16))
flat = axes.ravel()
for ax, c in zip(flat, CIRCS):
    draw(ax, c)
for ax in flat[len(CIRCS):]:
    ax.axis("off")
fig.suptitle("Per-step active-state size:  Clifft 2^k   vs   MDAM (ours)", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(f"{OUT}/bounded_vs_clifft_ALL_qubits.png", dpi=110)
plt.close(fig)
print("WROTE", OUT + f"/bounded_vs_clifft_<circuit>_qubits.png (x{len(CIRCS)}) + bounded_vs_clifft_ALL_qubits.png")
