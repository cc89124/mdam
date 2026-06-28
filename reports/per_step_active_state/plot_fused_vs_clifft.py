"""Per-step active-state size: Clifft (2^k) vs live fused-VA, one plot per circuit + a grid.
y = active-state size in qubits (= log2 dense-equivalent dimension). Reads the
fused_va_<circuit>_per_step.csv files (n_active = Clifft active rank k; fused_*_qubits)."""
import os, csv
os.chdir("/home/jung/clifft-paper")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "reports/per_step_active_state"
CIRCS = ["coherent_d3_r1", "coherent_d3_r3", "coherent_d5_r1", "coherent_d5_r5",
         "cultivation_d3", "cultivation_d5", "distillation", "surface_d7_r7"]
CL = "#c0392b"      # clifft crimson
FG = "#1e8449"      # fused green

def load(c):
    rows = list(csv.DictReader(open(f"{OUT}/fused_va_{c}_per_step.csv")))
    step = [int(r["step"]) for r in rows]
    k    = [int(r["n_active"]) for r in rows]
    tr   = [int(r["fused_transient_qubits"]) for r in rows]
    res  = [int(r["fused_resident_qubits"]) for r in rows]
    return step, k, tr, res

def draw(ax, c):
    step, k, tr, res = load(c)
    ax.fill_between(step, res, k, color=CL, alpha=0.12)          # the saving (clifft over fused)
    ax.plot(step, k,  color=CL, lw=1.6, label=f"Clifft  2^k  (peak {max(k)})")
    ax.plot(step, tr, color=FG, lw=1.4, ls="--", label=f"fused transient (peak {max(tr)})")
    ax.plot(step, res,color=FG, lw=1.8, label=f"fused resident (peak {max(res)})")
    ax.set_title(f"{c}   (Clifft k={max(k)} -> fused ws={max(tr)},  2^{max(k)-max(tr)}x state)",
                 fontsize=10)
    ax.set_xlabel("runtime step"); ax.set_ylabel("active-state size  (qubits = log2 dim)")
    ax.set_ylim(bottom=0); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25)

# per-circuit PNGs
for c in CIRCS:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    draw(ax, c)
    fig.tight_layout(); fig.savefig(f"{OUT}/fused_vs_clifft_{c}_qubits.png", dpi=130)
    plt.close(fig)

# overview grid
fig, axes = plt.subplots(4, 2, figsize=(15, 16))
for ax, c in zip(axes.ravel(), CIRCS):
    draw(ax, c)
fig.suptitle("Per-step active-state size: Clifft 2^k  vs  live fused virtual-axis", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(f"{OUT}/fused_vs_clifft_ALL_qubits.png", dpi=110)
plt.close(fig)
print("WROTE", OUT + "/fused_vs_clifft_<circuit>_qubits.png (x8) + fused_vs_clifft_ALL_qubits.png")
