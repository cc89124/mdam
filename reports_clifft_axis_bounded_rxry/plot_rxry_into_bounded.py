"""Per-step active-state size: Clifft (2^k) vs live clifft_axis_bounded on OFF-AXIS
(R_X / R_Y) coherent noise, d=3.  Writes into reports_clifft_axis_bounded/ with RXRY-suffixed
names so the R_Z bounded report is untouched.  y = active-state size in qubits (= log2 dim)."""
import os, csv
os.chdir("/home/jung/clifft-paper")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "reports_clifft_axis_bounded"
CIRCS = ["coherent_rx_d3_r1", "coherent_rx_d3_r3",
         "coherent_ry_d3_r1", "coherent_ry_d3_r3"]
CL = "#c0392b"      # clifft crimson
FG = "#1e8449"      # bounded green

def load(c):
    rows = list(csv.DictReader(open(f"{OUT}/bounded_{c}_per_step.csv")))
    step = [int(r["step"]) for r in rows]
    k    = [int(r["n_active"]) for r in rows]
    tr   = [int(r["bounded_transient_qubits"]) for r in rows]
    res  = [int(r["bounded_resident_qubits"]) for r in rows]
    return step, k, tr, res

def draw(ax, c):
    step, k, tr, res = load(c)
    ax.fill_between(step, res, k, color=CL, alpha=0.12)
    ax.plot(step, k,  color=CL, lw=1.6, label=f"Clifft  2^k  (peak {max(k)})")
    ax.plot(step, tr, color=FG, lw=1.4, ls="--", label=f"bounded transient (peak {max(tr)})")
    ax.plot(step, res,color=FG, lw=1.8, label=f"bounded resident (peak {max(res)})")
    ax.set_title(f"{c}   (Clifft k={max(k)} -> bounded ws={max(tr)},  2^{max(k)-max(tr)}x state)",
                 fontsize=10)
    ax.set_xlabel("runtime step"); ax.set_ylabel("active-state size  (qubits = log2 dim)")
    ax.set_ylim(bottom=0); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25)

for c in CIRCS:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    draw(ax, c)
    fig.tight_layout(); fig.savefig(f"{OUT}/bounded_vs_clifft_{c}_qubits.png", dpi=130)
    plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
for ax, c in zip(axes.ravel(), CIRCS):
    draw(ax, c)
fig.suptitle("Per-step active-state size: Clifft 2^k  vs  clifft_axis_bounded  "
             "(OFF-AXIS R_X / R_Y noise, d=3)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{OUT}/bounded_vs_clifft_RXRY_ALL_qubits.png", dpi=110)
plt.close(fig)
print("WROTE", OUT + "/bounded_vs_clifft_<rx|ry>_d3_*_qubits.png (x4) + bounded_vs_clifft_RXRY_ALL_qubits.png")
