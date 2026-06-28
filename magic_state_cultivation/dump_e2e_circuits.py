from pathlib import Path
from convert_s_to_t import (
    make_d3_end2end_expval,
    make_d5_end2end_expval,
    make_d3_end2end_expval_sproxy,
    make_d5_end2end_expval_sproxy,
)

OUT = Path("dumped_circuits")
OUT.mkdir(exist_ok=True)

noise = 0.001

circuits = {
    "t_gate_d3_e2e_p0.001.stim": make_d3_end2end_expval(noise_strength=noise),
    "t_gate_d5_e2e_p0.001.stim": make_d5_end2end_expval(noise_strength=noise),
    "s_proxy_d3_e2e_p0.001.stim": make_d3_end2end_expval_sproxy(noise_strength=noise),
    "s_proxy_d5_e2e_p0.001.stim": make_d5_end2end_expval_sproxy(noise_strength=noise),
}

for name, text in circuits.items():
    path = OUT / name
    path.write_text(text)
    print(path, "lines =", len(text.splitlines()))