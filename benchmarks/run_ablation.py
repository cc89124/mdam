"""Ablation runner for temporal-live carving seeders."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from temporal_carving.cost import CostModel
from temporal_carving.exact import exact_dp
from temporal_carving.pipeline import run
from temporal_carving.synth import planted_temporal_masking, random_brickwork, two_block_qec


GENERATORS = {
    "random_brickwork": random_brickwork,
    "planted_temporal_masking": planted_temporal_masking,
    "two_block_qec": two_block_qec,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="benchmarks/ablation.csv")
    p.add_argument("--sizes", nargs="*", type=int, default=[8, 12, 16])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exact-cap", type=int, default=16)
    args = p.parse_args()

    rows = []
    seeders = ["recursive_balanced_mincut", "louvain", "linear", "star", "random"]
    for gen_name, gen in GENERATORS.items():
        for n in args.sizes:
            if gen_name == "random_brickwork":
                trace = gen(n=n, depth=2 * n, seed=args.seed)
            elif gen_name == "two_block_qec":
                trace = gen(n=n, rounds=4, seed=args.seed)
            else:
                trace = gen(n=n, seed=args.seed)
            exact = None
            if n <= args.exact_cap:
                exact, _ = exact_dp(CostModel(trace), max_n=args.exact_cap)
            for seeder in seeders:
                t0 = time.perf_counter()
                res = run(trace, seeder=seeder, seed=args.seed, exact=False)
                rows.append({
                    "generator": gen_name,
                    "n": n,
                    "seeder": seeder,
                    "seed_peak": res["seed_peak"],
                    "refined_peak": res["refined_peak"],
                    "exact_opt": exact if exact is not None else "",
                    "ratio_to_opt": (res["refined_peak"] / exact if exact else ""),
                    "runtime": time.perf_counter() - t0,
                })
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
