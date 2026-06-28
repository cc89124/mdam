"""End-to-end temporal-live carving layout pipeline."""

from __future__ import annotations

import argparse
import json
import time

from .cost import CostModel
from .exact import exact_dp
from .io import load_trace, save_tree
from .refine import refine
from .seed import build_seed


def run(trace, seeder="recursive_balanced_mincut", refine_moves=("nni", "spr"),
        seed=0, partitioner="networkx", exact=False, max_exact_n=20):
    cost = CostModel(trace)
    t0 = time.perf_counter()
    tree = build_seed(trace, seeder, seed=seed, partitioner=partitioner)
    seed_peak = cost.tree_peak(tree)
    accepted = 0
    refined_peak = seed_peak
    if refine_moves:
        tree, refined_peak, accepted = refine(tree, cost, moves=tuple(refine_moves), max_moves=100)
    exact_peak = None
    if exact:
        exact_peak, _ = exact_dp(cost, max_n=max_exact_n)
    return {
        "tree": tree,
        "seed_peak": seed_peak,
        "refined_peak": refined_peak,
        "accepted_moves": accepted,
        "exact_peak": exact_peak,
        "profile": cost.tree_profile(tree),
        "elapsed_s": time.perf_counter() - t0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True)
    p.add_argument("--seeder", default="recursive_balanced_mincut")
    p.add_argument("--refine", default="nni,spr")
    p.add_argument("--anneal", action="store_true", help="reserved; default off")
    p.add_argument("--exact", action="store_true")
    p.add_argument("--max-exact-n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--partitioner", default="networkx")
    p.add_argument("--out-tree", default="tree.json")
    args = p.parse_args()

    trace = load_trace(args.trace)
    moves = tuple(x for x in args.refine.split(",") if x)
    result = run(
        trace,
        seeder=args.seeder,
        refine_moves=moves,
        seed=args.seed,
        partitioner=args.partitioner,
        exact=args.exact,
        max_exact_n=args.max_exact_n,
    )
    save_tree(result["tree"], args.out_tree)
    printable = {k: v for k, v in result.items() if k != "tree"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
