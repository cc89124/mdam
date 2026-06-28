import unittest

from temporal_carving.cost import CostModel, Trace
from temporal_carving.exact import exact_dp, fixed_tree_dp_decomposition_peak
from temporal_carving.refine import refine
from temporal_carving.seed import build_seed
from temporal_carving.synth import planted_temporal_masking, random_brickwork
from temporal_carving.tree import balanced_tree, canonical_newick, random_tree, TreeNode


def tiny_trace():
    return Trace(
        axes=(0, 1),
        dims={0: 2, 1: 2},
        timeline=(0, 1, 2),
        live_sets={
            0: frozenset([0, 1]),
            1: frozenset([0]),
            2: frozenset([0, 1]),
        },
        events={
            0: ((0, 1),),
            1: (),
            2: ((0, 1),),
        },
    )


def cap_trace():
    return Trace(
        axes=(0, 1),
        dims={0: 2, 1: 2},
        timeline=(0, 1, 2),
        live_sets={t: frozenset([0, 1]) for t in (0, 1, 2)},
        events={t: ((0, 1),) for t in (0, 1, 2)},
    )


class TestTemporalCarving(unittest.TestCase):
    def test_cost_unit_reset(self):
        cm = CostModel(tiny_trace())
        self.assertEqual(list(cm.live_cumulative_cut_pressure({0})), [1.0, 0.0, 1.0])
        self.assertEqual(list(cm.rhat({0})), [1.0, 0.0, 1.0])
        self.assertEqual(cm.nodecost({0}, {1}), 2.0)

    def test_cost_unit_min_cap(self):
        cm = CostModel(cap_trace())
        self.assertEqual(list(cm.live_cumulative_cut_pressure({0})), [1.0, 2.0, 3.0])
        # cap by ell({0}) = ell({1}) = 1
        self.assertEqual(list(cm.rhat({0})), [1.0, 1.0, 1.0])

    def test_objective_consistency(self):
        trace = random_brickwork(n=8, depth=10, seed=3)
        cm = CostModel(trace)
        for seed in range(5):
            tree = random_tree(trace.axes, seed=seed)
            self.assertAlmostEqual(cm.tree_peak(tree), fixed_tree_dp_decomposition_peak(cm, tree))

    def test_exact_vs_pipeline(self):
        for trace in (random_brickwork(n=8, depth=8, seed=1), planted_temporal_masking(n=8, seed=2)):
            cm = CostModel(trace)
            opt, _tree = exact_dp(cm, max_n=12)
            seed_tree = build_seed(trace, "recursive_balanced_mincut", seed=0)
            refined, peak, _moves = refine(seed_tree, cm, moves=("nni", "spr"), max_moves=20)
            self.assertGreaterEqual(peak + 1e-12, opt)
            self.assertLessEqual(peak / opt, 4.0)

    def test_temporal_advantage(self):
        trace = planted_temporal_masking(n=10, seed=0)
        cm = CostModel(trace)
        X = set(range(5))
        Y = set(range(5, 10))
        tree = TreeNode.join(balanced_tree(sorted(X)), balanced_tree(sorted(Y)))
        true_peak = cm.tree_peak(tree)
        union_peak = cm.union_graph_objective(tree)
        self.assertLess(true_peak, union_peak)
        self.assertGreater(union_peak - true_peak, 4.0)

        # Persistence case: one crossing at t=0 still contributes at t=1 while
        # both sides remain live, so instantaneous crossing underestimates.
        trace2 = Trace(
            axes=(0, 1),
            dims={0: 2, 1: 2},
            timeline=(0, 1),
            live_sets={0: frozenset([0, 1]), 1: frozenset([0, 1])},
            events={0: ((0, 1),), 1: ()},
        )
        cm2 = CostModel(trace2)
        self.assertEqual(list(cm2.rhat({0})), [1.0, 1.0])

    def test_seed_monotone_refine(self):
        trace = random_brickwork(n=10, depth=14, seed=9)
        cm = CostModel(trace)
        tree = build_seed(trace, "recursive_balanced_mincut", seed=5)
        seed_peak = cm.tree_peak(tree)
        _refined, refined_peak, _ = refine(tree, cm, moves=("nni", "spr"), max_moves=20)
        self.assertLessEqual(refined_peak, seed_peak + 1e-12)

    def test_reproducibility(self):
        trace = random_brickwork(n=10, depth=12, seed=2)
        a = build_seed(trace, "recursive_balanced_mincut", seed=11)
        b = build_seed(trace, "recursive_balanced_mincut", seed=11)
        self.assertEqual(canonical_newick(a), canonical_newick(b))


if __name__ == "__main__":
    unittest.main()
