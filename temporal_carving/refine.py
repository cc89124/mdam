"""NNI/SPR local refinement on the true objective."""

from __future__ import annotations

from .tree import TreeNode, iter_internal, iter_nodes, replace_subtree


def nni_candidates(tree: TreeNode):
    """Generate standard NNI alternatives around internal child edges."""
    out = []

    def walk(node):
        if node.is_leaf:
            return
        X, Y = node.left, node.right
        if not X.is_leaf:
            a, b, c = X.left, X.right, Y
            out.append(replace_subtree(tree, node.leaves(), TreeNode.join(TreeNode.join(a.clone(), c.clone()), b.clone())))
            out.append(replace_subtree(tree, node.leaves(), TreeNode.join(TreeNode.join(b.clone(), c.clone()), a.clone())))
        if not Y.is_leaf:
            a, b, c = Y.left, Y.right, X
            out.append(replace_subtree(tree, node.leaves(), TreeNode.join(TreeNode.join(a.clone(), c.clone()), b.clone())))
            out.append(replace_subtree(tree, node.leaves(), TreeNode.join(TreeNode.join(b.clone(), c.clone()), a.clone())))
        walk(node.left)
        walk(node.right)

    walk(tree)
    return out


def _all_subtree_leafsets(tree):
    return [n.leaves() for n in iter_nodes(tree) if n.leaves() != tree.leaves()]


def spr_candidates(tree: TreeNode, limit=200):
    """Generate a bounded set of SPR-like prune/regraft candidates.

    This is a conservative implementation: prune one subtree and regraft it by
    joining it with another existing subtree. It is still an SPR local move
    family, bounded for compile-time safety.
    """
    nodes = list(iter_nodes(tree))
    subtrees = [(n.leaves(), n.clone()) for n in nodes if n.leaves() != tree.leaves()]
    out = []
    for S, sub in subtrees:
        for T, target in subtrees:
            if S == T or S & T:
                continue
            repl = TreeNode.join(sub.clone(), target.clone())
            cand = replace_subtree(tree, T, repl)
            if cand.leaves() == tree.leaves():
                out.append(cand)
                if len(out) >= limit:
                    return out
    return out


def refine(tree: TreeNode, cost_model, moves=("nni", "spr"), max_moves=100, first_improvement=True):
    cur = tree.clone()
    cur_peak = cost_model.tree_peak(cur)
    accepted = 0
    while accepted < int(max_moves):
        cands = []
        if "nni" in moves:
            cands.extend(nni_candidates(cur))
        if "spr" in moves:
            cands.extend(spr_candidates(cur))
        best = (cur_peak, None)
        improved = False
        for cand in cands:
            try:
                peak = cost_model.tree_peak(cand)
            except Exception:
                continue
            if peak + 1e-12 < cur_peak:
                if first_improvement:
                    cur, cur_peak = cand, peak
                    accepted += 1
                    improved = True
                    break
                if peak < best[0]:
                    best = (peak, cand)
        if improved:
            continue
        if best[1] is not None:
            cur, cur_peak = best[1], best[0]
            accepted += 1
            continue
        break
    return cur, cur_peak, accepted
