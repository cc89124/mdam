"""Binary carving tree utilities.

Leaves are circuit axes. Internal nodes represent binary joins. The tree is a
carving decomposition candidate, not a branch decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import random


@dataclass
class TreeNode:
    axis: int | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @staticmethod
    def leaf(axis: int) -> "TreeNode":
        return TreeNode(axis=int(axis))

    @staticmethod
    def join(left: "TreeNode", right: "TreeNode") -> "TreeNode":
        return TreeNode(axis=None, left=left, right=right)

    @property
    def is_leaf(self) -> bool:
        return self.axis is not None

    def leaves(self) -> frozenset[int]:
        if self.is_leaf:
            return frozenset([int(self.axis)])
        return self.left.leaves() | self.right.leaves()

    def clone(self) -> "TreeNode":
        return copy.deepcopy(self)

    def to_dict(self):
        if self.is_leaf:
            return {"axis": int(self.axis)}
        return {"left": self.left.to_dict(), "right": self.right.to_dict()}

    @staticmethod
    def from_dict(d) -> "TreeNode":
        if "axis" in d:
            return TreeNode.leaf(int(d["axis"]))
        return TreeNode.join(TreeNode.from_dict(d["left"]), TreeNode.from_dict(d["right"]))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "TreeNode":
        return TreeNode.from_dict(json.loads(s))


def balanced_tree(axes) -> TreeNode:
    axes = list(map(int, axes))
    if not axes:
        raise ValueError("cannot build tree over empty axis set")
    if len(axes) == 1:
        return TreeNode.leaf(axes[0])
    mid = len(axes) // 2
    return TreeNode.join(balanced_tree(axes[:mid]), balanced_tree(axes[mid:]))


def caterpillar_tree(axes) -> TreeNode:
    axes = list(map(int, axes))
    if not axes:
        raise ValueError("cannot build tree over empty axis set")
    node = TreeNode.leaf(axes[0])
    for axis in axes[1:]:
        node = TreeNode.join(node, TreeNode.leaf(axis))
    return node


def random_tree(axes, seed=0) -> TreeNode:
    rng = random.Random(seed)
    nodes = [TreeNode.leaf(a) for a in axes]
    while len(nodes) > 1:
        i = rng.randrange(len(nodes))
        a = nodes.pop(i)
        j = rng.randrange(len(nodes))
        b = nodes.pop(j)
        nodes.append(TreeNode.join(a, b))
    return nodes[0]


def iter_nodes(node: TreeNode):
    yield node
    if not node.is_leaf:
        yield from iter_nodes(node.left)
        yield from iter_nodes(node.right)


def iter_internal(node: TreeNode):
    if not node.is_leaf:
        yield node
        yield from iter_internal(node.left)
        yield from iter_internal(node.right)


def tree_depth(node: TreeNode) -> int:
    if node.is_leaf:
        return 0
    return 1 + max(tree_depth(node.left), tree_depth(node.right))


def canonical_newick(node: TreeNode) -> str:
    if node.is_leaf:
        return str(node.axis)
    parts = sorted([canonical_newick(node.left), canonical_newick(node.right)])
    return "(" + ",".join(parts) + ")"


def replace_subtree(node: TreeNode, target_leaves: frozenset[int], replacement: TreeNode) -> TreeNode:
    if node.leaves() == target_leaves:
        return replacement.clone()
    if node.is_leaf:
        return node.clone()
    return TreeNode.join(
        replace_subtree(node.left, target_leaves, replacement),
        replace_subtree(node.right, target_leaves, replacement),
    )
