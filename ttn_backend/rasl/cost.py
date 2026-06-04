"""TTN-layout proxy cost model for RASL candidates."""

from __future__ import annotations

from collections import deque
import math


class LayoutCost:
    def __init__(self, spec, homing):
        self.spec = spec
        self.homing = homing
        self.home = homing["home"]
        self.owned = {int(k): list(v) for k, v in homing["owned_phys"].items()}
        self.n_bags = int(spec["union"]["n_bags"])
        self.adj = {i: [] for i in range(self.n_bags)}
        self.sep_bits = {}
        for i, j, sep in spec["union"]["bag_edges"]:
            s = len(sep)
            self.adj[i].append(j)
            self.adj[j].append(i)
            self.sep_bits[tuple(sorted((i, j)))] = s
        self.r_resident = self._resident_exp()

    def _resident_exp(self) -> int:
        best = 0
        for b in range(self.n_bags):
            exp = len(self.owned.get(b, []))
            for nb in self.adj[b]:
                exp += self.sep_bits[tuple(sorted((b, nb)))]
            best = max(best, exp)
        return int(best)

    def tree_path_bags(self, src: int, dst: int) -> list[int]:
        if src == dst:
            return [src]
        parent = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                break
            for v in self.adj[u]:
                if v not in parent:
                    parent[v] = u
                    q.append(v)
        if dst not in parent:
            return []
        out = []
        cur = dst
        while cur is not None:
            out.append(cur)
            cur = parent[cur]
        return list(reversed(out))

    def path_cost_idents(self, u: int, v: int) -> float:
        hu = self.home[u]
        hv = self.home[v]
        if hu == hv:
            return 0.0
        path = self.tree_path_bags(hu, hv)
        return float(sum(self.sep_bits[tuple(sorted((a, b)))]
                         for a, b in zip(path, path[1:])))

    def workspace_idents(self, u: int, v: int) -> float:
        hu = self.home[u]
        hv = self.home[v]
        region = set(self.tree_path_bags(hu, hv))
        if not region:
            return 0.0
        p_sum = sum(len(self.owned.get(b, [])) for b in region)
        boundary = 0
        for b in region:
            for nb in self.adj[b]:
                if nb not in region:
                    boundary += self.sep_bits[tuple(sorted((b, nb)))]
        return float(p_sum + boundary)

    def score_candidate(self, cand, axis_to_ident: dict[int, int], default_resident: float | None = None):
        path = 0.0
        workspace = 0.0
        for op in cand.ops:
            if not op.is_2q():
                continue
            if op.a not in axis_to_ident or op.b not in axis_to_ident:
                continue
            u = axis_to_ident[op.a]
            v = axis_to_ident[op.b]
            path += self.path_cost_idents(u, v)
            workspace = max(workspace, self.workspace_idents(u, v))
        cand.proxy_path_cost = path
        cand.proxy_workspace = workspace
        cand.refactor_cost = path + cand.num_2q_ops()
        cand.proxy_resident_bound = self.r_resident if default_resident is None else default_resident
        return cand

