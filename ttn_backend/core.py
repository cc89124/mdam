"""
ttn_backend.py -- Canonical TTN backend for Clifft.

Convention 1: each ident has EXACTLY ONE physical axis at its home bag.
Bag tensor axis order: [own_idents...] + [bond axes in neighbors order].
Bonds are static (every tree edge has a bond axis); when a side has no state,
the bond has size 1.

Invariants:
  I1. Each active ident appears in exactly one bag's own_idents.
  I2. For active ident u, u in bags[home[u]].own_idents.
  I3. For each tree edge (i,j), bag_i and bag_j have matching bond dim.

"""

from __future__ import annotations
import math
import os
import time
import gc
import numpy as np

from .clifford_frame import RegionLinearFrame


INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _thin_qr(M, rtol=1e-12, atol=1e-14):
    """Reduced QR with numerically-zero R rows removed.

    This is not a bond cap or approximate truncation policy; it only removes
    rows that QR has exposed as zero-rank within floating-point tolerance.
    """
    svd_rtol = os.environ.get("TTN_SVD_TRUNC_RTOL")
    if svd_rtol:
        min_elems = int(os.environ.get("TTN_SVD_TRUNC_MIN_MATRIX_ELEMS", "0"))
        if min_elems and int(M.size) < min_elems:
            svd_rtol = None
    if svd_rtol:
        rel = float(svd_rtol)
        abs_tol = float(os.environ.get("TTN_SVD_TRUNC_ATOL", str(atol)))
        try:
            U, s, Vh = np.linalg.svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            Q, R = np.linalg.qr(M, mode='reduced')
            return Q, R
        if s.size == 0:
            return U[:, :1], np.zeros((1, M.shape[1]), dtype=M.dtype)
        threshold = max(abs_tol, rel * float(s[0]))
        rank = int(np.count_nonzero(s > threshold))
        rank = max(rank, 1)
        R = s[:rank, None] * Vh[:rank, :]
        return U[:, :rank], R
    Q, R = np.linalg.qr(M, mode='reduced')
    if R.shape[0] == 0:
        return Q, R
    row_norms = np.linalg.norm(R, axis=1)
    scale = float(row_norms.max()) if row_norms.size else 0.0
    keep = row_norms > max(atol, rtol * scale)
    rank = int(np.count_nonzero(keep))
    if rank <= 0:
        rank = 1
    return Q[:, :rank], R[:rank, :]


def _edge_key(a, b):
    a = int(a); b = int(b)
    return (a, b) if a < b else (b, a)


def _prod(xs):
    out = 1
    for x in xs:
        out *= int(x)
    return int(out)


class TTNBag:
    def __init__(self, bag_id, neighbors):
        self.bag_id = bag_id
        self.neighbors = sorted(neighbors)
        self.own_idents = []
        shape = (1,) * len(self.neighbors) if self.neighbors else ()
        self._dense = np.ones(shape, dtype=np.complex128) if shape \
                      else np.array(1.0 + 0.0j, dtype=np.complex128)
        self._store = None  # BlockTensorStore when spilled out-of-core

    # `tensor` is a materializing view: reading it brings a spilled bag back
    # into RAM (and drops the store), so all existing op code is unchanged.
    # Metric/estimate/accounting code must use the non-materializing `shape`
    # and `resident_bytes` instead.
    @property
    def tensor(self):
        if self._dense is None and self._store is not None:
            self._dense = self._store.to_dense()
            self._store.close(unlink=True)
            self._store = None
        return self._dense

    @tensor.setter
    def tensor(self, value):
        if self._store is not None:
            self._store.close(unlink=True)
            self._store = None
        self._dense = value

    @property
    def is_spilled(self):
        return self._store is not None

    @property
    def shape(self):
        return self._store.shape if self._store is not None else self._dense.shape

    @property
    def resident_bytes(self):
        """RAM footprint: one block when spilled, full tensor when dense."""
        if self._store is not None:
            return int(self._store.ram_bytes)
        return int(self._dense.nbytes)

    @property
    def full_bytes(self):
        if self._store is not None:
            return int(self._store.ooc_bytes)
        return int(self._dense.nbytes)

    def spill(self, path, block_cap_bytes):
        """Move the dense tensor out-of-core, keeping only a block in RAM."""
        if self._store is not None or self._dense is None or self._dense.ndim == 0:
            return False
        from .block_tensor_store import (
            BlockTensorStore, choose_block_axis, block_size_for_cap,
        )
        arr = self._dense
        ax = choose_block_axis(arr.shape)
        if int(arr.shape[ax]) <= 1:
            return False
        bs = block_size_for_cap(arr.shape, ax, block_cap_bytes)
        self._store = BlockTensorStore.from_dense(arr, ax, bs, path)
        self._dense = None
        return True

    def n_own(self):
        return len(self.own_idents)

    def bond_axis_pos(self, neighbor_id):
        return self.n_own() + self.neighbors.index(neighbor_id)

    def ident_axis_pos(self, ident):
        return self.own_idents.index(ident)

    def __repr__(self):
        return (f"TTNBag(id={self.bag_id}, own={self.own_idents}, "
                f"nbrs={self.neighbors}, shape={self.shape})")


class TTNState:
    def __init__(self, bag_neighbors, home, capture_peak_snapshot=False,
                 trace_recorder=None):
        self.n_bags = len(bag_neighbors)
        self.bags = [TTNBag(i, bag_neighbors[i]) for i in range(self.n_bags)]
        self.home = dict(home)
        self.center_bag = None
        self.current_step = None
        self.current_op_kind = None
        self.executor_context = {}
        self.capture_peak_snapshot = bool(capture_peak_snapshot)
        self.trace_recorder = trace_recorder
        self._trace_event_index = 0
        self.metrics = {
            "peak_stored_bytes": self._stored_bytes(),
            "peak_pair_workspace_bytes": 0,
            "workspace_actual_peak_bytes": 0,
            "workspace_actual_peak_log2_numel": None,
            "actual_total_peak_bytes": self._stored_bytes(),
            "actual_total_peak_log2_numel": 0.0,
            "actual_total_peak_step": None,
            "actual_total_peak_kind": "init",
            "actual_total_peak_events": [],
            "destructive_total_peak_bytes": self._stored_bytes(),
            "destructive_total_peak_log2_numel": 0.0,
            "destructive_total_peak_step": None,
            "destructive_total_peak_kind": "init",
            "destructive_total_peak_debug": {},
            "resident_actual_peak_numel": 1,
            "resident_actual_peak_log2_numel": 0.0,
            "resident_actual_peak_bytes": 16,
            "actual_peak_offender_bag": None,
            "actual_peak_offender_step": None,
            "actual_peak_offender_shape": (),
            "actual_peak_offender_p_B": 0,
            "actual_peak_offender_incident_bond_dims": [],
            "actual_peak_offender_incident_edge_ids": [],
            "max_bond_dim": self._max_bond_dim(),
            "max_bond_dim_observed": self._max_bond_dim(),
            "max_separator_size_observed": self._max_separator_size(),
            "max_bag_degree_observed": self._max_bag_degree(),
            "edge_max_bond_dim": {},
            "edge_hit_count": {},
            "edge_rank_weighted_hits": {},
            "top5_bag_sizes": self._top_bag_sizes(),
            "top5_pair_workspace": [],
            "n_transports": 0,
            "n_qr": 0,
            "n_svd": 0,
            "num_path_contract": 0,
            "num_center_move": 0,
            "num_refactor": 0,
            "sum_path_length": 0,
            "sum_rank_weighted_path_length": 0.0,
            "sum_refactor_input_numel": 0,
            "max_refactor_input_numel": 0,
            "qr_work_proxy": 0.0,
            "multicnot_region_fused": 0,
            "multicnot_region_controls": 0,
            "multicnot_region_fallback": 0,
            "multicnot_region_workspace_peak_bytes": 0,
            "cap_infeasible_exact_count": 0,
            "cap_infeasible_exact_events": [],
            "num_frame_updates": 0,
            "num_frame_materializations": 0,
            "num_avoided_tensor_applies": 0,
            "num_avoided_open_close": 0,
            "num_avoided_qr_estimate": 0,
            "frame_lifted_windows": 0,
            "num_bag_fissions": 0,
            "bag_fission_events": [],
            "bag_fission_temp_peak_bytes": 0,
            "actual_step_peaks": {},
            "actual_step_workspace_peaks": {},
            "peak_snapshot": None,
        }
        self._record_metrics()

    def _jsonable(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (str, bool)) or obj is None:
            return obj
        if isinstance(obj, dict):
            return {str(k): self._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._jsonable(v) for v in obj]
        return str(obj)

    def _record_total_peak(self, total_bytes, kind="runtime", **debug):
        total_bytes = int(total_bytes)
        if total_bytes >= int(self.metrics.get("actual_total_peak_bytes", 0)):
            self.metrics["actual_total_peak_bytes"] = total_bytes
            elems = total_bytes / float(np.dtype(np.complex128).itemsize)
            self.metrics["actual_total_peak_log2_numel"] = (
                float(math.log2(elems)) if elems > 0 else None
            )
            self.metrics["actual_total_peak_step"] = self.current_step
            self.metrics["actual_total_peak_kind"] = kind
            peak_bag = self._current_peak_bag_actual()
            event = dict(
                step=self.current_step,
                opcode=self.current_op_kind,
                kind=kind,
                actual_total_peak_bytes=total_bytes,
                actual_total_peak_log2_numel=self.metrics["actual_total_peak_log2_numel"],
                stored_bytes=int(self._stored_bytes()),
                selected_executor=self.executor_context.get("selected_executor"),
                executor_context=self._jsonable(self.executor_context),
                debug=self._jsonable(debug),
                offender_bag=None if peak_bag is None else int(peak_bag["bag"]),
                offender_shape=None if peak_bag is None else list(peak_bag["shape"]),
                offender_p_B=None if peak_bag is None else int(peak_bag["p_B"]),
                offender_incident_bond_dims=(
                    None if peak_bag is None else list(peak_bag["incident_bond_dims"])
                ),
                offender_incident_edge_ids=(
                    None if peak_bag is None else list(peak_bag["incident_edge_ids"])
                ),
            )
            events = self.metrics.setdefault("actual_total_peak_events", [])
            events.append(event)
            if len(events) > 200:
                del events[:-200]

    def _record_destructive_total_peak(self, total_bytes, kind="runtime", **debug):
        total_bytes = int(total_bytes)
        if total_bytes >= int(self.metrics.get("destructive_total_peak_bytes", 0)):
            self.metrics["destructive_total_peak_bytes"] = total_bytes
            elems = total_bytes / float(np.dtype(np.complex128).itemsize)
            self.metrics["destructive_total_peak_log2_numel"] = (
                float(math.log2(elems)) if elems > 0 else None
            )
            self.metrics["destructive_total_peak_step"] = self.current_step
            self.metrics["destructive_total_peak_kind"] = kind
            self.metrics["destructive_total_peak_debug"] = {
                k: (int(v) if isinstance(v, (np.integer, int)) else v)
                for k, v in debug.items()
            }

    def _stored_bytes(self):
        return int(sum(b.resident_bytes for b in self.bags))

    def _max_bond_dim(self):
        m = 1
        for b in self.bags:
            bshape = b.shape
            for nb in b.neighbors:
                m = max(m, int(bshape[b.bond_axis_pos(nb)]))
        return m

    def _max_separator_size(self):
        dim = self._max_bond_dim()
        if dim <= 1:
            return 0
        return int(math.ceil(math.log2(dim)))

    def _max_bag_degree(self):
        return max((len(b.neighbors) for b in self.bags), default=0)

    def _record_edge_bonds(self):
        edge_dims = self.metrics.setdefault("edge_max_bond_dim", {})
        for b in self.bags:
            for nb in b.neighbors:
                if nb < b.bag_id:
                    continue
                dim = int(b.shape[b.bond_axis_pos(nb)])
                key = f"{b.bag_id}-{nb}"
                edge_dims[key] = max(int(edge_dims.get(key, 1)), dim)

    def _top_bag_sizes(self, k=5):
        rows = [
            dict(bag=b.bag_id, degree=len(b.neighbors), bytes=int(b.resident_bytes),
                 shape=tuple(int(x) for x in b.shape), spilled=bool(b.is_spilled))
            for b in self.bags
        ]
        rows.sort(key=lambda x: x["bytes"], reverse=True)
        return rows[:k]

    def _record_pair_workspace(self, src, dst, nbytes, k=5):
        if not nbytes:
            return
        rows = list(self.metrics.get("top5_pair_workspace", []))
        rows.append(dict(src=int(src), dst=int(dst), bytes=int(nbytes)))
        rows.sort(key=lambda x: x["bytes"], reverse=True)
        self.metrics["top5_pair_workspace"] = rows[:k]

    def _record_cap_infeasible_exact(self, total_bytes, cap_bytes, reason, **detail):
        self.metrics["cap_infeasible_exact_count"] = int(
            self.metrics.get("cap_infeasible_exact_count", 0)) + 1
        events = self.metrics.setdefault("cap_infeasible_exact_events", [])
        events.append(dict(
            step=self.current_step,
            opcode=self.current_op_kind,
            selected_executor=self.executor_context.get("selected_executor"),
            reason=str(reason),
            total_bytes=int(total_bytes),
            cap_bytes=int(cap_bytes),
            executor_context=self._jsonable(self.executor_context),
            detail=self._jsonable(detail),
        ))
        if len(events) > 500:
            del events[:-500]

    def _edge_rows(self):
        rows = []
        for b in self.bags:
            for nb in b.neighbors:
                if nb < b.bag_id:
                    continue
                dim = int(b.shape[b.bond_axis_pos(nb)])
                rows.append(dict(
                    edge_id=f"{b.bag_id}-{nb}",
                    a=int(b.bag_id),
                    b=int(nb),
                    chi=dim,
                    log2_chi=float(math.log2(dim)) if dim > 0 else None,
                    live=bool(dim > 1),
                ))
        return rows

    def _emit_trace(self, pair_workspace_bytes=0, pair_src=None, pair_dst=None):
        if self.trace_recorder is None:
            return
        bags = [self._bag_actual_row(b) for b in self.bags]
        edges = self._edge_rows()
        live_axes = sorted({
            int(ident)
            for b in self.bags
            for ident in b.own_idents
        })
        live_bags = sorted({
            int(row["bag"])
            for row in bags
            if int(row["p_B"]) > 0 or any(int(x) > 1 for x in row["incident_bond_dims"])
        })
        peak_bag = max(bags, key=lambda r: (int(r["numel"]), int(r["bytes"]), -int(r["bag"])))
        row = dict(
            event_index=int(self._trace_event_index),
            step_id=self.current_step,
            op_kind=self.current_op_kind,
            stored_bytes=int(self._stored_bytes()),
            pair_workspace_bytes=int(pair_workspace_bytes or 0),
            pair_src=None if pair_src is None else int(pair_src),
            pair_dst=None if pair_dst is None else int(pair_dst),
            live_axes=live_axes,
            live_bags=live_bags,
            bags=bags,
            edges=edges,
            peak_bag=peak_bag,
        )
        self._trace_event_index += 1
        self.trace_recorder(row)

    def _bag_actual_row(self, b):
        shape = tuple(int(x) for x in b.shape)
        numel = int(np.prod(shape)) if shape else 1
        bytes_ = int(b.full_bytes)
        bond_dims = []
        edge_ids = []
        for nb in b.neighbors:
            pos = b.bond_axis_pos(nb)
            dim = int(shape[pos]) if pos < len(shape) else 1
            bond_dims.append(dim)
            edge_ids.append(f"{min(b.bag_id, nb)}-{max(b.bag_id, nb)}")
        log2_numel = float(math.log2(numel)) if numel > 0 else float("-inf")
        return dict(
            bag=int(b.bag_id),
            p_B=int(b.n_own()),
            shape=shape,
            numel=numel,
            bytes=bytes_,
            log2_numel=log2_numel,
            incident_bond_dims=bond_dims,
            incident_edge_ids=edge_ids,
        )

    def _current_peak_bag_actual(self):
        rows = [self._bag_actual_row(b) for b in self.bags]
        if not rows:
            return None
        rows.sort(key=lambda r: (r["numel"], r["bytes"], -r["bag"]), reverse=True)
        return rows[0]

    def _record_actual_resident(self):
        row = self._current_peak_bag_actual()
        if row is None:
            return
        step = self.current_step
        op_kind = self.current_op_kind
        step_key = str(step) if step is not None else "init"
        prev_step = self.metrics["actual_step_peaks"].get(step_key)
        if prev_step is None or row["numel"] >= int(prev_step["resident_actual_peak_numel"]):
            self.metrics["actual_step_peaks"][step_key] = dict(
                step_id=step,
                op_kind=op_kind,
                resident_actual_peak_numel=row["numel"],
                resident_actual_peak_log2_numel=row["log2_numel"],
                resident_actual_peak_bytes=row["bytes"],
                peak_offender_bag=row["bag"],
                peak_offender_shape=list(row["shape"]),
                peak_offender_p_B=row["p_B"],
                peak_offender_incident_bond_dims=list(row["incident_bond_dims"]),
                peak_offender_incident_edge_ids=list(row["incident_edge_ids"]),
            )
        if row["numel"] >= int(self.metrics.get("resident_actual_peak_numel", 0)):
            self.metrics["resident_actual_peak_numel"] = row["numel"]
            self.metrics["resident_actual_peak_log2_numel"] = row["log2_numel"]
            self.metrics["resident_actual_peak_bytes"] = row["bytes"]
            self.metrics["actual_peak_offender_bag"] = row["bag"]
            self.metrics["actual_peak_offender_step"] = step
            self.metrics["actual_peak_offender_shape"] = list(row["shape"])
            self.metrics["actual_peak_offender_p_B"] = row["p_B"]
            self.metrics["actual_peak_offender_incident_bond_dims"] = list(row["incident_bond_dims"])
            self.metrics["actual_peak_offender_incident_edge_ids"] = list(row["incident_edge_ids"])
            if self.capture_peak_snapshot:
                self.metrics["peak_snapshot"] = dict(
                    step_id=step,
                    op_kind=op_kind,
                    bags=[
                        dict(
                            bag_id=int(b.bag_id),
                            neighbors=list(map(int, b.neighbors)),
                            own_idents=list(map(int, b.own_idents)),
                            tensor=b.tensor.copy(),
                        )
                        for b in self.bags
                    ],
                )

    def _record_metrics(self, pair_workspace_bytes=0, qr_count=0, transport_count=0,
                        pair_src=None, pair_dst=None, center_move_count=0,
                        workspace_live_in_total=True,
                        live_total_bytes_override=None,
                        stored_outside_open_regions=None,
                        absorbed_region_stored_bytes=0):
        stored = self._stored_bytes()
        live_ws = int(pair_workspace_bytes or 0) if workspace_live_in_total else 0
        if live_total_bytes_override is None:
            total_for_peak = stored + live_ws
            stored_outside_dbg = stored
        else:
            total_for_peak = int(live_total_bytes_override)
            stored_outside_dbg = int(stored_outside_open_regions or 0)
        self._record_total_peak(
            total_for_peak,
            kind="record_metrics",
            stored_outside_open_regions=stored_outside_dbg,
            absorbed_region_stored_bytes=int(absorbed_region_stored_bytes or 0),
            open_region_bytes=live_ws,
            temporary_bytes=0,
            pair_workspace_bytes=int(pair_workspace_bytes or 0),
            pair_src=None if pair_src is None else int(pair_src),
            pair_dst=None if pair_dst is None else int(pair_dst),
            workspace_live_in_total=bool(workspace_live_in_total),
        )
        self._record_destructive_total_peak(
            total_for_peak,
            kind="record_metrics",
            stored_outside_open_regions=stored_outside_dbg,
            absorbed_region_stored_bytes=int(absorbed_region_stored_bytes or 0),
            open_region_bytes=live_ws,
            temporary_bytes=0,
        )
        self._record_edge_bonds()
        self._record_actual_resident()
        if stored >= self.metrics["peak_stored_bytes"]:
            self.metrics["peak_stored_bytes"] = stored
            self.metrics["top5_bag_sizes"] = self._top_bag_sizes()
        self.metrics["peak_pair_workspace_bytes"] = max(
            self.metrics["peak_pair_workspace_bytes"], int(pair_workspace_bytes))
        self.metrics["workspace_actual_peak_bytes"] = max(
            self.metrics["workspace_actual_peak_bytes"], int(pair_workspace_bytes))
        if self.metrics["workspace_actual_peak_bytes"]:
            elems = self.metrics["workspace_actual_peak_bytes"] // np.dtype(np.complex128).itemsize
            self.metrics["workspace_actual_peak_log2_numel"] = (
                float(math.log2(elems)) if elems > 0 else None
            )
        if pair_src is not None and pair_dst is not None:
            self._record_pair_workspace(pair_src, pair_dst, pair_workspace_bytes)
        if pair_workspace_bytes:
            step_key = str(self.current_step) if self.current_step is not None else "init"
            prev_ws = self.metrics["actual_step_workspace_peaks"].get(step_key)
            if prev_ws is None or int(pair_workspace_bytes) >= int(prev_ws["workspace_actual_peak_bytes"]):
                self.metrics["actual_step_workspace_peaks"][step_key] = dict(
                    step_id=self.current_step,
                    op_kind=self.current_op_kind,
                    workspace_actual_peak_bytes=int(pair_workspace_bytes),
                    pair_src=None if pair_src is None else int(pair_src),
                    pair_dst=None if pair_dst is None else int(pair_dst),
                )
        max_bond = self._max_bond_dim()
        self.metrics["max_bond_dim"] = max(self.metrics["max_bond_dim"], max_bond)
        self.metrics["max_bond_dim_observed"] = max(
            self.metrics["max_bond_dim_observed"], max_bond)
        self.metrics["max_separator_size_observed"] = max(
            self.metrics["max_separator_size_observed"], self._max_separator_size())
        self.metrics["max_bag_degree_observed"] = max(
            self.metrics["max_bag_degree_observed"], self._max_bag_degree())
        self.metrics["n_qr"] += int(qr_count)
        self.metrics["n_transports"] += int(transport_count)
        self.metrics["num_path_contract"] += int(transport_count)
        self.metrics["num_center_move"] += int(center_move_count)
        self._emit_trace(
            pair_workspace_bytes=pair_workspace_bytes,
            pair_src=pair_src,
            pair_dst=pair_dst,
        )
        self.maybe_fission_large_bags()

    def _record_path_refactor(self, path):
        if not path or len(path) < 2:
            return
        self.metrics["num_refactor"] += 1
        path_length = len(path) - 1
        self.metrics["sum_path_length"] += int(path_length)
        rank_weight = 0.0
        edge_hits = self.metrics.setdefault("edge_hit_count", {})
        weighted = self.metrics.setdefault("edge_rank_weighted_hits", {})
        for a, b in zip(path, path[1:]):
            ba = self.bags[a]
            dim = int(ba.shape[ba.bond_axis_pos(b)])
            log_dim = float(math.log2(dim)) if dim > 0 else 0.0
            rank_weight += log_dim
            key = f"{min(a, b)}-{max(a, b)}"
            edge_hits[key] = int(edge_hits.get(key, 0)) + 1
            weighted[key] = float(weighted.get(key, 0.0)) + log_dim
        self.metrics["sum_rank_weighted_path_length"] += rank_weight

    def check_invariant_I1(self):
        seen = {}
        for b in self.bags:
            for ident in b.own_idents:
                if ident in seen:
                    raise AssertionError(
                        f"I1: ident {ident} in B{seen[ident]} and B{b.bag_id}")
                seen[ident] = b.bag_id

    def check_invariant_I2(self):
        for b in self.bags:
            for ident in b.own_idents:
                if self.home.get(ident) != b.bag_id:
                    raise AssertionError(
                        f"I2: ident {ident} in B{b.bag_id} but home={self.home.get(ident)}")

    def check_invariant_I3(self):
        for b in self.bags:
            for nb in b.neighbors:
                if nb <= b.bag_id: continue
                other = self.bags[nb]
                dh = b.shape[b.bond_axis_pos(nb)]
                do = other.shape[other.bond_axis_pos(b.bag_id)]
                if dh != do:
                    raise AssertionError(
                        f"I3: edge B{b.bag_id}-B{nb}, dim_here={dh}, dim_other={do}")

    def check_all_invariants(self):
        self.check_invariant_I1()
        self.check_invariant_I2()
        self.check_invariant_I3()

    def _axis_labels_for_bag(self, bag_id):
        bag = self.bags[int(bag_id)]
        labels = [('own', int(x)) for x in bag.own_idents]
        labels += [('bond', int(nb)) for nb in bag.neighbors]
        return labels

    def _reorder_tensor_to_neighbors(self, bag_id, tensor, labels):
        bag = self.bags[int(bag_id)]
        target = [('own', int(x)) for x in bag.own_idents]
        target += [('bond', int(nb)) for nb in bag.neighbors]
        missing = [x for x in target if x not in labels]
        extra = [x for x in labels if x not in target]
        if missing or extra:
            raise RuntimeError(
                f"reorder B{bag_id}: label mismatch missing={missing} extra={extra}")
        return np.transpose(tensor, [labels.index(x) for x in target])

    def _replace_neighbor_reference(self, bag_id, old_nb, new_nb):
        bag = self.bags[int(bag_id)]
        old_labels = self._axis_labels_for_bag(bag_id)
        new_neighbors = [int(new_nb) if int(x) == int(old_nb) else int(x)
                         for x in bag.neighbors]
        if len(set(new_neighbors)) != len(new_neighbors):
            raise RuntimeError(f"B{bag_id}: duplicate neighbor after retarget")
        bag.neighbors = sorted(new_neighbors)
        new_labels = []
        for lab in old_labels:
            if lab[0] == 'bond' and int(lab[1]) == int(old_nb):
                new_labels.append(('bond', int(new_nb)))
            else:
                new_labels.append(lab)
        bag.tensor = self._reorder_tensor_to_neighbors(bag_id, bag.tensor, new_labels)

    def _best_exact_bond_fission(self, bag_id):
        bag = self.bags[int(bag_id)]
        n_own = bag.n_own()
        n_bonds = len(bag.neighbors)
        if n_bonds < 2:
            return None
        if bag.tensor.ndim != n_own + n_bonds:
            return None
        own_idx = tuple(range(n_own))
        bond_idx = tuple(range(n_own, n_own + n_bonds))
        old_numel = int(bag.tensor.size)
        best = None
        best_key = None
        # Keep all physical axes on the original bag. Split incident bond axes only.
        # Canonicalize by forcing the first bond to stay on the old bag.
        rest = list(range(1, n_bonds))
        for mask in range(1 << len(rest)):
            left_bonds = [0]
            right_bonds = []
            for bit, local_bond in enumerate(rest, start=1):
                if (mask >> (bit - 1)) & 1:
                    left_bonds.append(local_bond)
                else:
                    right_bonds.append(local_bond)
            if not right_bonds:
                continue
            left = own_idx + tuple(bond_idx[i] for i in left_bonds)
            right = tuple(bond_idx[i] for i in right_bonds)
            order = list(left) + list(right)
            T = np.transpose(bag.tensor, order)
            left_shape = T.shape[:len(left)]
            right_shape = T.shape[len(left):]
            d_left = _prod(left_shape) if left_shape else 1
            d_right = _prod(right_shape) if right_shape else 1
            M = T.reshape(d_left, d_right)
            U, s, Vh = np.linalg.svd(M, full_matrices=False)
            if s.size:
                threshold = max(1e-14, 1e-12 * float(s[0]))
                rank = max(1, int(np.count_nonzero(s > threshold)))
            else:
                rank = 1
            peak_numel = max(d_left * rank, rank * d_right)
            total_numel = d_left * rank + rank * d_right
            if peak_numel >= old_numel:
                continue
            key = (peak_numel, total_numel, rank)
            if best_key is None or key < best_key:
                best_key = key
                best = dict(
                    left=left,
                    right=right,
                    left_bond_locals=left_bonds,
                    right_bond_locals=right_bonds,
                    left_shape=left_shape,
                    right_shape=right_shape,
                    rank=rank,
                    U=U[:, :rank],
                    s=s[:rank],
                    Vh=Vh[:rank, :],
                    matrix_shape=(d_left, d_right),
                    peak_numel=peak_numel,
                    total_numel=total_numel,
                    temp_bytes=int(M.nbytes),
                )
        return best

    def fission_bag_exact(self, bag_id, min_gain=1.05):
        bag_id = int(bag_id)
        bag = self.bags[bag_id]
        old_tensor = bag.tensor
        old_neighbors = list(map(int, bag.neighbors))
        old_own = list(map(int, bag.own_idents))
        old_numel = int(old_tensor.size)
        cand = self._best_exact_bond_fission(bag_id)
        if cand is None:
            return False
        if old_numel / int(cand["peak_numel"]) < float(min_gain):
            return False

        new_id = len(self.bags)
        rank = int(cand["rank"])
        sqrt_s = np.sqrt(cand["s"])
        left_tensor = (cand["U"] * sqrt_s[None, :]).reshape(cand["left_shape"] + (rank,))
        right_tensor = (sqrt_s[:, None] * cand["Vh"]).reshape((rank,) + cand["right_shape"])

        left_neighbor_set = {old_neighbors[i] for i in cand["left_bond_locals"]}
        right_neighbor_set = {old_neighbors[i] for i in cand["right_bond_locals"]}

        # Add new empty bag.
        new_bag = TTNBag(new_id, [])
        new_bag.own_idents = []
        self.bags.append(new_bag)
        self.n_bags += 1

        # Update topology.
        bag.neighbors = sorted(left_neighbor_set | {new_id})
        new_bag.neighbors = sorted(right_neighbor_set | {bag_id})
        for nb in sorted(right_neighbor_set):
            self._replace_neighbor_reference(nb, bag_id, new_id)

        old_labels = [('own', int(x)) for x in old_own]
        old_labels += [('bond', int(nb)) for nb in old_neighbors]
        left_labels = [old_labels[i] for i in cand["left"]] + [('bond', new_id)]
        right_labels = [('bond', bag_id)] + [old_labels[i] for i in cand["right"]]

        bag.own_idents = old_own
        bag.tensor = self._reorder_tensor_to_neighbors(bag_id, left_tensor, left_labels)
        new_bag.tensor = self._reorder_tensor_to_neighbors(new_id, right_tensor, right_labels)
        if self.center_bag == bag_id:
            self.center_bag = bag_id

        event = dict(
            step=self.current_step,
            opcode=self.current_op_kind,
            old_bag=bag_id,
            new_bag=new_id,
            old_bytes=int(old_tensor.nbytes),
            old_shape=list(map(int, old_tensor.shape)),
            old_neighbors=old_neighbors,
            left_neighbors=sorted(left_neighbor_set),
            right_neighbors=sorted(right_neighbor_set),
            internal_bond_dim=rank,
            new_peak_child_bytes=max(int(bag.tensor.nbytes), int(new_bag.tensor.nbytes)),
            new_total_child_bytes=int(bag.tensor.nbytes + new_bag.tensor.nbytes),
            temp_bytes=int(cand["temp_bytes"]),
            matrix_shape=list(map(int, cand["matrix_shape"])),
        )
        self.metrics["num_bag_fissions"] = int(self.metrics.get("num_bag_fissions", 0)) + 1
        self.metrics.setdefault("bag_fission_events", []).append(event)
        self.metrics["bag_fission_temp_peak_bytes"] = max(
            int(self.metrics.get("bag_fission_temp_peak_bytes", 0)), int(cand["temp_bytes"]))
        # Temporal-carving layouts may use a static home map that is not identical
        # to the current owning bag for every active axis. Bag fission only changes
        # local topology and bond shapes, so validate the invariants it is
        # responsible for instead of enforcing the stronger homing invariant here.
        self.check_invariant_I1()
        self.check_invariant_I3()
        return True

    def _estimate_transport_workspace_bytes(self, ident, src, dst):
        """Estimate theta workspace for moving ident across adjacent edge src->dst."""
        src = int(src); dst = int(dst); ident = int(ident)
        if dst not in self.bags[src].neighbors:
            return None
        src_bag = self.bags[src]
        dst_bag = self.bags[dst]
        if ident not in src_bag.own_idents:
            return None
        src_shape = src_bag.shape
        dst_shape = dst_bag.shape
        left_dim = 1
        for x in src_bag.own_idents:
            if int(x) != ident:
                left_dim *= int(src_shape[src_bag.ident_axis_pos(x)])
        for nb in src_bag.neighbors:
            if int(nb) != dst:
                left_dim *= int(src_shape[src_bag.bond_axis_pos(nb)])
        right_dim = 1
        for x in dst_bag.own_idents:
            if int(x) != ident:
                right_dim *= int(dst_shape[dst_bag.ident_axis_pos(x)])
        for nb in dst_bag.neighbors:
            if int(nb) != src:
                right_dim *= int(dst_shape[dst_bag.bond_axis_pos(nb)])
        return int(left_dim * 2 * right_dim * np.dtype(np.complex128).itemsize)

    def maybe_prefission_transport_edge(self, ident, src, dst):
        cap_env = os.environ.get("TTN_PREFISSION_TRANSPORT_CAP_BYTES")
        if not cap_env or getattr(self, "_in_bag_fission", False):
            return False
        est = self._estimate_transport_workspace_bytes(ident, src, dst)
        if est is None or est <= int(cap_env):
            return False
        min_gain = float(os.environ.get("TTN_PREFISSION_MIN_GAIN", "1.01"))
        candidates = sorted(
            [int(src), int(dst)],
            key=lambda b: int(self.bags[b].full_bytes),
            reverse=True,
        )
        self.metrics["prefission_transport_attempts"] = int(
            self.metrics.get("prefission_transport_attempts", 0)) + 1
        for bag_id in candidates:
            before_neighbors = tuple(self.bags[bag_id].neighbors)
            before_bytes = int(self.bags[bag_id].full_bytes)
            if self.fission_bag_exact(bag_id, min_gain=min_gain):
                self.metrics["prefission_transport_success"] = int(
                    self.metrics.get("prefission_transport_success", 0)) + 1
                self.metrics.setdefault("prefission_transport_events", []).append(dict(
                    step=self.current_step,
                    opcode=self.current_op_kind,
                    ident=int(ident),
                    src=int(src),
                    dst=int(dst),
                    fission_bag=int(bag_id),
                    estimated_workspace_bytes=int(est),
                    old_bag_bytes=int(before_bytes),
                    old_neighbors=list(map(int, before_neighbors)),
                ))
                return True
        self.metrics["prefission_transport_failed"] = int(
            self.metrics.get("prefission_transport_failed", 0)) + 1
        return False

    def maybe_fission_large_bags(self):
        cap_env = os.environ.get("TTN_BAG_FISSION_CAP_BYTES")
        # The staged-transport output-fission policy reuses the same exact local
        # bond fission, keyed off the global exact memory cap, so that Q/R output
        # bags exceeding the cap are stored as microtrees.
        if not cap_env and os.environ.get("TTN_STAGED_OUTPUT_FISSION", "0") not in (
                "", "0", "false", "False"):
            cap_env = os.environ.get("TTN_EXACT_TOTAL_CAP_BYTES")
        if (not cap_env or getattr(self, "_in_bag_fission", False)
                or getattr(self, "_suspend_bag_fission", False)):
            return
        if self.current_step is None:
            return
        cap = int(cap_env)
        max_axes = int(os.environ.get("TTN_BAG_FISSION_MAX_AXES", "6"))
        min_gain = float(os.environ.get("TTN_BAG_FISSION_MIN_GAIN", "1.05"))
        self._in_bag_fission = True
        try:
            while True:
                offenders = [
                    b for b in self.bags
                    if int(b.full_bytes) > cap and len(b.shape) <= max_axes
                ]
                if not offenders:
                    break
                bag = max(offenders, key=lambda x: int(x.full_bytes))
                if not self.fission_bag_exact(bag.bag_id, min_gain=min_gain):
                    break
        finally:
            self._in_bag_fission = False

    # ---- resident out-of-core streaming -----------------------------------
    def _resident_stream_cap(self):
        v = os.environ.get("TTN_RESIDENT_STREAM_CAP_BYTES")
        return int(v) if v else None

    def _spill_dir_path(self):
        d = getattr(self, "_spill_dir", None)
        if d is None:
            import tempfile
            d = tempfile.mkdtemp(prefix="ttn_resident_")
            self._spill_dir = d
        return d

    def spill_idle_large_bags(self, keep=(), keep_center=True, boundary=False):
        """Move dense bags larger than the resident cap out-of-core. `keep` is a
        set of bag ids the current op needs (its operands); those stay dense.
        The block cache stays in RAM; the full tensor lives on disk until an op
        materializes it again.

        Called both at instruction boundaries and, more importantly, before each
        transport's metric is recorded — the 67 MB co-residence forms *within* a
        single MULTI_CNOT instruction, so boundary-only spilling cannot catch it."""
        cap = self._resident_stream_cap()
        if cap is None or getattr(self, "_in_bag_fission", False):
            return
        block_cap = int(os.environ.get("TTN_RESIDENT_STREAM_BLOCK_BYTES",
                                       str(max(cap // 8, 4 * 1024 * 1024))))
        if boundary:
            bmax = max((int(b.full_bytes) for b in self.bags), default=0)
            self.metrics["resident_boundary_max_bag_bytes"] = max(
                int(self.metrics.get("resident_boundary_max_bag_bytes", 0)), bmax)
        keep = set(int(x) for x in keep)
        d = None
        for b in self.bags:
            if b.is_spilled or int(b.full_bytes) <= cap:
                continue
            if b.bag_id in keep:
                continue
            if keep_center and b.bag_id == self.center_bag:
                continue
            if d is None:
                d = self._spill_dir_path()
            path = os.path.join(d, f"bag_{b.bag_id}_{self.current_step}.dat")
            if b.spill(path, block_cap):
                self.metrics["resident_num_spills"] = int(
                    self.metrics.get("resident_num_spills", 0)) + 1
        ooc = self.out_of_core_bytes()
        self.metrics["resident_out_of_core_peak_bytes"] = max(
            int(self.metrics.get("resident_out_of_core_peak_bytes", 0)), int(ooc))

    def out_of_core_bytes(self):
        return int(sum(b.full_bytes for b in self.bags if b.is_spilled))

    def cleanup_spills(self):
        for b in self.bags:
            if b.is_spilled:
                b._store.close(unlink=True)
                b._store = None
        d = getattr(self, "_spill_dir", None)
        if d and os.path.isdir(d):
            try:
                os.rmdir(d)
            except OSError:
                pass
        self._spill_dir = None

    def contract_into_one(self):
        """BFS-contract entire TTN into a single dense tensor.
        Returns (tensor, ordered_idents)."""
        start = None
        for b in self.bags:
            if b.own_idents:
                start = b.bag_id; break
        if start is None:
            return np.array(1.0+0j, dtype=np.complex128), []

        merged = self.bags[start].tensor.copy()
        merged_own = list(self.bags[start].own_idents)
        merged_bonds = list(self.bags[start].neighbors)
        absorbed = {start}
        frontier = list(self.bags[start].neighbors)

        while frontier:
            nb_id = frontier.pop(0)
            if nb_id in absorbed: continue
            nb = self.bags[nb_id]
            connect_to = next((a for a in absorbed if a in nb.neighbors), None)
            if connect_to is None: continue

            cur_bond_pos = len(merged_own) + merged_bonds.index(nb_id)
            nb_bond_pos = len(nb.own_idents) + nb.neighbors.index(connect_to)

            m_t = np.moveaxis(merged, cur_bond_pos, -1)
            n_t = np.moveaxis(nb.tensor, nb_bond_pos, 0)
            new = np.tensordot(m_t, n_t, axes=([m_t.ndim - 1], [0]))

            new_merged_bonds = [b for b in merged_bonds if b != nb_id]
            new_merged_bonds += [b for b in nb.neighbors if b != connect_to]

            n_old_own = len(merged_own)
            n_old_bonds = len(merged_bonds) - 1
            n_new_own = len(nb.own_idents)
            if n_new_own > 0:
                src = list(range(n_old_own + n_old_bonds,
                                 n_old_own + n_old_bonds + n_new_own))
                dst = list(range(n_old_own, n_old_own + n_new_own))
                new = np.moveaxis(new, src, dst)
            merged = new
            merged_own = merged_own + list(nb.own_idents)
            merged_bonds = new_merged_bonds
            absorbed.add(nb_id)
            for x in nb.neighbors:
                if x not in absorbed and x not in frontier:
                    frontier.append(x)

        # Squeeze trivial remaining bonds
        n_own_total = len(merged_own)
        for i in reversed(range(n_own_total, merged.ndim)):
            if merged.shape[i] == 1:
                merged = np.squeeze(merged, axis=i)
            else:
                merged = merged.sum(axis=i)

        return merged, merged_own


# ==========================================================================
# Expand handler
# ==========================================================================

def _expand_method(self, ident):
    """Promote `ident` as a new physical axis in |+> at its home bag.
    
    home[ident] must be set externally before calling this.
    New axis is inserted at position n_own (i.e., at the END of own block,
    BEFORE bond axes). After the call, own_idents has ident appended.
    
    The new |+> axis is uncorrelated with everything else, so tensor doubles:
       T_new[..., x, b...] = T_old[..., b...] * (1/sqrt(2))    for x in {0, 1}
    
    Bond dims are unchanged.
    """
    home_id = self.home.get(ident)
    if home_id is None:
        raise ValueError(f"expand: ident {ident} has no home")
    bag = self.bags[home_id]
    if ident in bag.own_idents:
        raise ValueError(f"expand: ident {ident} already in B{home_id}")

    # Insert axis at position n_own (before any bond axes)
    new_axis_pos = bag.n_own()
    T = bag.tensor
    if T.ndim == 0:
        # scalar -> shape (2,)
        bag.tensor = np.array([INV_SQRT2, INV_SQRT2], dtype=np.complex128) * T
    else:
        # broadcast: new shape = T.shape[:new_axis_pos] + (2,) + T.shape[new_axis_pos:]
        new_shape = T.shape[:new_axis_pos] + (2,) + T.shape[new_axis_pos:]
        new_T = np.empty(new_shape, dtype=np.complex128)
        sl0 = [slice(None)] * len(new_shape); sl0[new_axis_pos] = 0
        sl1 = [slice(None)] * len(new_shape); sl1[new_axis_pos] = 1
        new_T[tuple(sl0)] = T * INV_SQRT2
        new_T[tuple(sl1)] = T * INV_SQRT2
        bag.tensor = new_T

    bag.own_idents.append(ident)

    # Initialize center if first expand
    if self.center_bag is None:
        self.center_bag = home_id
    self._record_metrics()


def _apply_diag_method(self, ident, c0, c1):
    """Multiply axis of `ident` by diag(c0, c1) at its home bag."""
    home_id = self.home[ident]
    bag = self.bags[home_id]
    pos = bag.ident_axis_pos(ident)
    sl0 = [slice(None)] * bag.tensor.ndim; sl0[pos] = 0
    sl1 = [slice(None)] * bag.tensor.ndim; sl1[pos] = 1
    bag.tensor[tuple(sl0)] *= c0
    bag.tensor[tuple(sl1)] *= c1
    self._record_metrics()


def _apply_1q_method(self, ident, U):
    """Apply 2x2 unitary U to ident's axis at home bag."""
    home_id = self.home[ident]
    bag = self.bags[home_id]
    pos = bag.ident_axis_pos(ident)
    T = np.moveaxis(bag.tensor, pos, -1)
    T = np.matmul(T, U.T)
    bag.tensor = np.moveaxis(T, -1, pos)
    self._record_metrics()


# Attach
TTNState.expand = _expand_method
TTNState.apply_diag = _apply_diag_method
TTNState.apply_1q = _apply_1q_method


# ==========================================================================
# Canonical center + QR sweep
# ==========================================================================

def _tree_path(self, src, dst):
    """BFS path in bag tree from src to dst."""
    if src == dst:
        return [src]
    parent = {src: None}
    queue = [src]
    while queue:
        u = queue.pop(0)
        if u == dst:
            break
        for v in self.bags[u].neighbors:
            if v not in parent:
                parent[v] = u
                queue.append(v)
    if dst not in parent:
        return None
    path = []
    cur = dst
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return list(reversed(path))


def _shift_center_one_edge(self, src, dst):
    """Move canonical center from src to dst (must be adjacent).
    
    src.tensor is reshaped as (rest | bond_to_dst). QR: M = Q R.
      Q (rest_dim, chi_new) -> new src.tensor (rest_shape, chi_new).
      R (chi_new, chi_old) -> absorbed into dst's bond_to_src axis.
    """
    src_bag = self.bags[src]
    dst_bag = self.bags[dst]
    if dst not in src_bag.neighbors:
        raise ValueError(f"shift_center: B{dst} not adjacent to B{src}")

    # 1) Move src.tensor's bond_to_dst axis to last
    bond_pos = src_bag.bond_axis_pos(dst)
    T = np.moveaxis(src_bag.tensor, bond_pos, -1)
    rest_shape = T.shape[:-1]
    chi_old = T.shape[-1]
    rest_dim = int(np.prod(rest_shape)) if rest_shape else 1

    # 2) Reshape and QR
    M = T.reshape(rest_dim, chi_old)
    Q, R = _thin_qr(M)   # Q: (rest_dim, chi_new), R: (chi_new, chi_old)
    chi_new = Q.shape[1]
    self._record_metrics(qr_count=1)

    # 3) Q -> src.tensor (rest_shape + (chi_new,))
    new_src_T = Q.reshape(rest_shape + (chi_new,))
    # Move last axis back to bond_pos
    new_src_T = np.moveaxis(new_src_T, -1, bond_pos)
    src_bag.tensor = new_src_T

    # 4) R -> absorb into dst's bond_to_src axis.
    # dst.tensor has bond_to_src axis at position dst_bag.bond_axis_pos(src) -- size chi_old.
    # We need to contract R (chi_new, chi_old) along chi_old with dst.tensor's bond axis.
    dst_bond_pos = dst_bag.bond_axis_pos(src)
    D = np.moveaxis(dst_bag.tensor, dst_bond_pos, 0)
    # D shape: (chi_old, ...rest...)
    new_D = np.tensordot(R, D, axes=([1], [0]))
    # new_D shape: (chi_new, ...rest...)
    new_D = np.moveaxis(new_D, 0, dst_bond_pos)
    dst_bag.tensor = new_D

    self.center_bag = dst
    self._record_metrics(center_move_count=1)


def _move_center_method(self, target):
    """Move canonical center to target bag, via tree path."""
    if self.center_bag is None:
        # No state; nothing to do
        self.center_bag = target
        return
    if self.center_bag == target:
        return
    path = self._tree_path(self.center_bag, target)
    if path is None:
        raise RuntimeError(f"no path from B{self.center_bag} to B{target}")
    for i in range(len(path) - 1):
        self._shift_center_one_edge(path[i], path[i+1])


TTNState._tree_path = _tree_path
TTNState._shift_center_one_edge = _shift_center_one_edge
TTNState.move_center = _move_center_method


# ==========================================================================
# Measurement
# ==========================================================================

def _measure_z_method(self, ident, rng):
    """Measure `ident` in Z basis. Returns outcome (0 or 1).
    
    Procedure:
      1. move_center(home[ident])
      2. compute P(x_ident = 1) from home tensor: |tensor with that axis=1|^2 sum
      3. sample outcome
      4. project + renormalize
      5. remove ident axis
    """
    home_id = self.home[ident]
    self.move_center(home_id)
    bag = self.bags[home_id]
    pos = bag.ident_axis_pos(ident)

    # P(x=1): contract |T|^2 over all other axes with x=1
    sl1 = [slice(None)] * bag.tensor.ndim
    sl1[pos] = 1
    slab1 = bag.tensor[tuple(sl1)]
    p1 = float(np.vdot(slab1.ravel(), slab1.ravel()).real)

    # Sanity: total norm should be 1
    total = float(np.vdot(bag.tensor.ravel(), bag.tensor.ravel()).real)
    if total > 0:
        p1 = p1 / total
    p1 = max(0.0, min(1.0, p1))

    outcome = 1 if rng.random() < p1 else 0
    p = p1 if outcome == 1 else (1.0 - p1)
    if p < 1e-300:
        raise RuntimeError(f"measure: impossible outcome p={p}")

    # Project: slice tensor at ident axis = outcome, divide by sqrt(p)
    sl = [slice(None)] * bag.tensor.ndim
    sl[pos] = outcome
    bag.tensor = bag.tensor[tuple(sl)] / math.sqrt(p)
    del bag.own_idents[pos]
    self._record_metrics()

    # ident is no longer in any bag's own_idents; remove from home
    # (we keep home dict as-is for record; just no longer active)

    return outcome


TTNState.measure_z = _measure_z_method


# ==========================================================================
# Two-qubit gates
# ==========================================================================

def _apply_2q_local_method(self, bag_id, ident_a, ident_b, U4):
    """Apply a 4x4 unitary to two physical axes currently in one bag."""
    self.move_center(bag_id)
    bag = self.bags[bag_id]
    pa = bag.ident_axis_pos(ident_a)
    pb = bag.ident_axis_pos(ident_b)
    T = np.moveaxis(bag.tensor, [pa, pb], [-2, -1])
    sh = T.shape
    out = np.matmul(T.reshape(-1, 4), U4.T)
    T = out.reshape(sh)
    bag.tensor = np.moveaxis(T, [-2, -1], [pa, pb])
    self._record_metrics()


def _apply_2q_class_A_method(self, ident_a, ident_b, U4):
    """Both idents have the same static home bag. Apply 4x4 unitary locally."""
    home = self.home[ident_a]
    assert self.home[ident_b] == home, "class A requires same home"
    self._apply_2q_local(home, ident_a, ident_b, U4)


def _connected_region_for_bags_method(self, bag_ids):
    """Return the minimal connected bag-tree region spanning `bag_ids`."""
    ids = [int(x) for x in bag_ids]
    if not ids:
        return set()
    region = {ids[0]}
    for dst in ids[1:]:
        src = next(iter(region))
        best_path = None
        for r in region:
            p = self._tree_path(r, dst)
            if p is not None and (best_path is None or len(p) < len(best_path)):
                best_path = p
        if best_path is None:
            raise RuntimeError(f"no tree path to bag {dst}")
        region.update(best_path)
    return set(region)


def _region_labels_for_bag(self, bag_id):
    bag = self.bags[int(bag_id)]
    labels = [('own', int(x)) for x in bag.own_idents]
    labels += [('bond',) + _edge_key(bag.bag_id, nb) for nb in bag.neighbors]
    return labels


def _estimate_region_workspace_bytes_method(self, region):
    """Workspace size after contracting internal bonds of `region`."""
    region = set(map(int, region))
    own = 0
    boundary_dims = []
    seen_boundary = set()
    for bid in region:
        bag = self.bags[bid]
        own += bag.n_own()
        for nb in bag.neighbors:
            if nb in region:
                continue
            key = _edge_key(bid, nb)
            if key in seen_boundary:
                continue
            seen_boundary.add(key)
            boundary_dims.append(int(bag.shape[bag.bond_axis_pos(nb)]))
    return int((2 ** own) * _prod(boundary_dims) * np.dtype(np.complex128).itemsize)


def _contract_region_method(self, region):
    """Contract a connected region into one tensor with labeled open axes."""
    region = set(map(int, region))
    if not region:
        raise ValueError("contract_region: empty region")
    root = min(region)
    tensor = self.bags[root].tensor.copy()
    labels = self._region_labels_for_bag(root)
    absorbed = {root}
    frontier = [nb for nb in self.bags[root].neighbors if nb in region]

    while frontier:
        nb = frontier.pop(0)
        if nb in absorbed:
            continue
        parent = next((x for x in self.bags[nb].neighbors if x in absorbed), None)
        if parent is None:
            frontier.append(nb)
            continue
        edge_label = ('bond',) + _edge_key(parent, nb)
        cur_axis = labels.index(edge_label)
        nb_labels = self._region_labels_for_bag(nb)
        nb_axis = nb_labels.index(edge_label)
        left = np.moveaxis(tensor, cur_axis, -1)
        right = np.moveaxis(self.bags[nb].tensor, nb_axis, 0)
        tensor = np.tensordot(left, right, axes=([left.ndim - 1], [0]))
        labels = [x for x in labels if x != edge_label] + [
            x for x in nb_labels if x != edge_label
        ]
        absorbed.add(nb)
        for x in self.bags[nb].neighbors:
            if x in region and x not in absorbed and x not in frontier:
                frontier.append(x)
    if absorbed != region:
        raise RuntimeError(f"contract_region: disconnected remainder {sorted(region - absorbed)}")
    return tensor, labels


def _apply_2q_on_labeled_tensor(tensor, labels, ident_a, ident_b, U4):
    ca = labels.index(('own', int(ident_a)))
    ta = labels.index(('own', int(ident_b)))
    T = np.moveaxis(tensor, [ca, ta], [-2, -1])
    sh = T.shape
    out = np.matmul(T.reshape(-1, 4), U4.T)
    T = out.reshape(sh)
    return np.moveaxis(T, [-2, -1], [ca, ta])


def _apply_cnot_on_labeled_tensor(tensor, labels, ctrl_ident, target_ident):
    return _apply_2q_on_labeled_tensor(
        tensor, labels, ctrl_ident, target_ident, _CNOT)


def _apply_diag_on_labeled_tensor(tensor, labels, ident, c0, c1):
    pos = labels.index(('own', int(ident)))
    sl0 = [slice(None)] * tensor.ndim
    sl1 = [slice(None)] * tensor.ndim
    sl0[pos] = 0
    sl1[pos] = 1
    tensor[tuple(sl0)] *= c0
    tensor[tuple(sl1)] *= c1
    return tensor


def _rooted_region_children(self, region, root):
    region = set(map(int, region))
    parent = {int(root): None}
    order = [int(root)]
    for u in order:
        for v in self.bags[u].neighbors:
            if v in region and v not in parent:
                parent[v] = u
                order.append(v)
    children = {u: [] for u in order}
    for v, p in parent.items():
        if p is not None:
            children[p].append(v)
    for u in children:
        children[u].sort()
    descendants = {}
    for u in reversed(order):
        s = {u}
        for c in children[u]:
            s.update(descendants[c])
        descendants[u] = s
    return parent, children, descendants


def _labels_for_region_subtree(self, nodes, full_region):
    nodes = set(map(int, nodes))
    full_region = set(map(int, full_region))
    labels = []
    for bid in sorted(nodes):
        labels += [('own', int(x)) for x in self.bags[bid].own_idents]
    for bid in sorted(nodes):
        for nb in self.bags[bid].neighbors:
            if nb in nodes:
                continue
            if nb not in full_region:
                lab = ('bond',) + _edge_key(bid, nb)
                if lab not in labels:
                    labels.append(lab)
    return labels


def _split_region_back_method(self, tensor, labels, region, root=None):
    """Split a contracted region back onto the existing bag-tree skeleton."""
    region = set(map(int, region))
    if root is None:
        root = min(region)
    root = int(root)
    _, children, descendants = self._rooted_region_children(region, root)
    assigned = {}
    qr_count = 0
    qr_work = 0.0

    def split_node(node, cur_tensor, cur_labels):
        nonlocal qr_count, qr_work
        node = int(node)
        for child in children[node]:
            child_labels = self._labels_for_region_subtree(descendants[child], region)
            child_set = set(child_labels)
            if not child_labels:
                continue
            rest_labels = [x for x in cur_labels if x not in child_set]
            if not rest_labels:
                continue
            order = rest_labels + child_labels
            cur_tensor = np.transpose(cur_tensor, [cur_labels.index(x) for x in order])
            rest_shape = cur_tensor.shape[:len(rest_labels)]
            child_shape = cur_tensor.shape[len(rest_labels):]
            rest_dim = _prod(rest_shape) if rest_shape else 1
            child_dim = _prod(child_shape) if child_shape else 1
            M = cur_tensor.reshape(rest_dim, child_dim)
            # QR/SVD computational proxy. For QR, O(m n^2) with reduced side.
            m = max(1, rest_dim); n = max(1, min(rest_dim, child_dim))
            qr_work += float(m) * float(n) * float(n)
            Q, R = _thin_qr(M)
            qr_count += 1
            chi = int(Q.shape[1])
            edge_label_parent = ('bond',) + _edge_key(node, child)
            cur_tensor = Q.reshape(rest_shape + (chi,))
            cur_labels = rest_labels + [edge_label_parent]
            child_tensor = R.reshape((chi,) + child_shape)
            child_cur_labels = [edge_label_parent] + child_labels
            split_node(child, child_tensor, child_cur_labels)

        target = [('own', int(x)) for x in self.bags[node].own_idents]
        target += [('bond',) + _edge_key(node, nb) for nb in self.bags[node].neighbors]
        missing = [x for x in target if x not in cur_labels]
        extra = [x for x in cur_labels if x not in target]
        if missing or extra:
            raise RuntimeError(
                f"split_region_back B{node}: label mismatch missing={missing} extra={extra}")
        assigned[node] = np.transpose(cur_tensor, [cur_labels.index(x) for x in target])

    split_node(root, tensor, list(labels))
    for bid, T in assigned.items():
        self.bags[bid].tensor = T
    self.center_bag = root
    return qr_count, qr_work


def _apply_multicnot_region_method(self, ctrl_idents, target_ident, cap_bytes=None,
                                   total_cap_bytes=None):
    """Fuse a MULTI_CNOT step by opening one connected region once.

    Returns (applied: bool, reason: str). This is an exact executor when it
    succeeds; otherwise callers should use the existing per-control path.
    """
    active_ctrls = [int(x) for x in ctrl_idents if int(x) != int(target_ident)]
    if not active_ctrls:
        return False, "no_controls"
    idents = active_ctrls + [int(target_ident)]
    homes = [self.home[int(x)] for x in idents]
    region = self._connected_region_for_bags(homes)
    est_bytes = self._estimate_region_workspace_bytes(region)
    stored_before = self._stored_bytes()
    region_stored = int(sum(self.bags[bid].resident_bytes for bid in region))
    destructive_est_total = int(stored_before - region_stored + est_bytes)
    if cap_bytes is not None and est_bytes > int(cap_bytes):
        return False, "cap"
    if total_cap_bytes is not None and destructive_est_total > int(total_cap_bytes):
        return False, "total_cap"
    self.move_center(next(iter(region)))
    stored_before = self._stored_bytes()
    region_stored = int(sum(self.bags[bid].resident_bytes for bid in region))
    tensor, labels = self._contract_region(region)
    actual_workspace = int(tensor.nbytes)
    destructive_total = int(stored_before - region_stored + actual_workspace)
    if cap_bytes is not None and actual_workspace > int(cap_bytes):
        return False, "cap_actual"
    if total_cap_bytes is not None and destructive_total > int(total_cap_bytes):
        return False, "total_cap_actual"
    destructive_open = os.environ.get("TTN_DESTRUCTIVE_OPEN", "0") not in ("", "0", "false", "False")
    if destructive_open:
        for bid in region:
            self.bags[bid].tensor = np.array(0.0 + 0.0j, dtype=np.complex128)
        gc.collect()
        live_open_total = int(self._stored_bytes() + actual_workspace)
        self.metrics["multicnot_destructive_open_enabled"] = 1
    else:
        live_open_total = int(stored_before + actual_workspace)
        self.metrics["multicnot_destructive_open_enabled"] = 0
    self.executor_context = dict(
        selected_executor="single_multicnot_region",
        region=sorted(map(int, region)),
        controls=list(map(int, active_ctrls)),
        target=int(target_ident),
    )
    self._record_total_peak(
        live_open_total,
        kind="multicnot_region_open",
        stored_before=stored_before,
        stored_outside_open_regions=int(stored_before - region_stored),
        absorbed_region_stored_bytes=region_stored,
        open_region_bytes=actual_workspace,
        temporary_bytes=0,
        region_size=len(region),
        region=sorted(map(int, region)),
    )
    self._record_destructive_total_peak(
        destructive_total,
        kind="multicnot_region_open",
        stored_before=stored_before,
        stored_outside_open_regions=int(stored_before - region_stored),
        absorbed_region_stored_bytes=region_stored,
        open_region_bytes=actual_workspace,
        temporary_bytes=0,
        region_size=len(region),
        region=sorted(map(int, region)),
    )
    for ctrl in active_ctrls:
        tensor = _apply_cnot_on_labeled_tensor(tensor, labels, ctrl, target_ident)
    qr_count, qr_work = self._split_region_back(tensor, labels, region, root=min(region))
    self.metrics["multicnot_region_fused"] = int(self.metrics.get("multicnot_region_fused", 0)) + 1
    self.metrics["multicnot_region_controls"] = int(
        self.metrics.get("multicnot_region_controls", 0)) + len(active_ctrls)
    self.metrics["multicnot_region_workspace_peak_bytes"] = max(
        int(self.metrics.get("multicnot_region_workspace_peak_bytes", 0)),
        actual_workspace,
    )
    self.metrics["qr_work_proxy"] = float(self.metrics.get("qr_work_proxy", 0.0)) + qr_work
    self._record_metrics(
        pair_workspace_bytes=actual_workspace,
        qr_count=qr_count,
        transport_count=0,
        pair_src=min(region),
        pair_dst=max(region),
        workspace_live_in_total=False,
    )
    return True, "fused"


def _multicnot_region_feasible_method(self, ctrl_idents, target_ident, cap_bytes=None,
                                      total_cap_bytes=None):
    active_ctrls = [int(x) for x in ctrl_idents if int(x) != int(target_ident)]
    if not active_ctrls:
        return False, "no_controls"
    idents = active_ctrls + [int(target_ident)]
    homes = [self.home[int(x)] for x in idents]
    region = self._connected_region_for_bags(homes)
    est_bytes = self._estimate_region_workspace_bytes(region)
    stored = self._stored_bytes()
    region_stored = int(sum(self.bags[bid].resident_bytes for bid in region))
    destructive_total = int(stored - region_stored + est_bytes)
    if cap_bytes is not None and est_bytes > int(cap_bytes):
        return False, "cap"
    if total_cap_bytes is not None and destructive_total > int(total_cap_bytes):
        return False, "total_cap"
    return True, "ok"


def _apply_2q_class_B_2bag_method(self, ident_u, ident_v, U4):
    """Class B with 2-bag path: home[u] != home[v], directly adjacent.
    
    Procedure:
      1. Move center to home[u] (arbitrary choice, either home works)
      2. Contract home[u] and home[v] along their bond -> Theta
      3. Apply U4 to (u, v) axes of Theta
      4. QR-split back: cut between (own_u + outer_bonds_u) | (own_v + outer_bonds_v)
      5. Q -> new home[u].tensor; R -> new home[v].tensor
      6. center becomes home[v] (the side with R, which carries the norm)
    """
    hu = self.home[ident_u]; hv = self.home[ident_v]
    assert hu != hv, "class B requires different homes"
    if hv not in self.bags[hu].neighbors:
        raise ValueError(f"class B 2-bag requires adjacent homes; B{hu} - B{hv} not edge")

    self.move_center(hu)

    bag_l = self.bags[hu]
    bag_r = self.bags[hv]
    L_n_own = bag_l.n_own()
    R_n_own = bag_r.n_own()

    # 1) Move inner bonds to last (L) and first (R)
    L_inner_pos = bag_l.bond_axis_pos(hv)
    R_inner_pos = bag_r.bond_axis_pos(hu)
    L_t = np.moveaxis(bag_l.tensor, L_inner_pos, -1)
    R_t = np.moveaxis(bag_r.tensor, R_inner_pos, 0)
    # L_t shape: [L_own..., L_outer_bonds..., chi_inner]
    # R_t shape: [chi_inner, R_own..., R_outer_bonds...]

    # 2) Contract along inner bond
    theta = np.tensordot(L_t, R_t, axes=([L_t.ndim - 1], [0]))
    # theta shape: [L_own..., L_outer_bonds..., R_own..., R_outer_bonds...]

    L_outer_bonds = [b for b in bag_l.neighbors if b != hv]
    R_outer_bonds = [b for b in bag_r.neighbors if b != hu]
    L_outer_n = len(L_outer_bonds)
    R_outer_n = len(R_outer_bonds)

    # 3) Apply U4 on (u, v) axes
    pu = bag_l.ident_axis_pos(ident_u)   # in L_own block: pu in [0, L_n_own)
    pv = (L_n_own + L_outer_n) + bag_r.ident_axis_pos(ident_v)
    amp = np.moveaxis(theta, [pu, pv], [-2, -1])
    sh = amp.shape
    out = np.matmul(amp.reshape(-1, 4), U4.T)
    amp = out.reshape(sh)
    theta_new = np.moveaxis(amp, [-2, -1], [pu, pv])

    # 4) Reshape for QR: [L_own + L_outer_bonds] | [R_own + R_outer_bonds]
    left_axes = L_n_own + L_outer_n
    left_shape = theta_new.shape[:left_axes]
    right_shape = theta_new.shape[left_axes:]
    left_dim = int(np.prod(left_shape)) if left_axes else 1
    right_dim = int(np.prod(right_shape)) if right_shape else 1
    M = theta_new.reshape(left_dim, right_dim)

    Q, R = _thin_qr(M)
    chi_new = Q.shape[1]

    # 5) Place Q into bag_l: shape = left_shape + (chi_new,)
    L_new = Q.reshape(left_shape + (chi_new,))
    # Currently axes: [L_own, L_outer_bonds, bond_to_R]
    # bond_to_R should be at position L_n_own + L_outer_bonds.index(hv)
    # But L_outer_bonds doesn't include hv; we need to insert bond_to_R at the
    # position in neighbors order. bag_l.neighbors is sorted (canonical).
    # Bond axis order should be sorted by neighbor id; we need new bond to hv at
    # its sorted position.
    # Strategy: build axis order according to bag_l.neighbors.
    # Current axes order: L_own (0..L_n_own-1), then L_outer_bonds in some order, then bond_to_R last.
    # L_outer_bonds was built as [b for b in bag_l.neighbors if b != hv], preserving sort.
    # We need axes order to be: L_own, then for each n in bag_l.neighbors, the bond axis.
    # Determine permutation:
    cur_bond_neighbors_in_L_new = L_outer_bonds + [hv]
    target_bond_order = list(bag_l.neighbors)
    perm_bonds = [cur_bond_neighbors_in_L_new.index(n) for n in target_bond_order]
    # Apply permutation to bond block
    full_perm = list(range(L_n_own)) + [L_n_own + p for p in perm_bonds]
    L_new = np.transpose(L_new, full_perm)
    bag_l.tensor = L_new

    # 6) Place R into bag_r: shape = (chi_new,) + right_shape
    R_new = R.reshape((chi_new,) + right_shape)
    # Current axes: [bond_to_L, R_own, R_outer_bonds]
    # Target axes: [R_own, bond axes in bag_r.neighbors sorted order]
    # Move bond_to_L to its target position.
    # First move bond_to_L (axis 0) past R_own to position R_n_own.
    R_new = np.moveaxis(R_new, 0, R_n_own)
    # Now axes: [R_own, bond_to_L, R_outer_bonds]
    cur_bond_neighbors_in_R_new = [hu] + R_outer_bonds
    target_bond_order_R = list(bag_r.neighbors)
    perm_bonds_R = [cur_bond_neighbors_in_R_new.index(n) for n in target_bond_order_R]
    full_perm_R = list(range(R_n_own)) + [R_n_own + p for p in perm_bonds_R]
    R_new = np.transpose(R_new, full_perm_R)
    bag_r.tensor = R_new

    # center is now at bag_r (R carries the non-isometric part)
    self.center_bag = hv


TTNState._apply_2q_local = _apply_2q_local_method
TTNState.apply_2q_class_A = _apply_2q_class_A_method
TTNState._connected_region_for_bags = _connected_region_for_bags_method
TTNState._region_labels_for_bag = _region_labels_for_bag
TTNState._estimate_region_workspace_bytes = _estimate_region_workspace_bytes_method
TTNState._contract_region = _contract_region_method
TTNState._rooted_region_children = _rooted_region_children
TTNState._labels_for_region_subtree = _labels_for_region_subtree
TTNState._split_region_back = _split_region_back_method
TTNState.apply_multicnot_region = _apply_multicnot_region_method
TTNState.multicnot_region_feasible = _multicnot_region_feasible_method
TTNState.apply_2q_class_B_2bag = _apply_2q_class_B_2bag_method


# Standard gates
def _gate_cnot():
    U = np.zeros((4, 4), dtype=np.complex128)
    U[0, 0] = 1; U[1, 1] = 1; U[3, 2] = 1; U[2, 3] = 1
    return U

def _gate_cz():
    U = np.eye(4, dtype=np.complex128); U[3, 3] = -1
    return U

_CNOT = _gate_cnot()
_CZ = _gate_cz()


def _u2_node_matrix_and_frame(prog, cp_idx, frame, axis):
    """Return the Clifft fused-U2 matrix selected by the incoming frame state."""
    nodes = getattr(prog, "fused_u2_nodes", None)
    if nodes is None:
        raise RuntimeError("Program does not expose fused_u2_nodes")
    node = nodes[int(cp_idx)]
    in_state = (2 if frame.zb(axis) else 0) | (1 if frame.xb(axis) else 0)
    U = np.asarray(node["matrices"][in_state], dtype=np.complex128).reshape(2, 2)
    out = int(node["out_states"][in_state])
    return U, out


def _u4_node_matrix_and_frame(prog, cp_idx, frame, axis_lo, axis_hi):
    """Return the Clifft fused-U4 matrix for basis |hi,lo>, lo as LSB."""
    nodes = getattr(prog, "fused_u4_nodes", None)
    if nodes is None:
        raise RuntimeError("Program does not expose fused_u4_nodes")
    in_state = ((8 if frame.zb(axis_hi) else 0) |
                (4 if frame.xb(axis_hi) else 0) |
                (2 if frame.zb(axis_lo) else 0) |
                (1 if frame.xb(axis_lo) else 0))
    entry = nodes[int(cp_idx)]["entries"][in_state]
    U = np.asarray(entry["matrix"], dtype=np.complex128)
    out = int(entry["out_state"])
    return U, out


# ==========================================================================
# N-bag path Class B
# ==========================================================================

def _insert_ident_sorted(own_idents, ident):
    """Insert ident into own_idents in a stable canonical position."""
    pos = 0
    while pos < len(own_idents) and own_idents[pos] < ident:
        pos += 1
    own_idents.insert(pos, ident)


def _staged_rank_from_gram(G, rtol, atol, svd_rtol, cond_max):
    """Eigendecompose a Hermitian Gram, return (eigvecs, singvals, keep_mask).

    Floors the kept rank at the Gram precision sqrt(eps)*smax and raises when
    real singular values sit near that noise floor (ambiguous rank).
    """
    G = (G + G.conj().T) * 0.5
    lam, U = np.linalg.eigh(G)
    lam = np.clip(lam.real, 0.0, None)
    s = np.sqrt(lam)
    smax = float(s[-1]) if s.size else 0.0
    eps = float(np.finfo(np.float64).eps)
    gram_floor = 8.0 * math.sqrt(eps) * smax
    rel = float(svd_rtol) if svd_rtol else float(rtol)
    thr = max(float(atol), rel * smax, gram_floor)
    keep = s > thr
    if not bool(np.any(keep)):
        keep = np.zeros(G.shape[0], dtype=bool)
        keep[-1] = True
    sk = s[keep]
    cond = float(smax / sk.min()) if sk.size and float(sk.min()) > 0 else float("inf")
    if cond > float(cond_max):
        raise np.linalg.LinAlgError(
            f"staged transport ill-conditioned (cond~{cond:.2e} > {cond_max:.2e})")
    return U, s, keep, cond


def _staged_factor_blocks(src_m, dst_m, left_dim, right_without_dim,
                          right_source_shape, perm, right_shape, right_dim,
                          block_target_bytes, rtol=1e-12, atol=1e-14,
                          svd_rtol=None, cond_max=1e6, force_reorth=False):
    """Exact block-streamed factorization M = Q @ R, Q orthonormal columns.

    M is the (left_dim x right_dim) transport matrix obtained by contracting
    `src_m` (left_dim, 2, chi_old) with `dst_m` (chi_old, right_without_dim) and
    reshaping/transposing by `perm`. The full M / theta is never materialized.

    The Gram is always formed on the *smaller* side so a wide M does not blow up:
      - left_dim >= right_dim: stream row blocks, Gram = M^H M (right_dim^2),
        Q = M (Vk/sk) (needs a CholeskyQR2 reorthonormal pass when conditioned).
      - left_dim <  right_dim: stream source-column blocks, Gram = M M^H
        (left_dim^2), Q = U_k directly (no division, no reorth needed).

    Returns (Q, R, meta). The caller may catch exceptions and fall back to a
    dense factorization.
    """
    left_dim = int(left_dim)
    n = int(right_dim)
    chi_old = int(src_m.shape[2])
    C = np.dtype(np.complex128).itemsize

    def row_block_fn(r0, r1):
        bt3 = np.tensordot(src_m[r0:r1], dst_m, axes=([2], [0]))  # (b,2,rwd)
        b = int(bt3.shape[0])
        bt = bt3.reshape((b,) + right_source_shape)
        return np.transpose(bt, perm).reshape(b, n)

    max_block_bytes = 0
    reorth = False

    if left_dim >= n:
        # ---- tall/square: Gram on the right (M^H M) via row streaming ----
        block_rows = max(1, int(block_target_bytes) // max(n * C, 1))

        def _rows():
            r0 = 0
            while r0 < left_dim:
                yield r0, min(r0 + block_rows, left_dim)
                r0 += block_rows

        G = np.zeros((n, n), dtype=np.complex128)
        for r0, r1 in _rows():
            Mb = row_block_fn(r0, r1)
            max_block_bytes = max(max_block_bytes, int(Mb.nbytes))
            G += np.matmul(Mb.conj().T, Mb)
        V, s, keep, cond = _staged_rank_from_gram(G, rtol, atol, svd_rtol, cond_max)
        Vk = V[:, keep]
        sk = s[keep]
        chi = int(Vk.shape[1])
        Vk_over_s = Vk / sk[None, :]
        Q = np.empty((left_dim, chi), dtype=np.complex128)
        for r0, r1 in _rows():
            Q[r0:r1] = np.matmul(row_block_fn(r0, r1), Vk_over_s)
        if force_reorth or cond > 1e3:
            reorth = True
            G2 = np.zeros((chi, chi), dtype=np.complex128)
            for r0, r1 in _rows():
                G2 += np.matmul(Q[r0:r1].conj().T, Q[r0:r1])
            G2 = (G2 + G2.conj().T) * 0.5
            lam2, V2 = np.linalg.eigh(G2)
            s2 = np.sqrt(np.clip(lam2.real, 0.0, None))
            s2_safe = np.where(s2 > max(atol, rtol * (float(s2[-1]) if s2.size else 1.0)),
                               s2, 1.0)
            W2 = V2 / s2_safe[None, :]
            for r0, r1 in _rows():
                Q[r0:r1] = np.matmul(Q[r0:r1], W2)
        R = np.zeros((chi, n), dtype=np.complex128)
        for r0, r1 in _rows():
            R += np.matmul(Q[r0:r1].conj().T, row_block_fn(r0, r1))
        gram_bytes = int(G.nbytes + V.nbytes + Vk_over_s.nbytes)
    else:
        # ---- wide: Gram on the left (M M^H) via source-column streaming ----
        # Source columns are (x in {0,1}) x (j in right_without_dim); their order
        # differs from M's column order only by `perm`, and M M^H is invariant to
        # column order, so the Gram is built without perm. R is assembled in the
        # natural source order, then permuted to M's right-label order.
        rwd = int(right_without_dim)
        block_cols = max(1, int(block_target_bytes) // max(left_dim * C, 1))

        def _col_blocks():
            for x in (0, 1):
                j0 = 0
                while j0 < rwd:
                    j1 = min(j0 + block_cols, rwd)
                    Tb = np.matmul(src_m[:, x, :], dst_m[:, j0:j1])   # (left_dim, j1-j0)
                    yield x, j0, j1, Tb
                    j0 = j1

        G = np.zeros((left_dim, left_dim), dtype=np.complex128)
        for x, j0, j1, Tb in _col_blocks():
            max_block_bytes = max(max_block_bytes, int(Tb.nbytes))
            G += np.matmul(Tb, Tb.conj().T)
        U, s, keep, cond = _staged_rank_from_gram(G, rtol, atol, svd_rtol, cond_max)
        Q = U[:, keep]                       # (left_dim, chi) orthonormal directly
        chi = int(Q.shape[1])
        QH = Q.conj().T
        # R_source[:, x, j] = Q^H @ M_source[:, x, j]
        R_source = np.empty((chi, 2, rwd), dtype=np.complex128)
        for x, j0, j1, Tb in _col_blocks():
            R_source[:, x, j0:j1] = np.matmul(QH, Tb)
        # reorder source axes -> right_source_shape -> M right-label order
        R_src = R_source.reshape((chi,) + right_source_shape)
        R = np.transpose(R_src, [0] + [perm[k] for k in range(1, len(perm))]).reshape(chi, n)
        gram_bytes = int(G.nbytes + U.nbytes + R_source.nbytes)

    meta = dict(
        chi=int(chi),
        mode="row" if left_dim >= n else "col",
        max_block_bytes=int(max_block_bytes),
        gram_bytes=int(gram_bytes),
        qr_temp_bytes=int(max_block_bytes + gram_bytes),
        q_bytes=int(Q.nbytes),
        r_bytes=int(R.nbytes),
        cond=float(cond),
        reorth=bool(reorth),
    )
    return Q, R, meta


def _transport_ident_across_edge_method(self, ident, src, dst):
    """Move one physical ident axis across an adjacent bag edge by 2-bag QR.

    The static `home` map is intentionally not changed. During a Class B/C
    sweep I2 is temporarily violated, then restored by the reverse transport.
    """
    if dst not in self.bags[src].neighbors:
        raise ValueError(f"transport: B{src} and B{dst} are not adjacent")
    src_bag = self.bags[src]
    dst_bag = self.bags[dst]
    if ident not in src_bag.own_idents:
        raise ValueError(f"transport: ident {ident} not in B{src}")

    # Resident streaming: spill idle large bags BEFORE measuring stored_before,
    # keeping only this transport's operands. The 67 MB co-residence forms
    # within a MULTI_CNOT's internal transports, and the peak accounting uses
    # stored_before, so the spill must happen here (start), not at the end.
    if self._resident_stream_cap() is not None:
        self.spill_idle_large_bags(keep={int(src), int(dst)}, keep_center=False)

    self.move_center(src)
    src_bag = self.bags[src]
    dst_bag = self.bags[dst]
    stored_before_transport = int(self._stored_bytes())
    absorbed_pair_bytes = int(src_bag.resident_bytes + dst_bag.resident_bytes)

    src_outer = [nb for nb in src_bag.neighbors if nb != dst]
    dst_outer = [nb for nb in dst_bag.neighbors if nb != src]
    src_labels = [('own', src, x) for x in src_bag.own_idents]
    src_labels += [('bond', src, nb) for nb in src_bag.neighbors]
    dst_labels = [('own', dst, x) for x in dst_bag.own_idents]
    dst_labels += [('bond', dst, nb) for nb in dst_bag.neighbors]

    new_src_own = [x for x in src_bag.own_idents if x != ident]
    new_dst_own = list(dst_bag.own_idents)
    _insert_ident_sorted(new_dst_own, ident)

    left_labels = [('own', src, x) for x in new_src_own]
    left_labels += [('bond', src, nb) for nb in src_outer]
    right_labels = [('own', dst, x) for x in new_dst_own]
    right_labels += [('bond', dst, nb) for nb in dst_outer]

    src_order = left_labels + [('own', src, ident), ('bond', src, dst)]
    src_t = np.transpose(src_bag.tensor, [src_labels.index(x) for x in src_order])
    left_shape = src_t.shape[:len(left_labels)]
    left_dim = int(np.prod(left_shape)) if left_shape else 1
    chi_old = src_t.shape[-1]
    src_m = src_t.reshape(left_dim, 2, chi_old)

    right_without_u = [x for x in right_labels if x != ('own', dst, ident)]
    dst_order = [('bond', dst, src)] + right_without_u
    dst_t = np.transpose(dst_bag.tensor, [dst_labels.index(x) for x in dst_order])
    right_without_shape = dst_t.shape[1:]
    right_without_dim = int(np.prod(right_without_shape)) if right_without_shape else 1
    dst_m = dst_t.reshape(chi_old, right_without_dim)

    # Label/shape structure of the factorized matrix M (left_dim x right_dim).
    # These depend only on the bag labels, not on tensor values, so they are
    # computed before any large contraction and shared by the dense and the
    # block-streamed paths.
    right_source_labels = [('own', dst, ident)] + right_without_u
    right_source_shape = (2,) + tuple(right_without_shape)
    perm = [0] + [1 + right_source_labels.index(x) for x in right_labels]
    right_shape = tuple(right_source_shape[right_source_labels.index(x)] for x in right_labels)
    left_dim = int(np.prod(left_shape)) if left_shape else 1
    right_dim = int(_prod(right_shape)) if right_shape else 1

    full_theta_bytes = int(left_dim * 2 * right_without_dim
                           * np.dtype(np.complex128).itemsize)
    cap_env = os.environ.get("TTN_EXACT_TOTAL_CAP_BYTES")

    # --- decide staged (block-streamed) vs dense transport -----------------
    def _flag(name):
        return os.environ.get(name, "0") not in ("", "0", "false", "False")

    staged_mode = _flag("TTN_STAGED_TRANSPORT")
    staged_force = _flag("TTN_STAGED_TRANSPORT_FORCE")
    staged_min = int(os.environ.get("TTN_STAGED_MIN_BYTES", str(32 * 1024 * 1024)))
    staged_threshold = int(cap_env) if cap_env else staged_min
    # The streamed Gram lives on the smaller matrix side; staging only helps when
    # min(left,right)^2 is well below the full theta, i.e. M is far from square.
    C_ITEM = np.dtype(np.complex128).itemsize
    gram_side_bytes = int(C_ITEM * (min(left_dim, right_dim) ** 2))
    gram_cap = int(cap_env) if cap_env else staged_min
    use_staged = staged_mode and (
        staged_force
        or (full_theta_bytes > staged_threshold and gram_side_bytes <= gram_cap)
    )

    svd_rtol = os.environ.get("TTN_SVD_TRUNC_RTOL")
    if svd_rtol:
        min_elems = int(os.environ.get("TTN_SVD_TRUNC_MIN_MATRIX_ELEMS", "0"))
        if min_elems and int(left_dim * right_dim) < min_elems:
            svd_rtol = None

    Q = R = None
    staged_meta = None
    if use_staged:
        default_block = max(int(cap_env) // 8 if cap_env else 0, 8 * 1024 * 1024)
        block_target_bytes = int(os.environ.get("TTN_STAGED_BLOCK_BYTES", str(default_block)))
        cond_max = float(os.environ.get("TTN_STAGED_COND_MAX", "1e6"))
        force_reorth = _flag("TTN_STAGED_FORCE_REORTH")
        try:
            Q, R, staged_meta = _staged_factor_blocks(
                src_m, dst_m, left_dim, right_without_dim,
                right_source_shape, perm, right_shape, right_dim,
                block_target_bytes, svd_rtol=svd_rtol,
                cond_max=cond_max, force_reorth=force_reorth,
            )
        except Exception as exc:  # pragma: no cover - numerical fallback
            self.metrics["staged_fallback_count"] = int(
                self.metrics.get("staged_fallback_count", 0)) + 1
            self.metrics.setdefault("staged_fallback_events", []).append(dict(
                step=self.current_step, src=int(src), dst=int(dst),
                ident=int(ident), error=str(exc),
            ))
            Q = R = staged_meta = None
            use_staged = False

    if use_staged and Q is not None:
        chi_new = int(staged_meta["chi"])
        # Staged peak under the same destructive-open liveness model as the dense
        # path: the open region (Q + R outputs + one streaming block + Gram)
        # logically replaces the absorbed src/dst bag tensors. This is consistent
        # with the dense convention (which also subtracts the absorbed pair while
        # the theta workspace is open). When destructive-open is off, count the
        # old tensors and the new outputs together.
        transient_bytes = int(staged_meta["max_block_bytes"] + staged_meta["gram_bytes"])
        workspace_bytes = transient_bytes
        open_region_bytes = int(staged_meta["q_bytes"] + staged_meta["r_bytes"]
                                + transient_bytes)
        if _flag("TTN_DESTRUCTIVE_OPEN"):
            staged_live_total = int(stored_before_transport - absorbed_pair_bytes
                                    + open_region_bytes)
        else:
            staged_live_total = int(stored_before_transport + open_region_bytes)
        # metrics specific to staged transport
        self.metrics["staged_transport_count"] = int(
            self.metrics.get("staged_transport_count", 0)) + 1
        self.metrics["staged_max_theta_block_bytes"] = max(
            int(self.metrics.get("staged_max_theta_block_bytes", 0)),
            int(staged_meta["max_block_bytes"]))
        self.metrics["staged_qr_temp_peak_bytes"] = max(
            int(self.metrics.get("staged_qr_temp_peak_bytes", 0)),
            int(staged_meta["qr_temp_bytes"]))
        self.metrics["staged_max_q_bytes"] = max(
            int(self.metrics.get("staged_max_q_bytes", 0)), int(staged_meta["q_bytes"]))
        self.metrics["staged_max_r_bytes"] = max(
            int(self.metrics.get("staged_max_r_bytes", 0)), int(staged_meta["r_bytes"]))
        self.metrics["staged_max_full_theta_bytes_avoided"] = max(
            int(self.metrics.get("staged_max_full_theta_bytes_avoided", 0)),
            int(full_theta_bytes))
        self.metrics["staged_reorth_count"] = int(
            self.metrics.get("staged_reorth_count", 0)) + (1 if staged_meta["reorth"] else 0)
        self.metrics["sum_refactor_input_numel"] += int(left_dim * right_dim)
        self.metrics["max_refactor_input_numel"] = max(
            int(self.metrics["max_refactor_input_numel"]),
            int(staged_meta["max_block_bytes"] // np.dtype(np.complex128).itemsize))
        if cap_env and staged_live_total > int(cap_env):
            self._record_cap_infeasible_exact(
                staged_live_total, int(cap_env),
                reason="staged_transport_total_cap",
                transport_ident=int(ident), src=int(src), dst=int(dst),
                stored_bytes=int(stored_before_transport),
                workspace_bytes=int(transient_bytes),
                q_bytes=int(staged_meta["q_bytes"]), r_bytes=int(staged_meta["r_bytes"]),
            )
    else:
        theta3 = np.tensordot(src_m, dst_m, axes=([2], [0]))
        workspace_bytes = theta3.nbytes
        staged_live_total = None
        if cap_env:
            cap_bytes = int(cap_env)
            destructive_pair = _flag("TTN_DESTRUCTIVE_OPEN")
            if destructive_pair:
                live_total = int(stored_before_transport - absorbed_pair_bytes + workspace_bytes)
            else:
                live_total = int(stored_before_transport + workspace_bytes)
            if live_total > cap_bytes:
                self._record_cap_infeasible_exact(
                    live_total,
                    cap_bytes,
                    reason="adjacent_transport_total_cap",
                    transport_ident=int(ident),
                    src=int(src),
                    dst=int(dst),
                    stored_bytes=int(stored_before_transport),
                    absorbed_pair_bytes=int(absorbed_pair_bytes if destructive_pair else 0),
                    workspace_bytes=int(workspace_bytes),
                    src_shape=tuple(int(x) for x in src_bag.shape),
                    dst_shape=tuple(int(x) for x in dst_bag.shape),
                )
                if _flag("TTN_EXACT_CAP_STRICT"):
                    raise RuntimeError(
                        "CAP_INFEASIBLE_EXACT adjacent_transport "
                        f"step={self.current_step} op={self.current_op_kind} "
                        f"B{src}->B{dst} ident={ident} total={live_total} cap={cap_bytes}"
                    )
        self.metrics["sum_refactor_input_numel"] += int(theta3.size)
        self.metrics["max_refactor_input_numel"] = max(
            int(self.metrics["max_refactor_input_numel"]), int(theta3.size))

        theta = theta3.reshape((left_dim,) + right_source_shape)
        theta = np.transpose(theta, perm)
        M = theta.reshape(left_dim, right_dim)
        if os.environ.get("TTN_DEBUG_TRANSPORT"):
            size = left_dim * right_dim
            if size >= int(os.environ.get("TTN_DEBUG_TRANSPORT_MIN_SIZE", "1048576")):
                print(
                    f"[transport] ident={ident} B{src}->B{dst} "
                    f"M={left_dim}x{right_dim} elems={size} "
                    f"stored={self._stored_bytes()} max_bond={self._max_bond_dim()}",
                    flush=True,
                )
        Q, R = _thin_qr(M)
        chi_new = Q.shape[1]

    src_bag.own_idents = new_src_own
    dst_bag.own_idents = new_dst_own

    q_labels = left_labels + [('bond', src, dst)]
    src_target = [('own', src, x) for x in src_bag.own_idents]
    src_target += [('bond', src, nb) for nb in src_bag.neighbors]
    q_tensor = Q.reshape(left_shape + (chi_new,))
    src_bag.tensor = np.transpose(q_tensor, [q_labels.index(x) for x in src_target])

    r_labels = [('bond', dst, src)] + right_labels
    dst_target = [('own', dst, x) for x in dst_bag.own_idents]
    dst_target += [('bond', dst, nb) for nb in dst_bag.neighbors]
    r_tensor = R.reshape((chi_new,) + right_shape)
    dst_bag.tensor = np.transpose(r_tensor, [r_labels.index(x) for x in dst_target])

    self.center_bag = dst
    ctx = dict(self.executor_context or {})
    ctx.update(dict(
        transport_ident=int(ident),
        transport_src=int(src),
        transport_dst=int(dst),
        selected_executor=ctx.get("selected_executor", "adjacent_transport"),
    ))
    self.executor_context = ctx
    if staged_meta is not None:
        # Staged path never materializes the full theta. Report the streaming
        # peak (open region = new Q/R + one block + Gram) under the same
        # destructive-open liveness as the dense path, and the small transient
        # (block + Gram) as the workspace.
        _destructive = _flag("TTN_DESTRUCTIVE_OPEN")
        self._record_metrics(
            pair_workspace_bytes=workspace_bytes,
            qr_count=1,
            transport_count=1,
            pair_src=src,
            pair_dst=dst,
            live_total_bytes_override=int(staged_live_total),
            stored_outside_open_regions=int(
                stored_before_transport - absorbed_pair_bytes if _destructive
                else stored_before_transport),
            absorbed_region_stored_bytes=int(absorbed_pair_bytes if _destructive else 0),
        )
    else:
        self._record_metrics(
            pair_workspace_bytes=workspace_bytes,
            qr_count=1,
            transport_count=1,
            pair_src=src,
            pair_dst=dst,
            live_total_bytes_override=int(stored_before_transport - absorbed_pair_bytes + workspace_bytes)
            if os.environ.get("TTN_DESTRUCTIVE_OPEN", "0") not in ("", "0", "false", "False")
            else None,
            stored_outside_open_regions=int(stored_before_transport - absorbed_pair_bytes),
            absorbed_region_stored_bytes=int(absorbed_pair_bytes),
        )


def _apply_2q_class_B_path_method(self, ident_u, ident_v, U4, path):
    """Class B/C gate via adjacent 2-bag transport sweep."""
    hu = self.home[ident_u]; hv = self.home[ident_v]
    if path[0] != hu and path[0] != hv:
        raise ValueError(f"path[0] must be a home of one of the idents")
    # Standardize: path[0] = home_u, path[-1] = home_v
    if path[0] == hv:
        path = list(reversed(path))
        ident_u, ident_v = ident_v, ident_u
        hu, hv = hv, hu  # re-fetch homes after swap

    # Dynamic bag fission can replace an original edge (a,b) by a micro-path.
    # Expand stale precomputed path segments against the current bag tree.
    expanded = [int(path[0])]
    for a, b in zip(path, path[1:]):
        a = int(a); b = int(b)
        if b in self.bags[a].neighbors:
            seg = [a, b]
        else:
            seg = self._tree_path(a, b)
            if seg is None:
                raise RuntimeError(f"dynamic path expansion failed for B{a}->B{b}")
        expanded.extend(seg[1:])
    path = expanded

    self.executor_context = dict(
        selected_executor="class_BC_path_transport",
        idents=[int(ident_u), int(ident_v)],
        homes=[int(hu), int(hv)],
        path=list(map(int, path)),
        path_length=max(0, len(path) - 1),
    )
    self._record_path_refactor(path)
    self.move_center(path[0])

    def _transport_until(ident, start, target):
        cur = int(start)
        target = int(target)
        guard = 0
        while cur != target:
            seg = self._tree_path(cur, target)
            if not seg or len(seg) < 2:
                raise RuntimeError(f"dynamic transport path failed for B{cur}->B{target}")
            nxt = int(seg[1])
            if self.maybe_prefission_transport_edge(ident, cur, nxt):
                guard += 1
                if guard > 4 * max(1, len(self.bags)):
                    raise RuntimeError(f"dynamic prefission loop for B{start}->B{target}")
                continue
            self.transport_ident_across_edge(ident, cur, nxt)
            cur = nxt
            guard += 1
            if guard > 4 * max(1, len(self.bags)):
                raise RuntimeError(f"dynamic transport path loop for B{start}->B{target}")
        return cur

    _transport_until(ident_u, path[0], path[-1])
    self._apply_2q_local(path[-1], ident_u, ident_v, U4)
    _transport_until(ident_u, path[-1], path[0])

    self.check_invariant_I1()
    self.check_invariant_I3()
    self._record_metrics()


TTNState.transport_ident_across_edge = _transport_ident_across_edge_method
TTNState.apply_2q_class_B_path = _apply_2q_class_B_path_method


# ==========================================================================
# Full bytecode dispatch
# ==========================================================================

# Try import for frame layer; gracefully degrade if not available
try:
    import clifft
    from . import treewidth as T_mod
    from . import frame_layer as ds_mod
    _HAVE_CLIFFT = True
    _FLAG_SIGN = ds_mod.FLAG_SIGN
    _T_PHASE = ds_mod.T_PHASE
    _T_PHASE_DAG = ds_mod.T_PHASE_DAG
    _H_MAT = np.array([[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]], dtype=np.complex128)
except ImportError:
    _HAVE_CLIFFT = False


class TTNBackend:
    """Full bytecode-driven TTN simulator.
    
    Wraps TTNState with: ident allocation tracking, Pauli frame, op dispatch.
    """
    def __init__(self, spec, homing, capture_peak_snapshot=False,
                 trace_recorder=None):
        self.spec = spec
        self.homing = homing
        self.capture_peak_snapshot = bool(capture_peak_snapshot)
        self.trace_recorder = trace_recorder
        self.last_metrics = None
        # Build bag adjacency from spec
        bag_neighbors = [[] for _ in range(spec["union"]["n_bags"])]
        for i, j, _ in spec["union"]["bag_edges"]:
            bag_neighbors[i].append(j)
            bag_neighbors[j].append(i)
        self.bag_neighbors = bag_neighbors
        self.home_of = homing["home"]
        
        # Build step -> ident map (from lifecycle)
        self.step_to_ident_expand = {}
        for ident, lc in spec["lifecycle"].items():
            self.step_to_ident_expand[lc["promote_step"]] = ident
        
        # op_classes indexed by step + axes pair (for two-axis ops)
        self.op_class_by_step_axes = {}
        for r in homing["op_classes"]:
            if r["kind"] == "two":
                key = (r["step"], frozenset(r["axes"]))
                self.op_class_by_step_axes[key] = r

    def _path_for_ordered_idents(self, ident_u, ident_v, opc):
        path = list(opc['path_bags'])
        hu = self.home_of[ident_u]
        hv = self.home_of[ident_v]
        if path and path[0] == hu and path[-1] == hv:
            return path
        if path and path[0] == hv and path[-1] == hu:
            return list(reversed(path))
        return path

    def _edge_chi(self, a, b):
        bag = self.state.bags[int(a)]
        return int(bag.shape[bag.bond_axis_pos(int(b))])

    def _rank_weighted_path_cost(self, src, dst, edge_base=1.0):
        """Transport-cost proxy for moving an ident src->dst on the current tree:
        sum over path edges of (edge_base + log2(current chi_e)). Uses live bond
        dims so the selector reflects which edges are actually expensive now."""
        src = int(src); dst = int(dst)
        if src == dst:
            return 0.0
        path = self.state._tree_path(src, dst)
        if not path or len(path) < 2:
            return 0.0
        c = 0.0
        for a, b in zip(path, path[1:]):
            chi = self._edge_chi(int(a), int(b))
            c += float(edge_base) + math.log2(max(1, chi))
        return c

    def _apply_cnot_idents(self, ctrl_ident, target_ident, step):
        """Apply a single CNOT(ctrl_ident -> target_ident) on the active state,
        using the same Class A / path-transport dispatch the per-control
        MULTI_CNOT loop uses. Does NOT touch the Pauli frame; the caller updates
        the frame by slot so frame and state stay consistent."""
        opc = self.op_class_by_step_axes.get(
            (step, frozenset({ctrl_ident, target_ident})))
        if opc is None:
            hu = self.home_of[ctrl_ident]
            hv = self.home_of[target_ident]
            if hu == hv:
                self.state.apply_2q_class_A(ctrl_ident, target_ident, _CNOT)
            else:
                path = self.state._tree_path(hu, hv)
                self.state.apply_2q_class_B_path(ctrl_ident, target_ident, _CNOT, path)
        elif opc['cls'] == 'A':
            self.state.apply_2q_class_A(ctrl_ident, target_ident, _CNOT)
        else:
            path = self._path_for_ordered_idents(ctrl_ident, target_ident, opc)
            self.state.apply_2q_class_B_path(ctrl_ident, target_ident, _CNOT, path)

    def _build_steiner_children(self, h_t, control_bags):
        """Rooted Steiner tree of {h_t} U control_bags on the bag tree.

        Returns (children, parent): children[node] = bags one step away from h_t,
        parent[node] = bag one step toward h_t. Only nodes on some
        control->h_t path are included.
        """
        parent = {int(h_t): None}
        children = {}
        for cbag in control_bags:
            cbag = int(cbag)
            if cbag == int(h_t):
                continue
            path = self.state._tree_path(int(h_t), cbag)
            if not path or len(path) < 2:
                continue
            for a, b in zip(path, path[1:]):
                a = int(a); b = int(b)
                if b not in parent:
                    parent[b] = a
                    children.setdefault(a, []).append(b)
        return children, parent

    def _execute_multicnot_parity_gather(self, target_slot, target_ident,
                                         ctrl_pairs, step):
        """Execute the parity-gather rewrite of one MULTI_CNOT. Exact.

        General recursive form: build the Steiner tree of the control home bags
        rooted at the target home, and recursively fold each subtree's controls
        into a single representative *control* (never through a non-control
        Steiner ident), so each tree edge is crossed once per subtree instead of
        once per control. Per-gateway-subtree the fold is applied only when a
        rank-weighted transport-cost proxy says it is cheaper than direct CNOTs,
        so the rewrite can never regress a subtree that has no expensive edge."""
        h_t = int(self.home_of[target_ident])
        controls_at = {}
        direct = []
        for cslot, cid in ctrl_pairs:
            h_c = int(self.home_of[cid])
            if h_c == h_t:
                direct.append((cslot, cid))     # already at target's bag
            else:
                controls_at.setdefault(h_c, []).append((cslot, cid))

        min_gain = float(os.environ.get("TTN_MULTICNOT_PARITY_MIN_GAIN", "1.0"))
        children, _parent = self._build_steiner_children(h_t, controls_at.keys())

        counters = dict(local=0, crossing=0, folded=0, direct=0, max_depth=0)

        def emit(pair_src, pair_dst):
            cslot, cid = pair_src
            dslot, did = pair_dst
            self._apply_cnot_idents(cid, did, step)
            self.frame.cnot(cslot, dslot)

        def cost1(a_bag, b_bag):
            return self._rank_weighted_path_cost(a_bag, b_bag)

        def home(pair):
            return int(self.home_of[pair[1]])

        def gather(node, depth):
            """Return (rep_pair, fold_ops, members) for the subtree rooted at
            `node` (away from h_t). fold_ops are (src_pair, dst_pair) CNOTs that
            collapse the subtree's controls onto rep_pair; members is every
            control pair in the subtree. All folds are control-into-control."""
            counters["max_depth"] = max(counters["max_depth"], depth)
            reps = []
            fold_ops = []
            members = []
            for ch in children.get(node, []):
                r, e, mem = gather(ch, depth + 1)
                fold_ops += e
                members += mem
                if r is not None:
                    reps.append(r)
            here = controls_at.get(node, [])
            reps += here
            members += here
            if not reps:
                return None, fold_ops, members
            # representative = rep closest (rank-weighted) to the target hub, so
            # the carry up the parent edge is the cheapest available control.
            rep = min(reps, key=lambda p: cost1(home(p), h_t))
            for other in reps:
                if other is rep:
                    continue
                fold_ops.append((other, rep))   # CNOT(other -> rep): rep absorbs
            return rep, fold_ops, members

        # controls already co-located with the target: always plain CNOT
        for pair in direct:
            emit(pair, (target_slot, target_ident))
            counters["crossing"] += 1

        # each immediate subtree of the target hub is decided independently
        for gw in children.get(h_t, []):
            rep, fold_ops, members = gather(gw, 1)
            if rep is None or not members:
                continue
            naive_cost = sum(2.0 * cost1(home(p), h_t) for p in members)
            # fold op executes twice (compute + uncompute), each a round trip
            fold_cost = (2.0 * cost1(home(rep), h_t)
                         + sum(4.0 * cost1(home(s), home(d)) for s, d in fold_ops))
            if (fold_cost > 0 and naive_cost > fold_cost
                    and naive_cost >= min_gain * fold_cost):
                for s, d in fold_ops:                 # fold (compute)
                    emit(s, d)
                    counters["local"] += 1
                emit(rep, (target_slot, target_ident))   # single carry to target
                counters["crossing"] += 1
                for s, d in reversed(fold_ops):       # uncompute (restores all)
                    emit(s, d)
                    counters["local"] += 1
                counters["folded"] += 1
            else:
                for pair in members:                  # not worth folding: direct
                    emit(pair, (target_slot, target_ident))
                    counters["crossing"] += 1
                counters["direct"] += 1

        m = self.state.metrics
        m["multicnot_parity_rewrite_windows"] = int(
            m.get("multicnot_parity_rewrite_windows", 0)) + 1
        m["multicnot_parity_local_cnots"] = int(
            m.get("multicnot_parity_local_cnots", 0)) + counters["local"]
        m["multicnot_parity_crossing_cnots"] = int(
            m.get("multicnot_parity_crossing_cnots", 0)) + counters["crossing"]
        m["multicnot_parity_groups_folded"] = int(
            m.get("multicnot_parity_groups_folded", 0)) + counters["folded"]
        m["multicnot_parity_groups_direct"] = int(
            m.get("multicnot_parity_groups_direct", 0)) + counters["direct"]
        m["multicnot_parity_max_fold_depth"] = max(
            int(m.get("multicnot_parity_max_fold_depth", 0)), counters["max_depth"])

    def _try_persistent_multicnot_window(self, prog, start_step, cap_bytes=None,
                                         total_cap_bytes=None):
        max_steps = int(os.environ.get("TTN_PERSISTENT_MULTICNOT_MAX_STEPS", "40"))
        min_multis = int(os.environ.get("TTN_PERSISTENT_MULTICNOT_MIN_MULTIS", "2"))
        include_active_clifford = os.environ.get(
            "TTN_PERSISTENT_INCLUDE_ARRAY_CLIFFORD", "0"
        ) not in ("", "0", "false", "False")
        events = []
        support = set()
        multi_count = 0
        end_step = start_step

        for step in range(start_step, min(len(prog), start_step + max_steps)):
            inst = prog[step]
            name = T_mod._opname(inst.opcode)
            a1 = int(inst.axis_1); a2 = int(inst.axis_2)
            if name in ds_mod.IGNORE_OPS:
                events.append(("ignore", step, name, a1, a2, None))
                end_step = step
                continue
            if name in ("OP_FRAME_H", "OP_FRAME_S", "OP_FRAME_S_DAG",
                        "OP_FRAME_CNOT", "OP_FRAME_CZ", "OP_FRAME_SWAP"):
                events.append(("frame", step, name, a1, a2, None))
                end_step = step
                continue
            if name == "OP_ARRAY_ROT":
                ident = self.slot2id.get(a1)
                if ident is not None:
                    support.add(int(ident))
                events.append(("rot", step, name, a1, a2, inst))
                end_step = step
                continue
            if include_active_clifford and name in ("OP_ARRAY_CNOT", "OP_ARRAY_CZ"):
                u = self.slot2id.get(a1)
                v = self.slot2id.get(a2)
                if u is None or v is None:
                    events.append(("two_noop", step, name, a1, a2, None))
                    end_step = step
                    continue
                support.add(int(u))
                support.add(int(v))
                events.append(("two_clifford", step, name, a1, a2,
                               (int(u), int(v))))
                end_step = step
                continue
            if name == "OP_ARRAY_MULTI_CNOT":
                d = ds_mod._d(inst)
                target_ident = self.slot2id.get(a1)
                if target_ident is None:
                    events.append(("multi_noop", step, name, a1, a2, inst))
                    end_step = step
                    continue
                ctrl_slots = []
                ctrl_idents = []
                for ctrl_slot in ds_mod._bits(int(d["mask"])):
                    if ctrl_slot == a1:
                        continue
                    ci = self.slot2id.get(ctrl_slot)
                    if ci is None:
                        continue
                    ctrl_slots.append(int(ctrl_slot))
                    ctrl_idents.append(int(ci))
                    support.add(int(ci))
                support.add(int(target_ident))
                events.append(("multi", step, name, a1, a2,
                               (int(target_ident), ctrl_slots, ctrl_idents)))
                multi_count += 1
                end_step = step
                continue
            break

        if multi_count < min_multis or len(support) <= 1:
            return False
        region = self.state._connected_region_for_bags([self.home_of[i] for i in support])
        est = self.state._estimate_region_workspace_bytes(region)
        stored = self.state._stored_bytes()
        region_stored = int(sum(self.state.bags[bid].resident_bytes for bid in region))
        destructive_est_total = int(stored - region_stored + est)
        if cap_bytes is not None and est > int(cap_bytes):
            return False
        if total_cap_bytes is not None and destructive_est_total > int(total_cap_bytes):
            return False

        self.state.move_center(next(iter(region)))
        stored_before = self.state._stored_bytes()
        region_stored = int(sum(self.state.bags[bid].resident_bytes for bid in region))
        tensor, labels = self.state._contract_region(region)
        actual_workspace = int(tensor.nbytes)
        destructive_total = int(stored_before - region_stored + actual_workspace)
        if cap_bytes is not None and actual_workspace > int(cap_bytes):
            return False
        if total_cap_bytes is not None and destructive_total > int(total_cap_bytes):
            return False

        destructive_open = os.environ.get("TTN_DESTRUCTIVE_OPEN", "0") not in ("", "0", "false", "False")
        if destructive_open:
            for bid in region:
                self.state.bags[bid].tensor = np.array(0.0 + 0.0j, dtype=np.complex128)
            gc.collect()
            live_total = int(self.state._stored_bytes() + actual_workspace)
            self.state.metrics["multicnot_destructive_open_enabled"] = 1
        else:
            live_total = int(stored_before + actual_workspace)
            self.state.metrics["multicnot_destructive_open_enabled"] = 0
        self.state.executor_context = dict(
            selected_executor="persistent_multicnot_region",
            region=sorted(map(int, region)),
            start_step=int(start_step),
            end_step=int(end_step),
            multi_count=int(multi_count),
            support=sorted(map(int, support)),
        )
        self.state._record_total_peak(
            live_total,
            kind="persistent_multicnot_open",
            stored_before=stored_before,
            stored_outside_open_regions=int(stored_before - region_stored),
            absorbed_region_stored_bytes=region_stored,
            open_region_bytes=actual_workspace,
            temporary_bytes=0,
            region_size=len(region),
            region=sorted(map(int, region)),
            start_step=start_step,
            end_step=end_step,
            multi_count=multi_count,
        )
        self.state._record_destructive_total_peak(
            destructive_total,
            kind="persistent_multicnot_open",
            stored_before=stored_before,
            stored_outside_open_regions=int(stored_before - region_stored),
            absorbed_region_stored_bytes=region_stored,
            open_region_bytes=actual_workspace,
            temporary_bytes=0,
            region_size=len(region),
            region=sorted(map(int, region)),
            start_step=start_step,
            end_step=end_step,
            multi_count=multi_count,
        )

        frame_lift = os.environ.get("TTN_CLIFFORD_FRAME_LIFT", "0") not in (
            "", "0", "false", "False"
        )
        pending_linear_frame = None
        pending_axis_support = set()

        def _materialize_pending_frame(reason):
            nonlocal tensor, pending_linear_frame, pending_axis_support
            if pending_linear_frame is None:
                return
            axis_map = {}
            for ident in pending_linear_frame.support_idents:
                lab = ('own', int(ident))
                if lab not in labels:
                    raise RuntimeError(
                        f"pending frame ident {ident} absent at materialization {reason}")
                axis_map[int(ident)] = labels.index(lab)
            tensor = pending_linear_frame.materialize_to_tensor(tensor, axis_map)
            self.state.metrics["num_frame_materializations"] = int(
                self.state.metrics.get("num_frame_materializations", 0)) + 1
            self.state.metrics["frame_materialization_reasons"] = {
                **self.state.metrics.get("frame_materialization_reasons", {}),
                reason: int(self.state.metrics.get("frame_materialization_reasons", {}).get(reason, 0)) + 1,
            }
            pending_linear_frame = None
            pending_axis_support = set()

        def _ensure_pending_frame(idents):
            nonlocal pending_linear_frame, pending_axis_support
            idents = {int(x) for x in idents}
            if not idents:
                return None
            if pending_linear_frame is None:
                pending_axis_support = set(idents)
                pending_linear_frame = RegionLinearFrame(sorted(pending_axis_support))
                return pending_linear_frame
            if not idents.issubset(pending_axis_support):
                _materialize_pending_frame("support_growth")
                pending_axis_support = set(idents)
                pending_linear_frame = RegionLinearFrame(sorted(pending_axis_support))
            return pending_linear_frame

        executed_steps = []
        total_controls = 0
        for kind, step, name, a1, a2, payload in events:
            if kind == "ignore":
                executed_steps.append(step)
                continue
            if kind == "frame":
                if name == "OP_FRAME_H":
                    self.frame.h(a1)
                elif name in ("OP_FRAME_S", "OP_FRAME_S_DAG"):
                    self.frame.s_gate(a1)
                elif name == "OP_FRAME_CNOT":
                    self.frame.cnot(a1, a2)
                elif name == "OP_FRAME_CZ":
                    self.frame.cz(a1, a2)
                elif name == "OP_FRAME_SWAP":
                    self.frame.swap(a1, a2)
                executed_steps.append(step)
                continue
            if kind == "rot":
                _materialize_pending_frame("non_clifford_rot")
                ident = self.slot2id.get(a1)
                if ident is not None:
                    d = ds_mod._d(payload)
                    z = complex(d["weight_re"], d["weight_im"])
                    if self.frame.xb(a1):
                        z = z.conjugate()
                    tensor = _apply_diag_on_labeled_tensor(tensor, labels, ident, 1.0, z)
                executed_steps.append(step)
                continue
            if kind == "two_noop":
                executed_steps.append(step)
                continue
            if kind == "two_clifford":
                u, v = payload
                if frame_lift and name == "OP_ARRAY_CNOT":
                    lf = _ensure_pending_frame([u, v])
                    lf.compose_cnot(u, v)
                    self.state.metrics["num_frame_updates"] = int(
                        self.state.metrics.get("num_frame_updates", 0)) + 1
                    self.state.metrics["num_avoided_tensor_applies"] = int(
                        self.state.metrics.get("num_avoided_tensor_applies", 0)) + 1
                else:
                    _materialize_pending_frame("unsupported_two_clifford")
                    U = _CNOT if name == "OP_ARRAY_CNOT" else _CZ
                    tensor = _apply_2q_on_labeled_tensor(tensor, labels, u, v, U)
                if name == "OP_ARRAY_CNOT":
                    self.frame.cnot(a1, a2)
                else:
                    self.frame.cz(a1, a2)
                executed_steps.append(step)
                continue
            if kind == "multi_noop":
                executed_steps.append(step)
                continue
            if kind == "multi":
                target_ident, ctrl_slots, ctrl_idents = payload
                if frame_lift and ctrl_idents:
                    lf = _ensure_pending_frame([target_ident] + list(ctrl_idents))
                    lf.compose_multicnot(target_ident, ctrl_idents)
                    self.state.metrics["num_frame_updates"] = int(
                        self.state.metrics.get("num_frame_updates", 0)) + 1
                    self.state.metrics["num_avoided_tensor_applies"] = int(
                        self.state.metrics.get("num_avoided_tensor_applies", 0)) + len(ctrl_idents)
                    for ctrl_slot in ctrl_slots:
                        self.frame.cnot(ctrl_slot, a1)
                    total_controls += len(ctrl_idents)
                else:
                    _materialize_pending_frame("frame_lift_disabled")
                    for ctrl_slot, ctrl_ident in zip(ctrl_slots, ctrl_idents):
                        tensor = _apply_cnot_on_labeled_tensor(
                            tensor, labels, ctrl_ident, target_ident)
                        self.frame.cnot(ctrl_slot, a1)
                        total_controls += 1
                executed_steps.append(step)

        _materialize_pending_frame("window_close")
        if frame_lift:
            self.state.metrics["frame_lifted_windows"] = int(
                self.state.metrics.get("frame_lifted_windows", 0)) + 1
            self.state.metrics["num_avoided_open_close"] = int(
                self.state.metrics.get("num_avoided_open_close", 0)) + max(0, multi_count - 1)
            self.state.metrics["num_avoided_qr_estimate"] = int(
                self.state.metrics.get("num_avoided_qr_estimate", 0)) + max(0, multi_count - 1)

        qr_count, qr_work = self.state._split_region_back(
            tensor, labels, region, root=min(region))
        self.state.metrics["persistent_multicnot_windows"] = int(
            self.state.metrics.get("persistent_multicnot_windows", 0)) + 1
        self.state.metrics["persistent_multicnot_steps"] = int(
            self.state.metrics.get("persistent_multicnot_steps", 0)) + multi_count
        self.state.metrics["persistent_multicnot_controls"] = int(
            self.state.metrics.get("persistent_multicnot_controls", 0)) + total_controls
        self.state.metrics["qr_work_proxy"] = float(
            self.state.metrics.get("qr_work_proxy", 0.0)) + qr_work
        self.state._record_metrics(
            pair_workspace_bytes=actual_workspace,
            qr_count=qr_count,
            transport_count=0,
            pair_src=min(region),
            pair_dst=max(region),
            workspace_live_in_total=False,
        )
        self._skip_steps.update(s for s in executed_steps if s != start_step)
        return True
    
    def _reset(self):
        self.state = TTNState(
            self.bag_neighbors,
            self.home_of,
            capture_peak_snapshot=self.capture_peak_snapshot,
            trace_recorder=self.trace_recorder,
        )
        self.frame = ds_mod.PauliFrame()
        self.record = {}
        self.slot2id = {}
        self._skip_steps = set()
        self.last_metrics = None
    
    def _finish_metrics(self, steps_completed, total_steps, timeout, elapsed_time_seconds):
        self.state._record_metrics()
        m = dict(self.state.metrics)
        m["steps_completed"] = int(steps_completed)
        m["total_steps"] = int(total_steps)
        m["timeout"] = bool(timeout)
        m["elapsed_time_seconds"] = float(elapsed_time_seconds)
        self.last_metrics = m
        return m

    def run_shot(self, prog, seed, runtime_timeout=None, check_interval=1,
                 rasl_exec_decisions=None, rasl_exec_max_changes=None,
                 rasl_fallback_default_on_unsafe=True, max_steps=None):
        rng = np.random.default_rng(seed)
        self._reset()
        rasl_exec_decisions = rasl_exec_decisions or {}
        rasl_changes_used = 0
        rasl_changes_skipped = 0
        t_start = time.perf_counter()
        total_steps = len(prog)
        if max_steps is None:
            env_max = os.environ.get("TTN_MAX_STEPS")
            max_steps = int(env_max) if env_max else None
        run_steps = total_steps if max_steps is None else min(total_steps, int(max_steps))
        
        noise_sampler = ds_mod.ClifftNoiseSampler(prog, rng)

        for step in range(run_steps):
            if step in self._skip_steps:
                continue
            if runtime_timeout is not None and step % max(1, int(check_interval)) == 0:
                elapsed = time.perf_counter() - t_start
                if elapsed >= runtime_timeout:
                    self._finish_metrics(step, total_steps, True, elapsed)
                    self.last_metrics["truncated_by_max_steps"] = False
                    self.state.cleanup_spills()
                    return self.record
            inst = prog[step]
            name = T_mod._opname(inst.opcode)
            self.state.current_step = step
            self.state.current_op_kind = name
            self.state.executor_context = {}
            # Resident streaming: push idle large bags out-of-core before this
            # op. The op materializes only the operands it touches (via the
            # bag.tensor property), so bags not used this step stay on disk and
            # never co-reside in RAM. No-op unless TTN_RESIDENT_STREAM_CAP_BYTES.
            if self.state._resident_stream_cap() is not None:
                self.state.spill_idle_large_bags(boundary=True)
            if name in ds_mod.IGNORE_OPS:
                continue
            a1 = int(inst.axis_1); a2 = int(inst.axis_2)
            flags = int(getattr(inst, 'flags', 0))
            sign = 1 if (flags & _FLAG_SIGN) else 0
            
            # Frame ops
            if name == "OP_FRAME_H":      self.frame.h(a1); continue
            if name in ("OP_FRAME_S","OP_FRAME_S_DAG"): self.frame.s_gate(a1); continue
            if name == "OP_FRAME_CNOT":   self.frame.cnot(a1, a2); continue
            if name == "OP_FRAME_CZ":     self.frame.cz(a1, a2); continue
            if name == "OP_FRAME_SWAP":   self.frame.swap(a1, a2); continue
            
            if name == "OP_APPLY_PAULI":
                d = ds_mod._d(inst); cond = d.get('condition_idx'); mask = d.get('cp_mask_idx')
                if cond is not None and mask is not None and int(self.record.get(int(cond), 0)) == 1:
                    ds_mod._apply_cp_mask(prog, int(mask), self.frame, rng)
                continue
            if name == "OP_NOISE":
                d = ds_mod._d(inst); site = d.get('noise_site_idx')
                if site is not None:
                    ds_mod._apply_noise_site(prog, int(site), self.frame, rng, noise_sampler)
                continue
            if name == "OP_NOISE_BLOCK":
                d = ds_mod._d(inst)
                start = d.get('start_site', d.get('noise_site_idx', d.get('block_idx')))
                count = d.get('count', 1)
                if start is not None:
                    for site in range(int(start), int(start) + int(count)):
                        ds_mod._apply_noise_site(prog, site, self.frame, rng, noise_sampler)
                continue
            if name == "OP_READOUT_NOISE":
                d = ds_mod._d(inst)
                entry_idx = d.get('readout_noise_idx')
                entries = getattr(prog, 'readout_noise', None)
                if entry_idx is not None and entries is not None:
                    entry = entries[int(entry_idx)]
                    meas_idx = int(entry['meas_idx'])
                    if float(rng.random()) < float(entry['prob']):
                        self.record[meas_idx] = int(self.record.get(meas_idx, 0)) ^ 1
                continue
            
            # Dormant measurements
            if name in ("OP_MEAS_DORMANT_STATIC","OP_MEAS_DORMANT_STATIC_FORCED"):
                d = ds_mod._d(inst); cidx = int(d.get('classical_idx',0))
                self.record[cidx] = self.frame.xb(a1) ^ sign
                continue
            if name in ("OP_MEAS_DORMANT_RANDOM","OP_MEAS_DORMANT_RANDOM_FORCED"):
                d = ds_mod._d(inst); cidx = int(d.get('classical_idx',0))
                m_abs = int(rng.integers(0, 2))
                self.record[cidx] = m_abs ^ sign
                self.frame.set_xz(a1, m_abs, 0)
                continue
            
            # Active ops: EXPAND family
            if name == "OP_EXPAND":
                ident = self.step_to_ident_expand.get(step)
                if ident is None: continue
                self.slot2id[a1] = ident
                self.state.expand(ident)
                continue
            if name == "OP_EXPAND_T":
                ident = self.step_to_ident_expand.get(step)
                if ident is None: continue
                self.slot2id[a1] = ident
                self.state.expand(ident)
                self.state.apply_diag(ident, 1.0,
                                      _T_PHASE_DAG if self.frame.xb(a1) else _T_PHASE)
                continue
            if name == "OP_EXPAND_T_DAG":
                ident = self.step_to_ident_expand.get(step)
                if ident is None: continue
                self.slot2id[a1] = ident
                self.state.expand(ident)
                self.state.apply_diag(ident, 1.0,
                                      _T_PHASE if self.frame.xb(a1) else _T_PHASE_DAG)
                continue
            if name == "OP_EXPAND_ROT":
                d = ds_mod._d(inst)
                ident = self.step_to_ident_expand.get(step)
                if ident is None: continue
                self.slot2id[a1] = ident
                self.state.expand(ident)
                z = complex(d["weight_re"], d["weight_im"])
                if self.frame.xb(a1):
                    z = z.conjugate()
                self.state.apply_diag(ident, 1.0, z)
                continue
            
            # Phase ops on active idents
            if name == "OP_PHASE_T":
                ident = self.slot2id.get(a1)
                if ident is not None: self.state.apply_diag(ident, 1.0, _T_PHASE)
                continue
            if name == "OP_PHASE_T_DAG":
                ident = self.slot2id.get(a1)
                if ident is not None: self.state.apply_diag(ident, 1.0, _T_PHASE_DAG)
                continue
            if name == "OP_PHASE_ROT":
                d = ds_mod._d(inst); ident = self.slot2id.get(a1)
                if ident is not None:
                    self.state.apply_diag(ident, 1.0, complex(d["weight_re"], d["weight_im"]))
                continue
            
            # Single-axis active gates
            if name == "OP_ARRAY_H":
                ident = self.slot2id.get(a1)
                if ident is not None: self.state.apply_1q(ident, _H_MAT)
                self.frame.h(a1)
                continue
            if name == "OP_ARRAY_S":
                ident = self.slot2id.get(a1)
                if ident is not None: self.state.apply_diag(ident, 1.0, 1j)
                self.frame.s_gate(a1)
                continue
            if name == "OP_ARRAY_S_DAG":
                ident = self.slot2id.get(a1)
                if ident is not None: self.state.apply_diag(ident, 1.0, -1j)
                self.frame.s_gate(a1)
                continue
            if name == "OP_ARRAY_T":
                ident = self.slot2id.get(a1)
                if ident is not None:
                    self.state.apply_diag(ident, 1.0,
                                          _T_PHASE_DAG if self.frame.xb(a1) else _T_PHASE)
                continue
            if name == "OP_ARRAY_T_DAG":
                ident = self.slot2id.get(a1)
                if ident is not None:
                    self.state.apply_diag(ident, 1.0,
                                          _T_PHASE if self.frame.xb(a1) else _T_PHASE_DAG)
                continue
            if name == "OP_ARRAY_ROT":
                d = ds_mod._d(inst); ident = self.slot2id.get(a1)
                if ident is not None:
                    z = complex(d["weight_re"], d["weight_im"])
                    if self.frame.xb(a1):
                        z = z.conjugate()
                    self.state.apply_diag(ident, 1.0, z)
                continue
            if name == "OP_ARRAY_U2":
                d = ds_mod._d(inst); ident = self.slot2id.get(a1)
                U, out = _u2_node_matrix_and_frame(prog, d["cp_idx"], self.frame, a1)
                if ident is not None:
                    self.state.apply_1q(ident, U)
                self.frame.set_xz(a1, out & 1, (out >> 1) & 1)
                continue

            # Two-axis active gates
            if name in ("OP_ARRAY_CNOT", "OP_ARRAY_CZ"):
                u = self.slot2id.get(a1); v = self.slot2id.get(a2)
                if u is None or v is None: continue
                U = _CNOT if name == "OP_ARRAY_CNOT" else _CZ
                opc = self.op_class_by_step_axes.get((step, frozenset({u, v})))
                if opc is None:
                    raise RuntimeError(f"step {step}: no op_class for ({u},{v})")
                cls = opc['cls']
                if cls == 'A':
                    self.state.apply_2q_class_A(u, v, U)
                elif cls in ('B', 'C'):
                    path = self._path_for_ordered_idents(u, v, opc)
                    self.state.apply_2q_class_B_path(u, v, U, path)
                if name == "OP_ARRAY_CNOT":
                    self.frame.cnot(a1, a2)
                else:
                    self.frame.cz(a1, a2)
                continue

            if name == "OP_ARRAY_MULTI_CNOT":
                d = ds_mod._d(inst)
                target_slot = a1
                target_ident = self.slot2id.get(target_slot)
                if target_ident is None:
                    continue
                override_ops = rasl_exec_decisions.get(step)
                slots = []
                if override_ops is not None and (
                    rasl_exec_max_changes is None or rasl_changes_used < int(rasl_exec_max_changes)
                ):
                    # Conservative executable RASL subset: only replace a default
                    # active-only MULTI_CNOT localization window by an explicit
                    # CNOT sequence over currently-active slots. Correctness is
                    # checked by the caller before large runs.
                    safe = all(getattr(op, "name", None) == "CNOT" and op.b is not None
                               for op in override_ops)
                    if safe:
                        slots = [(int(op.a), int(op.b)) for op in override_ops]
                        rasl_changes_used += 1
                    else:
                        rasl_changes_skipped += 1
                        if not rasl_fallback_default_on_unsafe:
                            raise RuntimeError(f"unsafe RASL exec decision at step {step}")
                if not slots:
                    slots = [(ctrl_slot, target_slot)
                             for ctrl_slot in ds_mod._bits(int(d["mask"]))
                             if ctrl_slot != target_slot]
                if (override_ops is None and slots and
                        os.environ.get("TTN_FUSE_MULTICNOT", "0") not in ("", "0", "false", "False")):
                    cap_env = os.environ.get("TTN_FUSE_MULTICNOT_CAP_BYTES")
                    cap = int(cap_env) if cap_env else None
                    total_cap_env = os.environ.get("TTN_FUSE_MULTICNOT_TOTAL_CAP_BYTES")
                    total_cap = int(total_cap_env) if total_cap_env else None
                    if os.environ.get("TTN_PERSISTENT_MULTICNOT", "0") not in ("", "0", "false", "False"):
                        if self._try_persistent_multicnot_window(
                                prog, step, cap_bytes=cap, total_cap_bytes=total_cap):
                            continue
                    ctrl_idents = []
                    live_slots = []
                    for ctrl_slot, target_slot_cur in slots:
                        if target_slot_cur != target_slot:
                            break
                        ci = self.slot2id.get(ctrl_slot)
                        if ci is not None:
                            ctrl_idents.append(ci)
                            live_slots.append(ctrl_slot)
                    else:
                        applied, reason = self.state.apply_multicnot_region(
                            ctrl_idents, target_ident, cap_bytes=cap,
                            total_cap_bytes=total_cap)
                        if applied:
                            for ctrl_slot in live_slots:
                                self.frame.cnot(ctrl_slot, target_slot)
                            continue
                        self.state.metrics["multicnot_region_fallback"] = int(
                            self.state.metrics.get("multicnot_region_fallback", 0)) + 1
                        fb = self.state.metrics.setdefault("multicnot_region_fallback_reasons", {})
                        fb[reason] = int(fb.get(reason, 0)) + 1
                        if os.environ.get("TTN_FUSE_MULTICNOT_BATCH", "1") not in ("", "0", "false", "False"):
                            pending = list(zip(live_slots, ctrl_idents))
                            executed = set()
                            while pending:
                                batch = []
                                batch_slots = []
                                for ctrl_slot_i, ctrl_ident_i in list(pending):
                                    ok, _ = self.state.multicnot_region_feasible(
                                        batch + [ctrl_ident_i],
                                        target_ident,
                                        cap_bytes=cap,
                                        total_cap_bytes=total_cap,
                                    )
                                    if ok:
                                        batch.append(ctrl_ident_i)
                                        batch_slots.append(ctrl_slot_i)
                                if not batch:
                                    break
                                applied_b, reason_b = self.state.apply_multicnot_region(
                                    batch,
                                    target_ident,
                                    cap_bytes=cap,
                                    total_cap_bytes=total_cap,
                                )
                                if not applied_b:
                                    break
                                self.state.metrics["multicnot_region_batches"] = int(
                                    self.state.metrics.get("multicnot_region_batches", 0)) + 1
                                for ctrl_slot_i in batch_slots:
                                    self.frame.cnot(ctrl_slot_i, target_slot)
                                    executed.add((ctrl_slot_i, target_slot))
                                pending = [(s, i) for s, i in pending if (s, target_slot) not in executed]
                            if executed:
                                slots = [x for x in slots if x not in executed]
                # Exact parity-gather rewrite of the remaining per-control CNOTs:
                # fold same-subtree controls into one accumulator so each tree
                # edge incident to the target hub is crossed once per subtree
                # instead of once per control. Net linear map is unchanged.
                if (os.environ.get("TTN_MULTICNOT_PARITY_REWRITE", "0")
                        not in ("", "0", "false", "False")
                        and override_ops is None
                        and slots
                        and all(ts == target_slot for _, ts in slots)):
                    tid = self.slot2id.get(target_slot)
                    ctrl_pairs = []
                    for ctrl_slot, _ts in slots:
                        if ctrl_slot == target_slot:
                            continue
                        cid = self.slot2id.get(ctrl_slot)
                        if cid is not None:
                            ctrl_pairs.append((ctrl_slot, cid))
                    if tid is not None and ctrl_pairs:
                        self._execute_multicnot_parity_gather(
                            target_slot, tid, ctrl_pairs, step)
                        continue
                for ctrl_slot, target_slot_cur in slots:
                    if ctrl_slot == target_slot_cur:
                        continue
                    target_ident_cur = self.slot2id.get(target_slot_cur)
                    if target_ident_cur is None:
                        continue
                    ctrl_ident = self.slot2id.get(ctrl_slot)
                    if ctrl_ident is None:
                        continue
                    opc = self.op_class_by_step_axes.get(
                        (step, frozenset({ctrl_ident, target_ident_cur})))
                    if opc is None:
                        hu = self.home_of[ctrl_ident]
                        hv = self.home_of[target_ident_cur]
                        if hu == hv:
                            self.state.apply_2q_class_A(ctrl_ident, target_ident_cur, _CNOT)
                        else:
                            path = self.state._tree_path(hu, hv)
                            self.state.apply_2q_class_B_path(
                                ctrl_ident, target_ident_cur, _CNOT, path)
                    elif opc['cls'] == 'A':
                        self.state.apply_2q_class_A(ctrl_ident, target_ident_cur, _CNOT)
                    else:
                        path = self._path_for_ordered_idents(ctrl_ident, target_ident_cur, opc)
                        self.state.apply_2q_class_B_path(
                            ctrl_ident, target_ident_cur, _CNOT, path)
                    self.frame.cnot(ctrl_slot, target_slot_cur)
                continue

            if name == "OP_ARRAY_MULTI_CZ":
                d = ds_mod._d(inst)
                ctrl_ident = self.slot2id.get(a1)
                if ctrl_ident is None:
                    continue
                for target_slot in ds_mod._bits(int(d["mask"])):
                    if target_slot == a1:
                        continue
                    target_ident = self.slot2id.get(target_slot)
                    if target_ident is None:
                        continue
                    opc = self.op_class_by_step_axes.get(
                        (step, frozenset({ctrl_ident, target_ident})))
                    if opc is None:
                        raise RuntimeError(
                            f"step {step}: no op_class for multi CZ ({ctrl_ident},{target_ident})")
                    if opc['cls'] == 'A':
                        self.state.apply_2q_class_A(ctrl_ident, target_ident, _CZ)
                    else:
                        path = self._path_for_ordered_idents(ctrl_ident, target_ident, opc)
                        self.state.apply_2q_class_B_path(
                            ctrl_ident, target_ident, _CZ, path)
                    self.frame.cz(a1, target_slot)
                continue

            if name == "OP_ARRAY_U4":
                d = ds_mod._d(inst)
                ident_lo = self.slot2id.get(a1)
                ident_hi = self.slot2id.get(a2)
                if ident_lo is None or ident_hi is None:
                    continue
                U, out = _u4_node_matrix_and_frame(prog, d["cp_idx"], self.frame, a1, a2)
                opc = self.op_class_by_step_axes.get((step, frozenset({ident_lo, ident_hi})))
                if opc is None:
                    raise RuntimeError(f"step {step}: no op_class for U4 ({ident_lo},{ident_hi})")
                # Clifft U4 matrix basis is |hi,lo> with lo as the least-significant bit.
                if opc['cls'] == 'A':
                    self.state.apply_2q_class_A(ident_hi, ident_lo, U)
                else:
                    path = self._path_for_ordered_idents(ident_hi, ident_lo, opc)
                    self.state.apply_2q_class_B_path(ident_hi, ident_lo, U, path)
                self.frame.set_xz(a1, out & 1, (out >> 1) & 1)
                self.frame.set_xz(a2, (out >> 2) & 1, (out >> 3) & 1)
                continue

            # Active measurements
            if name in ("OP_MEAS_ACTIVE_DIAGONAL","OP_MEAS_ACTIVE_DIAGONAL_FORCED"):
                d = ds_mod._d(inst); cidx = int(d.get('classical_idx',0))
                ident = self.slot2id.get(a1)
                if ident is None: continue
                b = self.state.measure_z(ident, rng)
                del self.slot2id[a1]
                m_abs = b ^ self.frame.xb(a1)
                self.record[cidx] = m_abs ^ sign
                self.frame.set_xz(a1, m_abs, 0)
                continue
            if name in ("OP_MEAS_ACTIVE_INTERFERE","OP_MEAS_ACTIVE_INTERFERE_FORCED"):
                d = ds_mod._d(inst); cidx = int(d.get('classical_idx',0))
                ident = self.slot2id.get(a1)
                if ident is None: continue
                self.state.apply_1q(ident, _H_MAT)
                b_x = self.state.measure_z(ident, rng)
                del self.slot2id[a1]
                m_abs = b_x ^ self.frame.zb(a1)
                self.record[cidx] = m_abs ^ sign
                self.frame.set_xz(a1, m_abs, 0)
                continue
            
            # ARRAY_SWAP: relabel slot2id, frame swap. State tensor unchanged (axis label).
            if name == "OP_ARRAY_SWAP":
                i1 = self.slot2id.get(a1); i2 = self.slot2id.get(a2)
                if i1 is not None: del self.slot2id[a1]
                if i2 is not None: del self.slot2id[a2]
                if i1 is not None: self.slot2id[a2] = i1
                if i2 is not None: self.slot2id[a1] = i2
                self.frame.swap(a1, a2)
                continue
            
            # SWAP_MEAS_INTERFERE: swap, then INTERFERE on swap_to
            if name in ("OP_SWAP_MEAS_INTERFERE","OP_SWAP_MEAS_INTERFERE_FORCED"):
                d = ds_mod._d(inst); cidx = int(d.get('classical_idx',0))
                # array swap
                i1 = self.slot2id.get(a1); i2 = self.slot2id.get(a2)
                if i1 is not None: del self.slot2id[a1]
                if i2 is not None: del self.slot2id[a2]
                if i1 is not None: self.slot2id[a2] = i1
                if i2 is not None: self.slot2id[a1] = i2
                self.frame.swap(a1, a2)
                # measure at slot a2 (now i1)
                ident = self.slot2id.get(a2)
                if ident is None: continue
                self.state.apply_1q(ident, _H_MAT)
                b_x = self.state.measure_z(ident, rng)
                del self.slot2id[a2]
                m_abs = b_x ^ self.frame.zb(a2)
                self.record[cidx] = m_abs ^ sign
                self.frame.set_xz(a2, m_abs, 0)
                continue
            
            # Unsupported opcodes currently fall through.
        
        self._finish_metrics(
            run_steps,
            total_steps,
            False,
            time.perf_counter() - t_start,
        )
        self.last_metrics["truncated_by_max_steps"] = bool(run_steps < total_steps)
        self.last_metrics["rasl_exec_changes_used"] = int(rasl_changes_used)
        self.last_metrics["rasl_exec_changes_skipped"] = int(rasl_changes_skipped)
        self.state.cleanup_spills()
        return self.record
    
    def sample(self, prog, shots, seed=None, num_measurements=None):
        master = np.random.default_rng(seed)
        if num_measurements is None:
            num_measurements = prog.num_measurements
        out = np.zeros((shots, num_measurements), dtype=np.uint8)
        for sh in range(shots):
            sd = int(master.integers(0, 2**63 - 1))
            rec = self.run_shot(prog, sd)
            for cidx, bit in rec.items():
                if 0 <= cidx < num_measurements:
                    out[sh, cidx] = bit
        return out
