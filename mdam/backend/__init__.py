"""mdam.backend -- a complete near-Clifford simulation backend for clifft
bytecode, sibling to (and independent of) the tensor TTN backend.

The active state is kept in the near-Clifford form  U_C (|0>^{n-k} (x) |phi>_M):
a Clifford tableau that absorbs all Clifford structure for free, plus a dense
magic register over only the |M| <= k qubits that non-Clifford rotations promote.
For the coherent-error QEC circuits k is tiny (0 for the coherent families), so
this backend produces clifft's exact measurement distribution without ever paying
the stabilizer-entanglement bond-dimension cost the tensor backend pays.

Public API:
    NearClifford          -- the verified near-Clifford core (simulator.py)
    NearCliffordBackend   -- the full clifft-bytecode backend (backend.py)
"""
from mdam.backend.simulator import NearClifford
from mdam.backend.backend import NearCliffordBackend, count_idents

__all__ = ["NearClifford", "NearCliffordBackend", "count_idents"]
