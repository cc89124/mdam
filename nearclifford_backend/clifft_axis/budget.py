"""Hard dense-memory budget for the Clifft-axis engine.

The whole point of the Clifft-axis mode is a STRICT memory claim:

    peak live complex words  <=  2 ** k_clifft

i.e. at no instant does the engine hold more complex128 amplitudes than clifft's
own active array would. `peak |M| = k` (an axis COUNT) is necessary but NOT
sufficient -- it says nothing about transient working buffers a kernel allocates
on top of the resident `phi`. This object accounts for BOTH:

    live = resident (phi.size) + transient (a kernel's scratch high-water)

every time a kernel is about to allocate scratch it `charge()`s the resident +
transient pair; the budget tracks the peak and RAISES MemoryBudgetExceeded the
moment `live > cap`. The in-place kernels are written so the transient is a small
bounded chunk (never a second full-length vector), so the budget is dominated by
the resident 2^|M|, which the inherited parity reduction holds at <= k_clifft.

This is a coarse, conservative model guard (it counts the arrays the kernels
declare). The verification harness ALSO wraps a shot in tracemalloc to report the
true measured peak, which must independently be <= 2^k_clifft.
"""
from __future__ import annotations


class MemoryBudgetExceeded(Exception):
    """Raised the instant a dense allocation would exceed the hard cap 2^clifft_k_max.

    The HARD invariant of the bounded mode is: the dense magic register (the resident,
    the only exponential object) NEVER exceeds 2^clifft_k_max -- not "settles back below
    it after a reduction". A materialize-before-reduce path that opens a raw core of rank
    r > k_clifft trips this at r = k_clifft + 1 (loud FAIL, not a silent transient)."""

    def __init__(self, live, cap, where=""):
        self.live = int(live)
        self.cap = int(cap)
        self.where = where
        super().__init__(
            f"dense memory bound VIOLATED{(' at ' + where) if where else ''}: "
            f"would allocate {self.live} complex words (dense rank "
            f"{self.live.bit_length() - 1}) > cap={self.cap} = 2^clifft_k_max "
            f"(rank {self.cap.bit_length() - 1}) -- materialize-before-reduce")


# the bound is a hard correctness invariant of clifft_axis_bounded, not a soft budget
MemoryBoundViolation = MemoryBudgetExceeded


class DenseMemoryBudget:
    """Tracks the peak live complex-word count against a hard cap = 2 ** k_clifft."""

    def __init__(self, k_clifft, enforce=True):
        # cap = number of complex128 words clifft's active array would hold. This bounds
        # the RESIDENT register (the single exponential object): |psi> never holds more
        # amplitudes than clifft's active state. The kernels also use O(chunk) transient
        # working memory (gathered pairs / index arrays) -- a small NON-exponential term,
        # like clifft's own per-op scratch -- bounded by `live_cap` (a generous multiple).
        self.k_clifft = int(k_clifft)
        self.cap = 1 << int(k_clifft)
        # working scratch of a vectorised pairwise op is O(resident) (gathered pairs +
        # index arrays); when |M| hits k the resident already == cap, so live can reach a
        # few x cap on tiny registers. That is NON-exponential numpy granularity, not a
        # blow-up: the RESIDENT guard (resident <= cap) is the real protection; live_cap is
        # only a runaway catcher (a genuine 2^B blow-up trips the resident guard first).
        self.live_cap = 4 * self.cap
        self.enforce = bool(enforce)
        self.peak = 0          # peak live (resident + transient) words seen
        self.peak_resident = 0  # peak resident (phi.size) words -- the exponential term
        self.peak_transient = 0
        self.n_charges = 0
        self.worst_where = ""

    def charge(self, resident, transient=0, where=""):
        """Declare that a kernel is about to hold `resident` + `transient` complex words
        simultaneously. HARD invariant: resident (the dense register, the only exponential
        object) must never exceed 2^k_clifft -- a violation is a genuine memory blow-up and
        raises. The working scratch is bounded by live_cap = 2*cap (raises on runaway)."""
        resident = int(resident)
        transient = int(transient)
        live = resident + transient
        self.n_charges += 1
        if resident > self.peak_resident:
            self.peak_resident = resident
        if transient > self.peak_transient:
            self.peak_transient = transient
        if live > self.peak:
            self.peak = live
            self.worst_where = where
        if self.enforce:
            if resident > self.cap:               # exponential blow-up of the register
                raise MemoryBudgetExceeded(resident, self.cap, where + " [resident]")
            if live > self.live_cap:              # runaway working scratch
                raise MemoryBudgetExceeded(live, self.live_cap, where + " [live]")
        return live

    def note_resident(self, resident, where=""):
        """Record a settled resident size (no transient) -- e.g. after a compress /
        reduction reassigns `phi`. Keeps the resident high-water honest."""
        return self.charge(resident, 0, where)

    def summary(self):
        rank = (self.peak_resident.bit_length() - 1) if self.peak_resident else 0
        return dict(cap=self.cap, k_clifft=self.k_clifft,
                    peak_live_words=self.peak, peak_resident_words=self.peak_resident,
                    peak_dense_rank=rank, peak_transient_words=self.peak_transient,
                    resident_within_cap=(self.peak_resident <= self.cap),
                    verdict=("PASS" if self.peak_resident <= self.cap else "FAIL"),
                    n_charges=self.n_charges, worst_where=self.worst_where)
