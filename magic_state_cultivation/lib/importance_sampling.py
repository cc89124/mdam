"""Stratified importance sampling utilities for magic state cultivation.

Provides:
- Binomial PMF computation (uniform fault probabilities)
- Stratified ratio estimator with Delta Method error bars
- Reweighting across physical error rates
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom


def binomial_pmf(N_sites: int, p: float, max_k: int) -> np.ndarray:
    """Compute the fault-count PMF P(K=k) for k=0..max_k.

    All noise sites share the same probability p (uniform depolarizing
    noise), so K follows a Binomial(N_sites, p) distribution.

    Args:
        N_sites: Number of noise sites in the circuit.
        p: Per-site fault probability.
        max_k: Maximum fault count to compute.

    Returns:
        1D array of shape (max_k + 1,) with P(K=k).
    """
    return binom.pmf(np.arange(max_k + 1), N_sites, p)


class StratumResult:
    """Per-stratum sampling results for the ratio estimator.

    Stores sufficient statistics (sum and sum-of-squares of the metric)
    computed analytically, avoiding large array allocations.
    """

    __slots__ = (
        "k",
        "total_shots",
        "passed_shots",
        "sum_U",
        "sum_U_sq",
    )

    def __init__(
        self,
        k: int,
        total_shots: int,
        passed_shots: int,
        *,
        n_errors: int | None = None,
        U_survivors: np.ndarray | None = None,
    ):
        """
        Args:
            k: Fault count for this stratum.
            total_shots: Total shots attempted (N).
            passed_shots: Shots surviving post-selection (n).
            n_errors: Number of logical errors among survivors (binary metric).
                For binary error counting, sum_U = sum_U_sq = n_errors since
                U_i^2 = U_i for U_i in {0, 1}. This avoids allocating a
                large numpy array.
            U_survivors: 1D array of per-survivor metric values. Use this
                for continuous metrics (e.g., infidelity). For binary errors,
                prefer n_errors instead.

        Exactly one of n_errors or U_survivors must be provided.
        """
        if n_errors is not None and U_survivors is not None:
            raise ValueError("Provide n_errors or U_survivors, not both")
        if n_errors is None and U_survivors is None:
            raise ValueError("Must provide n_errors or U_survivors")

        self.k = k
        self.total_shots = total_shots
        self.passed_shots = passed_shots

        if n_errors is not None:
            self.sum_U = float(n_errors)
            self.sum_U_sq = float(n_errors)  # U_i^2 = U_i for binary
        else:
            self.sum_U = float(np.sum(U_survivors))
            self.sum_U_sq = float(np.sum(U_survivors**2))

    @property
    def s_k(self) -> float:
        """Per-stratum survival rate: n / N."""
        return self.passed_shots / self.total_shots if self.total_shots > 0 else 0.0

    @property
    def u_k(self) -> float:
        """Per-stratum mean metric (over all attempts): sum(U) / N."""
        return self.sum_U / self.total_shots if self.total_shots > 0 else 0.0

    @property
    def var_S(self) -> float:
        """Sample variance of S_i over all N attempts (Bessel-corrected)."""
        N = self.total_shots
        if N <= 1:
            return 0.0
        s = self.s_k
        return s * (1.0 - s) * N / (N - 1)

    @property
    def var_U(self) -> float:
        """Sample variance of U_i over all N attempts (Bessel-corrected)."""
        N = self.total_shots
        if N <= 1:
            return 0.0
        u = self.u_k
        return (self.sum_U_sq / N - u**2) * N / (N - 1)

    @property
    def cov_SU(self) -> float:
        """Sample covariance of (S_i, U_i) over all N attempts (Bessel-corrected).

        Since U_i = 0 for discarded shots (S_i=0) and S_i=1 for survivors:
        Cov(S,U) = E[S*U] - E[S]*E[U] = sum(U)/N - s_k * u_k
        """
        N = self.total_shots
        if N <= 1:
            return 0.0
        return (self.sum_U / N - self.s_k * self.u_k) * N / (N - 1)


def ratio_estimate(
    P_K: np.ndarray,
    strata: list[StratumResult],
) -> tuple[float, float]:
    """Compute the stratified ratio estimate with Delta Method error bars.

    Estimates I = mu_U / mu_S where:
        mu_S = sum_k P(K=k) * s_k   (weighted survival rate)
        mu_U = sum_k P(K=k) * u_k   (weighted metric rate)

    Args:
        P_K: 1D array of PMF weights P(K=k).
        strata: List of StratumResult objects (one per sampled k).

    Returns:
        (estimate, std_error): The ratio estimate and its 1-sigma error bar.
    """
    mu_S = 0.0
    mu_U = 0.0
    var_S_global = 0.0
    var_U_global = 0.0
    cov_SU_global = 0.0

    for sr in strata:
        k = sr.k
        if k >= len(P_K):
            continue
        w = P_K[k]
        if w < 1e-30 or sr.total_shots == 0:
            continue

        mu_S += w * sr.s_k
        mu_U += w * sr.u_k

        w2_over_N = w**2 / sr.total_shots
        var_S_global += w2_over_N * sr.var_S
        var_U_global += w2_over_N * sr.var_U
        cov_SU_global += w2_over_N * sr.cov_SU

    if mu_S == 0:
        return 0.0, 0.0

    estimate = mu_U / mu_S

    # Delta Method for Var(ratio)
    var_ratio = (1.0 / mu_S**2) * (
        var_U_global
        - 2.0 * estimate * cov_SU_global
        + estimate**2 * var_S_global
    )
    std_err = float(np.sqrt(max(0.0, var_ratio)))

    return estimate, std_err


def survival_rate(
    P_K: np.ndarray,
    strata: list[StratumResult],
) -> float:
    """Compute the overall weighted survival rate."""
    total = 0.0
    for sr in strata:
        k = sr.k
        if k >= len(P_K):
            continue
        w = P_K[k]
        if w < 1e-30 or sr.total_shots == 0:
            continue
        total += w * sr.s_k
    return total
