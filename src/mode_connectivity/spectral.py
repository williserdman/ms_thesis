"""Spectral-filter analysis helpers (pure numpy).

These describe the *explicit linear coefficient axis* gamma -> g(lambda) that the
PoC substrate model exposes. They are framework-agnostic (numpy only) so the
filter-response and conditioning math can be unit-tested on a CPU-only box and
reused by the analysis side of every experiment.

Two filter bases are supported, both linear in the coefficient vector gamma:

  - "mono"  monomial:  g(lambda) = sum_k gamma_k * lambda^k
  - "cheb"  Chebyshev (1st kind): g(lambda) = sum_k gamma_k * T_k(lambda)

Two spectral domains:

  - "adj"  symmetric-normalized adjacency, eigenvalues already in [-1, 1].
  - "lap"  symmetric-normalized Laplacian, eigenvalues in [0, 2]; for the
           Chebyshev basis we rescale to [-1, 1] via lambda_tilde = lambda - 1.
"""

from __future__ import annotations

import numpy as np


def _rescale_for_basis(eigs: np.ndarray, basis: str, domain: str) -> np.ndarray:
    """Map eigenvalues into the basis's natural argument range."""
    eigs = np.asarray(eigs, dtype=float)
    if basis == "cheb" and domain == "lap":
        # Laplacian eigenvalues in [0,2] -> [-1,1] for Chebyshev stability.
        return eigs - 1.0
    return eigs


def basis_matrix(eigs: np.ndarray, K: int, basis: str = "cheb", domain: str = "adj") -> np.ndarray:
    """(n, K+1) matrix B with B[i, k] = phi_k(lambda_i).

    g(lambda) = B @ gamma. This is the Vandermonde matrix (monomial) or the
    Chebyshev design matrix; its conditioning controls how stably a target
    filter response can be turned into coefficients.
    """
    x = _rescale_for_basis(eigs, basis, domain)
    n = x.shape[0]
    B = np.zeros((n, K + 1), dtype=float)
    if basis == "mono":
        for k in range(K + 1):
            B[:, k] = x ** k
    elif basis == "cheb":
        # T_0 = 1, T_1 = x, T_k = 2 x T_{k-1} - T_{k-2}
        if K >= 0:
            B[:, 0] = 1.0
        if K >= 1:
            B[:, 1] = x
        for k in range(2, K + 1):
            B[:, k] = 2.0 * x * B[:, k - 1] - B[:, k - 2]
    else:
        raise ValueError(f"unknown basis {basis!r}")
    return B


def filter_response(gamma: np.ndarray, eigs: np.ndarray, basis: str = "cheb", domain: str = "adj") -> np.ndarray:
    """Realized filter response g(lambda_i) for coefficients gamma."""
    gamma = np.asarray(gamma, dtype=float)
    B = basis_matrix(eigs, len(gamma) - 1, basis=basis, domain=domain)
    return B @ gamma


def condition_number(eigs: np.ndarray, K: int, basis: str = "cheb", domain: str = "adj") -> float:
    """Condition number kappa of the basis design matrix on these eigenvalues.

    High kappa (monomial / Laplacian domain at large K) means the coefficient
    -> response map is ill-conditioned; the idea-09/idea-16 hypotheses tie this
    to weight-space mode-connectivity barrier height.
    """
    B = basis_matrix(eigs, K, basis=basis, domain=domain)
    s = np.linalg.svd(B, compute_uv=False)
    smin = s[-1]
    if smin <= 0:
        return float("inf")
    return float(s[0] / smin)


def response_distance(gamma_a: np.ndarray, gamma_b: np.ndarray, eigs: np.ndarray,
                      basis: str = "cheb", domain: str = "adj") -> float:
    """L2 distance between two filters in *response* space (not coefficient space).

    Used by the filter-space (vs weight-space) barrier comparisons.
    """
    ga = filter_response(gamma_a, eigs, basis, domain)
    gb = filter_response(gamma_b, eigs, basis, domain)
    return float(np.linalg.norm(ga - gb) / np.sqrt(len(eigs)))
