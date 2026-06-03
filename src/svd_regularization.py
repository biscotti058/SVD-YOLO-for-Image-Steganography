"""
SVD-based regularization tools, aligned with the course program.

This module fills the gap between the steganography pipeline and the
"SVD for rectangular linear systems" content of the course
(Popolizio, dec. 2025 lecture):

    * Moore–Penrose pseudoinverse  ──  ``pseudoinverse``
    * Condition number κ(A)=σ₁/σₙ   ──  ``condition_number``
    * Truncated SVD pseudoinverse   ──  ``tsvd_pseudoinverse``
    * TSVD denoising (rank-k filt.) ──  ``tsvd_denoise``
    * Least-squares solver          ──  ``solve_least_squares``
    * Minimum-norm solver           ──  ``solve_min_norm``
    * σᵢ = √λᵢ(AᵀA) numerical check ──  ``verify_sigma_lambda``
    * Compression footprint formula ──  ``compression_footprint``
    * TSVD-regularized decoder      ──  ``tsvd_regularized_decode``

These tools are used by `large_scale_eval` and by the notebook to
verify the course theorems numerically and to show TSVD as a
*regularization* of the steganographic decoder against attack noise —
exactly the role TSVD plays in the slide "Overdetermined Systems and
Conditioning".
"""

from __future__ import annotations

from typing import Tuple, Dict, List
import numpy as np


# ---------------------------------------------------------------------------
# Pseudoinverse, condition number, truncated SVD
# ---------------------------------------------------------------------------


def pseudoinverse(A: np.ndarray, tol: float | None = None) -> np.ndarray:
    """
    Moore–Penrose pseudoinverse via SVD.

    For A = U Σ Vᵀ define
        Σ⁺ = diag(1/σ₁, …, 1/σᵣ, 0, …, 0)
    and  A⁺ = V Σ⁺ Uᵀ.

    Values σᵢ ≤ tol are treated as zero.  When tol is None the standard
    NumPy default `max(m,n) · σ₁ · ε_machine` is used.
    """
    A = np.asarray(A, dtype=np.float64)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    if tol is None:
        eps = np.finfo(A.dtype).eps
        tol = max(A.shape) * s[0] * eps
    s_inv = np.where(s > tol, 1.0 / np.maximum(s, tol), 0.0)
    return Vt.T @ (s_inv[:, None] * U.T)


def condition_number(A: np.ndarray) -> float:
    """
    Spectral condition number κ(A) = σ₁(A)/σₙ(A).

    For ill-conditioned matrices (large κ) least-squares is very
    sensitive to perturbations on the right-hand side — the central
    motivation for TSVD regularization (slide 21).
    """
    s = np.linalg.svd(np.asarray(A, dtype=np.float64), compute_uv=False)
    if s[-1] == 0.0:
        return float("inf")
    return float(s[0] / s[-1])


def tsvd_pseudoinverse(A: np.ndarray, k: int) -> np.ndarray:
    """
    Truncated SVD pseudoinverse:
        A_k⁺ = V_k Σ_k⁺ U_kᵀ,
    obtained by keeping only the first k singular components.

    This regularizes the inverse: small (noisy) singular values are
    set to zero rather than inverted to large values.
    """
    A = np.asarray(A, dtype=np.float64)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    k = max(1, min(int(k), len(s)))
    s_inv = np.zeros_like(s)
    s_inv[:k] = 1.0 / s[:k]
    return Vt.T @ (s_inv[:, None] * U.T)


def tsvd_denoise(matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Rank-k TSVD filter of a 2-D matrix:
        A_k = U_k Σ_k V_kᵀ.

    By Eckart–Young (slide 12) this is the optimal rank-k approximation
    in spectral and Frobenius norm; in the presence of noise it removes
    the noise components living in the directions of small σᵢ.
    """
    A = np.asarray(matrix, dtype=np.float64)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    k = max(1, min(int(k), len(s)))
    return (U[:, :k] * s[:k]) @ Vt[:k, :]


# ---------------------------------------------------------------------------
# Rectangular linear systems  Ax = b
# ---------------------------------------------------------------------------


def solve_least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Overdetermined system (m > n):
        x* = arg min ||A x − b||₂ = A⁺ b.

    The solution exists uniquely when rank(A) = n.
    """
    return pseudoinverse(A) @ np.asarray(b, dtype=np.float64)


def solve_min_norm(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Underdetermined system (m < n) consistent:
        x† = arg min { ||x||₂ : A x = b } = A⁺ b.

    Same closed form as least-squares; the SVD separates the column
    space (active) from the null space (unconstrained), see slide 22.
    """
    return pseudoinverse(A) @ np.asarray(b, dtype=np.float64)


def solve_tsvd(A: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """
    Regularized least-squares via truncated SVD (TSVD), slide 21:
        x*_k = A_k⁺ b.

    Smaller k → smoother solution, more regularization, smaller
    sensitivity to noise on b but larger approximation error.
    """
    return tsvd_pseudoinverse(A, k) @ np.asarray(b, dtype=np.float64)


# ---------------------------------------------------------------------------
# SVD ↔ eigen-decomposition  (slide 17)
# ---------------------------------------------------------------------------


def verify_sigma_lambda(A: np.ndarray) -> Dict[str, np.ndarray | float]:
    """
    Numerical verification of  σᵢ(A) = √λᵢ(AᵀA),  cf. slide 17.

    Returns a dictionary with the singular values of A, the
    eigenvalues of AᵀA in decreasing order, the implied σ values
    √λᵢ, and the maximum absolute discrepancy.
    """
    A = np.asarray(A, dtype=np.float64)
    s = np.linalg.svd(A, compute_uv=False)
    eigvals = np.linalg.eigvalsh(A.T @ A)
    eigvals = np.sort(eigvals)[::-1]  # decreasing
    # Eigenvalues of AᵀA can pick up tiny negative numerical errors.
    eigvals = np.clip(eigvals, 0.0, None)
    sigma_from_lambda = np.sqrt(eigvals[: len(s)])
    return {
        "sigma": s,
        "lambda": eigvals[: len(s)],
        "sigma_from_lambda": sigma_from_lambda,
        "max_abs_error": float(np.max(np.abs(s - sigma_from_lambda))),
    }


# ---------------------------------------------------------------------------
# Compression footprint  (slide 16)
# ---------------------------------------------------------------------------


def compression_footprint(m: int, n: int, k: int) -> Dict[str, float]:
    """
    Memory footprint of a rank-k truncated SVD of an m×n matrix,
    compared with the raw mn footprint  (slide 16).

    Returns the ratio  2k (1/m + 1/n)  (the figure used in the slide
    for fairly square matrices).
    """
    raw = float(m * n)
    compressed = float(k * (m + n + 1))
    return {
        "raw_bytes": raw,
        "compressed_bytes": compressed,
        "ratio": compressed / raw if raw > 0 else float("inf"),
        "slide_ratio_2k_invm_plus_invn": 2.0 * k * (1.0 / m + 1.0 / n),
    }


# ---------------------------------------------------------------------------
# TSVD-regularized steganographic decoder
# ---------------------------------------------------------------------------


def tsvd_regularized_decode(stego: np.ndarray,
                            cover_singular_values: List[np.ndarray],
                            secret_U: List[np.ndarray],
                            secret_Vt: List[np.ndarray],
                            alpha_eff: float,
                            k: int) -> np.ndarray:
    """
    TSVD-regularized version of the steganographic decoder.

    Standard decoder:
        Σ̃_S = (σ(stego) − σ(cover)) / α_eff
        Ŝ   = U_S diag(Σ̃_S) V_Sᵀ

    Under attack the small components of Σ̃_S are noise-dominated
    (slide 21: "small singular values amplify noise").  Truncating
    Σ̃_S to its top k entries before reconstruction yields the
    rank-k optimal denoised secret (Eckart–Young, slide 12):

        Ŝ_k = U_S^{(k)} diag(Σ̃_S^{(k)}) V_S^{(k)ᵀ}.

    Parameters
    ----------
    stego : (H, W, 3) uint8 or float ndarray
    cover_singular_values : list of three 1-D arrays (per channel)
    secret_U, secret_Vt   : SVD factors of the secret (per channel)
    alpha_eff : float, the effective embedding strength
    k : int, the truncation rank (Σ̃_S is kept to its top k entries)

    Returns
    -------
    recovered_secret : (H, W, 3) uint8 ndarray
    """
    if stego.ndim != 3:
        raise ValueError("stego must be HxWx3.")
    stego = stego.astype(np.float64)
    h, w, c = stego.shape
    recovered = np.zeros_like(stego)
    for ch in range(c):
        _, s_st, _ = np.linalg.svd(stego[:, :, ch], full_matrices=False)
        sigma_s_recovered = (s_st - cover_singular_values[ch]) / alpha_eff
        # Truncate to the top-k components: TSVD denoising of the
        # recovered singular values of the secret.
        if 0 < k < len(sigma_s_recovered):
            sigma_s_recovered = sigma_s_recovered.copy()
            sigma_s_recovered[k:] = 0.0
        recovered[:, :, ch] = (
            secret_U[ch] @ np.diag(sigma_s_recovered) @ secret_Vt[ch]
        )
    return np.clip(recovered, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def per_channel_condition_number(image: np.ndarray) -> List[float]:
    """Condition number of every colour channel of an RGB image."""
    return [condition_number(image[:, :, c]) for c in range(image.shape[2])]


def per_channel_rank(image: np.ndarray, tol: float | None = None) -> List[int]:
    """
    Numerical rank of every channel, computed as the number of singular
    values above `tol` (default: standard NumPy threshold).
    """
    ranks = []
    for c in range(image.shape[2]):
        s = np.linalg.svd(image[:, :, c].astype(np.float64), compute_uv=False)
        if tol is None:
            eps = np.finfo(np.float64).eps
            t = max(image.shape[:2]) * s[0] * eps
        else:
            t = tol
        ranks.append(int(np.sum(s > t)))
    return ranks
