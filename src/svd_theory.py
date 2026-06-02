"""
Mathematical Analysis of SVD-Based Steganography
=================================================

This module provides the theoretical tools used to justify the SVD-based
embedding scheme and to compare *measured* metrics against *predicted*
metrics derived from classical perturbation theory.

The mathematical objects involved are:

    C  ∈ R^{m×n}   cover image channel
    S  ∈ R^{m×n}   secret image channel (resized to match C)
    C = U_C Σ_C V_C^T,    S = U_S Σ_S V_S^T          (SVD)
    C* = U_C (Σ_C + α Σ_S) V_C^T                      (stego, per channel)

The following classical results are used:

1.  **Eckart–Young–Mirsky theorem**.
    For every k ≤ rank(A),

        min_{rank(B) ≤ k}  ‖A − B‖_2 = σ_{k+1}(A),
        min_{rank(B) ≤ k}  ‖A − B‖_F = ( Σ_{i>k} σ_i(A)^2 )^{1/2}.

    Consequence: the SVD truncated to k components is the optimal rank-k
    approximation in both spectral and Frobenius norms.  This is the
    formal reason why "embedding into the singular values" is justified
    as the perturbation that maximally preserves the dominant structure
    of the image.

2.  **Weyl's inequality** for singular values.  For any A, B,

        | σ_i(A + B) − σ_i(A) | ≤ ‖B‖_2 = σ_1(B)        for every i.

    Applied with A = C, B = α S, this gives the per-singular-value
    perturbation bound used during embedding.

3.  **Mirsky's theorem**.

        ( Σ_i ( σ_i(A + B) − σ_i(A) )^2 )^{1/2}  ≤  ‖B‖_F.

    Applied with B = α S, this gives the total Frobenius distortion bound
    used to derive the lower bound on PSNR.

4.  **Frobenius-norm identity for the stego image**.
    Because C* − C = α · U_C Σ_S V_C^T and U_C, V_C are orthogonal,

        ‖C* − C‖_F = α · ‖Σ_S‖_F = α · ‖S‖_F.

    This identity is *exact* (not just a bound) for the embedding scheme
    used here, and gives a closed-form expression for the MSE and hence
    for PSNR.

Symbols used throughout this module:

    sigma         singular-value vector (1D ndarray, sorted descending)
    Sigma_C       singular values of C       (np.ndarray)
    Sigma_S       singular values of S       (np.ndarray)
    alpha         embedding strength         (float, > 0)

All functions are pure (no global state) and operate on real-valued
2-D or 3-D arrays representing single channels or full-colour images.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Core SVD helpers
# ---------------------------------------------------------------------------


def channel_svd(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the (thin) SVD of a 2D real matrix."""
    U, s, Vt = np.linalg.svd(matrix.astype(np.float64), full_matrices=False)
    return U, s, Vt


def singular_values_per_channel(image: np.ndarray) -> List[np.ndarray]:
    """Singular values for every colour channel of a H×W×C image."""
    if image.ndim == 2:
        return [np.linalg.svd(image.astype(np.float64), compute_uv=False)]
    return [
        np.linalg.svd(image[:, :, c].astype(np.float64), compute_uv=False)
        for c in range(image.shape[2])
    ]


# ---------------------------------------------------------------------------
# Eckart–Young–Mirsky: optimal low-rank approximation
# ---------------------------------------------------------------------------


def low_rank_approximation(matrix: np.ndarray, k: int) -> np.ndarray:
    """Optimal rank-k approximation of `matrix` (Eckart–Young–Mirsky)."""
    U, s, Vt = channel_svd(matrix)
    k = max(1, min(k, len(s)))
    return (U[:, :k] * s[:k]) @ Vt[:k, :]


def eckart_young_error(sigma: np.ndarray, k: int) -> Dict[str, float]:
    """
    Theoretical error of the best rank-k approximation of a matrix whose
    singular values are `sigma` (sorted descending).

    Returns the spectral norm error σ_{k+1} and the Frobenius norm error
    sqrt( Σ_{i>k} σ_i^2 ).
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    if k >= len(sigma):
        return {"spectral": 0.0, "frobenius": 0.0}
    return {
        "spectral": float(sigma[k]),
        "frobenius": float(np.sqrt(np.sum(sigma[k:] ** 2))),
    }


def energy_capture_curve(sigma: np.ndarray) -> np.ndarray:
    """
    Cumulative fraction of energy captured by the first k singular values:
        E(k) = ( Σ_{i≤k} σ_i^2 ) / ( Σ_i σ_i^2 ).
    Returns a vector of length len(sigma).
    """
    sigma2 = np.asarray(sigma, dtype=np.float64) ** 2
    total = float(np.sum(sigma2))
    if total == 0.0:
        return np.zeros_like(sigma2)
    return np.cumsum(sigma2) / total


def rank_for_energy(sigma: np.ndarray, fraction: float = 0.95) -> int:
    """
    Smallest k such that the rank-k truncation captures at least `fraction`
    of the energy (squared Frobenius norm) of the matrix.
    """
    curve = energy_capture_curve(sigma)
    idx = int(np.searchsorted(curve, fraction) + 1)
    return min(idx, len(sigma))


# ---------------------------------------------------------------------------
# Perturbation theory: Weyl / Mirsky bounds applied to embedding
# ---------------------------------------------------------------------------


def weyl_bound(Sigma_S: np.ndarray, alpha: float) -> float:
    """
    Per-singular-value perturbation upper bound for the embedding
    Σ_stego = Σ_C + α Σ_S.

    Weyl:  |σ_i(C + αS) − σ_i(C)|  ≤  ‖αS‖_2  =  α · σ_1(S).
    """
    return float(alpha) * float(np.max(Sigma_S))


def mirsky_bound(Sigma_S: np.ndarray, alpha: float) -> float:
    """
    Mirsky bound on the L2 norm of the singular-value perturbation:

        ( Σ_i ( σ_i(C + αS) − σ_i(C) )^2 )^{1/2}  ≤  α · ‖Σ_S‖_2.
    """
    return float(alpha) * float(np.linalg.norm(Sigma_S, 2))


def frobenius_distortion(Sigma_S_per_channel: List[np.ndarray],
                         alpha: float) -> float:
    """
    Exact Frobenius distortion ‖C* − C‖_F summed over channels, using

        ‖C* − C‖_F^2 = α^2 · Σ_i σ_i(S)^2 = α^2 · ‖S‖_F^2 .

    This is an *equality*, not a bound, for the embedding used here.
    """
    total = 0.0
    for s in Sigma_S_per_channel:
        total += float(np.sum(np.asarray(s, dtype=np.float64) ** 2))
    return float(alpha) * float(np.sqrt(total))


# ---------------------------------------------------------------------------
# Closed-form PSNR / MSE predictions
# ---------------------------------------------------------------------------


def predicted_mse(secret_frob_norm: float,
                  alpha: float,
                  num_pixels: int,
                  num_channels: int = 3) -> float:
    """
    Predicted MSE between cover and stego (per-pixel, per-channel)
    derived from the Frobenius identity.

        MSE = ‖C* − C‖_F^2 / (H·W·C)
            = (α · ‖S‖_F)^2 / (H·W·C)            (single channel)
            = (α^2 · Σ_c ‖S_c‖_F^2) / (H·W·C)    (multi-channel)

    Parameters
    ----------
    secret_frob_norm : float
        Frobenius norm of the secret image, summed over channels:
        sqrt( Σ_c Σ_i σ_i(S_c)^2 ).
    alpha : float
        Embedding strength.
    num_pixels : int
        H · W (single-channel pixel count).
    num_channels : int
        Number of channels (default 3).
    """
    return (alpha ** 2) * (secret_frob_norm ** 2) / (num_pixels * num_channels)


def predicted_psnr(secret_frob_norm: float,
                   alpha: float,
                   num_pixels: int,
                   num_channels: int = 3,
                   max_value: float = 255.0) -> float:
    """
    Predicted PSNR derived from `predicted_mse`.

        PSNR_pred = 10 · log10( MAX^2 / MSE_pred ).

    Note
    ----
    The measured PSNR is computed on the *clipped* uint8 stego image, so
    it deviates slightly from this prediction whenever clipping removes
    perturbation energy.  The prediction is therefore a *lower bound* on
    the achievable PSNR (the clipping only ever improves the PSNR by
    truncating outliers), and an *upper bound* on the worst-case PSNR.
    """
    mse = predicted_mse(secret_frob_norm, alpha, num_pixels, num_channels)
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(max_value ** 2 / mse))


def secret_frobenius_norm(secret_img: np.ndarray) -> float:
    """
    Total Frobenius norm of a (possibly multi-channel) secret image:
        sqrt( Σ_c ‖S_c‖_F^2 ).
    """
    arr = secret_img.astype(np.float64)
    return float(np.linalg.norm(arr))


# ---------------------------------------------------------------------------
# Theoretical alpha selection
# ---------------------------------------------------------------------------


def alpha_for_target_psnr(secret_img: np.ndarray,
                          target_psnr_db: float,
                          max_value: float = 255.0) -> float:
    """
    Invert the closed-form PSNR formula to obtain the largest α such that
    the predicted PSNR is at least `target_psnr_db`.

        α* = MAX / ‖S‖_F · sqrt(H·W·C) · 10^(−PSNR_target / 20).

    This gives a principled, image-dependent starting point for the
    embedding strength, replacing the previous "try a few values" rule
    of thumb.
    """
    arr = secret_img.astype(np.float64)
    if arr.ndim == 2:
        h, w = arr.shape
        c = 1
    else:
        h, w, c = arr.shape
    norm = secret_frobenius_norm(arr)
    if norm <= 0.0:
        return float("inf")
    n_pix = h * w * c
    return float(
        max_value * np.sqrt(n_pix)
        * (10.0 ** (-target_psnr_db / 20.0))
        / norm
    )


def alpha_window(secret_img: np.ndarray,
                 psnr_min_db: float = 30.0,
                 psnr_max_db: float = 50.0) -> Tuple[float, float]:
    """
    Recommended interval [α_min, α_max] derived from imperceptibility
    targets (default PSNR ∈ [30, 50] dB):

        α_max = alpha_for_target_psnr(secret, psnr_min_db)   # imperceptibility floor
        α_min = alpha_for_target_psnr(secret, psnr_max_db)   # near-lossless ceiling

    Robustness considerations (QR decodability) further constrain α_min
    from below; this is treated experimentally.
    """
    a_max = alpha_for_target_psnr(secret_img, psnr_min_db)
    a_min = alpha_for_target_psnr(secret_img, psnr_max_db)
    return a_min, a_max


# ---------------------------------------------------------------------------
# Decoding stability analysis
# ---------------------------------------------------------------------------


def decoding_error_bound(noise_frob_norm: float, alpha: float) -> float:
    """
    Worst-case error on the recovered singular values of the secret when
    an attack introduces a perturbation E with ‖E‖_F = noise_frob_norm.

    The decoder recovers  Σ̂_S = (Σ_stego − Σ_C) / α  from the *attacked*
    stego image.  By Mirsky's theorem applied to (C* + E),

        ‖Σ̂_S − Σ_S‖_2  ≤  ‖E‖_F / α.

    The reconstructed secret then satisfies

        ‖Ŝ − S‖_F  ≤  ‖E‖_F / α,

    because the reconstruction Ŝ = U_S diag(Σ̂_S) V_S^T uses the original
    orthogonal U_S, V_S and orthogonal matrices preserve Frobenius norm.

    This formalises the "α large → robust, α small → fragile" intuition.
    """
    if alpha <= 0.0:
        return float("inf")
    return float(noise_frob_norm) / float(alpha)


# ---------------------------------------------------------------------------
# Empirical vs theoretical comparison
# ---------------------------------------------------------------------------


def compare_predicted_measured(cover: np.ndarray,
                               secret: np.ndarray,
                               stego: np.ndarray,
                               alpha: float) -> Dict[str, float]:
    """
    Compare the closed-form prediction against the measured distortion.

    Returns a dict with keys:
        frob_predicted, frob_measured,
        psnr_predicted, psnr_measured,
        weyl_bound_max, mirsky_bound_max
    """
    cover_f = cover.astype(np.float64)
    stego_f = stego.astype(np.float64)
    secret_f = secret.astype(np.float64)

    diff = stego_f - cover_f
    frob_measured = float(np.linalg.norm(diff))
    frob_predicted = float(alpha) * secret_frobenius_norm(secret_f)

    n_pix = cover_f.shape[0] * cover_f.shape[1]
    n_ch = cover_f.shape[2] if cover_f.ndim == 3 else 1

    psnr_predicted = predicted_psnr(
        secret_frobenius_norm(secret_f), alpha, n_pix, n_ch
    )
    mse_measured = float(np.mean(diff ** 2))
    psnr_measured = (
        float("inf") if mse_measured == 0.0
        else float(10.0 * np.log10(255.0 ** 2 / mse_measured))
    )

    # Per-channel Weyl / Mirsky bounds (worst over channels)
    weyl_max = 0.0
    mirsky_max = 0.0
    if secret_f.ndim == 3:
        for c in range(secret_f.shape[2]):
            s = np.linalg.svd(secret_f[:, :, c], compute_uv=False)
            weyl_max = max(weyl_max, weyl_bound(s, alpha))
            mirsky_max = max(mirsky_max, mirsky_bound(s, alpha))
    else:
        s = np.linalg.svd(secret_f, compute_uv=False)
        weyl_max = weyl_bound(s, alpha)
        mirsky_max = mirsky_bound(s, alpha)

    return {
        "frob_predicted": frob_predicted,
        "frob_measured": frob_measured,
        "psnr_predicted": psnr_predicted,
        "psnr_measured": psnr_measured,
        "weyl_bound_max": weyl_max,
        "mirsky_bound_max": mirsky_max,
    }
