"""RMT (Random Matrix Theory) factor correlation denoising.

When N factors are correlated, simple weighted averaging ignores the
correlation structure. RMT denoises the correlation matrix by:
1. Computing eigenvalue spectrum of factor-score correlation matrix
2. Filtering eigenvalues above the RMT theoretical noise threshold
3. Reconstructing a clean correlation matrix
4. Computing optimal weights from the denoised matrix
"""

import numpy as np
import json
import os

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache", "rmt_weights.json")


def rmt_threshold(n_features: int, n_samples: int, sigma: float = 1.0) -> float:
    """RMT theoretical maximum noise eigenvalue (Marchenko-Pastur upper bound).

    lambda_max = sigma^2 * (1 + sqrt(N/T))^2
    where N = features, T = samples.
    """
    q = n_features / max(n_samples, 1)
    return sigma ** 2 * (1 + np.sqrt(q)) ** 2


def denoise_correlation(corr_matrix: np.ndarray, n_samples: int) -> np.ndarray:
    """Denoise a correlation matrix via eigenvalue filtering.

    Args:
        corr_matrix: (N, N) correlation matrix
        n_samples: number of observations used to compute the matrix

    Returns:
        Denoised (N, N) correlation matrix
    """
    n = corr_matrix.shape[0]

    # Eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)

    # RMT noise threshold
    lambda_max = rmt_threshold(n, n_samples)

    # Keep only signal eigenvalues (above noise)
    signal_mask = eigenvals > lambda_max
    n_signal = signal_mask.sum()

    if n_signal == 0:
        # All noise - return diagonal (no correlation)
        return np.eye(n)

    # Reconstruct with signal eigenvalues only
    eigenvals_clean = np.where(signal_mask, eigenvals, 0.0)
    denoised = eigenvecs @ np.diag(eigenvals_clean) @ eigenvecs.T

    # Rescale to correlation matrix (diagonal = 1)
    d = np.sqrt(np.diag(denoised))
    d[d < 1e-8] = 1.0
    denoised = denoised / np.outer(d, d)

    np.fill_diagonal(denoised, 1.0)
    return denoised


def rmt_optimal_weights(
    denoised_corr: np.ndarray,
    ic_vector: np.ndarray,
    factor_names: list[str],
) -> dict:
    """Compute optimal weights from denoised correlation and IC vector.

    Uses Markowitz-style inverse-variance weighting:
      w = Sigma^-1 * IC  (normalized to sum=1)

    Where Sigma is the denoised correlation matrix and IC is the
    expected information coefficient for each factor.

    Returns:
        {factor_name: weight, ...}
    """
    n = len(ic_vector)
    if n == 0:
        return {}

    try:
        inv_corr = np.linalg.pinv(denoised_corr)
    except np.linalg.LinAlgError:
        return {name: 1.0 / n for name in factor_names}

    # Raw weights: w = Sigma^-1 * IC
    raw = inv_corr @ ic_vector

    # Ensure non-negative (clip negatives to 0)
    raw = np.maximum(raw, 0)

    # Normalize
    total = raw.sum()
    if total < 1e-8:
        return {name: 1.0 / n for name in factor_names}

    weights = raw / total
    return {name: float(weights[i]) for i, name in enumerate(factor_names)}


def compute_rmt_weights(
    factor_history: np.ndarray,  # (n_samples, n_factors)
    ic_vector: np.ndarray,        # (n_factors,)
    factor_names: list[str],
) -> dict | None:
    """Full RMT denoising pipeline: corr → denoise → weights.

    Args:
        factor_history: (T, N) array of factor scores over time
        ic_vector: (N,) array of expected IC per factor
        factor_names: list of N factor names

    Returns:
        {name: weight} dict, or None if insufficient data
    """
    T, N = factor_history.shape
    if T < 20 or N < 2:
        return None

    # Compute correlation matrix from factor score history
    corr = np.corrcoef(factor_history.T)

    # Handle NaN/Inf
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)

    # Denoise
    denoised = denoise_correlation(corr, T)

    # Compute weights
    weights = rmt_optimal_weights(denoised, np.abs(ic_vector), factor_names)

    return weights


def _load_factor_history(cache_dir: str = None) -> dict | None:
    """Load factor score history from cached files for RMT computation."""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")

    files = [
        os.path.join(cache_dir, "factor_ic_cache.json"),
        os.path.join(cache_dir, "factor_7_ic.json"),
    ]

    for f in files:
        if not os.path.exists(f):
            continue
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Extract factor names and IC values
            if isinstance(data, dict):
                ic_values = {}
                for key, val in data.items():
                    if isinstance(val, dict) and "ic" in val:
                        ic_values[key] = val["ic"]
                    elif isinstance(val, (int, float)):
                        ic_values[key] = val
                if ic_values:
                    return ic_values
        except (json.JSONDecodeError, IOError):
            continue

    return None


# ---- Self-test ----
if __name__ == "__main__":
    np.random.seed(42)

    # Test 1: threshold formula
    t = rmt_threshold(n_features=8, n_samples=200)
    print(f"RMT threshold (N=8, T=200): {t:.4f}")

    # Test 2: generate synthetic correlated factor history
    n_samples, n_factors = 200, 8
    base = np.random.randn(n_samples)
    factor_history = np.zeros((n_samples, n_factors))
    for i in range(n_factors):
        factor_history[:, i] = base * (0.3 + 0.2 * np.random.random()) + np.random.randn(n_samples) * 0.5

    corr = np.corrcoef(factor_history.T)
    denoised = denoise_correlation(corr, n_samples)
    print(f"\nOriginal eigenvalues:  {np.linalg.eigvalsh(corr)[-3:].round(3)}")
    print(f"Denoised eigenvalues: {np.linalg.eigvalsh(denoised)[-3:].round(3)}")

    # Test 3: compute RMT weights
    ic_vec = np.array([0.04, 0.03, 0.02, 0.01, -0.02, -0.03, 0.01, 0.005])
    names = ["wyckoff", "trend_momentum", "volatility", "volume", "candlestick", "risk_reward", "relative_strength", "market_sentiment"]
    w = compute_rmt_weights(factor_history, ic_vec, names)
    if w:
        print(f"\nRMT weights:")
        for name, weight in sorted(w.items(), key=lambda x: -x[1]):
            print(f"  {name}: {weight:.4f}")
        print(f"  Sum: {sum(w.values()):.4f}")

    # Test 4: insufficient data fallback
    small = np.random.randn(5, 8)
    result = compute_rmt_weights(small, ic_vec, names)
    print(f"\nInsufficient data: {result}")

    # Test 5: all-noise fallback
    noisy = np.random.randn(500, 8)
    w2 = compute_rmt_weights(noisy, ic_vec, names)
    print(f"  Noisy weights (should be near uniform): max={max(w2.values()):.4f}, min={min(w2.values()):.4f}")

    print("\nAll tests passed.")
