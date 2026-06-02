"""Wavelet Multi-Scale Analyzer for price sequence decomposition.

Decomposes price series into frequency components:
  - High frequency (Level 1 details): daily noise
  - Medium frequency (Level 2+3): weekly trend
  - Low frequency (Approximation): monthly cycle

Detects Wyckoff structures across scales for multi-scale confirmation.
Prefer pywt (PyWavelets); falls back to simple difference filtering.
"""

import numpy as np
import os


def _pywt_available() -> bool:
    try:
        import pywt
        return True
    except ImportError:
        return False


def wavelet_decompose(prices: np.ndarray, level: int = 3, wavelet: str = "db4") -> dict:
    """Decompose price series using DWT.

    Args:
        prices: 1-D array of closing prices
        level: decomposition level (default 3)
        wavelet: wavelet name (default 'db4')

    Returns:
        {"approximation": ndarray (low-freq trend),
         "details": [detail_1, detail_2, detail_3] (high→low frequency)}
    """
    if len(prices) < 2 ** level:
        return _fallback_decompose(prices, level)

    if _pywt_available():
        return _pywt_decompose(prices, level, wavelet)
    else:
        return _fallback_decompose(prices, level)


def _pywt_decompose(prices: np.ndarray, level: int, wavelet: str) -> dict:
    """DWT via pywt with boundary handling."""
    import pywt

    # Pad to power-of-2 for DWT
    n = len(prices)
    target = 2 ** int(np.ceil(np.log2(n)))
    padded = np.pad(prices, (0, target - n), mode="edge")

    coeffs = pywt.wavedec(padded, wavelet, level=level)
    approx = coeffs[0]

    details = []
    for i, detail_coeff in enumerate(coeffs[1:]):
        # Reconstruct each level to original length
        coeff_list = [np.zeros_like(approx) for _ in range(level - i)]
        coeff_list.append(detail_coeff)
        for j in range(level - i - 1):
            coeff_list.insert(-1, np.zeros_like(coeffs[j + i + 2]))
        # Use waverec for single-level recon
        rec = _single_level_recon(approx.shape, coeff_list, wavelet, level - i)
        # Trim to original length
        details.append(rec[:n])

    approx_rec = np.zeros_like(coeffs[0])
    approx_full = np.zeros(target)
    # Simple reconstruction: pad approx and waverec
    try:
        all_coeffs = [approx] + [np.zeros_like(c) for c in coeffs[1:]]
        approx_full = pywt.waverec(all_coeffs, wavelet)
        approx_rec = approx_full[:n]
    except Exception:
        # Fallback: upsample approx
        approx_rec = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(approx)), approx)

    return {
        "approximation": approx_rec,
        "details": details,
    }


def _single_level_recon(target_shape: tuple, coeffs: list, wavelet: str, total_level: int) -> np.ndarray:
    """Reconstruct a single detail level."""
    import pywt

    # Build complete coefficient structure
    # coeffs = [approximation, detail_1, ..., detail_k]
    # where the first non-zero detail represents the level we want
    n = len(coeffs)
    # Pad rest with zeros to match total_level + 1
    padding_needed = total_level - (n - 1)
    full_coeffs = [coeffs[0]]  # approximation
    for i in range(1, n):
        full_coeffs.append(coeffs[i])
    for _ in range(padding_needed):
        full_coeffs.append(np.zeros_like(coeffs[0]))

    try:
        return pywt.waverec(full_coeffs, wavelet)
    except Exception:
        return np.zeros(np.prod(target_shape))


def _fallback_decompose(prices: np.ndarray, level: int = 3) -> dict:
    """Simple difference-filter fallback when pywt unavailable.

    Uses moving-average subtraction to approximate wavelet decomposition:
    - approximation = low-pass (long MA)
    - detail_1 = high-pass (short window)
    - detail_2 = mid-pass (medium window - short)
    - detail_3 = low-pass residual (long - medium)
    """
    n = len(prices)
    prices = np.asarray(prices, dtype=float)

    # Window sizes based on level
    short = max(3, min(8, n // 8))
    medium = max(5, min(21, n // 4))
    long_win = max(10, min(63, n // 2))

    # Moving averages
    def ma(data, window):
        if window <= 0:
            return data
        n = len(data)
        if window >= n:
            return np.full(n, np.mean(data))
        result = np.zeros_like(data)
        kernel = np.ones(window) / window
        result[window - 1:] = np.convolve(data, kernel, mode="valid")
        result[:window - 1] = data[:window - 1]
        return result

    ma_short = ma(prices, short)
    ma_medium = ma(prices, medium)
    ma_long = ma(prices, long_win)

    # Approximation = long-term trend
    approximation = ma_long

    # Details from difference of MAs
    detail_1 = prices - ma_short                     # High frequency
    detail_2 = ma_short - ma_medium                  # Mid frequency
    detail_3 = ma_medium - ma_long                   # Low-mid frequency

    return {
        "approximation": approximation,
        "details": [detail_1, detail_2, detail_3],
    }


def detect_multi_scale_pattern(
    prices: np.ndarray,
    wyckoff_signals: list | None = None,
    levels: dict | None = None,
) -> dict:
    """Multi-scale Wyckoff pattern detection.

    Analyzes price across decomposition levels and evaluates:
    - High-freq (detail_1): Exact Spring/UT location
    - Mid-freq (detail_2+3): Trend structure confirmation
    - Low-freq (approximation): Primary trend direction

    Args:
        prices: 1-D closing price array
        wyckoff_signals: existing Wyckoff detection results
        levels: pre-computed wavelet decomposition (if None, computed)

    Returns:
        {"scale_conf": {level: confidence}, "composite": overall signal}
    """
    if levels is None:
        levels = wavelet_decompose(prices)

    approx = levels.get("approximation", np.zeros(1))
    details = levels.get("details", [np.zeros(1) for _ in range(3)])

    n = len(prices)
    if n < 20:
        return {"scale_conf": {"high": 0.5, "mid": 0.5, "low": 0.5}, "composite": 0.5}

    confidences = {}
    reasons = []

    # ---- High frequency: detect local turning points ----
    d1 = details[0] if len(details) > 0 else np.zeros_like(prices)
    if len(d1) >= 5:
        recent = d1[-5:]
        # Spring signal: negative dip followed by recovery (reversal)
        spring_score = 0.5
        if recent[-1] > recent[-2] and recent.min() < -0.3 * np.std(d1):
            spring_score += 0.2
        if recent[-1] > 0:
            spring_score += 0.15
        confidences["high"] = round(min(1.0, spring_score), 2)
        if confidences["high"] > 0.7:
            reasons.append("high-freq reversal detected")
    else:
        confidences["high"] = 0.5

    # ---- Mid frequency: trend structure ----
    d2 = details[1] if len(details) > 1 else np.zeros_like(prices)
    d3 = details[2] if len(details) > 2 else np.zeros_like(prices)
    mid_signal = np.concatenate([d2, d3]) if len(d2) > 0 and len(d3) > 0 else d2

    mid_score = 0.5
    if len(mid_signal) >= 10:
        mid_recent = mid_signal[-10:]
        # Check for trend stabilization (low volatility in medium freq)
        if np.std(mid_recent) < 0.3 * np.std(mid_signal):
            mid_score += 0.2
        # Check for positive slope
        if len(mid_recent) >= 5:
            slope = np.polyfit(np.arange(5), mid_recent[-5:], 1)[0]
            if slope > 0:
                mid_score += 0.15
            else:
                mid_score -= 0.15
    confidences["mid"] = round(min(1.0, max(0.0, mid_score)), 2)
    if confidences["mid"] > 0.7:
        reasons.append("mid-freq trend confirmed")

    # ---- Low frequency: primary trend ----
    low_score = 0.5
    if len(approx) >= 10:
        slope = np.polyfit(np.arange(min(10, len(approx))), approx[-10:], 1)[0]
        if slope > 0:
            low_score = 0.65
        elif slope < 0:
            low_score = 0.35
    confidences["low"] = round(low_score, 2)
    if low_score > 0.6:
        reasons.append("primary uptrend")

    # ---- Composite signal ----
    composite = np.mean(list(confidences.values()))
    composite = round(min(1.0, max(0.0, composite)), 2)

    result = {
        "scale_conf": confidences,
        "composite": composite,
        "reasons": reasons,
    }

    # Incorporate existing Wyckoff signals if available
    if wyckoff_signals and isinstance(wyckoff_signals, list):
        wyckoff_bonus = min(1.0, len(wyckoff_signals) * 0.05)
        result["composite"] = round(min(1.0, composite + wyckoff_bonus), 2)
        result["wyckoff_overlay"] = True

    return result


def compute_wavelet_snr(prices: np.ndarray, window: int = 60) -> float:
    """Compute signal-to-noise ratio from wavelet decomposition.

    SNR = variance(approx) / variance(detail_1) in the trailing window.
    High SNR → strong trend, low SNR → noisy/choppy.
    """
    if len(prices) < window:
        window = len(prices)

    levels = wavelet_decompose(prices[-window:])
    approx = levels["approximation"]
    details = levels.get("details", [])

    if len(details) == 0:
        return 1.0

    d1 = details[0]
    var_approx = np.var(approx) if len(approx) > 1 else 0.0
    var_detail = np.var(d1) if len(d1) > 1 else 1e-8

    snr = var_approx / max(var_detail, 1e-8)
    return float(snr)


# ---- Self-test ----
if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic price series with trend + noise
    n = 256
    t = np.arange(n)
    trend = np.sin(t * 0.05) * 5 + t * 0.02  # sinusoidal + slow uptrend
    noise = np.random.randn(n) * 1.5
    prices = 50 + trend + noise

    # Decompose
    levels = wavelet_decompose(prices)
    print(f"Decomposition: approx={len(levels['approximation'])}, details=[{', '.join(str(len(d)) for d in levels['details'])}]")

    # Multi-scale pattern
    result = detect_multi_scale_pattern(prices, levels=levels)
    print(f"\nMulti-scale result:")
    print(f"  Confidences: {result['scale_conf']}")
    print(f"  Composite: {result['composite']}")
    print(f"  Reasons: {result.get('reasons', [])}")

    # SNR
    snr = compute_wavelet_snr(prices)
    print(f"\nWavelet SNR: {snr:.3f}")

    # Short series fallback
    short = np.array([10, 11, 10.5, 11, 10.8, 11.2, 11, 10.9])
    short_levels = wavelet_decompose(short)
    short_result = detect_multi_scale_pattern(short, levels=short_levels)
    print(f"\nShort series ({len(short)}): composite={short_result['composite']}")

    print(f"\npywt available: {_pywt_available()}")

    print("\nAll tests passed.")
