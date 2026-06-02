from dataclasses import dataclass
import numpy as np
import chaos_theory as ct

@dataclass
class ChaosSystemState:
    embed_dim: int = 3
    embed_tau: int = 1
    structure_confidence: float = 0.0
    attractor_ratio: float = 1.0
    lyapunov: float = 0.0
    lyapunov_change: float = 0.0
    hurst: float = 0.5
    correlation_dim: float = 3.0
    svd_entropy: float = 0.5
    snr_db: float = 0.0
    npe_ratio: float = 1.0
    memory_decay_exp: float = 2.0
    mutual_info_lambda: float = 10.0
def analyze_stock_chaos(close):
    s = np.asarray(close, dtype=float)
    s = s[np.isfinite(s)]
    state = ChaosSystemState()
    def safe(fn, *a, **kw):
        try: return fn(*a, **kw)
        except: return None
    r = safe(ct.estimate_optimal_params, s)
    if r: state.embed_dim, state.embed_tau, state.structure_confidence = r
    r = safe(ct.attractor_shape_ratio, s, state.embed_dim, state.embed_tau)
    if r is not None: state.attractor_ratio = r
    r = safe(ct.calc_lyapunov_exponent, s, state.embed_dim, state.embed_tau)
    if r is not None: state.lyapunov = r
    r = safe(ct.calc_lyapunov_change_rate, s, state.embed_dim, state.embed_tau)
    if r is not None: state.lyapunov_change = r
    r = safe(ct.calc_hurst_exponent, s)
    if r is not None: state.hurst = r
    r = safe(ct.calc_correlation_dimension, s)
    if r is not None: state.correlation_dim = r
    r = safe(ct.compute_svd_entropy, s, state.embed_dim, state.embed_tau)
    if r is not None: state.svd_entropy = r
    r = safe(ct.estimate_snr, s, state.embed_dim, state.embed_tau)
    if r is not None: state.snr_db = r
    r = safe(ct.compute_npe_ratio, s, state.embed_dim, state.embed_tau)
    if r is not None: state.npe_ratio = r
    r = safe(ct.compute_memory_decay_exponent, s, state.embed_dim, state.embed_tau)
    if r is not None: state.memory_decay_exp = r
    r = safe(ct.compute_mutual_info_decay, s)
    if r is not None: state.mutual_info_lambda = r
    return state
def strategy_selector(state):
    svd, npe, cd, lyap, hst, ar = state.svd_entropy, state.npe_ratio, state.correlation_dim, state.lyapunov, state.hurst, state.attractor_ratio
    def conf(conds):
        scores = []
        for d, t in conds:
            r = abs(d) / max(abs(t), 0.01)
            scores.append(min(1.0, max(0.0, 0.5 + 0.25 * (r - 1.0))))
        return round(float(np.mean(scores)), 2) if scores else 0.5
    if svd > 0.8 and npe < 1.1:
        return {"recommended": "no_trade", "rationale": "pure noise (high SVD entropy + low NPE)", "confidence": 0.85}
    if cd > 5 and npe < 1.3:
        return {"recommended": "stat_arb", "rationale": "high-dim noise, statistical arbitrage only", "confidence": conf([(cd, 5.0), (1.0/max(npe,0.01), 1.0/1.3)])}
    if lyap > 0.3 and hst > 0.6:
        return {"recommended": "momentum", "rationale": "strong chaotic trend", "confidence": conf([(lyap, 0.3), (hst, 0.6)])}
    if cd < 3 and ar > 3:
        return {"recommended": "wyckoff_trend", "rationale": "low-dim with dominant mode", "confidence": conf([(3.0/max(cd,0.1), 1.0), (ar, 3.0)])}
    if cd < 3 and ar < 1.5:
        return {"recommended": "wyckoff_reversal", "rationale": "low-dim balanced, favor reversals", "confidence": conf([(3.0/max(cd,0.1), 1.0), (1.5/max(ar,0.1), 1.0)])}
    return {"recommended": "wyckoff_trend", "rationale": "default", "confidence": 0.40}


def position_adjuster(state, base_pct):
    multipliers = []
    if state.lyapunov_change > 1.5: multipliers.append(0.5)
    if state.snr_db < 0: multipliers.append(0.6)
    elif state.snr_db > 10: multipliers.append(1.3)
    if state.npe_ratio < 1.2: multipliers.append(0.7)
    elif state.npe_ratio > 2: multipliers.append(1.2)
    if state.attractor_ratio > 3: multipliers.append(1.2)
    if state.memory_decay_exp > 2: multipliers.append(0.8)
    if not multipliers: multipliers = [1.0]
    total = np.prod(multipliers)
    return round(float(base_pct * max(0.3, min(1.5, total))), 4)


def weight_adjuster(state, base_weights):
    adj = dict(base_weights)
    tech_keys = {"wyckoff","trend_momentum","candlestick","volume","volatility","relative_strength","orbit_compression","lyapunov","hurst","fractal_dim","attractor_shape"}
    morph_keys = {"wyckoff","candlestick","orbit_compression","attractor_shape"}
    tech_mult = 1.2 if state.snr_db > 10 else (0.7 if state.snr_db < 0 else 1.0)
    morph_mult = 0.7 if state.correlation_dim > 5 else 1.0
    for key in adj:
        if key in tech_keys: adj[key] *= tech_mult
        if key in morph_keys: adj[key] *= morph_mult
    total = sum(adj.values())
    if total > 1e-8:
        for key in adj: adj[key] = round(adj[key] / total, 4)
    return adj


def holding_period_suggestion(lambda_val, beta):
    days = max(3, min(int(lambda_val * beta * 5), 60))
    conf = "high" if days <= 5 else ("medium" if days <= 15 else "low")
    return {"days": days, "confidence": conf}


def sector_sync_score(sector_codes, price_dict):
    codes = list(sector_codes)
    n = len(codes)
    if n < 2:
        return {"sync_index": 1.0, "sync_std": 0.0, "top_pairs": [], "recommendation": "insufficient data"}
    scores, pairs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            a_c, b_c = codes[i], codes[j]
            if a_c not in price_dict or b_c not in price_dict: continue
            a_d = np.asarray(price_dict[a_c], dtype=float)
            b_d = np.asarray(price_dict[b_c], dtype=float)
            if len(a_d) < 10 or len(b_d) < 10: continue
            try:
                sync = ct.compute_phase_synchronization(a_d, b_d)
                scores.append(sync)
                pairs.append((a_c, b_c, round(float(sync), 3)))
            except: pass
    if not scores:
        return {"sync_index": 0.5, "sync_std": 0.0, "top_pairs": [], "recommendation": "insufficient data"}
    sync_mean = round(float(np.mean(scores)), 3)
    sync_std = round(float(np.std(scores)), 3)
    pairs.sort(key=lambda x: -x[2])
    if sync_mean > 0.7: rec = "sector cohesion strong, rotation effective"
    elif sync_mean < 0.3: rec = "individual divergence high, stock-picking focus"
    else: rec = "moderate cohesion, selective rotation"
    return {"sync_index": sync_mean, "sync_std": sync_std, "top_pairs": pairs[:5], "recommendation": rec}