"""Chaos Theory Core - pure numpy+scipy, no system dependencies.
References: Rosenstein(1993), Higuchi(1988), Grassberger-Procaccia(1983),
            Sugihara(2012) CCM, Kim(1999) C-C method.
"""
import numpy as np
from scipy import signal, stats, spatial



# ============================================================
# Group A: Phase Space Reconstruction (App points 1,2,3,7)
# ============================================================

def phase_space_reconstruct(series, dim=3, tau=1):
    s = np.asarray(series, dtype=float)
    n = len(s)
    if n < 4:
        return None
    min_pts = 6
    if n < (dim - 1) * tau + min_pts:
        dim = max(2, (n - min_pts) // tau + 1)
    if n < (dim - 1) * tau + 1:
        return None
    N = n - (dim - 1) * tau
    traj = np.zeros((N, dim))
    for d in range(dim):
        traj[:, d] = s[d * tau: d * tau + N]
    return traj


def estimate_optimal_params(series, max_dim=10):
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 60:
        return (max(2, max_dim//2), 1, 0.0)
    s = (s - s.mean()) / (s.std() or 1.0)
    sig = 1.0
    max_tau = min(50, n // 4)
    tau_range = list(range(1, max_tau + 1))
    m_values = [2, 3, 4, 5]
    r_vals = [0.5 * sig, 1.0 * sig, 1.5 * sig, 2.0 * sig]
    delta_S = np.zeros(len(tau_range))
    for ti, tau in enumerate(tau_range):
        S_m = np.zeros((len(m_values), len(r_vals)))
        for mi, m in enumerate(m_values):
            if n < (m - 1) * tau + 30:
                S_m[mi, :] = np.nan
                continue
            traj = phase_space_reconstruct(s, dim=m, tau=tau)
            if traj is None:
                S_m[mi, :] = np.nan
                continue
            N_pts = len(traj)
            if N_pts > 300:
                idx = np.random.choice(N_pts, 300, replace=False)
                pts = traj[idx]
            else:
                pts = traj
            dists = spatial.distance.pdist(pts, metric="euclidean")
            for ri, r_val in enumerate(r_vals):
                C_r = np.mean(dists < r_val)
                if 0 < C_r < 1.0:
                    S_m[mi, ri] = C_r * (1 - C_r)
                else:
                    S_m[mi, ri] = 0.0
        valid = np.isfinite(S_m)
        if np.any(valid):
            delta_S[ti] = np.nanmax(S_m, axis=0).mean() - np.nanmin(S_m, axis=0).mean()
        else:
            delta_S[ti] = 1.0
    tau_opt = 1
    for ti in range(1, len(delta_S) - 1):
        if delta_S[ti] < delta_S[ti-1] and delta_S[ti] < delta_S[ti+1]:
            tau_opt = tau_range[ti]
            break
    if tau_opt == 1 and len(delta_S) >= 10:
        tau_opt = max(1, tau_range[max(1, np.argmin(delta_S[1:10]) + 1)])
    m_opt = min(max_dim, max(3, int(n ** 0.25) + 1))
    if n >= 120:
        recent_std = np.std(s[-60:])
        older_std = np.std(s[:-60])
        if older_std > 1e-8:
            confidence = round(float(abs(recent_std - older_std) / older_std * 2.0), 2)
        else:
            confidence = 0.0
    else:
        confidence = 0.0
    return (m_opt, tau_opt, confidence)


def compute_mutual_info_decay(series, max_lag=50):
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < max_lag + 10:
        max_lag = max(5, min(30, n // 4))
    n_bins = max(4, min(20, int(np.sqrt(n))))
    bins = np.linspace(s.min() - 1e-8, s.max() + 1e-8, n_bins + 1)
    digitized = np.digitize(s, bins) - 1
    digitized = np.clip(digitized, 0, n_bins - 1)
    mi_values = []
    lags = list(range(1, max_lag + 1))
    for tau in lags:
        n_pairs = n - tau
        if n_pairs < 20:
            continue
        joint = np.zeros((n_bins, n_bins))
        for t in range(n_pairs):
            joint[min(digitized[t], n_bins-1), min(digitized[t+tau], n_bins-1)] += 1
        joint = joint / max(n_pairs, 1)
        p_x = np.sum(joint, axis=1)
        p_y = np.sum(joint, axis=0)
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint[i,j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += joint[i,j] * np.log(joint[i,j] / (p_x[i] * p_y[j]))
        mi_values.append(max(0.0, mi))
    if len(mi_values) < 3:
        return 10.0
    mi_values = np.array(mi_values)
    lags_arr = np.array(lags[:len(mi_values)])
    valid_mi = mi_values > 0
    if valid_mi.sum() < 3:
        return 5.0
    y = np.log(mi_values[valid_mi])
    x = lags_arr[valid_mi]
    slope = np.polyfit(x, y, 1)[0]
    if abs(slope) < 1e-8:
        return 30.0
    lam = -1.0 / slope
    return float(max(1.0, min(60.0, lam)))


def attractor_shape_ratio(series, dim=3, tau=1):
    traj = phase_space_reconstruct(series, dim=dim, tau=tau)
    if traj is None or len(traj) < 5:
        return 1.0
    traj = traj - traj.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(traj, full_matrices=False)
    except np.linalg.LinAlgError:
        return 1.0
    if len(S) < 2 or S[1] < 1e-8:
        return 10.0
    return float(S[0] / S[1])


# ============================================================
# Group B: Lyapunov Exponents (App points 4, 12)
# ============================================================

def calc_lyapunov_exponent(series, dim=3, tau=1, window=20):
    s = np.asarray(series, dtype=float)[-window:]
    s = s[np.isfinite(s)]
    n = len(s)
    if n < dim + tau + 5:
        return 0.0
    traj = phase_space_reconstruct(s, dim=dim, tau=tau)
    if traj is None or len(traj) < 10:
        return 0.0
    N = len(traj)
    d = np.diff(s)
    d = d - np.mean(d)
    crossings = np.where(np.diff(np.signbit(d)))[0]
    t_sep = max(tau, int(np.mean(np.diff(crossings)) / 2)) if len(crossings) > 1 else tau * 2
    max_evolve = min(30, N - t_sep - 5)
    if max_evolve < 3:
        return 0.0
    divergence = np.zeros(max_evolve)
    count = 0
    for j in range(min(N - max_evolve, N)):
        dists = np.linalg.norm(traj[:N - max_evolve] - traj[j], axis=1)
        lo = max(0, j - t_sep)
        hi = min(N - max_evolve, j + t_sep + 1)
        dists[lo:hi] = np.inf
        dists[np.isnan(dists)] = np.inf
        min_idx = np.argmin(dists)
        if min_idx < 0 or min_idx >= N - max_evolve or np.isinf(dists[min_idx]):
            continue
        d_j = np.zeros(max_evolve)
        for k in range(max_evolve):
            if j + k < N and min_idx + k < N:
                d_j[k] = np.linalg.norm(traj[j + k] - traj[min_idx + k])
        if np.any(d_j > 0):
            divergence += np.log(d_j + 1e-8)
            count += 1
    if count < 3:
        return 0.0
    log_div = divergence / max(count, 1)
    use_n = min(len(log_div), 15)
    use_n = max(3, use_n)
    slope = np.polyfit(np.arange(use_n), log_div[:use_n], 1)[0]
    return float(slope)


def calc_lyapunov_change_rate(series, dim=3, tau=1, window=40):
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < window:
        window = max(10, n // 2)
    if n < window:
        return 1.0
    half = window // 2
    older = calc_lyapunov_exponent(s[-window:-half], dim=dim, tau=tau, window=min(half, len(s[-window:-half])))
    recent = calc_lyapunov_exponent(s[-half:], dim=dim, tau=tau, window=min(half, len(s[-half:])))
    denom = max(abs(older), 0.001)
    return float(max(0.0, recent / denom))


# ============================================================
# Group C: Hurst + Fractal + Correlation Dimension (App points 5, 14)
# ============================================================

def calc_hurst_exponent(series, max_lag=None):
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 30:
        return 0.5
    if max_lag is None:
        max_lag = min(n // 3, 100)
    lags = np.unique(np.logspace(np.log10(10), np.log10(max_lag), num=min(12, max_lag//3)).astype(int))
    lags = lags[lags < n // 2]
    if len(lags) < 3:
        return 0.5
    rs_values = np.zeros(len(lags))
    for li, lag in enumerate(lags):
        n_segments = n // lag
        if n_segments < 2:
            rs_values[li] = np.nan
            continue
        rs_sum = 0.0
        div_by = 0
        for seg in range(n_segments):
            chunk = s[seg * lag: (seg + 1) * lag]
            if len(chunk) < 5:
                continue
            mean = chunk.mean()
            cum_dev = np.cumsum(chunk - mean)
            R = cum_dev.max() - cum_dev.min()
            S = chunk.std()
            if S > 1e-8:
                rs_sum += R / S
                div_by += 1
        if div_by > 0:
            rs_values[li] = rs_sum / div_by
    valid = rs_values > 0
    if valid.sum() < 3:
        return 0.5
    x = np.log(lags[valid])
    y = np.log(rs_values[valid])
    return float(max(0.1, min(1.0, np.polyfit(x, y, 1)[0])))


def calc_fractal_dimension(series):
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 30:
        return 1.5
    kmax = min(10, (n - 1) // 4)
    if kmax < 2:
        return 1.5
    L_k = np.zeros(kmax)
    for k in range(1, kmax + 1):
        L_sum = 0.0
        for m in range(k):
            idx = np.arange(m, n - k, k)
            if len(idx) < 2:
                continue
            sub = s[idx]
            L_m = np.sum(np.abs(np.diff(sub))) * (n - 1) / (len(idx) * k)
            L_sum += L_m
        L_k[k - 1] = L_sum / max(k, 1)
    valid = L_k > 0
    if valid.sum() < 3:
        return 1.5
    x = np.log(1.0 / np.arange(1, kmax + 1)[valid])
    y = np.log(L_k[valid])
    return float(max(1.0, min(2.0, abs(np.polyfit(x, y, 1)[0]))))


def calc_correlation_dimension(series, max_dim=10):
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 60:
        return 3.0
    s = (s - s.mean()) / (s.std() or 1.0)
    d_vals = []
    for m in range(2, min(max_dim + 1, 8)):
        traj = phase_space_reconstruct(s, dim=m, tau=1)
        if traj is None or len(traj) < 20:
            continue
        N_pts = len(traj)
        if N_pts > 200:
            idx = np.sort(np.random.choice(N_pts, 200, replace=False))
            pts = traj[idx]
        else:
            pts = traj
        dists = spatial.distance.pdist(pts, metric="euclidean")
        dists_sort = np.sort(dists)
        lo_r = max(dists_sort[0] * 2, dists_sort[max(1, int(len(dists_sort) * 0.05))] + 1e-8)
        hi_r = min(dists_sort[-1] * 0.5, dists_sort[min(len(dists_sort) - 1, int(len(dists_sort) * 0.95))])
        if hi_r <= lo_r:
            d_vals.append(1.0)
            continue
        rs = np.exp(np.linspace(np.log(lo_r), np.log(hi_r), num=15))
        Cr = np.array([np.mean(dists < r_val) for r_val in rs])
        valid = (Cr > 0.001) & (Cr < 0.999)
        if valid.sum() < 3:
            d_vals.append(1.0)
            continue
        x = np.log(rs[valid])
        y = np.log(Cr[valid])
        mid_s = max(1, len(x) // 4)
        mid_e = min(len(x) - 1, 3 * len(x) // 4)
        if mid_e - mid_s < 2:
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = np.polyfit(x[mid_s:mid_e], y[mid_s:mid_e], 1)[0]
        d_vals.append(float(max(0.5, min(m * 2, slope))))
    if not d_vals:
        return 3.0
    d_vals = np.array(d_vals)
    if len(d_vals) >= 2:
        d_changes = np.abs(np.diff(d_vals[-3:])) if len(d_vals) >= 3 else np.abs(np.diff(d_vals[-2:]))
        D2 = float(d_vals[-1]) if np.max(d_changes) < 0.5 else float(d_vals[np.argmin(d_changes) + 1])
    else:
        D2 = float(d_vals[-1])
    return D2


# ============================================================
# Group D: RQA (App points 2, 8, 9, 10)
# ============================================================

def build_recurrence_matrix(series, dim=3, tau=1, threshold=None):
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < dim + tau + 5:
        return np.eye(1)
    traj = phase_space_reconstruct(s, dim=dim, tau=tau)
    if traj is None:
        return np.eye(1)
    N_pts = len(traj)
    if N_pts > 200:
        idx = np.sort(np.random.choice(N_pts, 200, replace=False))
        pts = traj[idx]
    else:
        pts = traj
        idx = np.arange(N_pts)
    P = len(pts)
    dists = np.zeros((P, P))
    for i in range(P):
        dists[i] = np.linalg.norm(pts[i] - pts, axis=1)
    upper_tri = dists[np.triu_indices(P, k=1)]
    if len(upper_tri) == 0:
        return np.eye(1)
    if threshold is None:
        threshold = np.mean(upper_tri) * 0.15
        if threshold < 1e-8:
            threshold = np.std(upper_tri) * 0.1 + 1e-8
    R = (dists < threshold).astype(int)
    np.fill_diagonal(R, 0)
    return R


def _extract_line_lengths(R):
    N = R.shape[0]
    diag_lengths = []
    vert_lengths = []
    for k in range(-(N - 1), N):
        d = np.diag(R, k)
        if len(d) < 2:
            continue
        d_pad = np.concatenate([[0], d, [0]])
        edges = np.diff(d_pad.astype(int))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        for s_i, e_i in zip(starts, ends):
            l = e_i - s_i
            if l >= 2:
                diag_lengths.append(l)
    for j in range(N):
        col = R[:, j]
        if len(col) < 2:
            continue
        c_pad = np.concatenate([[0], col, [0]])
        edges = np.diff(c_pad.astype(int))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        for s_i, e_i in zip(starts, ends):
            l = e_i - s_i
            if l >= 2:
                vert_lengths.append(l)
    return diag_lengths, vert_lengths


def compute_rqa_features(series, dim=3, tau=1, window=20):
    s = np.asarray(series, dtype=float)[-window:]
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 10:
        return {"RR": 0.0, "DET": 0.0, "LAM": 0.0, "avg_diag": 0.0}
    R = build_recurrence_matrix(s, dim=min(dim, max(2, n // 3)), tau=min(tau, max(1, n // 3)))
    if R.size <= 1:
        return {"RR": 0.0, "DET": 0.0, "LAM": 0.0, "avg_diag": 0.0}
    N = R.shape[0]
    total = N * (N - 1)
    n_recurrence = R.sum()
    RR = n_recurrence / max(total, 1)
    diag_lens, vert_lens = _extract_line_lengths(R)
    DET = sum(diag_lens) / max(n_recurrence, 1) if diag_lens else 0.0
    LAM = sum(vert_lens) / max(n_recurrence, 1) if vert_lens else 0.0
    avg_diag = float(np.mean(diag_lens)) if diag_lens else 0.0
    return {
        "RR": round(float(RR), 4),
        "DET": round(float(DET), 4),
        "LAM": round(float(LAM), 4),
        "avg_diag": round(float(avg_diag), 2),
    }


def compute_memory_decay_exponent(series, dim=3, tau=1):
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < dim + tau + 10:
        return 2.0
    R = build_recurrence_matrix(s, dim=dim, tau=tau)
    if R.size <= 1:
        return 2.0
    diag_lens, _ = _extract_line_lengths(R)
    if len(diag_lens) < 5:
        return 2.0
    max_len = max(diag_lens)
    if max_len <= 2:
        return 2.0
    hist = np.bincount(diag_lens, minlength=max_len + 1)
    valid = hist[2:] > 0
    if valid.sum() < 3:
        return 2.0
    l_vals = np.arange(2, max_len + 1)[valid]
    p_vals = hist[2:][valid]
    x = np.log(l_vals)
    y = np.log(p_vals)
    beta = float(-np.polyfit(x, y, 1)[0])
    return max(0.5, min(4.0, beta))


# ============================================================
# Group E: SVD Spectral Analysis (App points 6, 14)
# ============================================================

def compute_svd_entropy(series, dim=3, tau=1):
    traj = phase_space_reconstruct(series, dim=dim, tau=tau)
    if traj is None or len(traj) < 5:
        return 0.5
    traj = traj - traj.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(traj, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.5
    S_sum = S.sum()
    if S_sum < 1e-8:
        return 0.5
    p = S / S_sum
    H_val = 0.0
    for pi in p:
        if pi > 1e-8:
            H_val -= pi * np.log(pi)
    H_norm = H_val / max(np.log(len(S)), 1e-8)
    return float(max(0.0, min(1.0, H_norm)))


def estimate_snr(series, dim=3, tau=1):
    traj = phase_space_reconstruct(series, dim=dim, tau=tau)
    if traj is None or len(traj) < 5:
        return 0.0
    traj = traj - traj.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(traj, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0
    if len(S) < 3:
        return 0.0
    S_norm = S / max(S[0], 1e-8)
    slopes = np.abs(np.diff(S_norm))
    plateau_idx = len(S) - 1
    for i in range(len(slopes) - 1):
        if slopes[i] < 0.01 and slopes[i + 1] < 0.01:
            plateau_idx = i + 1
            break
    p_idx = max(1, min(plateau_idx, len(S) - 2))
    S_signal = S[:p_idx]
    S_noise = S[p_idx:]
    if len(S_noise) == 0:
        return 20.0
    var_noise = float(np.mean(S_noise ** 2))
    var_signal = max(float(np.mean(S_signal ** 2)) - var_noise, 1e-8)
    snr = 10.0 * np.log10(var_signal / max(var_noise, 1e-8))
    return float(max(-20.0, min(60.0, snr)))


# ============================================================
# Group F: Nonlinear Prediction Error (App point 13)
# ============================================================

def compute_npe_ratio(series, dim=3, tau=1, n_shuffle=30):
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < dim + tau + 10:
        return 1.0
    traj = phase_space_reconstruct(s, dim=dim, tau=tau)
    if traj is None:
        return 1.0
    N = len(traj)
    pred_errors = []
    for i in range(N - 1):
        dists_i = np.linalg.norm(traj[:-1] - traj[i], axis=1)
        dists_i[i] = np.inf
        dists_i[np.isnan(dists_i)] = np.inf
        if np.all(np.isinf(dists_i)):
            continue
        nn_idx = np.argmin(dists_i)
        if nn_idx >= N - 1:
            continue
        pi = i + (dim - 1) * tau + tau
        pn = nn_idx + (dim - 1) * tau + tau
        if pi >= n or pn >= n:
            continue
        pred_errors.append((s[pn] - s[pi]) ** 2)
    if not pred_errors:
        return 1.0
    real_mse = float(np.mean(pred_errors))
    if real_mse < 1e-8:
        return 1.0
    shuffle_mses = []
    for _ in range(n_shuffle):
        shuffled = s.copy()
        np.random.shuffle(shuffled)
        if len(shuffled) < dim + tau + 10:
            continue
        traj_s = phase_space_reconstruct(shuffled, dim=dim, tau=tau)
        if traj_s is None:
            continue
        Ns = len(traj_s)
        s_errors = []
        for i in range(min(100, Ns - 1)):
            dists_i = np.linalg.norm(traj_s[:-1] - traj_s[i], axis=1)
            dists_i[i] = np.inf
            dists_i[np.isnan(dists_i)] = np.inf
            if np.all(np.isinf(dists_i)):
                continue
            nn_idx = np.argmin(dists_i)
            if nn_idx >= Ns - 1:
                continue
            pi = i + (dim - 1) * tau + tau
            pn = nn_idx + (dim - 1) * tau + tau
            if pi >= n or pn >= n:
                continue
            s_errors.append((shuffled[pn] - shuffled[pi]) ** 2)
        if s_errors:
            shuffle_mses.append(float(np.mean(s_errors)))
    if not shuffle_mses:
        return 1.0
    shuffle_mse = float(np.mean(shuffle_mses))
    return float(max(0.1, min(10.0, shuffle_mse / max(real_mse, 1e-8))))


# ============================================================
# Group G: Causality & Synchronization (App points 11, 12)
# ============================================================

def compute_ccm_causality(source, target, dim=3, tau=1, lib_size=None):
    src = np.asarray(source, dtype=float)
    tgt = np.asarray(target, dtype=float)
    src = src[np.isfinite(src)]
    tgt = tgt[np.isfinite(tgt)]
    min_len = min(len(src), len(tgt))
    src = src[:min_len]
    tgt = tgt[:min_len]
    n = min_len
    if n < dim + tau + 10:
        return {"strength": 0.0, "cross_map_skill": 0.0}
    if lib_size is None:
        lib_size = min(n // 2, 100)
    traj = phase_space_reconstruct(src, dim=dim, tau=tau)
    if traj is None:
        return {"strength": 0.0, "cross_map_skill": 0.0}
    N = len(traj)
    lib_n = min(lib_size, N - 1)
    predictions = []
    actuals = []
    for i in range(N):
        nn_dists = np.linalg.norm(traj[:lib_n] - traj[i], axis=1)
        nn_dists[np.isnan(nn_dists)] = np.inf
        k = min(5, lib_n)
        if k < 1:
            continue
        nn_idx = np.argpartition(nn_dists, k)[:k]
        pred = 0.0
        count = 0
        for idx in nn_idx:
            ti = idx + (dim - 1) * tau
            if ti < n:
                pred += tgt[ti]
                count += 1
        if count > 0:
            pred /= count
            ai = i + (dim - 1) * tau
            if ai < n:
                predictions.append(pred)
                actuals.append(tgt[ai])
    if len(predictions) < 5:
        return {"strength": 0.0, "cross_map_skill": 0.0}
    pred_arr = np.array(predictions)
    act_arr = np.array(actuals)
    valid = np.isfinite(pred_arr) & np.isfinite(act_arr)
    if valid.sum() < 5:
        return {"strength": 0.0, "cross_map_skill": 0.0}
    r = stats.pearsonr(pred_arr[valid], act_arr[valid])[0]
    if np.isnan(r):
        r = 0.0
    return {"strength": round(float(r), 4), "cross_map_skill": round(float(abs(r)), 4)}


def compute_phase_synchronization(series_a, series_b, dim=3, tau=1):
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n = max(min(len(a), len(b)), 1)
    a = a[:n]
    b = b[:n]
    if n < 10:
        return 0.5
    a = a - a.mean()
    b = b - b.mean()
    try:
        analytic_a = signal.hilbert(a)
        analytic_b = signal.hilbert(b)
    except Exception:
        return 0.5
    phi_a = np.arctan2(analytic_a.imag, analytic_a.real)
    phi_b = np.arctan2(analytic_b.imag, analytic_b.real)
    delta_phi = phi_a - phi_b
    R = np.abs(np.mean(np.exp(1j * delta_phi)))
    return float(max(0.0, min(1.0, R)))
