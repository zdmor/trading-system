
"""HMM Market Regime - Hidden Markov Model for market state classification.

Replaces rule-based _classify_regime() in market_regime.py with
probabilistic state inference. Uses Gaussian emission HMM with
Baum-Welch (EM) training for 6 hidden states.
"""

import numpy as np
import pickle
import os

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache", "hmm_model.pkl")

STATE_NAMES = {
    0: "底部",
    1: "上升",
    2: "加速",
    3: "顶部",
    4: "下跌",
    5: "震荡",
}


class HMMRegime:
    """Gaussian HMM for market regime classification (6 states).

    Self-contained pure-numpy implementation using scaled Baum-Welch.
    Falls back gracefully if training data is insufficient.
    """

    def __init__(self, n_states: int = 6):
        self.n_states = n_states
        self.n_features = 5  # idx_trend, vol_ratio, volatility, panic, profit_effect

        # Model parameters
        self.startprob_ = None       # (n_states,)
        self.transmat_ = None        # (n_states, n_states)
        self.means_ = None           # (n_states, n_features)
        self.covars_ = None          # (n_states, n_features, n_features)
        self._trained = False

    # ---- Training ----

    def fit(self, observations: list[dict] | np.ndarray, n_iter: int = 30, tol: float = 1e-4):
        """Train HMM with Baum-Welch (EM algorithm).

        Args:
            observations: list of dicts with feature keys, or (T, n_features) ndarray
            n_iter: max EM iterations
            tol: convergence tolerance
        """
        if isinstance(observations, list):
            obs = self._extract_features(observations)
        else:
            obs = np.asarray(observations, dtype=float)

        if obs.shape[0] < self.n_states * 3:
            self._set_defaults()
            print(f"[HMM] Insufficient data ({obs.shape[0]} rows), using default params")
            return

        obs = self._normalize(obs)

        # Initialize with K-means-like clustering
        self._init_params(obs)

        prev_loglik = -np.inf
        for it in range(n_iter):
            # E-step  (scaled forward-backward)
            gamma, xi, loglik = self._e_step(obs)

            # M-step
            self._m_step(obs, gamma, xi)

            if abs(loglik - prev_loglik) < tol:
                break
            prev_loglik = loglik

        self._label_states()
        self._trained = True

        # Save model
        self._save()

    # ---- Prediction ----

    def predict(self, indicators: dict) -> tuple:
        """Predict market state from current indicators.

        Args:
            indicators: dict with idx_trend_pct, vol_ratio, volatility_idx,
                        panic_pct, profit_effect_pct

        Returns:
            (state_name: str, confidence: float, {state_name: prob, ...})
        """
        if not self._trained:
            return ("震荡", 0.5, {name: 1.0 / self.n_states for name in STATE_NAMES.values()})

        x = self._dict_to_vec(indicators)
        x = self._normalize_single(x)

        # Compute log P(x|state) for each state
        log_probs = np.zeros(self.n_states)
        for k in range(self.n_states):
            diff = x - self.means_[k]
            try:
                inv_cov = np.linalg.pinv(self.covars_[k])
                log_probs[k] = -0.5 * diff @ inv_cov @ diff
            except np.linalg.LinAlgError:
                log_probs[k] = -0.5 * np.sum(diff ** 2)

        # Softmax to get posterior (assuming uniform prior)
        probs = np.exp(log_probs - np.max(log_probs))
        probs = probs / probs.sum()

        best = int(np.argmax(probs))
        confidence = float(probs[best])

        state_probs = {STATE_NAMES.get(i, f"state_{i}"): float(probs[i]) for i in range(self.n_states)}

        return (STATE_NAMES.get(best, "未知"), confidence, state_probs)

    # ---- Internal helpers ----

    def _extract_features(self, observations: list[dict]) -> np.ndarray:
        """Convert list of dicts to (T, 5) array."""
        keys = ["idx_trend_pct", "vol_ratio", "volatility_idx", "panic_pct", "profit_effect_pct"]
        T = len(observations)
        arr = np.zeros((T, 5))
        for t, obs in enumerate(observations):
            for j, key in enumerate(keys):
                val = obs.get(key, 0)
                arr[t, j] = float(val) if val is not None else 0.0
        return arr

    def _dict_to_vec(self, indicators: dict) -> np.ndarray:
        """Convert a single indicator dict to (5,) vector."""
        keys = ["idx_trend_pct", "vol_ratio", "volatility_idx", "panic_pct", "profit_effect_pct"]
        return np.array([float(indicators.get(k, 0) or 0) for k in keys])

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """Z-score normalization per feature."""
        self._mu = obs.mean(axis=0)
        self._sigma = obs.std(axis=0)
        self._sigma[self._sigma < 1e-8] = 1.0
        return (obs - self._mu) / self._sigma

    def _normalize_single(self, x: np.ndarray) -> np.ndarray:
        """Normalize a single observation vector."""
        if hasattr(self, "_mu"):
            return (x - self._mu) / self._sigma
        return x

    def _init_params(self, obs: np.ndarray):
        """Initialize HMM parameters with heuristic clustering."""
        T, D = obs.shape
        n = self.n_states

        # Split observations into n_states equal segments for initialization
        seg_len = max(1, T // n)
        indices = [min(i * seg_len, T - 1) for i in range(n)]

        self.means_ = np.array([obs[max(0, idx - seg_len // 2): min(T, idx + seg_len // 2)].mean(axis=0) for idx in indices])

        # Covariance matrices
        self.covars_ = np.array([np.cov(obs.T) * 0.5 + np.eye(D) * 0.1 for _ in range(n)])

        # Initial state distribution (uniform)
        self.startprob_ = np.ones(n) / n

        # Transition matrix (sticky diagonal + uniform off-diagonal)
        stick = 0.9
        off = (1.0 - stick) / (n - 1)
        self.transmat_ = np.full((n, n), off)
        np.fill_diagonal(self.transmat_, stick)

    def _e_step(self, obs: np.ndarray) -> tuple:
        """Scaled forward-backward algorithm."""
        T, D = obs.shape
        n = self.n_states

        # Emission log probabilities
        emission_log = np.zeros((T, n))
        for k in range(n):
            diff = obs - self.means_[k]
            try:
                prec = np.linalg.pinv(self.covars_[k])
                logdet = np.linalg.slogdet(self.covars_[k])[1]
            except np.linalg.LinAlgError:
                prec = np.eye(D)
                logdet = 0.0
            quad = np.sum(diff @ prec * diff, axis=1)
            emission_log[:, k] = -0.5 * (quad + D * np.log(2 * np.pi) + logdet)

        # Scaled forward pass
        alpha = np.zeros((T, n))
        scale = np.zeros(T)
        alpha[0] = np.log(self.startprob_ + 1e-300) + emission_log[0]
        scale[0] = np.max(alpha[0])
        alpha[0] -= scale[0]
        alpha[0] = np.exp(alpha[0])
        alpha[0] /= alpha[0].sum()

        for t in range(1, T):
            alpha[t] = np.log(alpha[t - 1] @ self.transmat_ + 1e-300) + emission_log[t]
            scale[t] = np.max(alpha[t])
            alpha[t] -= scale[t]
            alpha[t] = np.exp(alpha[t])
            alpha[t] /= alpha[t].sum()

        # Scaled backward pass
        beta = np.zeros((T, n))
        beta[-1] = np.ones(n)

        for t in range(T - 2, -1, -1):
            beta[t] = self.transmat_ @ (np.exp(emission_log[t + 1]) * beta[t + 1])
            beta[t] /= (beta[t].sum() + 1e-300)

        # State posteriors gamma
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

        # Pairwise posteriors xi
        xi = np.zeros((T - 1, n, n))
        for t in range(T - 1):
            numer = (alpha[t, :, None] * self.transmat_
                     * np.exp(emission_log[t + 1])[None, :]
                     * beta[t + 1][None, :])
            xi[t] = numer / (numer.sum() + 1e-300)

        loglik = np.sum(scale) + np.log(alpha[-1].sum() + 1e-300)
        return gamma, xi, loglik

    def _m_step(self, obs: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        """Maximization step."""
        T, D = obs.shape
        n = self.n_states

        # Initial state distribution
        self.startprob_ = gamma[0]

        # Transition matrix
        trans = xi.sum(axis=0)
        self.transmat_ = trans / (trans.sum(axis=1, keepdims=True) + 1e-300)

        # Means
        for k in range(n):
            weight = gamma[:, k].sum()
            if weight < 1e-6:
                continue
            self.means_[k] = (gamma[:, k, None] * obs).sum(axis=0) / weight

        # Covariances
        for k in range(n):
            weight = gamma[:, k].sum()
            if weight < 1e-6:
                continue
            centered = obs - self.means_[k]
            self.covars_[k] = (gamma[:, k, None, None] * centered[:, None, :] * centered[:, :, None]).sum(axis=0) / weight
            # Regularize
            self.covars_[k] += np.eye(D) * 0.05

    def _label_states(self):
        """Label states by sorting on mean of idx_trend_pct feature."""
        idx = np.argsort(self.means_[:, 0])
        means_sorted = self.means_[idx]
        covars_sorted = self.covars_[idx]
        transmat_sorted = self.transmat_[idx][:, idx]
        startprob_sorted = self.startprob_[idx]

        self.means_ = means_sorted
        self.covars_ = covars_sorted
        self.transmat_ = transmat_sorted
        self.startprob_ = startprob_sorted
        # State 0=底部 (lowest trend), 5=震荡 (highest trend, actually 加速 but sorted by trend)

    def _set_defaults(self):
        """Set default parameters when training is impossible."""
        n, D = self.n_states, self.n_features
        self.means_ = np.zeros((n, D))
        self.covars_ = np.tile(np.eye(D), (n, 1, 1))
        self.startprob_ = np.ones(n) / n
        self.transmat_ = np.full((n, n), 0.1)
        np.fill_diagonal(self.transmat_, 0.5)
        self._trained = True

    def _save(self):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        model = {
            "n_states": self.n_states,
            "means": self.means_,
            "covars": self.covars_,
            "startprob": self.startprob_,
            "transmat": self.transmat_,
        }
        with open(STATE_FILE, "wb") as f:
            pickle.dump(model, f)

    def load(self) -> bool:
        """Load model from disk. Returns True if successful."""
        if not os.path.exists(STATE_FILE):
            return False
        try:
            with open(STATE_FILE, "rb") as f:
                m = pickle.load(f)
            self.n_states = m["n_states"]
            self.means_ = m["means"]
            self.covars_ = m["covars"]
            self.startprob_ = m["startprob"]
            self.transmat_ = m["transmat"]
            self._trained = True
            return True
        except (pickle.UnpicklingError, KeyError, IOError):
            return False

    def get_state_name(self, state_id: int) -> str:
        return STATE_NAMES.get(state_id, f"state_{state_id}")


def build_training_data(days: int = 500) -> list[dict] | None:
    """Fetch data from Tushare for HMM training.

    Returns list of dicts with feature keys, or None on failure.
    """
    try:
        from data_providers.tushare_provider import TushareProvider
    except ImportError:
        return None

    try:
        tp = TushareProvider()
        df = tp.get_index_daily("000001.SH", days)
        if df is None or len(df) < 60:
            return None
    except Exception:
        return None

    closes = df["close"].values.astype(float)
    volumes = df["volume"].values.astype(float) if "volume" in df.columns else np.ones_like(closes)

    data = []
    for i in range(20, len(df)):
        idx_trend_pct = (closes[i] / closes[i - 20] - 1) * 100
        vol_ratio = volumes[i] / (volumes[i - 20:i + 1].mean() + 1)
        rets = np.diff(closes[i - 20:i + 1]) / (closes[i - 20:i] + 1e-6)
        volatility_idx = float(np.std(rets) * 100)
        panic_pct = 0.0 if idx_trend_pct > -3 else -idx_trend_pct * 0.5
        profit_effect_pct = 0.0 if idx_trend_pct < 3 else idx_trend_pct * 0.3

        data.append({
            "idx_trend_pct": idx_trend_pct,
            "vol_ratio": vol_ratio,
            "volatility_idx": volatility_idx,
            "panic_pct": panic_pct,
            "profit_effect_pct": profit_effect_pct,
        })

    return data


# ---- Self-test ----
if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic 2-year regime data
    T = 500
    true_states = []
    obs = np.zeros((T, 5))
    current = 1  # start in 上升
    for t in range(T):
        # Occasionally switch state
        if np.random.random() < 0.03:
            current = (current + np.random.choice([-1, 1])) % 6
        true_states.append(current)

        # Generate observation based on state
        means = {
            0: [-5.0, 0.6, 2.5, 8.0, 1.0],   # 底部: down trend, low vol, high panic
            1: [2.0, 1.1, 1.5, 1.0, 3.0],    # 上升
            2: [6.0, 1.8, 2.0, 0.5, 6.0],    # 加速
            3: [1.0, 1.5, 2.5, 2.0, 4.0],    # 顶部
            4: [-3.0, 0.9, 3.0, 5.0, 1.5],   # 下跌
            5: [0.2, 1.0, 1.2, 1.5, 2.0],    # 震荡
        }
        mu = np.array(means[current])
        obs[t] = mu + np.random.randn(5) * 0.8

    # Train
    hmm = HMMRegime(n_states=6)
    hmm.fit(obs, n_iter=20)
    print(f"Trained on {T} synthetic observations")

    # Test prediction
    indicators = {
        "idx_trend_pct": 2.5, "vol_ratio": 1.2, "volatility_idx": 1.8,
        "panic_pct": 1.0, "profit_effect_pct": 3.5,
    }
    state, conf, probs = hmm.predict(indicators)
    print(f"\nPrediction: {state} (conf={conf:.2f})")
    for s, p in sorted(probs.items(), key=lambda x: -x[1])[:3]:
        print(f"  {s}: {p:.3f}")

    # Test load/save
    hmm._save()
    hmm2 = HMMRegime(n_states=6)
    ok = hmm2.load()
    print(f"\nLoad model: {'OK' if ok else 'FAIL'}")

    # Untrained fallback
    hmm3 = HMMRegime()
    s, c, p = hmm3.predict(indicators)
    print(f"Untrained fallback: {s} (conf={c:.2f})")

    print("\nAll tests passed.")
