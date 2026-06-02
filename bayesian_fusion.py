"""Bayesian Signal Fusion - Convert factor scores to win probability.

Replaces linear weighted scoring with Bayesian posterior:
  P(up | factors) ∝ P(factors | up) × P(up)

Each factor's likelihood is learned from historical backtest data
by binning scores and counting outcomes per bin.
"""

import numpy as np
import json
import os

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache", "bayesian_likelihoods.json")


class BayesianFusion:
    """Converts factor-score dict into P(up) via naive Bayes."""

    BIN_EDGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101]
    BIN_COUNT = len(BIN_EDGES) - 1

    def __init__(self, prior_prob: float = 0.5):
        self.prior = prior_prob
        self.likelihoods = {}  # {factor_key: {"up": [P(score|up) per bin], "down": [...]}}

    # ---- Training ----

    def fit(self, history: list[dict]):
        """Learn P(score_bin | outcome) from historical data.

        Args:
            history: [{"factors": {"wyckoff": 80, ...}, "outcome": 1|0}, ...]
                     where 1 = up, 0 = down.
        """
        if not history:
            return

        outcomes = np.array([h["outcome"] for h in history])

        # Get factor keys from first record
        factor_keys = list(history[0]["factors"].keys())
        up_mask = outcomes == 1
        down_mask = ~up_mask

        for key in factor_keys:
            scores = np.array([h["factors"].get(key, 50) for h in history])
            self.likelihoods[key] = {
                "up": self._bin_and_normalize(scores[up_mask]),
                "down": self._bin_and_normalize(scores[down_mask]),
            }

        # Update prior from data
        self.prior = up_mask.mean() if up_mask.sum() > 0 else 0.5

        self._save()

    def _bin_and_normalize(self, scores: np.ndarray) -> np.ndarray:
        """Bin scores and return normalized probability distribution."""
        counts = np.zeros(self.BIN_COUNT)
        for s in scores:
            s = max(0, min(100, s))
            for j in range(self.BIN_COUNT):
                if self.BIN_EDGES[j] <= s < self.BIN_EDGES[j + 1]:
                    counts[j] += 1
                    break
        # Laplace smoothing
        counts += 0.5
        return counts / counts.sum()

    # ---- Prediction ----

    def predict_proba(self, factor_scores: dict) -> float:
        """Compute P(up | all factors).

        Args:
            factor_scores: {"wyckoff": 75, "trend_momentum": 60, ...}

        Returns:
            Probability of upward move (0.0 - 1.0)
        """
        if not self.likelihoods:
            return self.prior

        log_up = np.log(max(self.prior, 1e-10))
        log_down = np.log(max(1 - self.prior, 1e-10))

        for key, likelihood in self.likelihoods.items():
            score = factor_scores.get(key, 50)
            score = max(0, min(100, score))

            # Find bin
            bin_idx = min(self.BIN_COUNT - 1, int(score // 10))

            p_up = likelihood["up"][bin_idx]
            p_down = likelihood["down"][bin_idx]

            log_up += np.log(max(p_up, 1e-10))
            log_down += np.log(max(p_down, 1e-10))

        # Normalize
        max_log = max(log_up, log_down)
        up = np.exp(log_up - max_log)
        down = np.exp(log_down - max_log)

        return up / (up + down + 1e-10)

    def predict_level(self, prob: float) -> str:
        """Convert probability to human-readable level."""
        if prob >= 0.65:
            return "高"
        elif prob >= 0.40:
            return "中"
        else:
            return "低"

    # ---- Persistence ----

    def _save(self):
        """Save fitted likelihoods to disk."""
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        data = {
            "prior": self.prior,
            "likelihoods": {k: {"up": v["up"].tolist(), "down": v["down"].tolist()} for k, v in self.likelihoods.items()},
        }
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self) -> bool:
        """Load fitted likelihoods from disk."""
        if not os.path.exists(CACHE_FILE):
            return False
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.prior = data["prior"]
            self.likelihoods = {k: {"up": np.array(v["up"]), "down": np.array(v["down"])} for k, v in data["likelihoods"].items()}
            return True
        except (json.JSONDecodeError, KeyError, IOError):
            return False


# ---- Self-test ----
if __name__ == "__main__":
    np.random.seed(42)
    bf = BayesianFusion(prior_prob=0.52)

    # Generate synthetic training data
    history = []
    for _ in range(200):
        factors = {
            "wyckoff": float(np.clip(np.random.normal(60, 15), 0, 100)),
            "trend_momentum": float(np.clip(np.random.normal(50, 12), 0, 100)),
            "volume": float(np.clip(np.random.normal(48, 14), 0, 100)),
            "volatility": float(np.clip(np.random.normal(52, 10), 0, 100)),
        }
        up_prob = 0.5 + 0.005 * (factors["wyckoff"] - 50) + 0.003 * (factors["trend_momentum"] - 50)
        outcome = 1 if np.random.random() < up_prob else 0
        history.append({"factors": factors, "outcome": outcome})

    bf.fit(history)
    print(f"Prior: {bf.prior:.3f}, Factors: {list(bf.likelihoods.keys())}")

    # Test predictions
    tests = [
        {"wyckoff": 85, "trend_momentum": 75, "volume": 60, "volatility": 55},
        {"wyckoff": 25, "trend_momentum": 30, "volume": 40, "volatility": 45},
        {"wyckoff": 50, "trend_momentum": 50, "volume": 50, "volatility": 50},
    ]
    for i, scores in enumerate(tests):
        prob = bf.predict_proba(scores)
        level = bf.predict_level(prob)
        print(f"  Test {i+1}: wyckoff={scores['wyckoff']} → P(up)={prob:.3f} ({level})")

    # Untrained fallback
    bf2 = BayesianFusion()
    print(f"  Untrained: P(up)={bf2.predict_proba(tests[0]):.3f}")

    print("\nAll tests passed.")
