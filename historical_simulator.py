# -*- coding: utf-8 -*-
"""
historical_simulator.py — 交易系统全历史绩效验证
50 stocks × 2018-2026 × daily scoring + 80 param grid

Usage:
  python historical_simulator.py --stocks 50 --full
  python historical_simulator.py --from-checkpoint
  python historical_simulator.py --sensitivity
  python historical_simulator.py --report-only
"""

import sys, os, json, pickle, time, random, argparse, math
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

# Add trading_system to path
TS_DIR = os.path.dirname(os.path.abspath(__file__)) or "."
sys.path.insert(0, TS_DIR)

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

CACHE_DIR = os.path.join(TS_DIR, "data_cache", "historical")
OUTPUT_DIR = os.path.join(TS_DIR, "output")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "historical_checkpoint.json")

START_DATE = "2018-01-01"
END_DATE = "2026-05-30"

# Existing 12 stocks in portfolio/watchlist
EXISTING_STOCKS = [
    "002050",  # 三花智控
    "600038",  # 中直股份
    "600011",  # 华能国际
    "002714",  # 牧原股份
    "600309",  # 万华化学
    "300750",  # 宁德时代
    "002709",  # 天赐材料
    "300274",  # 阳光电源
    "300308",  # 中际旭创
    "600415",  # 小商品城
    "605099",  # 共创草坪
    "603993",  # 洛阳钼业
]

# CSI 500 sample — pre-generated for reproducibility
CSI500_RANDOM = [
    "000039", "000066", "000100", "000630", "000723",
    "000830", "000878", "000927", "002028", "002074",
    "600023", "600079", "600132", "600153", "600160",
    "600176", "600208", "600233", "600256", "600282",
    "600316", "600329", "600338", "600346", "600352",
    "600380", "600392", "600426", "600502", "600521",
    "600535", "600546", "600563", "600572", "600580",
    "600583", "600588", "600660", "600673", "600685",
    "600699", "600711", "600736", "600783", "600801",
    "600873", "600884", "600909", "601000", "601058",
]

# Parameter grid for sensitivity analysis
PARAM_GRID = {
    "score_threshold": [35, 40, 45, 50, 55],
    "kelly_fraction":  [0.5, 0.75, 1.0, 1.25],
    "stop_loss_days":  [10, 15, 20, 25],
}

# Commission rates
COMMISSION = 0.0003     # 万三
STAMP_TAX = 0.001       # 千一（卖出）
SLIPPAGE = 0.005        # 万五

# Scoring weights (static, from factor_weights BASE_WEIGHTS)
SCORE_WEIGHTS = {
    "trend":        0.20,
    "momentum":     0.20,
    "volume":       0.15,
    "risk_reward":  0.25,
    "volatility":   0.10,
    "relative":     0.10,
}

# ──────────────────────────────────────────────────────────────
# Data Module
# ──────────────────────────────────────────────────────────────

def fetch_kline_akshare(symbol: str, start: str = START_DATE, end: str = END_DATE) -> Optional[pd.DataFrame]:
    """Fetch full daily K-line from akshare"""
    try:
        import akshare as ak
    except ImportError:
        return None
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                 start_date=start.replace("-", ""),
                                 end_date=end.replace("-", ""),
                                 adjust="qfq")
        if df is None or len(df) == 0:
            return None
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume"
        })
        for col in ["open", "close", "high", "low", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return None


def fetch_kline_tx(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch recent daily K-line from Tencent API (max ~320 rows)"""
    code = f"sh{symbol}" if symbol.startswith(("6", "9")) else f"sz{symbol}"
    url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,320,qfq"
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    try:
        stock_data = data.get("data", {})
        if not isinstance(stock_data, dict):
            return None
        days = stock_data.get(code, {})
        if not isinstance(days, dict):
            return None
        klines = days.get("qfqday") or days.get("day")
        if not klines:
            return None
    except (KeyError, IndexError, TypeError):
        return None

    rows = []
    for k in klines:
        try:
            rows.append({
                "date": k[0], "open": float(k[1]), "close": float(k[2]),
                "high": float(k[3]), "low": float(k[4]), "volume": float(k[5]),
            })
        except (ValueError, IndexError):
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_stock_data(symbol: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """Load stock data from cache or fetch. Priority: pickle cache > CSV cache > akshare > Tencent API"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{symbol}.pkl")

    # 1. Pickle cache (full historical range)
    if not force_refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                df = pickle.load(f)
                if len(df) > 500:  # Need substantial data
                    print(f"(pickle {len(df)}r)", end=" ", flush=True)
                    return df
        except Exception:
            pass

    # 2. CSV cache (try both sh_ and sz_ prefixes)
    for prefix in ("sh_", "sz_"):
        csv_path = os.path.join(TS_DIR, "data_cache", "daily", f"{prefix}{symbol}.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, encoding="gbk")
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date").reset_index(drop=True)
                    mask = (df["date"] >= START_DATE) & (df["date"] <= END_DATE)
                    df = df[mask].reset_index(drop=True)
                    if len(df) > 200:
                        with open(cache_path, "wb") as f:
                            pickle.dump(df, f)
                        print(f"(csv {len(df)}r)", end=" ", flush=True)
                        return df
            except Exception:
                pass

    # 3. akshare (full date range, slower but complete)
    df = fetch_kline_akshare(symbol)
    if df is not None and len(df) > 200:
        with open(cache_path, "wb") as f:
            pickle.dump(df, f)
        return df

    # 4. Tencent API fallback (limited to ~320 rows)
    df = fetch_kline_tx(symbol)
    if df is not None and len(df) > 100:
        with open(cache_path, "wb") as f:
            pickle.dump(df, f)
    return df


def prepare_all_stock_data(stocks: list, callback=None) -> dict:
    """Load all stocks into {symbol: DataFrame}"""
    result = {}
    for i, sym in enumerate(stocks):
        if callback:
            callback(i, len(stocks), sym)
        df = load_stock_data(sym)
        if df is not None and len(df) > 200:
            result[sym] = df
    return result


# ──────────────────────────────────────────────────────────────
# Scoring Module (simplified offline version of scoring.py)
# ──────────────────────────────────────────────────────────────

def calc_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Vectorized RSI calculation"""
    deltas = np.diff(closes, prepend=closes[0])
    gains = np.maximum(deltas, 0)
    losses = np.maximum(-deltas, 0)
    avg_gain = np.convolve(gains, np.ones(period) / period, mode='full')[:len(gains)]
    avg_loss = np.convolve(losses, np.ones(period) / period, mode='full')[:len(losses)]
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
    rsi[:period] = 50  # warmup
    rsi = np.where(np.isfinite(rsi), rsi, 50)
    return rsi


def calc_macd(closes: np.ndarray) -> tuple:
    """Return (macd, signal, histogram)"""
    ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean().values
    macd = ema12 - ema26
    signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
    hist = macd - signal
    return macd, signal, hist


def calc_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Compute ATR"""
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.abs(high - prev_close))
    tr = np.maximum(tr, np.abs(low - prev_close))
    atr = pd.Series(tr).rolling(window=period, min_periods=1).mean().values
    return atr


def score_single_stock(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily factor scores for a single stock DataFrame.
    
    Returns DataFrame with columns: date, trend_score, momentum_score, 
    volume_score, risk_reward_score, volatility_score, relative_score, composite
    """
    n = len(df)
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    v = df["volume"].values.astype(float)

    # Compute indicators
    rsi = calc_rsi(c, 14)
    macd, macd_signal, macd_hist = calc_macd(c)
    atr = calc_atr(df, 14)
    ma20 = pd.Series(c).rolling(20, min_periods=1).mean().values
    ma50 = pd.Series(c).rolling(50, min_periods=1).mean().values
    ma200 = pd.Series(c).rolling(200, min_periods=1).mean().values

    # Volatility (20-day annualized)
    ret = np.diff(c, prepend=c[0]) / np.maximum(c, 0.01)
    vol20 = pd.Series(ret).rolling(20, min_periods=1).std().values * np.sqrt(252)

    # Volume ratio vs 20-day average
    vol_ma20 = pd.Series(v).rolling(20, min_periods=1).mean().values
    vol_ratio = v / np.maximum(vol_ma20, 1)

    # ── Factor 1: Trend Score ──
    # Bullish when price > MA50 > MA200
    trend_score = np.full(n, 50.0)
    for i in range(1, n):
        s = 50
        if c[i] > ma20[i]: s += 8
        if c[i] > ma50[i]: s += 12
        if c[i] > ma200[i]: s += 10
        if ma20[i] > ma50[i]: s += 8
        if ma50[i] > ma200[i]: s += 7
        # Penalty for being below MAs
        if c[i] < ma20[i]: s -= 5
        if c[i] < ma50[i]: s -= 8
        trend_score[i] = max(0, min(100, s))

    # ── Factor 2: Momentum Score ──
    momentum_score = np.full(n, 50.0)
    for i in range(1, n):
        s = rsi[i]
        # MACD bonus/penalty
        if macd[i] > macd_signal[i] and macd[i] > 0:
            s += 10
        elif macd[i] < macd_signal[i] and macd[i] < 0:
            s -= 10
        # Recent price change (5-day)
        if i >= 5:
            pct5 = (c[i] - c[i-5]) / max(c[i-5], 0.01)
            if pct5 > 0.05: s += 8
            elif pct5 < -0.05: s -= 8
        momentum_score[i] = max(0, min(100, s))

    # ── Factor 3: Volume Score ──
    volume_score = np.full(n, 50.0)
    for i in range(5, n):
        s = 50
        # Volume ratio
        if vol_ratio[i] > 2.0: s += 15
        elif vol_ratio[i] > 1.5: s += 10
        elif vol_ratio[i] > 1.0: s += 5
        # Price-volume confirmation
        if vol_ratio[i] > 1.2 and c[i] > c[i-1]:
            s += 10
        elif vol_ratio[i] > 1.2 and c[i] < c[i-1]:
            s -= 10
        # Volume trend
        if i >= 5:
            vol_trend = np.mean(v[i-4:i+1]) / max(np.mean(v[i-20:i+1]), 1)
            if vol_trend > 1.3: s += 8
            elif vol_trend < 0.7: s -= 5
        volume_score[i] = max(0, min(100, s))

    # ── Factor 4: Risk-Reward Score ──
    rr_score = np.full(n, 50.0)
    for i in range(20, n):
        # Use 20-day low as stop, 20-day high as target
        stop = min(l[i-19:i+1])
        target = max(h[i-19:i+1])
        if c[i] <= stop:
            rr_score[i] = 10
            continue
        risk = c[i] - stop
        reward = target - c[i]
        if risk <= 0:
            rr_score[i] = 20
            continue
        rr = reward / risk
        if rr >= 3.0: base = 95
        elif rr >= 2.0: base = 80
        elif rr >= 1.5: base = 60
        elif rr >= 1.0: base = 40
        else: base = 20
        # Trend discount
        if c[i] < ma50[i]: base *= 0.7
        rr_score[i] = max(5, min(100, int(base)))

    # ── Factor 5: Volatility Score ──
    vol_score = np.full(n, 50.0)
    for i in range(20, n):
        # Lower volatility = higher score (stability premium)
        if vol20[i] < 0.15: s = 90
        elif vol20[i] < 0.25: s = 75
        elif vol20[i] < 0.35: s = 60
        elif vol20[i] < 0.50: s = 40
        else: s = 20
        # ATR expansion bonus (breakout signal)
        if atr[i] > np.mean(atr[max(0,i-20):i]) * 1.3:
            s = min(100, s + 10)
        vol_score[i] = s

    # ── Factor 6: Relative Strength (simplified — self-referential) ──
    rel_score = np.full(n, 50.0)
    for i in range(20, n):
        # 60-day price percentile within 1-year lookback
        lookback = min(i, 250)
        if lookback > 20:
            percentile = np.sum(c[i-lookback:i+1] <= c[i]) / lookback * 100
        else:
            percentile = 50
        if percentile > 80: s = 90
        elif percentile > 60: s = 70
        elif percentile > 40: s = 50
        elif percentile > 20: s = 30
        else: s = 15
        rel_score[i] = s

    # ── Composite Score ──
    composite = (
        trend_score     * SCORE_WEIGHTS["trend"] +
        momentum_score  * SCORE_WEIGHTS["momentum"] +
        volume_score    * SCORE_WEIGHTS["volume"] +
        rr_score        * SCORE_WEIGHTS["risk_reward"] +
        vol_score       * SCORE_WEIGHTS["volatility"] +
        rel_score       * SCORE_WEIGHTS["relative"]
    )

    return pd.DataFrame({
        "date": df["date"],
        "close": c,
        "trend_score": trend_score,
        "momentum_score": momentum_score,
        "volume_score": volume_score,
        "risk_reward_score": rr_score,
        "volatility_score": vol_score,
        "relative_score": rel_score,
        "composite": composite,
    })


# ──────────────────────────────────────────────────────────────
# Simulation Engine
# ──────────────────────────────────────────────────────────────

class HistoricalSimulator:
    """Walk-forward historical trading simulator"""
    
    def __init__(self, score_threshold: float = 55, kelly_fraction: float = 1.0,
                 stop_loss_days: int = 20, take_profit_pct: float = 0.15,
                 exit_threshold: float = None):
        self.score_threshold = score_threshold
        self.kelly_fraction = kelly_fraction
        self.stop_loss_days = stop_loss_days
        self.take_profit_pct = take_profit_pct
        self.exit_threshold = exit_threshold or (score_threshold - 10)
        self.max_holdings = 5  # Max concurrent positions

    def run(self, score_df: pd.DataFrame, symbol: str) -> dict:
        """Run simulation on a single stock's score data.
        
        Returns dict with:
            trades: list of trade dicts
            equity: list of daily equity values
            metrics: dict of summary metrics
        """
        n = len(score_df)
        if n < 100:
            return {"trades": [], "equity": [], "metrics": {"error": "insufficient_data"}}

        dates = score_df["date"].tolist()
        closes = score_df["close"].values
        composites = score_df["composite"].values

        initial_cash = 100000
        cash = initial_cash
        position = 0        # shares held
        entry_price = 0
        entry_date = None
        entry_idx = 0
        consecutive_losses = 0
        total_value = initial_cash

        trades = []
        equity = []
        daily_values = []

        # Warmup: first 50 days are for indicator warmup only
        for i in range(50, n):
            c = closes[i]
            score = composites[i]
            prev_c = closes[i-1] if i > 0 else c

            # ── Position management ──
            if position > 0:
                # Check exits
                exit_signal = None
                exit_reason = ""

                # 1. Score degradation
                if score < self.exit_threshold:
                    exit_signal = "score_fall"
                    exit_reason = f"score {score:.0f} < {self.exit_threshold}"

                # 2. Time stop
                days_held = i - entry_idx
                if days_held >= self.stop_loss_days:
                    exit_signal = "time_stop"
                    exit_reason = f"held {days_held}d >= {self.stop_loss_days}d"

                # 3. Profit taking (partial)
                pnl_pct = (c - entry_price) / entry_price
                if pnl_pct >= self.take_profit_pct and exit_signal is None:
                    # Sell 1/3, keep 2/3
                    sell_shares = int(position / 3)
                    if sell_shares > 0:
                        sell_price = prev_c * (1 - SLIPPAGE)
                        sell_proceeds = sell_shares * sell_price
                        cash += sell_proceeds * (1 - COMMISSION - STAMP_TAX)
                        position -= sell_shares
                        # Record partial trade
                        trades.append({
                            "symbol": symbol,
                            "entry_date": str(dates[entry_idx])[:10],
                            "exit_date": str(dates[i])[:10],
                            "entry_price": entry_price,
                            "exit_price": sell_price,
                            "shares": sell_shares,
                            "pnl_pct": pnl_pct * 100,
                            "pnl_amount": sell_proceeds - sell_shares * entry_price,
                            "days": days_held,
                            "reason": "partial_take_profit",
                        })
                    # After partial sell, continue with trailing stop
                    entry_price = c  # Update cost basis for remaining

                # 4. Consecutive loss freeze
                if consecutive_losses >= 3:
                    exit_signal = "loss_freeze"
                    exit_reason = "3 consecutive losses"

                if exit_signal:
                    sell_price = prev_c * (1 - SLIPPAGE)
                    sell_proceeds = position * sell_price
                    cash += sell_proceeds * (1 - COMMISSION - STAMP_TAX)
                    pnl = sell_proceeds - position * entry_price
                    pnl_pct = (sell_price - entry_price) / entry_price
                    trades.append({
                        "symbol": symbol,
                        "entry_date": str(dates[entry_idx])[:10],
                        "exit_date": str(dates[i])[:10],
                        "entry_price": entry_price,
                        "exit_price": sell_price,
                        "shares": position,
                        "pnl_pct": pnl_pct * 100,
                        "pnl_amount": pnl,
                        "days": days_held,
                        "reason": exit_reason,
                    })
                    if pnl < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    position = 0
                    entry_price = 0
                    entry_date = None

            # ── Buy signal ──
            elif score >= self.score_threshold:
                # Check for sufficient warmup and positive trend
                if i < 60:
                    pass  # Skip, not enough data
                else:
                    # Position size based on Kelly fraction
                    position_pct = self.kelly_fraction * 0.25  # 25% per position
                    invest_amount = cash * position_pct
                    buy_price = prev_c * (1 + SLIPPAGE)
                    shares = int(invest_amount / buy_price)
                    if shares >= 100:  # Minimum 1 lot (100 shares)
                        cost = shares * buy_price * (1 + COMMISSION)
                        if cost <= cash:
                            cash -= cost
                            position = shares
                            entry_price = buy_price
                            entry_date = dates[i]
                            entry_idx = i

            # ── Daily value ──
            position_value = position * c if position > 0 else 0
            total_value = cash + position_value
            daily_values.append({
                "date": dates[i],
                "cash": cash,
                "position_value": position_value,
                "total": total_value,
                "position": position,
                "score": score,
            })

        # Close any remaining position at last close
        if position > 0:
            last_close = closes[-1]
            sell_proceeds = position * last_close * (1 - SLIPPAGE - STAMP_TAX)
            cash += sell_proceeds
            pnl_pct = (last_close - entry_price) / entry_price
            trades.append({
                "symbol": symbol,
                "entry_date": str(dates[entry_idx])[:10],
                "exit_date": str(dates[-1])[:10],
                "entry_price": entry_price,
                "exit_price": last_close,
                "shares": position,
                "pnl_pct": pnl_pct * 100,
                "pnl_amount": sell_proceeds - position * entry_price,
                "days": n - entry_idx,
                "reason": "closeout",
            })
            total_value = cash

        # ── Calculate metrics ──
        final_value = total_value
        total_return = (final_value - initial_cash) / initial_cash
        n_years = (closes[0] if len(closes) > 0 else 0)  # placeholder
        trading_days = n - 50

        # Annualized return
        if trading_days > 0:
            years = trading_days / 252
            ann_return = (1 + total_return) ** (1 / max(years, 0.25)) - 1
        else:
            ann_return = 0

        # Max drawdown from equity curve
        values = np.array([d["total"] for d in daily_values]) if daily_values else np.array([])
        if len(values) > 0:
            peak = np.maximum.accumulate(values)
            drawdown = (values - peak) / peak
            max_dd = drawdown.min()
        else:
            max_dd = 0

        # Sharpe (daily returns)
        if len(values) > 10:
            daily_r = np.diff(values) / np.maximum(values[:-1], 1)
            mean_r = np.mean(daily_r)
            std_r = np.std(daily_r) if np.std(daily_r) > 0 else 0.0001
            sharpe = np.sqrt(252) * mean_r / std_r
            calmar = ann_return / abs(max_dd) if max_dd != 0 else 999
        else:
            sharpe = calmar = 0

        # Win rate
        wins = sum(1 for t in trades if t["pnl_amount"] > 0)
        total_trades = len(trades)
        win_rate = wins / total_trades if total_trades > 0 else 0

        # Profit factor
        gross_profit = sum(t["pnl_amount"] for t in trades if t["pnl_amount"] > 0)
        gross_loss = abs(sum(t["pnl_amount"] for t in trades if t["pnl_amount"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

        metrics = {
            "symbol": symbol,
            "total_return": total_return,
            "ann_return": ann_return,
            "sharpe": sharpe,
            "calmar": calmar,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            "avg_trade_pnl": np.mean([t["pnl_pct"] for t in trades]) if trades else 0,
            "avg_hold_days": np.mean([t["days"] for t in trades]) if trades else 0,
            "final_value": final_value,
        }

        return {
            "trades": trades,
            "equity": daily_values,
            "metrics": metrics,
        }

    def run_batch(self, stock_data: dict, verbose: bool = True) -> dict:
        """Run simulation on multiple stocks. stock_data: {symbol: DataFrame}"""
        all_metrics = []
        all_trades = []

        for i, (sym, df) in enumerate(stock_data.items()):
            if verbose:
                print(f"  [{i+1}/{len(stock_data)}] Simulating {sym}...", end=" ", flush=True)
            try:
                score_df = score_single_stock(df)
                result = self.run(score_df, sym)
                all_metrics.append(result["metrics"])
                all_trades.extend(result["trades"])
                if verbose:
                    m = result["metrics"]
                    print(f"return={m['total_return']*100:+.1f}%  "
                          f"sharpe={m['sharpe']:.2f}  "
                          f"win={m['win_rate']:.0%}  "
                          f"trades={m['total_trades']}")
            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}")
                all_metrics.append({"symbol": sym, "error": str(e)})

        return {"metrics": all_metrics, "trades": all_trades}


# ──────────────────────────────────────────────────────────────
# Analysis Module
# ──────────────────────────────────────────────────────────────

def analyze_performance(metrics_list: list) -> dict:
    """Compute aggregate performance statistics"""
    valid = [m for m in metrics_list if "error" not in m and m["total_trades"] > 0]
    if not valid:
        return {"error": "no_valid_results"}

    # Portfolio aggregate
    total_trades = sum(m["total_trades"] for m in valid)
    total_wins = sum(int(m["win_rate"] * m["total_trades"]) for m in valid)
    avg_return = np.mean([m["total_return"] for m in valid])
    avg_sharpe = np.mean([m["sharpe"] for m in valid])
    avg_calmar = np.mean([m["calmar"] for m in valid])
    avg_win_rate = total_wins / total_trades if total_trades > 0 else 0
    avg_max_dd = np.mean([m["max_drawdown"] for m in valid])

    # Distribution
    positive_stocks = sum(1 for m in valid if m["total_return"] > 0)
    negative_stocks = sum(1 for m in valid if m["total_return"] < 0)

    # Best and worst
    best = max(valid, key=lambda m: m["sharpe"])
    worst = min(valid, key=lambda m: m["total_return"])

    return {
        "total_stocks": len(metrics_list),
        "valid_stocks": len(valid),
        "total_trades": total_trades,
        "avg_return": avg_return,
        "avg_sharpe": avg_sharpe,
        "avg_calmar": avg_calmar,
        "avg_win_rate": avg_win_rate,
        "avg_max_drawdown": avg_max_dd,
        "positive_stocks": positive_stocks,
        "negative_stocks": negative_stocks,
        "best_stock": {"symbol": best["symbol"], "sharpe": best["sharpe"], "return": best["total_return"]},
        "worst_stock": {"symbol": worst["symbol"], "sharpe": worst["sharpe"], "return": worst["total_return"]},
        "valid_metrics": valid,
    }


def analyze_by_year(trades: list) -> dict:
    """Group trades by year"""
    by_year = defaultdict(lambda: {"trades": 0, "wins": 0, "total_pnl": 0.0, "total_pnl_pct": 0.0})
    for t in trades:
        year = t["entry_date"][:4]
        by_year[year]["trades"] += 1
        if t["pnl_amount"] > 0:
            by_year[year]["wins"] += 1
        by_year[year]["total_pnl"] += t["pnl_amount"]
        by_year[year]["total_pnl_pct"] += t["pnl_pct"]

    result = {}
    for year in sorted(by_year.keys()):
        d = by_year[year]
        result[year] = {
            "trades": d["trades"],
            "win_rate": d["wins"] / d["trades"] if d["trades"] > 0 else 0,
            "total_pnl_pct": d["total_pnl_pct"],
            "avg_pnl": d["total_pnl_pct"] / d["trades"] if d["trades"] > 0 else 0,
        }
    return result


# ──────────────────────────────────────────────────────────────
# Grid Search Module
# ──────────────────────────────────────────────────────────────

def run_grid_search(stock_data: dict, checkpoint_file: str = None) -> list:
    """Run parameter sensitivity grid search"""
    thresholds = PARAM_GRID["score_threshold"]
    kellys = PARAM_GRID["kelly_fraction"]
    stops = PARAM_GRID["stop_loss_days"]

    total = len(thresholds) * len(kellys) * len(stops)
    results = []
    idx = 0

    # Load checkpoint
    completed = set()
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                cp = json.load(f)
                completed = set(tuple(r[:3]) for r in cp.get("grid_completed", []))
                results = cp.get("grid_results", [])
        except Exception:
            pass

    # Pre-score all stocks (expensive, do once)
    print("Pre-scoring all stocks...")
    stock_scores = {}
    for sym, df in stock_data.items():
        try:
            stock_scores[sym] = score_single_stock(df)
        except Exception as e:
            print(f"  ERROR scoring {sym}: {e}")

    print(f"Scored {len(stock_scores)}/{len(stock_data)} stocks")

    for th in thresholds:
        for kf in kellys:
            for sd in stops:
                key = (th, kf, sd)
                idx += 1
                if key in completed:
                    print(f"  [{idx}/{total}] th={th} kf={kf} sd={sd} (cached)")
                    continue

                print(f"  [{idx}/{total}] th={th} kf={kf} sd={sd}...", end=" ", flush=True)
                sim = HistoricalSimulator(
                    score_threshold=th, kelly_fraction=kf,
                    stop_loss_days=sd
                )

                grid_metrics = []
                for sym, score_df in stock_scores.items():
                    result = sim.run(score_df, sym)
                    grid_metrics.append(result["metrics"])

                agg = analyze_performance(grid_metrics)
                entry = {
                    "score_threshold": th,
                    "kelly_fraction": kf,
                    "stop_loss_days": sd,
                    "avg_return": agg.get("avg_return", 0),
                    "avg_sharpe": agg.get("avg_sharpe", 0),
                    "avg_win_rate": agg.get("avg_win_rate", 0),
                    "avg_max_dd": agg.get("avg_max_drawdown", 0),
                    "total_trades": agg.get("total_trades", 0),
                    "positive_stocks": agg.get("positive_stocks", 0),
                }
                results.append(entry)
                print(f"sharpe={entry['avg_sharpe']:.2f} return={entry['avg_return']*100:+.1f}%")

                # Incremental checkpoint
                if checkpoint_file:
                    with open(checkpoint_file, "w") as f:
                        json.dump({
                            "grid_completed": [[r["score_threshold"], r["kelly_fraction"], r["stop_loss_days"]]
                                               for r in results],
                            "grid_results": results,
                        }, f, indent=2)

    return results


# ──────────────────────────────────────────────────────────────
# Report Generation
# ──────────────────────────────────────────────────────────────

def generate_report(agg: dict, all_trades: list, all_metrics: list,
                    grid_results: list = None, output_path: str = None) -> str:
    """Generate markdown report"""
    today = datetime.now().strftime("%Y-%m-%d")
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, f"historical_performance_{today}.md")

    lines = []
    lines.append(f"# 交易系统历史绩效验证报告\n")
    lines.append(f"**报告日期**: {today}  ")
    lines.append(f"**数据范围**: {START_DATE} ~ {END_DATE}  ")
    lines.append(f"**股票数量**: {agg.get('total_stocks', 0)} ({agg.get('valid_stocks', 0)} 有效)\n")

    # ── Section 1: Aggregate Performance ──
    lines.append("## 一、总体绩效摘要\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 有效股票数 | {agg.get('valid_stocks', 0)} |")
    lines.append(f"| 总交易次数 | {agg.get('total_trades', 0)} |")
    lines.append(f"| 平均收益率 | {agg.get('avg_return', 0)*100:+.2f}% |")
    lines.append(f"| 平均 Sharpe | {agg.get('avg_sharpe', 0):.2f} |")
    lines.append(f"| 平均 Calmar | {agg.get('avg_calmar', 0):.2f} |")
    lines.append(f"| 平均胜率 | {agg.get('avg_win_rate', 0)*100:.1f}% |")
    lines.append(f"| 平均最大回撤 | {agg.get('avg_max_drawdown', 0)*100:.1f}% |")
    lines.append(f"| 盈利股票 | {agg.get('positive_stocks', 0)} |")
    lines.append(f"| 亏损股票 | {agg.get('negative_stocks', 0)} |")
    best = agg.get("best_stock", {})
    worst = agg.get("worst_stock", {})
    lines.append(f"| 最佳股票 | {best.get('symbol', '-')} (Sharpe {best.get('sharpe', 0):.2f}) |")
    lines.append(f"| 最差股票 | {worst.get('symbol', '-')} (收益 {worst.get('return', 0)*100:.1f}%) |")
    lines.append("")

    # ── Section 2: Per-Stock Summary ──
    valid = agg.get("valid_metrics", [])
    if valid:
        lines.append("## 二、个股绩效明细\n")
        lines.append("| 股票 | 收益 | Sharpe | Calmar | 最大回撤 | 胜率 | 交易次数 |")
        lines.append("|------|------|--------|--------|----------|------|----------|")
        for m in sorted(valid, key=lambda x: x.get("sharpe", -99), reverse=True):
            lines.append(f"| {m['symbol']} | {m['total_return']*100:+.1f}% | "
                         f"{m['sharpe']:.2f} | {m['calmar']:.2f} | "
                         f"{m['max_drawdown']*100:.1f}% | {m['win_rate']*100:.0f}% | "
                         f"{m['total_trades']} |")
        lines.append("")

    # ── Section 3: By Year ──
    if all_trades:
        by_year = analyze_by_year(all_trades)
        lines.append("## 三、分年度绩效\n")
        lines.append("| 年份 | 交易次数 | 胜率 | 累计盈亏(%) | 平均盈亏(%) |")
        lines.append("|------|----------|------|-------------|-------------|")
        for year, d in sorted(by_year.items()):
            lines.append(f"| {year} | {d['trades']} | {d['win_rate']*100:.0f}% | "
                         f"{d['total_pnl_pct']:+.1f}% | {d['avg_pnl']:+.1f}% |")
        lines.append("")

    # ── Section 4: Best/Worst Trades ──
    if all_trades:
        sorted_trades = sorted(all_trades, key=lambda t: t["pnl_pct"], reverse=True)
        best5 = sorted_trades[:5]
        worst5 = sorted_trades[-5:]

        lines.append("## 四、最佳交易 (Top 5)\n")
        lines.append("| 股票 | 买入 | 卖出 | 持有天数 | 盈亏 | 原因 |")
        lines.append("|------|------|------|----------|------|------|")
        for t in best5:
            lines.append(f"| {t['symbol']} | {t['entry_date']} | {t['exit_date']} | "
                         f"{t['days']} | {t['pnl_pct']:+.1f}% | {t.get('reason','')} |")
        lines.append("")

        lines.append("## 五、最差交易 (Bottom 5)\n")
        lines.append("| 股票 | 买入 | 卖出 | 持有天数 | 盈亏 | 原因 |")
        lines.append("|------|------|------|----------|------|------|")
        for t in worst5:
            lines.append(f"| {t['symbol']} | {t['entry_date']} | {t['exit_date']} | "
                         f"{t['days']} | {t['pnl_pct']:+.1f}% | {t.get('reason','')} |")
        lines.append("")

    # ── Section 5: Parameter Sensitivity ──
    if grid_results:
        lines.append("## 六、参数敏感性分析\n")
        lines.append(f"共测试 {len(grid_results)} 组参数组合\n")
        lines.append("### Top 10 by Sharpe\n")
        lines.append("| Threshold | Kelly | Stop Days | Sharpe | Return | Win Rate | Max DD |")
        lines.append("|-----------|-------|-----------|--------|--------|----------|--------|")
        top10 = sorted(grid_results, key=lambda r: r.get("avg_sharpe", -99), reverse=True)[:10]
        for r in top10:
            lines.append(f"| {r['score_threshold']} | {r['kelly_fraction']} | {r['stop_loss_days']} | "
                         f"{r['avg_sharpe']:.2f} | {r['avg_return']*100:+.1f}% | "
                         f"{r['avg_win_rate']*100:.0f}% | {r['avg_max_dd']*100:.1f}% |")
        lines.append("")

        lines.append("### Top 10 by Return\n")
        lines.append("| Threshold | Kelly | Stop Days | Sharpe | Return | Win Rate |")
        lines.append("|-----------|-------|-----------|--------|--------|----------|")
        top10_r = sorted(grid_results, key=lambda r: r.get("avg_return", -99), reverse=True)[:10]
        for r in top10_r:
            lines.append(f"| {r['score_threshold']} | {r['kelly_fraction']} | {r['stop_loss_days']} | "
                         f"{r['avg_sharpe']:.2f} | {r['avg_return']*100:+.1f}% | "
                         f"{r['avg_win_rate']*100:.0f}% |")
        lines.append("")

    # ── Section 7: Conclusions ──
    lines.append("## 七、结论\n")
    if agg.get("avg_sharpe", 0) > 1.0:
        quality = "优秀"
    elif agg.get("avg_sharpe", 0) > 0.5:
        quality = "良好"
    elif agg.get("avg_sharpe", 0) > 0:
        quality = "一般"
    else:
        quality = "需改进"

    lines.append(f"- **系统评级**: {quality}（Sharpe {agg.get('avg_sharpe', 0):.2f}）")
    lines.append(f"- **盈利概率**: {agg.get('positive_stocks', 0)}/{agg.get('valid_stocks', 0)} 只股票盈利")
    lines.append(f"- **平均最大回撤**: {agg.get('avg_max_drawdown', 0)*100:.1f}%")

    if grid_results:
        best_param = max(grid_results, key=lambda r: r.get("avg_sharpe", -99))
        lines.append(f"- **最优参数**: threshold={best_param['score_threshold']}, "
                     f"kelly={best_param['kelly_fraction']}, stop_days={best_param['stop_loss_days']} "
                     f"(Sharpe {best_param['avg_sharpe']:.2f})")
    lines.append("")

    content = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path


# ──────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="交易系统全历史绩效验证")
    parser.add_argument("--stocks", type=int, default=50, help="股票数量")
    parser.add_argument("--full", action="store_true", help="全量运行")
    parser.add_argument("--from-checkpoint", action="store_true", help="从断点续跑")
    parser.add_argument("--sensitivity", action="store_true", help="参数敏感性分析")
    parser.add_argument("--report-only", action="store_true", help="只生成报告")
    parser.add_argument("--quick", type=int, default=0, help="快速测试N只股票")
    args = parser.parse_args()

    # Build stock list dynamically — prefer stocks in cache
    existing = EXISTING_STOCKS.copy()
    if args.quick:
        existing = existing[:args.quick]
        stocks = existing
    else:
        # Discover CSI500 stocks that are in the data cache
        csv_daily = os.path.join(TS_DIR, "data_cache", "daily")
        cached_codes = set()
        if os.path.exists(csv_daily):
            for f in os.listdir(csv_daily):
                if f.endswith('.csv'):
                    code = f.replace('sh_','').replace('sz_','').replace('.csv','')
                    cached_codes.add(code)
        # Filter CSI500 random to cached only, exclude existing
        csi_available = [c for c in CSI500_RANDOM if c in cached_codes and c not in existing]
        csi_needed = min(args.stocks - len(existing), len(csi_available))
        stocks = existing + csi_available[:csi_needed]
        print(f"CSI500 from cache: {len(csi_available)} available, using {csi_needed}")

    print(f"=== Historical Performance Validation ===")
    print(f"Stocks: {len(stocks)}")
    print(f"Period: {START_DATE} ~ {END_DATE}")
    print(f"Cache: {CACHE_DIR}")
    print()

    # Load checkpoint
    checkpoint_data = {"completed_stocks": [], "step": "data"}
    if args.from_checkpoint and os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            checkpoint_data = json.load(f)
        print(f"Resuming from checkpoint: {len(checkpoint_data['completed_stocks'])} stocks done")
        remaining = [s for s in stocks if s not in checkpoint_data["completed_stocks"]]
        if not remaining:
            print("All stocks completed, proceeding to analysis...")
        stocks = remaining

    # ── Step 1: Data Preparation ──
    print(f"\n[1/4] Loading data for {len(stocks)} stocks...")
    t0 = time.time()
    stock_data = {}
    for i, sym in enumerate(stocks):
        print(f"  [{i+1}/{len(stocks)}] {sym}...", end=" ", flush=True)
        df = load_stock_data(sym)
        if df is not None and len(df) > 200:
            stock_data[sym] = df
            print(f"{len(df)} rows OK")
        else:
            print(f"FAILED")
    print(f"  Done in {time.time()-t0:.1f}s — {len(stock_data)}/{len(stocks)} loaded")

    if args.report_only:
        # Load pre-computed results
        result_path = os.path.join(OUTPUT_DIR, "historical_results.json")
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                cached = json.load(f)
            agg = cached.get("aggregate", {})
            trades = cached.get("trades", [])
            metrics = cached.get("metrics", [])
            grid = cached.get("grid_results", [])
            report_path = generate_report(agg, trades, metrics, grid)
            print(f"\nReport written: {report_path}")
        else:
            print("No cached results found. Run --full first.")
        return

    if not stock_data:
        print("ERROR: No stock data loaded")
        return

    # ── Step 2: Simulation ──
    print(f"\n[2/4] Running simulation...")
    sim = HistoricalSimulator()
    results = sim.run_batch(stock_data)

    # Save intermediate results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "historical_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save trades CSV
    if results["trades"]:
        trades_df = pd.DataFrame(results["trades"])
        csv_path = os.path.join(OUTPUT_DIR, f"historical_trades_{datetime.now().strftime('%Y-%m-%d')}.csv")
        trades_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  Trades saved: {csv_path} ({len(trades_df)} records)")

    # ── Step 3: Analysis ──
    print(f"\n[3/4] Computing aggregate metrics...")
    agg = analyze_performance(results["metrics"])
    print(f"  Avg Sharpe: {agg.get('avg_sharpe', 0):.2f}")
    print(f"  Avg Return: {agg.get('avg_return', 0)*100:+.1f}%")
    print(f"  Win Rate:   {agg.get('avg_win_rate', 0)*100:.1f}%")
    print(f"  +Stocks:    {agg.get('positive_stocks', 0)}")
    print(f"  -Stocks:    {agg.get('negative_stocks', 0)}")

    # ── Step 4: Grid Search (optional) ──
    grid_results = None
    if args.sensitivity:
        print(f"\n[4/4] Parameter sensitivity analysis (up to 80 combos)...")
        grid_checkpoint = os.path.join(OUTPUT_DIR, "grid_checkpoint.json")
        grid_results = run_grid_search(stock_data, grid_checkpoint)
        print(f"  Grid search complete: {len(grid_results)} parameter combinations")
        try:
            from system_state import report_task
            best = max(grid_results, key=lambda r: r.get("avg_sharpe", -99)) if grid_results else {}
            report_task("grid_search",
                        f"completed {len(grid_results)}/80 combos | "
                        f"best_sharpe={best.get('avg_sharpe', 0):.2f} "
                        f"(th={best.get('score_threshold', '?')} "
                        f"kf={best.get('kelly_fraction', '?')} "
                        f"sd={best.get('stop_loss_days', '?')})")
        except Exception:
            pass

    # ── Report ──
    print(f"\nGenerating report...")
    report_path = generate_report(agg, results["trades"], results["metrics"],
                                  grid_results)
    print(f"Report: {report_path}")
    print(f"\n=== DONE ===")


if __name__ == "__main__":
    main()
