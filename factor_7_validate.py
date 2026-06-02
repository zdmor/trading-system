# -*- coding: utf-8 -*-
"""
7因子全量截面IC + 分层回测验证
一次缓存TOP500日线 → 逐截面计算7个因子 → IC/分层 → 汇总输出
输出: output_v2/factor_7_ic.json, factor_7_layer.json, 7factor_summary.md
"""
import sys, os, json, time, math, io
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from data_providers.tushare_provider import _get_pro

# ============================================================
# 因子计算（纯本地，无Tushare调用）
# ============================================================

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    c = np.array(closes, dtype=float)
    deltas = np.diff(c[-period-1:])
    gains = np.sum(np.maximum(deltas, 0))
    losses = np.sum(np.maximum(-deltas, 0))
    if losses < 1e-12:
        return 100.0
    rs = gains / losses
    return float(100.0 - 100.0 / (1.0 + rs))


def calc_atr(highs, lows, closes, period=14):
    n = min(len(highs), len(lows), len(closes))
    if n < 2:
        return max(highs[-1] - lows[-1], 0.01)
    period = min(period, n - 1)
    tr_list = []
    for i in range(1, n):
        hl = abs(highs[i] - lows[i])
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        tr = max(hl, hc, lc)
        tr_list.append(tr)
    use = tr_list[-period:] if len(tr_list) >= period else tr_list
    return float(np.mean(use)) if use else max(highs[-1] - lows[-1], 0.01)


def calc_wyckoff_factor(highs, lows, closes, opens, volumes):
    """威科夫因子值: 基于本地WyckoffAnalyzer"""
    try:
        from scanner import WyckoffAnalyzer
        sigs, phase = WyckoffAnalyzer.analyze_all(
            closes, highs, lows, opens, volumes
        )
        if not sigs:
            return 50.0
        best_sig, raw_score, _ = sigs[0]
        # 简化映射: 信号类型+强度归一化到0-100
        type_map = {"SOS": 85, "Spring": 80, "LPS": 70, "Compression": 60, "Markup": 75, "Upthrust": 35, "EVR": 40}
        base = type_map.get(best_sig.replace("弱", ""), 50)
        if best_sig.startswith("弱"):
            base -= 15
        # 原始分微调
        if 30 <= raw_score <= 100:
            if raw_score >= 80: base = min(100, base + 8)
            elif raw_score <= 50: base = max(0, base - 8)
        return float(max(0, min(100, base)))
    except Exception:
        return 50.0


def calc_risk_reward_factor(closes, highs, atr_val, trend_dir):
    """盈亏比因子: ATR止损 + 前高阻力"""
    price = closes[-1]
    stop = price - 2.5 * atr_val  # ATR止损
    risk = price - stop
    if risk <= 0:
        return 10.0
    
    # 简化: 用前20日最高价作为阻力
    if len(highs) >= 21:
        resistance = max(highs[-21:-1])
        if resistance <= price:
            resistance = price * 1.08
    else:
        resistance = price * 1.08
    
    reward = resistance - price
    rr_ratio = reward / risk
    
    discount = 0.7 if trend_dir == "空头" else 1.0
    if rr_ratio >= 3.0: base = 95
    elif rr_ratio >= 2.0: base = 80
    elif rr_ratio >= 1.5: base = 60
    elif rr_ratio >= 1.0: base = 40
    else: base = 20
    
    return float(max(5, min(100, base * discount)))


def calc_volume_factor(volumes, col_volumes=None):
    """量能因子: 量比 + 放量/缩量"""
    if len(volumes) < 21:
        return 50.0
    current = volumes[-1]
    ma20 = np.mean(volumes[-21:-1])
    ratio = current / max(ma20, 1)
    
    # 用历史分布做百分位
    hist_ratios = []
    for i in range(len(volumes) - 21, len(volumes)):
        ma_i = np.mean(volumes[max(0,i-20):i])
        hist_ratios.append(volumes[i] / max(ma_i, 1))
    
    pct = sum(1 for r in hist_ratios if r < ratio) / max(len(hist_ratios), 1) * 100
    
    if pct >= 90: base = 92
    elif pct >= 75: base = 80
    elif pct >= 50: base = 65
    elif pct >= 25: base = 45
    else: base = 20
    
    
    return float(base)


def calc_candlestick_factor(closes, opens, highs, lows):
    """K线形态因子: 从OHLC简化"""
    try:
        o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
        if len(closes) >= 2:
            po = opens[-2] if len(opens) >= 2 else o
            pc = closes[-2]
            ph = highs[-2] if len(highs) >= 2 else h
            pl = lows[-2] if len(lows) >= 2 else l
        else:
            po, pc, ph, pl = o, c, h, l
        
        body = abs(c - o)
        total = max(h - l, 0.001)
        upper_shadow = h - max(c, o)
        lower_shadow = min(c, o) - l
        
        score = 50.0
        
        # 锤子线 / 倒锤子 (单K)
        if body < total * 0.3:
            if lower_shadow > body * 2 and c > pc:
                score += 20  # 锤子+阳线
            elif upper_shadow > body * 2 and c < pc:
                score -= 15  # 倒锤+阴线
        
        # 吞没形态
        prev_body = abs(pc - po)
        if body > prev_body * 1.2:
            if c > o and pc < po:
                score += 25  # 阳包阴
            elif c < o and pc > po:
                score -= 25  # 阴包阳
        
        # 十字星
        if body < total * 0.1:
            score -= 10
        
        # 大阳线
        if body > total * 0.7 and c > o:
            score += 15
        
        return float(max(0, min(100, score)))
    except Exception:
        return 50.0


def calc_trend_momentum_factor(closes, highs, lows):
    """趋势动量因子: MA排列 + RSI + 价格位置 (自适应窗口)"""
    c = np.array(closes, dtype=float)
    n = len(c)
    if n < 10:
        return 50.0
    price = c[-1]
    
    # 自适应窗口: 用MA(5,20,50)或(5,10,20)取决于数据长度
    if n >= 50:
        ma_short = np.mean(c[-20:]) if n >= 20 else np.mean(c[-5:])
        ma_mid = np.mean(c[-50:])
        ma_long = np.mean(c[-100:]) if n >= 100 else ma_mid
    elif n >= 20:
        ma_short = np.mean(c[-5:])
        ma_mid = np.mean(c[-20:])
        ma_long = ma_mid
    else:
        ma_short = np.mean(c[-5:])
        ma_mid = np.mean(c[-10:])
        ma_long = ma_mid
    
    score = 50.0
    # MA排列
    if price > ma_mid > ma_long: score += 15
    elif price < ma_mid < ma_long: score -= 10
    # MA gap
    if ma_long > 0:
        gap = (ma_mid - ma_long) / ma_long * 100
        score += min(10, max(-5, gap))
    # RSI
    rsi = calc_rsi(closes, min(14, n-1))
    if rsi > 70: score += 10
    elif rsi > 60: score += 5
    elif rsi < 30: score -= 10
    elif rsi < 40: score -= 5
    # 价格 vs 短期MA
    if price > ma_short: score += 5
    else: score -= 5
    
    return float(max(0, min(100, score)))


def calc_relative_strength_factor(stock_closes, market_closes):
    """相对强度因子: 个股vs大盘"""
    if len(stock_closes) < 6 or len(market_closes) < 6:
        return 50.0
    s = np.array(stock_closes, dtype=float)
    m = np.array(market_closes, dtype=float)
    
    def rel_ret(n):
        if len(s) < n + 1 or len(m) < n + 1:
            return 0
        sr = (s[-1] / s[-(n+1)] - 1) * 100
        mr = (m[-1] / m[-(n+1)] - 1) * 100
        return sr - mr
    
    def diff_score(diff):
        if diff > 5: return 12
        if diff > 2: return 6
        if diff > 0.5: return 3
        if diff > -0.5: return 0
        if diff > -2: return -3
        if diff > -5: return -6
        return -12
    
    total = diff_score(rel_ret(1)) * 0.5 + diff_score(rel_ret(5)) * 0.3 + diff_score(rel_ret(20)) * 0.2
    return float(max(0, min(100, 50 + total)))


def calc_volatility_factor(closes):
    """波动率因子: 20日标准差对数收益"""
    if len(closes) < 21:
        return 50.0
    c = np.array(closes[-21:], dtype=float)
    log_ret = np.diff(np.log(c))
    vol = float(np.std(log_ret) * 100)
    
    # 低波动=好 (IC反转逻辑)
    if vol < 1.5: return 90.0
    if vol < 2.0: return 80.0
    if vol < 2.5: return 70.0
    if vol < 3.0: return 60.0
    if vol < 3.5: return 50.0
    if vol < 4.0: return 40.0
    if vol < 5.0: return 30.0
    return 20.0


# ============================================================
# 数据获取
# ============================================================

def get_stock_pool():
    pro = _get_pro()
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,list_date')
    if df is None or df.empty: return []
    cal = pro.trade_cal(exchange='SSE', is_open=1, start_date='20250801', end_date='20260101')
    today = str(cal['cal_date'].iloc[-1])
    pool = []
    for _, row in df.iterrows():
        code = str(row['symbol']).zfill(6)
        ts_code = str(row['ts_code'])
        name = str(row['name'])
        ld = str(row['list_date'])
        if code.startswith('8') or '.BJ' in ts_code: continue
        if 'ST' in name.upper() or '退' in name: continue
        if ld != 'nan' and int(today) - int(ld) < 60: continue
        pool.append({'code': code, 'name': name, 'ts_code': ts_code})
    return pool


def get_market_data():
    """获取上证指数日线 (腾讯API, 快速)"""
    try:
        import requests
        url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh000001,day,,,500,qfq"
        r = requests.get(url, timeout=15)
        data = r.json()
        klines = data.get("data", {}).get("sh000001", {}).get("qfqday") or \
                 data.get("data", {}).get("sh000001", {}).get("day", [])
        if klines:
            dates = []
            o_arr, h_arr, l_arr, c_arr, v_arr = [], [], [], [], []
            for k in klines:
                dates.append(str(k[0]))
                o_arr.append(float(k[1]))
                c_arr.append(float(k[2]))
                h_arr.append(float(k[3]))
                l_arr.append(float(k[4]))
                v_arr.append(float(k[5]))
            return {'dates': dates, 'open': o_arr, 'high': h_arr, 'low': l_arr,
                    'close': c_arr, 'volume': v_arr, 'name': '上证指数'}
    except Exception as e:
        print(f"市场数据获取失败: {e}")
    return None


def bulk_fetch_stocks(pool, start_date, end_date, top_n=500):
    pro = _get_pro()
    # 筛选成交额TOP N
    print("  [1/3] 筛选成交额TOP{}...".format(top_n))
    rank = []
    batch = 100
    total = min(len(pool), 2000)
    for bi in range(0, total, batch):
        for stock in pool[bi:bi+batch]:
            try:
                df = pro.daily(adj='qfq', ts_code=stock['ts_code'], start_date='20251220', end_date='20251231', fields='amount')
                if df is not None and not df.empty:
                    rank.append({**stock, 'avg_amount': float(df['amount'].mean())})
            except: pass
        if (bi + batch) % 200 == 0:
            print("    已筛选 {}/{} 上榜 {}".format(min(bi+batch, total), total, len(rank)))
        time.sleep(0.2)
    rank.sort(key=lambda x: x['avg_amount'], reverse=True)
    top = rank[:top_n]
    print("    TOP{}确认".format(len(top)))
    
    # 批量日线
    print("  [2/3] 拉取日线...")
    stock_data = {}
    for i, stock in enumerate(top):
        try:
            df = pro.daily(adj='qfq', ts_code=stock['ts_code'], start_date=start_date, end_date=end_date,
                           fields='trade_date,open,high,low,close,vol,amount')
            if df is not None and not df.empty and len(df) >= 50:
                df = df.sort_values('trade_date').reset_index(drop=True)
                stock_data[stock['code']] = {
                    'ts_code': stock['ts_code'], 'name': stock['name'],
                    'dates': df['trade_date'].tolist(),
                    'open': df['open'].values.astype(float).tolist(),
                    'high': df['high'].values.astype(float).tolist(),
                    'low': df['low'].values.astype(float).tolist(),
                    'close': df['close'].values.astype(float).tolist(),
                    'vol': df['vol'].values.astype(float).tolist(),
                }
        except: pass
        if (i+1) % 100 == 0:
            print("    日线 {}/{} 成功 {}".format(i+1, len(top), len(stock_data)))
            time.sleep(0.5)
    print("    缓存完成: {}只".format(len(stock_data)))
    return stock_data


def get_cross_section_dates(n=20):
    pro = _get_pro()
    cal = pro.trade_cal(exchange='SSE', is_open=1, start_date='20250901', end_date='20251231')
    all_dates = sorted(cal['cal_date'].tolist())
    step = max(1, len(all_dates) // n)
    return [all_dates[i] for i in range(0, len(all_dates), step)][-n:]


def get_idx_for_date(dates, target_date):
    """在dates列表中找到target_date的索引"""
    td_str = str(target_date)
    if td_str in dates:
        return dates.index(td_str)
    for i in range(len(dates)-1, -1, -1):
        if str(dates[i]) <= td_str:
            return i
    return None


def get_fwd_return(close_arr, idx, fwd_days=5):
    """前视收益"""
    if idx + fwd_days < len(close_arr):
        return (close_arr[idx + fwd_days] - close_arr[idx]) / close_arr[idx]
    return None


# ============================================================
# 因子计算引擎


def calc_orbit_compression_factor(closes):
    try:
        import chaos_theory as cth; import numpy as np
        c = np.array(closes[-60:], dtype=float) if len(closes) >= 60 else np.array(closes, dtype=float)
        traj = cth.phase_space_reconstruct(c, dim=3, tau=1)
        if traj is None or len(traj) < 10: return 50.0
        vols = []
        for d in range(2, min(5, len(traj) - d)):
            subset = traj[-d:]
            vol = np.prod(np.linalg.svd(subset, full_matrices=False)[1])
            vols.append(vol)
        if len(vols) < 2: return 50.0
        r = vols[-1] / max(vols[-2], 1e-10)
        if r < 0.85: return min(100.0, 80.0 + (0.85 - r) / 0.85 * 20.0)
        elif r <= 1.15: return max(0.0, min(100.0, 40.0 + (1.15 - r) / 0.3 * 40.0))
        else: return max(0.0, 40.0 - (r - 1.15) / 0.5 * 40.0)
    except Exception: return 50.0


def calc_lyapunov_factor(closes):
    try:
        import chaos_theory as cth; import numpy as np
        c = np.array(closes, dtype=float)
        le = cth.calc_lyapunov_exponent(c, dim=3, tau=1, window=min(60, len(c)))
        abs_le = abs(le)
        if abs_le < 0.01: return 80.0
        elif abs_le < 0.05: return 50.0 + (0.05 - abs_le) / 0.04 * 30.0
        elif abs_le < 0.1: return 20.0 + (0.1 - abs_le) / 0.05 * 30.0
        else: return max(0.0, 20.0 - abs_le * 50.0)
    except Exception: return 50.0


def calc_hurst_factor(closes):
    try:
        import chaos_theory as cth; import numpy as np
        c = np.array(closes, dtype=float)
        h = cth.calc_hurst_exponent(c)
        if h < 0.3: return max(0.0, h / 0.3 * 20.0)
        elif h < 0.4: return 20.0 + (h - 0.3) / 0.1 * 20.0
        elif h < 0.6: return 40.0 + (h - 0.4) / 0.2 * 20.0
        elif h < 0.7: return 60.0 + (h - 0.6) / 0.1 * 20.0
        else: return min(100.0, 80.0 + (h - 0.7) / 0.3 * 20.0)
    except Exception: return 50.0


def calc_fractal_dim_factor(closes):
    try:
        import chaos_theory as cth; import numpy as np
        c = np.array(closes, dtype=float)
        fd = cth.calc_fractal_dimension(c)
        if fd < 1.3: return min(100.0, 80.0 + (1.3 - fd) / 0.3 * 20.0)
        elif fd < 1.5: return 50.0 + (1.5 - fd) / 0.2 * 30.0
        elif fd < 1.7: return 20.0 + (1.7 - fd) / 0.2 * 30.0
        else: return max(0.0, 20.0 - (fd - 1.7) * 30.0)
    except Exception: return 50.0


def calc_attractor_shape_factor(closes):
    try:
        import chaos_theory as cth; import numpy as np
        c = np.array(closes, dtype=float)
        ratio = cth.attractor_shape_ratio(c)
        if ratio < 1.5: return max(0.0, 30.0 + (ratio - 1.0) / 0.5 * 20.0)
        elif ratio < 3.0: return 50.0 + (ratio - 1.5) / 1.5 * 20.0
        else: return min(100.0, 70.0 + (ratio - 3.0) / 7.0 * 30.0)
    except Exception: return 50.0


def compute_chaos_meta_for_stock(data, idx):
    try:
        import chaos_theory as cth; import numpy as np
        closes = np.array(data['close'][:idx+1], dtype=float)
        s = closes[np.isfinite(closes)]
        if len(s) < 30:
            return {"embed_dim": 3, "embed_tau": 1, "structure_confidence": 0.0,
                    "attractor_ratio": 1.0, "mutual_info_lambda": 10.0,
                    "memory_decay_beta": 2.0, "svd_entropy": 0.5,
                    "snr_db": 0.0, "npe_ratio": 1.0, "rqa_det": 0.0, "rqa_lam": 0.0}
        params = cth.estimate_optimal_params(s)
        rqa = cth.compute_rqa_features(s)
        return {
            "embed_dim": params[0], "embed_tau": params[1],
            "structure_confidence": float(params[2]),
            "attractor_ratio": float(cth.attractor_shape_ratio(s)),
            "mutual_info_lambda": float(cth.compute_mutual_info_decay(s)),
            "memory_decay_beta": float(cth.compute_memory_decay_exponent(s)),
            "svd_entropy": float(cth.compute_svd_entropy(s)),
            "snr_db": float(cth.estimate_snr(s)),
            "npe_ratio": float(cth.compute_npe_ratio(s)),
            "rqa_det": float(rqa["DET"]), "rqa_lam": float(rqa["LAM"]),
        }
    except Exception:
        return {"embed_dim": 3, "embed_tau": 1, "structure_confidence": 0.0,
                "attractor_ratio": 1.0, "mutual_info_lambda": 10.0,
                "memory_decay_beta": 2.0, "svd_entropy": 0.5,
                "snr_db": 0.0, "npe_ratio": 1.0, "rqa_det": 0.0, "rqa_lam": 0.0}

# ============================================================

FACTOR_DEFS = [
    ("wyckoff", calc_wyckoff_factor, ["high", "low", "close", "open", "vol"]),
    ("risk_reward", calc_risk_reward_factor, ["close", "high", "atr", "trend_dir"]),
    ("volume", calc_volume_factor, ["vol", None]),
    ("candlestick", calc_candlestick_factor, ["close", "open", "high", "low"]),
    ("trend_momentum", calc_trend_momentum_factor, ["close", "high", "low"]),
    ("relative_strength", calc_relative_strength_factor, ["close", "market_close"]),
    ("volatility", calc_volatility_factor, ["close"]),
    ("orbit_compression", calc_orbit_compression_factor, ["close"]),
    ("lyapunov", calc_lyapunov_factor, ["close"]),
    ("hurst", calc_hurst_factor, ["close"]),
    ("fractal_dim", calc_fractal_dim_factor, ["close"]),
    ("attractor_shape", calc_attractor_shape_factor, ["close"]),
]


def compute_factors_for_stock(data, idx, market_closes_at_idx):
    """为单只股票计算全部7个因子"""
    cs = data['close']
    hs = data['high']
    ls = data['low']
    vs = data['vol']
    os = data['open']
    
    # 判断趋势方向
    if len(cs) >= 200:
        ma50 = np.mean(cs[max(0,idx-49):idx+1])
        ma200 = np.mean(cs[max(0,idx-199):idx+1])
        if cs[idx] > ma50 > ma200: trend_dir = "多头"
        elif cs[idx] < ma50 < ma200: trend_dir = "空头"
        else: trend_dir = "震荡"
    else:
        trend_dir = "震荡"
    
    # ATR
    slice_h = hs[max(0,idx-14):idx+1]
    slice_l = ls[max(0,idx-14):idx+1]
    slice_c = cs[max(0,idx-14):idx+1]
    atr_val = calc_atr(slice_h, slice_l, slice_c)
    
    factors = {}
    for name, func, _ in FACTOR_DEFS:
        try:
            if name == "wyckoff":
                wc = cs[max(0,idx-210):idx+1]
                wh = hs[max(0,idx-210):idx+1]
                wl = ls[max(0,idx-210):idx+1]
                wo = os[max(0,idx-210):idx+1]
                wv = vs[max(0,idx-210):idx+1]
                factors[name] = func(wh, wl, wc, wo, wv)
            elif name == "risk_reward":
                rh = hs[max(0,idx-20):idx+1]
                rc = cs[max(0,idx-20):idx+1]
                factors[name] = func(rc, rh, atr_val, trend_dir)
            elif name == "volume":
                vv = vs[max(0,idx-120):idx+1]
                factors[name] = func(vv, vs[max(0,idx-120):idx+1])
            elif name == "candlestick":
                so = os[max(0,idx-2):idx+1]
                sh = hs[max(0,idx-2):idx+1]
                sl = ls[max(0,idx-2):idx+1]
                sc = cs[max(0,idx-2):idx+1]
                factors[name] = func(sc, so, sh, sl)
            elif name == "trend_momentum":
                tc = cs[max(0,idx-199):idx+1]
                th = hs[max(0,idx-199):idx+1]
                tl = ls[max(0,idx-199):idx+1]
                factors[name] = func(tc, th, tl)
            elif name == "relative_strength":
                sc = cs[max(0,idx-20):idx+1]
                factors[name] = func(sc, market_closes_at_idx)
            elif name == "volatility":
                vc = cs[max(0,idx-20):idx+1]
                factors[name] = func(vc)
            elif name in ("orbit_compression", "lyapunov", "hurst", "fractal_dim", "attractor_shape"):
                factors[name] = func(cs[:idx+1])
        except Exception:
            factors[name] = 50.0
    return factors


# ============================================================
# 主流程
# ============================================================

def run_all_factors():
    print("="*60)
    print("7因子全量截面IC + 分层回测验证")
    print("="*60)
    
    # 数据准备
    print("\n[Phase 1] 数据准备")
    pool = get_stock_pool()
    print("  股票池: {}只".format(len(pool)))
    
    print("  获取大盘指数...")
    market = get_market_data()
    m_dates = market['dates'] if market else []
    m_close = market['close'] if market else []
    
    # 缓存
    cache_path = os.path.join(os.path.dirname(__file__), 'output_v2', '_7factor_cache.json')
    if os.path.exists(cache_path):
        print("  加载缓存: {}".format(cache_path))
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_raw = json.load(f)
        stock_data = {}
        for code, d in cache_raw.items():
            stock_data[code] = {
                'ts_code': d['ts_code'], 'name': d['n'],
                'dates': d['d'], 'open': d['o'], 'high': d['h'],
                'low': d['l'], 'close': d['c'], 'vol': d['v'],
            }
    else:
        print("  无缓存，拉取新数据...")
        stock_data = bulk_fetch_stocks(pool, '20241201', '20251231', top_n=500)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_save = {}
        for code, d in stock_data.items():
            cache_save[code] = {'ts_code': d['ts_code'], 'n': d['name'],
                                'd': d['dates'], 'o': d['open'], 'h': d['high'],
                                'l': d['low'], 'c': d['close'], 'v': d['vol']}
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_save, f, ensure_ascii=False)
        print("  缓存已保存")
    
    if len(stock_data) < 100:
        print("  缓存不足({})".format(len(stock_data)))
        return
    
    # 截面日期
    cross_dates = get_cross_section_dates(20)
    print("  截面日期: {}".format(len(cross_dates)))
    
    # ========================
    # Phase 2: 截面IC
    # ========================
    print("\n[Phase 2] 截面IC计算")
    
    ic_results = {name: [] for name, _, _ in FACTOR_DEFS}
    
    for di, cross_date in enumerate(cross_dates):
        td_str = str(cross_date)
        print("\n  [{}/{}] {}".format(di+1, len(cross_dates), td_str))
        
        # 找大盘在该截面的数据位置
        m_idx = get_idx_for_date(m_dates, td_str) if m_dates else None
        mc_lines = m_close[:m_idx+1] if m_idx is not None and m_idx < len(m_close) else m_close
        
        factors_all = {name: [] for name, _, _ in FACTOR_DEFS}
        returns_all = []
        
        valid_count = 0
        for code, data in stock_data.items():
            idx = get_idx_for_date(data['dates'], td_str)
            if idx is None or idx < 20:
                continue
            
            fwd = get_fwd_return(data['close'], idx, 5)
            if fwd is None:
                continue
            
            fs = compute_factors_for_stock(data, idx, mc_lines)
            for name in factors_all:
                factors_all[name].append(fs.get(name, 50.0))
            returns_all.append(fwd)
            valid_count += 1
        
        if valid_count < 50:
            print("    样本不足({})".format(valid_count))
            continue
        
        # Rank IC
        from scipy.stats import rankdata
        rank_r = rankdata(returns_all)
        for name in factors_all:
            if len(factors_all[name]) < 50:
                continue
            rank_f = rankdata(factors_all[name])
            if np.std(rank_f) > 1e-12 and np.std(rank_r) > 1e-12:
                ic = float(np.corrcoef(rank_f, rank_r)[0, 1])
            else:
                ic = 0.0
            ic = 0.0 if math.isnan(ic) else ic
            ic_results[name].append(ic)
        
        # 打印摘要
        ic_line = "  ".join("{}:{:+.4f}".format(n[:6], ic_results[n][-1]) for n in ic_results if ic_results[n])
        print("    n={}  {}".format(valid_count, ic_line))
    
    # IC汇总
    print("\n  --- IC汇总 ---")
    ic_summary = {}
    for name in FACTOR_DEFS:
        name = name[0]
        series = ic_results[name]
        if not series:
            ic_summary[name] = {"error": "no_data"}
            continue
        mean_ic = np.mean(series)
        std_ic = np.std(series)
        ir = mean_ic / std_ic if std_ic > 0 else 0
        pos_ratio = sum(1 for i in series if i > 0) / len(series) * 100
        
        # 判定有效性
        abs_ic = abs(mean_ic)
        if abs_ic > 0.03 and abs(ir) > 0.5: grade = "有效"
        elif abs_ic > 0.01: grade = "弱有效"
        else: grade = "无效"
        
        ic_summary[name] = {
            "ic_mean": round(float(mean_ic), 4),
            "ic_std": round(float(std_ic), 4),
            "ir": round(float(ir), 4),
            "ic_pos_ratio": round(pos_ratio, 1),
            "n_cross_sections": len(series),
            "ic_series": [round(float(i), 4) for i in series],
            "grade": grade,
        }
        print("  {}: IC={:+.4f}  std={:.4f}  IR={:+.4f}  pos={:.0f}%  n={}  [{}]".format(
            name, mean_ic, std_ic, ir, pos_ratio, len(series), grade))
    
    # ========================
    # Phase 3: 分层回测
    # ========================
    print("\n[Phase 3] 分层回测")
    
    # 确定缓存数据范围
    sample_dates = sorted(set(d for data in stock_data.values() for d in data['dates']))
    cache_min_date = min(sample_dates)
    cache_max_date = max(sample_dates)
    print("  缓存日期范围: {} ~ {}".format(cache_min_date, cache_max_date))
    
    # 获取所有可能的再平衡日期(每月第一个交易日)
    pro = _get_pro()
    cal = pro.trade_cal(exchange='SSE', is_open=1, start_date=cache_min_date, end_date=cache_max_date)
    months = {}
    for _, row in cal.iterrows():
        d = str(row['cal_date'])
        if d[:6] not in months:
            months[d[:6]] = d
    all_rebal_dates = sorted(months.values())
    print("  日历候选月份: {}个 ({})".format(len(all_rebal_dates), all_rebal_dates))
    
    # 过滤: 只保留缓存中有数据的月份
    def date_in_cache_range(d):
        return get_idx_for_date(sample_dates, d) is not None
    
    rebal_dates = [d for d in all_rebal_dates if date_in_cache_range(d)]
    skipped_pre = len(all_rebal_dates) - len(rebal_dates)
    if skipped_pre > 0:
        print("  [警告] {}/{}个月不在缓存范围被跳过: {}".format(
            skipped_pre, len(all_rebal_dates),
            [d for d in all_rebal_dates if not date_in_cache_range(d)]))
    print("  有效再平衡月份: {}个".format(len(rebal_dates)))
    
    layer_results = {name: [] for name, _, _ in FACTOR_DEFS}
    skipped_months = []  # 记录被跳过的月份及原因
    
    N_LAYERS = 5
    STOCKS_PER_LAYER = 80
    MIN_STOCKS = N_LAYERS * STOCKS_PER_LAYER  # 至少需要400只
    
    planned_periods = len(rebal_dates) - 1
    
    for mi in range(len(rebal_dates) - 1):
        entry_date = rebal_dates[mi]
        exit_date = rebal_dates[mi + 1]
        print("\n  [{}/{}] {} -> {}".format(mi+1, planned_periods, entry_date, exit_date))
        
        # 找大盘截面位置
        m_entry_idx = get_idx_for_date(m_dates, entry_date) if m_dates else None
        mc_entry = m_close[:m_entry_idx+1] if m_entry_idx is not None else m_close
        
        # 计算所有股票的因子值
        stock_factors = {name: [] for name, _, _ in FACTOR_DEFS}
        for code, data in stock_data.items():
            idx = get_idx_for_date(data['dates'], entry_date)
            if idx is None or idx < 20:
                continue
            fs = compute_factors_for_stock(data, idx, mc_entry)
            
            # 获取出场价
            exit_idx = get_idx_for_date(data['dates'], exit_date)
            if exit_idx is None:
                continue
            entry_price = data['close'][idx]
            exit_price = data['close'][exit_idx]
            ret = (exit_price - entry_price) / entry_price
            
            for name in FACTOR_DEFS:
                n = name[0]
                stock_factors[n].append({
                    'code': code, 'factor': fs[n], 'ret': ret
                })
        
        # 检查每个因子的股票数量
        max_stocks = max(len(sf) for sf in stock_factors.values()) if stock_factors else 0
        
        if max_stocks < MIN_STOCKS:
            skip_reason = "样本不足: 最大{}只 < 需要{}".format(max_stocks, MIN_STOCKS)
            skipped_months.append((entry_date, exit_date, skip_reason))
            print("    [跳过] {}".format(skip_reason))
            # 即使跳过，也要记录available_stocks供调试
            for name in FACTOR_DEFS:
                n = name[0]
                count = len(stock_factors.get(n, []))
                if count < MIN_STOCKS:
                    print("      {}: {}只 (需{}只)".format(n, count, MIN_STOCKS))
            continue
        
        # 各因子独立计算分层
        valid_this_period = []
        for name in FACTOR_DEFS:
            n = name[0]
            sf = stock_factors[n]
            if len(sf) < MIN_STOCKS:
                skipped_months.append((entry_date, exit_date, "{}: 仅{}只".format(n, len(sf))))
                print("    [警告] {}: {}只 < {}只, 该因子本期跳过".format(n, len(sf), MIN_STOCKS))
                continue
            sf.sort(key=lambda x: x['factor'], reverse=True)
            per_layer = min(len(sf) // N_LAYERS, STOCKS_PER_LAYER)
            layer_rets = []
            for li in range(N_LAYERS):
                layer_stocks = sf[li * per_layer:(li + 1) * per_layer]
                avg_ret = np.mean([s['ret'] for s in layer_stocks])
                layer_rets.append(float(avg_ret))
            layer_results[n].append(layer_rets)
            valid_this_period.append(n)
        
        print("    完成, 有效因子: {}/7 可用股票: {}".format(len(valid_this_period), max_stocks))
    
    # 分层汇总
    layer_summary = {}
    print("\n  --- 分层汇总 (计划{}期, 实际按因子独立统计) ---".format(planned_periods))
    if skipped_months:
        print("  [跳过明细] 共{}个月被跳过:".format(len(skipped_months)))
        for entry_d, exit_d, reason in skipped_months:
            print("    {}->{}: {}".format(entry_d, exit_d, reason))
    
    for name in FACTOR_DEFS:
        n = name[0]
        monthly = layer_results[n]
        if not monthly:
            layer_summary[n] = {
                "n_months": 0, "planned_periods": planned_periods,
                "error": "no_data -- 所有月份样本不足",
            }
            print("  {}: 无数据".format(n))
            continue
        
        n_actual = len(monthly)
        # aggregate across months
        layer_avg = []
        for li in range(N_LAYERS):
            li_rets = [m[li] for m in monthly if len(m) > li]
            if li_rets:
                layer_avg.append(float(np.mean(li_rets)))
            else:
                layer_avg.append(0.0)
        
        # 单调性
        decreasing = all(layer_avg[i] >= layer_avg[i+1] for i in range(len(layer_avg)-1))
        increasing = all(layer_avg[i] <= layer_avg[i+1] for i in range(len(layer_avg)-1))
        if decreasing: mono = "下降(有效)"
        elif increasing: mono = "上升(反转)"
        else: mono = "无单调"
        
        layer_summary[n] = {
            "n_months": n_actual,
            "planned_periods": planned_periods,
            "completeness": "{}/{}".format(n_actual, planned_periods),
            "layer_avg_returns": layer_avg,
            "top_bottom_diff": round(layer_avg[0] - layer_avg[-1], 6) if len(layer_avg) >= 2 else 0,
            "monotonicity": mono,
            "monthly_details": [[round(float(r), 6) for r in m] for m in monthly],
        }
        print("  {} ({}): {}  layers={}  diff={:+.2%}".format(
            n, layer_summary[n]['completeness'], mono, 
            [round(x*100,1) for x in layer_avg],
            layer_avg[0]-layer_avg[-1] if layer_avg else 0))
    
    # ========================
    # Phase 4: 输出
    # ========================
    out_dir = os.path.join(os.path.dirname(__file__), 'output_v2')
    os.makedirs(out_dir, exist_ok=True)
    
    # IC JSON
    ic_path = os.path.join(out_dir, 'factor_7_ic.json')
    with open(ic_path, 'w', encoding='utf-8') as f:
        json.dump({
            "description": "7因子截面Rank IC验证",
            "method": "截面Spearman Rank IC (因子排名 vs 未来5日收益排名)",
            "sample": "A股成交额TOP500, 20个截面日期(2025Q3-Q4)",
            "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "factors": ic_summary,
        }, f, ensure_ascii=False, indent=2)
    print("\n  IC结果: {}".format(ic_path))
    
    # Layer JSON
    layer_path = os.path.join(out_dir, 'factor_7_layer.json')
    with open(layer_path, 'w', encoding='utf-8') as f:
        json.dump({
            "description": "7因子月度分层回测",
            "method": "每月初按因子值分5层(每层80只), 持有1月",
            "planned_periods": len(rebal_dates)-1,
            "rebalance_dates": rebal_dates,
            "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "factors": layer_summary,
        }, f, ensure_ascii=False, indent=2)
    print("  分层结果: {}".format(layer_path))
    
    # 汇总报告
    report = []
    report.append("# 7因子全量验证报告\n")
    report.append("> 运行时间: {} | 样本: A股TOP500 | 截面: {}个日期\n".format(
        time.strftime("%Y-%m-%d %H:%M"), len(cross_dates)))
    report.append("\n## IC结果\n\n")
    report.append("| 因子 | IC均值 | IC标准差 | IR | 胜率 | 截面数 | 判定 |\n")
    report.append("|------|--------|----------|-----|------|--------|------|\n")
    for name in FACTOR_DEFS:
        n = name[0]
        s = ic_summary.get(n, {})
        if "error" in s:
            report.append("| {} | - | - | - | - | - | 无数据 |\n".format(n))
        else:
            report.append("| {} | {:+.4f} | {:.4f} | {:+.4f} | {:.0f}% | {} | **{}** |\n".format(
                n, s['ic_mean'], s['ic_std'], s['ir'], s['ic_pos_ratio'], s['n_cross_sections'], s['grade']))
    
    report.append("\n## 分层回测\n\n")
    report.append("| 因子 | 有效期 | Layer1(高) | Layer2 | Layer3 | Layer4 | Layer5(低) | Top-Bottom | 单调性 |\n")
    report.append("|------|--------|-----------|--------|--------|--------|-----------|------------|--------|\n")
    for name in FACTOR_DEFS:
        n = name[0]
        s = layer_summary.get(n, {})
        if "error" in s:
            report.append("| {} | - | - | - | - | - | - | - | {} |\n".format(n, s.get('error','')))
        else:
            avgs = s.get('layer_avg_returns', [0]*5)
            report.append("| {} | {} | {:+.2%} | {:+.2%} | {:+.2%} | {:+.2%} | {:+.2%} | {:+.2%} | {} |\n".format(
                n, s.get('completeness','?'), avgs[0], avgs[1], avgs[2], avgs[3], avgs[4], s.get('top_bottom_diff', 0), s.get('monotonicity', '-')))
    
    report.append("\n## 综合判定\n\n")
    report.append("| 因子 | IC判定 | 分层判定(完整性) | 最终结论 |\n")
    report.append("|------|--------|----------|----------|\n")
    for name in FACTOR_DEFS:
        n = name[0]
        ic_g = ic_summary.get(n, {}).get('grade', '无数据')
        ls = layer_summary.get(n, {})
        l_g = ls.get('monotonicity', '无单调') if 'monotonicity' in ls else '无数据'
        comp = ls.get('completeness', '?')
        if ic_g == "有效": final = "✅ 有效"
        elif ic_g == "弱有效": final = "⚠ 弱有效"
        else: final = "❌ 无效"
        report.append("| {} | {} | {} ({}) | {} |\n".format(n, ic_g, l_g, comp, final))
    
    md_path = os.path.join(out_dir, '7factor_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    print("  汇总报告: {}".format(md_path))
    
    # 自检
    print("\n" + "="*60)
    print("自检")
    total_factors = len(FACTOR_DEFS)
    ok_ic = sum(1 for n,_,_ in FACTOR_DEFS if ic_summary.get(n, {}).get('grade') not in ('无效','无数据'))
    ok_layer = sum(1 for n,_,_ in FACTOR_DEFS if layer_summary.get(n, {}).get('monotonicity','无单调') not in ('无单调','无数据'))
    print("  IC有效/弱有效: {}/{}".format(ok_ic, total_factors))
    print("  分层有单调性: {}/{}".format(ok_layer, total_factors))
    
    return ic_summary, layer_summary


if __name__ == "__main__":
    try:
        run_all_factors()
    except Exception as e:
        print("ERROR: {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)